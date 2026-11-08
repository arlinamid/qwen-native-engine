//! The browser-facing engine: model session, KV-cache handling, generation loop and agent loop.

use std::rc::Rc;

use js_sys::{Array, ArrayBuffer, BigInt64Array, Float32Array, Function, Object, Promise, Reflect, Uint16Array};
use llguidance::Constraint;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokenizers::Tokenizer;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::{
    grammar::ToolGrammar,
    ortjs::{InferenceSession, Tensor, get, now_ms, read_gpu_range},
    prompt::{Message, render, render_system_block},
    sampling::{Sampler, SamplingOptions},
    toolcall::{ToolCall, parse_output, validate_call},
};

fn js_err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

fn to_js<T: Serialize>(value: &T) -> Result<JsValue, JsValue> {
    value.serialize(&serde_wasm_bindgen::Serializer::json_compatible()).map_err(js_err)
}

fn from_js<T: for<'de> Deserialize<'de>>(value: JsValue) -> Result<T, JsValue> {
    serde_wasm_bindgen::from_value(value).map_err(js_err)
}

#[wasm_bindgen(start)]
fn start() {
    console_error_panic_hook::set_once();
}

#[derive(Deserialize)]
#[serde(default)]
struct LoadOptions {
    /// Folder of the ONNX files, with a trailing slash.
    base_url: String,
    /// File name suffix of the quantization, for example "q4f16".
    dtype: String,
    /// "webgpu" or "wasm". The embedding table always runs on WebGPU: the browser's CPU build has
    /// no kernel for its quantized lookup.
    device: String,
    /// A second decoder session for prompt processing, typically "wasm": the CPU processes a
    /// whole prompt in one batch far faster than the GPU's per-token path.
    prefill_device: Option<String>,
    /// "gpu-buffer" or "cpu". Forced to "cpu" on the wasm device.
    kv_location: String,
    logits_location: String,
    n_layers: usize,
    full_attention_interval: usize,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self {
            base_url: "/models/qwen3.5-0.8b/onnx/".into(),
            dtype: "q4f16".into(),
            device: "webgpu".into(),
            prefill_device: None,
            kv_location: "gpu-buffer".into(),
            logits_location: "gpu-buffer".into(),
            n_layers: 24,
            full_attention_interval: 4,
        }
    }
}

/// `preferredOutputLocation` must name real outputs, and they are only known after loading, so
/// they are derived from the architecture: every Nth layer is full attention, the rest linear.
fn decoder_locations(options: &LoadOptions, kv_location: &str, logits_location: &str) -> Value {
    let mut locations = serde_json::Map::new();
    for i in 0..options.n_layers {
        let names: [String; 2] = if (i + 1) % options.full_attention_interval == 0 {
            [format!("present.{i}.key"), format!("present.{i}.value")]
        } else {
            [format!("present_conv.{i}"), format!("present_recurrent.{i}")]
        };
        for name in names {
            locations.insert(name, json!(kv_location));
        }
    }
    locations.insert("logits".into(), json!(logits_location));
    Value::Object(locations)
}

async fn open_session(base_url: &str, file: &str, device: &str, locations: Value) -> Result<InferenceSession, JsValue> {
    let data_file = format!("{file}_data");
    let options = json!({
        "executionProviders": [device],
        "graphOptimizationLevel": "all",
        "preferredOutputLocation": locations,
        "externalData": [{ "path": data_file, "data": format!("{base_url}{data_file}") }],
    });
    let options = js_sys::JSON::parse(&options.to_string())?;
    InferenceSession::create(&JsValue::from_str(&format!("{base_url}{file}")), &options).await
}

/// One recurrent or KV input of the decoder, and the output that replaces it on the next step.
struct StateSlot {
    past: String,
    present: String,
    dtype: String,
    /// Shape of the initial state. A growing KV-cache starts with length 0, a fixed one as zeros.
    dims: Vec<f64>,
}

fn state_slots(decoder: &InferenceSession) -> Result<Vec<StateSlot>, JsValue> {
    let mut slots = Vec::new();
    for meta in decoder.input_metadata().iter() {
        let name = get(&meta, "name")?.as_string().unwrap_or_default();
        let present = if let Some(rest) = name.strip_prefix("past_key_values.") {
            format!("present.{rest}")
        } else if let Some(rest) = name.strip_prefix("past_conv.") {
            format!("present_conv.{rest}")
        } else if let Some(rest) = name.strip_prefix("past_recurrent.") {
            format!("present_recurrent.{rest}")
        } else {
            continue;
        };
        let dtype = get(&meta, "type")?.as_string().ok_or("state input without a type")?;
        let shape = Array::from(&get(&meta, "shape")?);
        let dims = shape
            .iter()
            .map(|d| d.as_f64().unwrap_or(if d.as_string().as_deref() == Some("batch_size") { 1.0 } else { 0.0 }))
            .collect();
        slots.push(StateSlot { past: name, present, dtype, dims });
    }
    Ok(slots)
}

#[derive(Clone, Copy, PartialEq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Mode {
    /// No grammar.
    Free,
    /// Free text until `<tool_call>`, grammar afterwards.
    Lazy,
    /// Starts inside `<tool_call>`, so a call is mandatory.
    Required,
}

#[derive(Clone, Deserialize)]
#[serde(default)]
struct GenOptions {
    max_new_tokens: usize,
    /// Prompt tokens per forward pass; 0 sends the prompt at once. When unset: the whole prompt
    /// with a CPU prefill session, else 1, because on WebGPU the decoder has a fast path for one
    /// token while batches go through a `Scan` loop per linear-attention layer (measured >2x slower).
    prefill_chunk: Option<usize>,
    sampling: SamplingOptions,
}

impl Default for GenOptions {
    fn default() -> Self {
        Self { max_new_tokens: 128, prefill_chunk: None, sampling: SamplingOptions::default() }
    }
}

#[derive(Default, Serialize)]
struct GenStats {
    prompt_tokens: usize,
    cached_tokens: usize,
    new_tokens: usize,
    grammar_init_ms: f64,
    prefill_ms: f64,
    ttft_ms: f64,
    decode_ms: f64,
    tokens_per_s: f64,
    embed_ms: f64,
    run_ms: f64,
    read_ms: f64,
    sample_ms: f64,
    mask_ms: f64,
    stop: String,
}

#[derive(Default)]
struct StepTimes {
    embed_ms: f64,
    run_ms: f64,
    read_ms: f64,
}

/// Tensors feeding the next step. Snapshots owned by the prefix cache must outlive the step.
struct Kv {
    tensors: Vec<Tensor>,
    disposable: bool,
}

impl Kv {
    fn release(&mut self) {
        if self.disposable {
            self.tensors.iter().for_each(Tensor::dispose);
        }
        self.tensors.clear();
    }
}

struct PrefixCache {
    ids: Vec<u32>,
    tensors: Vec<Tensor>,
}

#[wasm_bindgen]
pub struct Engine {
    embed: InferenceSession,
    decoder: InferenceSession,
    prefill: Option<InferenceSession>,
    tokenizer: Rc<Tokenizer>,
    grammar: Option<ToolGrammar>,
    slots: Vec<StateSlot>,
    stop_tokens: [u32; 2],
    tool_call_open: u32,
    prefix: Option<PrefixCache>,
    logits: Vec<f32>,
    info: Value,
}

#[wasm_bindgen]
impl Engine {
    /// `tokenizer_json` holds the bytes of `tokenizer.json`.
    pub async fn load(tokenizer_json: Vec<u8>, options: JsValue) -> Result<Engine, JsValue> {
        let options: LoadOptions = if options.is_undefined() { LoadOptions::default() } else { from_js(options)? };
        let tokenizer = Tokenizer::from_bytes(&tokenizer_json).map_err(js_err)?;
        let token = |name: &str| tokenizer.token_to_id(name).ok_or_else(|| js_err(format!("tokenizer has no {name} token")));
        let stop_tokens = [token("<|im_end|>")?, token("<|endoftext|>")?];
        let tool_call_open = token("<tool_call>")?;

        let on_gpu = options.device == "webgpu";
        let kv_location = if on_gpu { options.kv_location.clone() } else { "cpu".to_string() };
        let logits_location = if on_gpu { options.logits_location.clone() } else { "cpu".to_string() };
        let prefill_device = options.prefill_device.clone().filter(|d| *d != options.device);
        // A CPU prefill session needs CPU embeddings; the GPU decoder uploads 4 kB per token.
        let embed_location = if on_gpu && prefill_device.is_none() { "gpu-buffer" } else { "cpu" };

        let embed_file = format!("embed_tokens_{}.onnx", options.dtype);
        let embed = open_session(&options.base_url, &embed_file, "webgpu", json!({ "inputs_embeds": embed_location })).await?;
        let decoder_file = format!("decoder_model_merged_{}.onnx", options.dtype);
        let locations = decoder_locations(&options, &kv_location, &logits_location);
        let decoder = open_session(&options.base_url, &decoder_file, &options.device, locations).await?;
        let prefill = match &prefill_device {
            Some(device) => Some(open_session(&options.base_url, &decoder_file, device, json!("cpu")).await?),
            None => None,
        };
        let slots = state_slots(&decoder)?;

        let info = json!({
            "dtype": options.dtype,
            "device": options.device,
            "prefill_device": prefill_device,
            "kv_location": kv_location,
            "logits_location": logits_location,
            "state_tensors": slots.len(),
            "state_dtype": slots.first().map(|s| s.dtype.clone()),
        });
        Ok(Engine {
            embed,
            decoder,
            prefill,
            tokenizer: Rc::new(tokenizer),
            grammar: None,
            slots,
            stop_tokens,
            tool_call_open,
            prefix: None,
            logits: Vec::new(),
            info,
        })
    }

    pub fn info(&self) -> Result<JsValue, JsValue> {
        to_js(&self.info)
    }

    /// Plain completion of an already formatted prompt, for benchmarks and diagnostics.
    pub async fn generate(&mut self, prompt: String, options: JsValue, on_token: Option<Function>) -> Result<JsValue, JsValue> {
        let options: GenOptions = from_js(options)?;
        let ids = self.encode(&prompt)?;
        let (text, stats) = self.generate_ids(&ids, 0, &options, &[], Mode::Free, on_token.as_ref()).await?;
        to_js(&json!({ "text": text, "stats": stats }))
    }

    /// Runs the agent loop. `tool_exec(name, argsJson)` may return a string or a promise of one.
    pub async fn agent(&mut self, request: JsValue, tool_exec: Function, on_event: Option<Function>) -> Result<JsValue, JsValue> {
        let request: AgentRequest = from_js(request)?;
        let outcome = self.run_agent(request, &tool_exec, on_event.as_ref()).await?;
        to_js(&outcome)
    }

    /// Builds the grammar tables now, so the first tool call does not pay for them.
    pub fn prepare_grammar(&mut self) -> Result<f64, JsValue> {
        let t0 = now_ms();
        self.ensure_grammar()?;
        Ok(now_ms() - t0)
    }

    pub fn clear_prefix_cache(&mut self) {
        if let Some(prefix) = self.prefix.take() {
            prefix.tensors.iter().for_each(Tensor::dispose);
        }
    }
}

impl Engine {
    fn encode(&self, text: &str) -> Result<Vec<u32>, JsValue> {
        Ok(self.tokenizer.encode(text, false).map_err(js_err)?.get_ids().to_vec())
    }

    fn ensure_grammar(&mut self) -> Result<(), JsValue> {
        if self.grammar.is_none() {
            self.grammar = Some(ToolGrammar::new(&self.tokenizer).map_err(js_err)?);
        }
        Ok(())
    }

    fn new_constraint(&self, tools: &[Value]) -> Result<Constraint, JsValue> {
        self.grammar.as_ref().ok_or("grammar not ready")?.constraint(tools).map_err(js_err)
    }

    fn empty_kv(&self) -> Result<Kv, JsValue> {
        let mut tensors = Vec::with_capacity(self.slots.len());
        for slot in &self.slots {
            let len = slot.dims.iter().product::<f64>() as u32;
            let data: JsValue = match slot.dtype.as_str() {
                "float16" => Uint16Array::new_with_length(len).into(),
                _ => Float32Array::new_with_length(len).into(),
            };
            let dims: Array = slot.dims.iter().map(|d| JsValue::from_f64(*d)).collect();
            tensors.push(Tensor::new(&slot.dtype, &data, &dims)?);
        }
        Ok(Kv { tensors, disposable: true })
    }

    fn int64_tensor(values: Vec<i64>, dims: &[f64]) -> Result<Tensor, JsValue> {
        let dims: Array = dims.iter().map(|d| JsValue::from_f64(*d)).collect();
        Tensor::new("int64", &BigInt64Array::from(values.as_slice()).into(), &dims)
    }

    /// One forward pass. Replaces `kv` with the new state, and leaves the last row of the logits
    /// in `self.logits` when `want_logits` is set. With `prefill` set the prefill session runs, if
    /// there is one; the GPU decoder uploads its CPU state on the first decoding step.
    async fn step(&mut self, ids: &[u32], kv: &mut Kv, past_len: usize, want_logits: bool, prefill: bool, times: &mut StepTimes) -> Result<(), JsValue> {
        let n = ids.len();
        let total = past_len + n;

        let t_embed = now_ms();
        let embed_feeds = Object::new();
        let input_ids = Self::int64_tensor(ids.iter().map(|&t| i64::from(t)).collect(), &[1.0, n as f64])?;
        Reflect::set(&embed_feeds, &JsValue::from_str("input_ids"), &input_ids)?;
        let embedded = self.embed.run(&embed_feeds).await?;
        let embeds = Tensor::unchecked_from_js(get(&embedded, "inputs_embeds")?);
        times.embed_ms += now_ms() - t_embed;

        let feeds = Object::new();
        let set = |name: &str, tensor: &Tensor| Reflect::set(&feeds, &JsValue::from_str(name), tensor);
        set("inputs_embeds", &embeds)?;
        set("attention_mask", &Self::int64_tensor(vec![1; total], &[1.0, total as f64])?)?;
        // One row per multimodal rope axis; for text all three are the plain position.
        let positions: Vec<i64> = (past_len as i64..total as i64).cycle().take(3 * n).collect();
        set("position_ids", &Self::int64_tensor(positions, &[3.0, 1.0, n as f64])?)?;
        for (slot, tensor) in self.slots.iter().zip(&kv.tensors) {
            set(&slot.past, tensor)?;
        }

        let t0 = now_ms();
        let session = if prefill { self.prefill.as_ref().unwrap_or(&self.decoder) } else { &self.decoder };
        let outputs = session.run(&feeds).await?;
        times.run_ms += now_ms() - t0;
        embeds.dispose();

        let next = self
            .slots
            .iter()
            .map(|slot| get(&outputs, &slot.present).map(Tensor::unchecked_from_js))
            .collect::<Result<Vec<_>, _>>()?;
        kv.release();
        kv.tensors = next;
        kv.disposable = true;

        let logits = Tensor::unchecked_from_js(get(&outputs, "logits")?);
        if want_logits {
            let t1 = now_ms();
            self.read_last_logits(&logits).await?;
            times.read_ms += now_ms() - t1;
        }
        logits.dispose();
        Ok(())
    }

    async fn read_last_logits(&mut self, logits: &Tensor) -> Result<(), JsValue> {
        let dims: Vec<f64> = logits.dims().iter().filter_map(|v| v.as_f64()).collect();
        let (rows, vocab) = (dims[1], dims[2]);
        let dtype = logits.dtype();
        let elem_bytes = if dtype == "float16" { 2.0 } else { 4.0 };
        let row_offset = (rows - 1.0) * vocab * elem_bytes;

        let (buffer, offset): (ArrayBuffer, f64) = if logits.location() == "gpu-buffer" {
            (read_gpu_range(logits, row_offset, vocab * elem_bytes).await?, 0.0)
        } else {
            let data = logits.get_data().await?;
            let base = get(&data, "byteOffset")?.as_f64().unwrap_or(0.0);
            (get(&data, "buffer")?.unchecked_into(), base + row_offset)
        };

        let n = vocab as usize;
        self.logits.resize(n, 0.0);
        if dtype == "float16" {
            let mut raw = vec![0u16; n];
            Uint16Array::new_with_byte_offset_and_length(&buffer, offset as u32, n as u32).copy_to(&mut raw);
            for (out, half) in self.logits.iter_mut().zip(raw) {
                *out = half_to_f32(half);
            }
        } else {
            Float32Array::new_with_byte_offset_and_length(&buffer, offset as u32, n as u32).copy_to(&mut self.logits);
        }
        Ok(())
    }

    /// Returns a KV-cache positioned after `ids`, computing and keeping a snapshot on a miss.
    async fn prefix_kv(&mut self, ids: &[u32], chunk: usize, stats: &mut GenStats) -> Result<Kv, JsValue> {
        if self.prefix.as_ref().is_some_and(|p| p.ids == ids) {
            stats.cached_tokens = ids.len();
        } else {
            self.clear_prefix_cache();
            let mut kv = self.empty_kv()?;
            let mut times = StepTimes::default();
            let mut pos = 0;
            for batch in ids.chunks(chunk) {
                self.step(batch, &mut kv, pos, false, true, &mut times).await?;
                pos += batch.len();
            }
            self.prefix = Some(PrefixCache { ids: ids.to_vec(), tensors: std::mem::take(&mut kv.tensors) });
        }
        let tensors = self.prefix.as_ref().map(|p| p.tensors.clone()).unwrap_or_default();
        Ok(Kv { tensors, disposable: false })
    }

    async fn generate_ids(
        &mut self,
        ids: &[u32],
        prefix_len: usize,
        options: &GenOptions,
        tools: &[Value],
        mode: Mode,
        on_token: Option<&Function>,
    ) -> Result<(String, GenStats), JsValue> {
        // Without a forward pass the logits would still be those of the previous request.
        if ids.is_empty() {
            return Err(js_err("empty prompt"));
        }
        let mut stats = GenStats { prompt_tokens: ids.len(), ..Default::default() };
        let mode = if tools.is_empty() { Mode::Free } else { mode };

        if mode != Mode::Free {
            let t = now_ms();
            self.ensure_grammar()?;
            stats.grammar_init_ms = now_ms() - t;
        }
        let mut constraint = if mode == Mode::Required { Some(self.new_constraint(tools)?) } else { None };

        let t0 = now_ms();
        let chunk = match options.prefill_chunk {
            Some(0) => ids.len().max(1),
            Some(n) => n,
            None if self.prefill.is_some() => ids.len().max(1),
            None => 1,
        };
        let mut kv = if prefix_len > 0 { self.prefix_kv(&ids[..prefix_len], chunk, &mut stats).await? } else { self.empty_kv()? };

        let mut pos = prefix_len;
        let rest = &ids[prefix_len..];
        let batches = rest.len().div_ceil(chunk);
        let mut prefill_times = StepTimes::default();
        for (i, batch) in rest.chunks(chunk).enumerate() {
            self.step(batch, &mut kv, pos, i + 1 == batches, true, &mut prefill_times).await?;
            pos += batch.len();
        }
        stats.prefill_ms = now_ms() - t0;

        let tokenizer = Rc::clone(&self.tokenizer);
        let mut stream = tokenizer.decode_stream(false);
        let mut sampler = Sampler::new(options.sampling.clone());
        let mut text = String::new();
        let mut times = StepTimes::default();
        let t_decode = now_ms();
        stats.stop = "max_tokens".into();

        while stats.new_tokens < options.max_new_tokens {
            let t_mask = now_ms();
            let mask = match constraint.as_mut() {
                Some(c) => {
                    let step = c.compute_mask().map_err(js_err)?;
                    if step.is_stop() {
                        stats.stop = "grammar_end".into();
                        break;
                    }
                    step.sample_mask.as_ref()
                }
                None => None,
            };
            let t_sample = now_ms();
            stats.mask_ms += t_sample - t_mask;
            let token = sampler.try_sample(&self.logits, mask).map_err(js_err)?;
            stats.sample_ms += now_ms() - t_sample;
            if stats.new_tokens == 0 {
                stats.ttft_ms = now_ms() - t0;
            }

            if self.stop_tokens.contains(&token) {
                stats.stop = "eos".into();
                break;
            }
            stats.new_tokens += 1;
            if let Some(piece) = stream.step(token).map_err(js_err)? {
                if let Some(callback) = on_token {
                    callback.call1(&JsValue::NULL, &JsValue::from_str(&piece))?;
                }
                text.push_str(&piece);
            }

            let mut grammar_done = false;
            if let Some(c) = constraint.as_mut() {
                let t = now_ms();
                grammar_done = c.commit_token(Some(token)).map_err(js_err)?.stop;
                stats.mask_ms += now_ms() - t;
            } else if mode == Mode::Lazy && token == self.tool_call_open {
                let t = now_ms();
                constraint = Some(self.new_constraint(tools)?);
                stats.mask_ms += now_ms() - t;
            }
            if grammar_done {
                stats.stop = "grammar_end".into();
                break;
            }
            if stats.new_tokens >= options.max_new_tokens {
                break;
            }

            self.step(&[token], &mut kv, pos, true, false, &mut times).await?;
            pos += 1;
        }
        kv.release();

        stats.decode_ms = now_ms() - t_decode;
        let steps = stats.new_tokens.saturating_sub(1) as f64;
        if steps > 0.0 {
            stats.tokens_per_s = steps / (stats.decode_ms / 1000.0);
            stats.embed_ms = times.embed_ms / steps;
            stats.run_ms = times.run_ms / steps;
            stats.read_ms = times.read_ms / steps;
        }
        if stats.new_tokens > 0 {
            stats.sample_ms /= stats.new_tokens as f64;
            stats.mask_ms /= stats.new_tokens as f64;
        }
        Ok((text, stats))
    }

    async fn run_agent(&mut self, request: AgentRequest, tool_exec: &Function, on_event: Option<&Function>) -> Result<AgentOutcome, JsValue> {
        let tools = request.tools;
        let mut messages = Vec::new();
        if let Some(system) = request.system {
            messages.push(Message::new("system", system));
        }
        messages.push(Message::new("user", request.user));

        let mut outcome = AgentOutcome::default();
        for step in 0..request.max_steps {
            let mode = if request.mode == Mode::Required && step > 0 { Mode::Lazy } else { request.mode };
            let mut prompt = render(&messages, &tools, true, request.thinking);
            if mode == Mode::Required && !tools.is_empty() {
                prompt.push_str("<tool_call>");
            }
            let ids = self.encode(&prompt)?;

            let mut prefix_len = 0;
            if request.prefix_cache {
                let block = self.encode(&render_system_block(&messages, &tools))?;
                if !block.is_empty() && ids.starts_with(&block) {
                    prefix_len = block.len();
                }
            }

            let (mut text, stats) = self.generate_ids(&ids, prefix_len, &request.generation, &tools, mode, None).await?;
            if mode == Mode::Required && !tools.is_empty() {
                text.insert_str(0, "<tool_call>");
            }
            let parsed = parse_output(&text, &tools);
            let natural_end = stats.stop == "eos";
            outcome.steps.push(AgentStep { text: text.clone(), stats });
            emit(on_event, &json!({ "type": "assistant", "step": step, "text": text }))?;

            if parsed.tool_calls.is_empty() && parsed.errors.is_empty() {
                // A reply cut off by the token limit is returned, but not as finished.
                outcome.answer = parsed.content;
                outcome.finished = natural_end;
                break;
            }

            messages.push(Message { role: "assistant".into(), content: parsed.content, tool_calls: parsed.tool_calls.clone() });
            for error in parsed.errors {
                outcome.parse_errors += 1;
                messages.push(Message::new("tool", json!({ "error": error }).to_string()));
            }
            for call in parsed.tool_calls {
                let (result, error) = match validate_call(&call, &tools) {
                    Ok(()) => (exec_tool(tool_exec, &call).await?, None),
                    Err(error) => (json!({ "error": error }).to_string(), Some(error)),
                };
                emit(on_event, &json!({ "type": "tool", "name": call.name, "arguments": call.arguments, "result": result, "error": error }))?;
                outcome.calls.push(AgentCall { name: call.name, arguments: call.arguments, error });
                messages.push(Message::new("tool", result));
            }
        }
        Ok(outcome)
    }
}

fn emit(on_event: Option<&Function>, event: &Value) -> Result<(), JsValue> {
    if let Some(callback) = on_event {
        callback.call1(&JsValue::NULL, &to_js(event)?)?;
    }
    Ok(())
}

async fn exec_tool(tool_exec: &Function, call: &ToolCall) -> Result<String, JsValue> {
    let returned = tool_exec.call2(&JsValue::NULL, &JsValue::from_str(&call.name), &JsValue::from_str(&call.arguments.to_string()))?;
    let resolved = JsFuture::from(Promise::resolve(&returned)).await?;
    Ok(resolved.as_string().unwrap_or_else(|| js_sys::JSON::stringify(&resolved).map(String::from).unwrap_or_default()))
}

fn half_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from((h >> 10) & 0x1f);
    let frac = u32::from(h & 0x3ff);
    match (exp, frac) {
        (0, 0) => f32::from_bits(sign),
        (0, _) => {
            let magnitude = frac as f32 * f32::from_bits(0x3380_0000); // 2^-24
            if sign != 0 { -magnitude } else { magnitude }
        }
        (31, _) => f32::from_bits(sign | 0x7f80_0000 | (frac << 13)),
        _ => f32::from_bits(sign | ((exp + 112) << 23) | (frac << 13)),
    }
}

fn default_max_steps() -> usize {
    4
}
fn default_true() -> bool {
    true
}
fn default_mode() -> Mode {
    Mode::Lazy
}

#[derive(Deserialize)]
struct AgentRequest {
    #[serde(default)]
    system: Option<String>,
    user: String,
    #[serde(default)]
    tools: Vec<Value>,
    #[serde(default = "default_mode")]
    mode: Mode,
    #[serde(default)]
    thinking: bool,
    #[serde(default = "default_max_steps")]
    max_steps: usize,
    #[serde(default = "default_true")]
    prefix_cache: bool,
    #[serde(default)]
    generation: GenOptions,
}

#[derive(Serialize)]
struct AgentCall {
    name: String,
    arguments: Value,
    error: Option<String>,
}

#[derive(Serialize)]
struct AgentStep {
    text: String,
    stats: GenStats,
}

#[derive(Default, Serialize)]
struct AgentOutcome {
    answer: String,
    finished: bool,
    parse_errors: usize,
    calls: Vec<AgentCall>,
    steps: Vec<AgentStep>,
}


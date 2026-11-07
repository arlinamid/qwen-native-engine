//! Native benchmark of the same ONNX model, outside the browser.
//!
//! It runs the generation loop of `src/engine.rs` against native ONNX Runtime, so the browser's
//! WebGPU cost can be told apart from the model's own cost. The runtime library is loaded at
//! startup (`--dll`), which lets one binary compare CPU, DirectML and native WebGPU (Dawn) builds.
//!
//! cargo run --release --bin qwen_bench -- --ep cpu --dll .toolchain/ort-dll/cpu/onnxruntime.dll

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    native::main()
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::{
        path::{Path, PathBuf},
        time::Instant,
    };

    use anyhow::{Context, Result, bail};
    use half::f16;
    use ort::{
        ep,
        memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType},
        environment::Environment,
        session::{Session, builder::GraphOptimizationLevel},
        value::{DynValue, Tensor, TensorElementType, ValueType},
    };
    use qwen_wasm::sampling::{Sampler, SamplingOptions};
    use serde_json::json;
    use tokenizers::Tokenizer;

    struct Args {
        ep: String,
        /// Runs prompt processing on a second session, e.g. batched on the CPU.
        prefill_ep: Option<String>,
        dll: PathBuf,
        model_dir: PathBuf,
        dtype: String,
        threads: usize,
        tokens: usize,
        prefill_chunk: usize,
        out: Option<PathBuf>,
        /// Writes a reference for the native engine: prompt ids, logits after the prompt and
        /// greedy ids, for `--prompt` (chat-formatted, thinking off).
        dump: Option<PathBuf>,
        prompt: String,
    }

    fn parse_args() -> Result<Args> {
        let mut args = Args {
            ep: "cpu".into(),
            prefill_ep: None,
            dll: PathBuf::from(".toolchain/ort-dll/cpu/onnxruntime.dll"),
            model_dir: PathBuf::from("models/qwen3.5-0.8b"),
            dtype: "q4f16".into(),
            threads: 4,
            tokens: 64,
            prefill_chunk: 1,
            out: None,
            dump: None,
            prompt: "Explain in simple terms why the sky is blue.".into(),
        };
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            let mut value = || it.next().with_context(|| format!("{flag} needs a value"));
            match flag.as_str() {
                "--ep" => args.ep = value()?,
                "--prefill-ep" => args.prefill_ep = Some(value()?),
                "--dll" => args.dll = value()?.into(),
                "--model-dir" => args.model_dir = value()?.into(),
                "--dtype" => args.dtype = value()?,
                "--threads" => args.threads = value()?.parse()?,
                "--tokens" => args.tokens = value()?.parse()?,
                "--prefill-chunk" => args.prefill_chunk = value()?.parse()?,
                "--out" => args.out = Some(value()?.into()),
                "--dump" => args.dump = Some(value()?.into()),
                "--prompt" => args.prompt = value()?,
                other => bail!("unknown flag {other}"),
            }
        }
        Ok(args)
    }

    /// One recurrent or KV input of the decoder, and the output that replaces it on the next step.
    struct Slot {
        past: String,
        present: String,
        dims: Vec<i64>,
        half: bool,
    }

    struct Model {
        embed: Session,
        decoder: Session,
        prefill: Option<Session>,
        slots: Vec<Slot>,
        state_mem: MemoryInfo<'static>,
        cpu_mem: MemoryInfo<'static>,
        logits: Vec<f32>,
    }

    #[derive(Default)]
    struct Times {
        embed_ms: f64,
        run_ms: f64,
        read_ms: f64,
        steps: usize,
    }

    fn ms(since: Instant) -> f64 {
        since.elapsed().as_secs_f64() * 1000.0
    }

    fn open_session(env: &Environment, path: &Path, ep_name: &str, threads: usize) -> ort::Result<Session> {
        let builder = Session::builder(env)?.with_optimization_level(GraphOptimizationLevel::Level3)?.with_intra_threads(threads)?;
        let mut builder = match ep_name {
            // DirectML needs memory patterns off and sequential execution.
            "directml" => builder
                .with_memory_pattern(false)?
                .with_parallel_execution(false)?
                .with_execution_providers([ep::DirectML::default().build().error_on_failure()])?,
            "webgpu" => builder.with_execution_providers([ep::WebGPU::default().build().error_on_failure()])?,
            _ => builder,
        };
        Ok(builder.commit_from_file(path)?)
    }

    impl Model {
        fn load(env: &Environment, args: &Args) -> Result<Self> {
            for name in std::iter::once(&args.ep).chain(&args.prefill_ep) {
                if !["cpu", "directml", "webgpu"].contains(&name.as_str()) {
                    bail!("unknown execution provider {name}");
                }
            }
            let onnx = args.model_dir.join("onnx");
            // The embedding is one table lookup per token, so it stays on the CPU for every EP.
            let embed = open_session(env, &onnx.join(format!("embed_tokens_{}.onnx", args.dtype)), "cpu", args.threads)?;
            let decoder_path = onnx.join(format!("decoder_model_merged_{}.onnx", args.dtype));
            let decoder = open_session(env, &decoder_path, &args.ep, args.threads)?;
            let prefill = match &args.prefill_ep {
                Some(name) if name != &args.ep => Some(open_session(env, &decoder_path, name, args.threads)?),
                _ => None,
            };

            let mut slots = Vec::new();
            for input in decoder.inputs() {
                let name = input.name();
                let present = if let Some(rest) = name.strip_prefix("past_key_values.") {
                    format!("present.{rest}")
                } else if let Some(rest) = name.strip_prefix("past_conv.") {
                    format!("present_conv.{rest}")
                } else if let Some(rest) = name.strip_prefix("past_recurrent.") {
                    format!("present_recurrent.{rest}")
                } else {
                    continue;
                };
                let ValueType::Tensor { ty, shape, .. } = input.dtype() else { bail!("{name} is not a tensor") };
                let half = match ty {
                    TensorElementType::Float16 => true,
                    TensorElementType::Float32 => false,
                    other => bail!("{name} is {other:?}; only float16 and float32 state is handled"),
                };
                // Dynamic dims: batch (first) is 1, a growing sequence length starts at 0.
                let dims = shape.iter().enumerate().map(|(i, &d)| if d >= 0 { d } else if i == 0 { 1 } else { 0 }).collect();
                slots.push(Slot { past: name.to_string(), present, dims, half });
            }

            let device = match args.ep.as_str() {
                "directml" => AllocationDevice::DIRECTML,
                "webgpu" => AllocationDevice::WEBGPU_BUFFER,
                _ => AllocationDevice::CPU,
            };
            let state_mem = MemoryInfo::new(device, 0, AllocatorType::Device, MemoryType::Default)?;
            let cpu_mem = MemoryInfo::new(AllocationDevice::CPU, 0, AllocatorType::Device, MemoryType::Default)?;
            Ok(Self { embed, decoder, prefill, slots, state_mem, cpu_mem, logits: Vec::new() })
        }

        fn empty_state(&self) -> Result<Vec<DynValue>> {
            self.slots
                .iter()
                .map(|s| {
                    let len = s.dims.iter().product::<i64>() as usize;
                    Ok(if s.half {
                        Tensor::from_array((s.dims.clone(), vec![f16::ZERO; len]))?.into_dyn()
                    } else {
                        Tensor::from_array((s.dims.clone(), vec![0f32; len]))?.into_dyn()
                    })
                })
                .collect()
        }

        /// One forward pass. A fresh binding per step gets fresh output buffers, so the state that
        /// is fed back as input is never overwritten by the run that reads it. With `prefill` set the
        /// prefill session runs, if there is one; its CPU state is uploaded by the first GPU step.
        fn step(&mut self, ids: &[u32], state: &mut Vec<DynValue>, past_len: usize, want_logits: bool, prefill: bool, t: &mut Times) -> Result<()> {
            let n = ids.len();
            let total = past_len + n;

            let t0 = Instant::now();
            let input_ids = Tensor::from_array(([1usize, n], ids.iter().map(|&x| i64::from(x)).collect::<Vec<_>>()))?;
            let mut embedded = self.embed.run(ort::inputs!["input_ids" => input_ids])?;
            let embeds = embedded.remove("inputs_embeds").context("no inputs_embeds")?;
            t.embed_ms += ms(t0);

            let (session, state_mem) = match (prefill, self.prefill.as_mut()) {
                (true, Some(session)) => (session, &self.cpu_mem),
                _ => (&mut self.decoder, &self.state_mem),
            };
            let mut binding = session.create_binding()?;
            binding.bind_input("inputs_embeds", &embeds)?;
            binding.bind_input("attention_mask", &Tensor::from_array(([1usize, total], vec![1i64; total]))?)?;
            let positions: Vec<i64> = (past_len as i64..total as i64).cycle().take(3 * n).collect();
            binding.bind_input("position_ids", &Tensor::from_array(([3usize, 1, n], positions))?)?;
            for (slot, value) in self.slots.iter().zip(state.iter()) {
                binding.bind_input(slot.past.as_str(), value)?;
                binding.bind_output_to_device(slot.present.as_str(), state_mem)?;
            }
            binding.bind_output_to_device("logits", &self.cpu_mem)?;

            let t1 = Instant::now();
            let mut outputs = session.run_binding(&binding)?;
            t.run_ms += ms(t1);

            let t2 = Instant::now();
            let mut next = Vec::with_capacity(self.slots.len());
            for slot in &self.slots {
                next.push(outputs.remove(slot.present.as_str()).with_context(|| format!("no {}", slot.present))?);
            }
            if want_logits {
                let logits = outputs.remove("logits").context("no logits")?;
                self.logits.clear();
                if let Ok((shape, data)) = logits.try_extract_tensor::<f16>() {
                    let (rows, vocab) = (shape[1] as usize, shape[2] as usize);
                    self.logits.extend(data[(rows - 1) * vocab..rows * vocab].iter().map(|h| h.to_f32()));
                } else {
                    let (shape, data) = logits.try_extract_tensor::<f32>()?;
                    let (rows, vocab) = (shape[1] as usize, shape[2] as usize);
                    self.logits.extend_from_slice(&data[(rows - 1) * vocab..rows * vocab]);
                }
            }
            drop(outputs);
            *state = next;
            t.read_ms += ms(t2);
            t.steps += 1;
            Ok(())
        }
    }

    struct RunResult {
        name: &'static str,
        text: String,
        prompt_tokens: usize,
        new_tokens: usize,
        prefill_ms: f64,
        decode_ms: f64,
        times: Times,
    }

    fn generate(model: &mut Model, tok: &Tokenizer, name: &'static str, prompt: &str, max_new: usize, chunk: usize) -> Result<RunResult> {
        let stop = [tok.token_to_id("<|im_end|>").unwrap(), tok.token_to_id("<|endoftext|>").unwrap()];
        let ids = tok.encode(prompt, false).map_err(anyhow::Error::msg)?.get_ids().to_vec();
        let chunk = if chunk == 0 { ids.len() } else { chunk };

        let mut state = model.empty_state()?;
        let mut prefill_times = Times::default();
        let t0 = Instant::now();
        let mut pos = 0;
        let batches = ids.len().div_ceil(chunk);
        for (i, batch) in ids.chunks(chunk).enumerate() {
            model.step(batch, &mut state, pos, i + 1 == batches, true, &mut prefill_times)?;
            pos += batch.len();
        }
        let prefill_ms = ms(t0);

        let mut sampler = Sampler::new(SamplingOptions { temperature: 0.0, ..Default::default() });
        let mut out = Vec::new();
        let mut times = Times::default();
        let t1 = Instant::now();
        while out.len() < max_new {
            let token = sampler.sample(&model.logits, None);
            if stop.contains(&token) {
                break;
            }
            out.push(token);
            if out.len() == max_new {
                break;
            }
            model.step(&[token], &mut state, pos, true, false, &mut times)?;
            pos += 1;
        }
        let decode_ms = ms(t1);
        let text = tok.decode(&out, false).map_err(anyhow::Error::msg)?;
        Ok(RunResult { name, text, prompt_tokens: ids.len(), new_tokens: out.len(), prefill_ms, decode_ms, times })
    }

    /// Prompt ids, the logits after the prompt (batched prefill) and `n` greedy ids.
    fn dump_reference(model: &mut Model, tok: &Tokenizer, user: &str, n: usize, path: &Path) -> Result<()> {
        let ids = tok.encode(chat(user).as_str(), false).map_err(anyhow::Error::msg)?.get_ids().to_vec();
        let stop = [tok.token_to_id("<|im_end|>").unwrap(), tok.token_to_id("<|endoftext|>").unwrap()];
        let mut state = model.empty_state()?;
        let mut times = Times::default();
        model.step(&ids, &mut state, 0, true, true, &mut times)?;
        let logits = model.logits.clone();
        let mut sampler = Sampler::new(SamplingOptions { temperature: 0.0, ..Default::default() });
        let mut out = Vec::new();
        let mut pos = ids.len();
        while out.len() < n {
            let token = sampler.sample(&model.logits, None);
            out.push(token);
            if stop.contains(&token) || out.len() == n {
                break;
            }
            model.step(&[token], &mut state, pos, true, false, &mut times)?;
            pos += 1;
        }
        let text = tok.decode(&out, false).map_err(anyhow::Error::msg)?;
        eprintln!("reference: {} prompt tokens -> {text:?}", ids.len());
        let report = json!({ "prompt": user, "prompt_ids": ids, "logits": logits, "ids": out, "text": text });
        std::fs::write(path, serde_json::to_string(&report)?)?;
        Ok(())
    }

    fn chat(user: &str) -> String {
        format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    }

    pub fn main() -> Result<()> {
        let args = parse_args()?;
        let dll = std::fs::canonicalize(&args.dll).with_context(|| format!("missing {}", args.dll.display()))?;
        // Dependent DLLs (DirectML.dll, dxcompiler.dll) are found next to onnxruntime.dll.
        let dir = dll.parent().unwrap_or(Path::new("."));
        let path = std::env::var_os("PATH").unwrap_or_default();
        unsafe { std::env::set_var("PATH", std::env::join_paths(std::iter::once(dir.to_path_buf()).chain(std::env::split_paths(&path)))?) };
        let env = ort::init_from(&dll).map_err(|e| anyhow::anyhow!("{e}"))?.build()?;

        let t_load = Instant::now();
        let tok = Tokenizer::from_file(args.model_dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        let mut model = Model::load(&env, &args)?;
        let load_ms = ms(t_load);
        eprintln!(
            "loaded {} {} (prefill: {}, {} state tensors) in {load_ms:.0} ms",
            args.dtype,
            args.ep,
            args.prefill_ep.as_deref().unwrap_or(&args.ep),
            model.slots.len()
        );

        if let Some(path) = &args.dump {
            return dump_reference(&mut model, &tok, &args.prompt, args.tokens, path);
        }

        let short = chat("Explain in simple terms why the sky is blue.");
        let long = chat(&format!(
            "{}\nSummarize the text above.",
            "The history of computing spans mechanical calculators, vacuum tubes, transistors, integrated circuits and the networked machines of today. ".repeat(24)
        ));

        let mut runs = Vec::new();
        for (name, prompt, tokens) in [("warmup", &short, 4), ("short", &short, args.tokens), ("long", &long, args.tokens)] {
            let r = generate(&mut model, &tok, name, prompt, tokens, args.prefill_chunk)?;
            let steps = r.times.steps.max(1) as f64;
            let tps = if r.new_tokens > 1 { (r.new_tokens - 1) as f64 / (r.decode_ms / 1000.0) } else { 0.0 };
            eprintln!(
                "{:<6} prompt {:>4}  prefill {:>8.0} ms ({:>6.1} tok/s)  decode {:>6.2} tok/s  per step: embed {:.1} run {:.1} read {:.1} ms",
                r.name,
                r.prompt_tokens,
                r.prefill_ms,
                r.prompt_tokens as f64 / (r.prefill_ms / 1000.0),
                tps,
                r.times.embed_ms / steps,
                r.times.run_ms / steps,
                r.times.read_ms / steps
            );
            runs.push(json!({
                "name": r.name, "text": r.text, "prompt_tokens": r.prompt_tokens, "new_tokens": r.new_tokens,
                "prefill_ms": r.prefill_ms, "decode_ms": r.decode_ms, "tokens_per_s": tps,
                "embed_ms": r.times.embed_ms / steps, "run_ms": r.times.run_ms / steps, "read_ms": r.times.read_ms / steps,
            }));
        }
        println!("{}", runs.last().and_then(|r| r["text"].as_str()).unwrap_or_default().chars().take(160).collect::<String>());

        if let Some(out) = &args.out {
            let report = json!({
                "ep": args.ep, "prefill_ep": args.prefill_ep, "dll": args.dll, "dtype": args.dtype, "threads": args.threads,
                "prefill_chunk": args.prefill_chunk, "load_ms": load_ms, "runs": runs,
            });
            std::fs::write(out, serde_json::to_string_pretty(&report)?)?;
        }
        Ok(())
    }
}

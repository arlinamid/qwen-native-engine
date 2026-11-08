//! The real model: loads the weights written by `scripts/convert_weights.py` and decodes with
//! the prototype's kernels. Prompt processing runs the same one-token graph once per token.

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::{
    batch::{self, BATCH},
    ARGMAX_GROUPS, ATTN_HEADS, Dispatch, GDN_HEADS, GDN_PROJ, Gpu, MIXER_OUT, QMat, Tile, VOCAB, map_read, shaders,
    shaders::{Epilogue, MAX_CONTEXT},
};

#[derive(Deserialize)]
struct TensorEntry {
    offset: u64,
    bytes: u64,
}

#[derive(Deserialize)]
struct MatrixShape {
    rows: u32,
    k: u32,
}

#[derive(Deserialize)]
struct IndexConfig {
    layers: u32,
    full_attention_every: u32,
    hidden: u32,
    #[serde(default)]
    intermediate: u32,
    vocab: u32,
    rope_theta: f64,
    rope_dim: u32,
}

#[derive(Deserialize)]
struct Index {
    config: IndexConfig,
    matrices: HashMap<String, MatrixShape>,
    tensors: HashMap<String, TensorEntry>,
}

impl Index {
    fn intermediate(&self) -> u32 {
        if self.config.intermediate != 0 { self.config.intermediate }
        else if self.config.hidden == 1024 { 3584 } else { 6144 }
    }
    fn validate(&self, bytes: usize) -> Result<()> {
        let c = &self.config;
        anyhow::ensure!(c.layers == 24 && c.full_attention_every == 4 && matches!((c.hidden, self.intermediate()), (1024, 3584) | (2048, 6144))
            && c.vocab == VOCAB && c.rope_dim == 64 && c.rope_theta == 10_000_000.0,
            "unsupported model architecture: these kernels require Qwen3.5-0.8B or Qwen3.5-2B");
        for (name, tensor) in &self.tensors {
            anyhow::ensure!(tensor.offset.checked_add(tensor.bytes).is_some_and(|end| end <= bytes as u64)
                && tensor.offset % 4 == 0 && tensor.bytes > 0 && tensor.bytes % 4 == 0,
                "invalid tensor range: {name}");
        }
        let tensor = |name: &str, expected: u64| -> Result<()> {
            anyhow::ensure!(self.tensors.get(name).is_some_and(|t| t.bytes == expected), "invalid tensor size: {name}");
            Ok(())
        };
        let matrix = |name: &str, rows: u32, k: u32| -> Result<()> {
            anyhow::ensure!(self.matrices.get(name).is_some_and(|s| s.rows == rows && s.k == k), "invalid matrix shape: {name}");
            let elements = u64::from(rows) * u64::from(k);
            tensor(&format!("{name}.w"), elements / 2)?;
            tensor(&format!("{name}.s"), elements / 16)?;
            tensor(&format!("{name}.z"), elements / 64)
        };
        matrix("embed", VOCAB, c.hidden)?;
        matrix("lm_head", VOCAB, c.hidden)?;
        tensor("final_norm", u64::from(c.hidden) * 4)?;
        for i in 0..24 {
            let attention = (i + 1) % 4 == 0;
            for name in ["input_norm", "post_norm"] { tensor(&format!("layer{i}.{name}"), u64::from(c.hidden) * 4)?; }
            matrix(&format!("layer{i}.mixer_in"), if attention {5120} else {GDN_PROJ}, c.hidden)?;
            matrix(&format!("layer{i}.mixer_out"), c.hidden, MIXER_OUT)?;
            matrix(&format!("layer{i}.gate_up"), self.intermediate() * 2, c.hidden)?;
            matrix(&format!("layer{i}.down"), c.hidden, self.intermediate())?;
            if attention { tensor(&format!("layer{i}.attn_norms"), 2048)?; }
            else { tensor(&format!("layer{i}.conv_w"), 98304)?; tensor(&format!("layer{i}.gdn_params"), 640)?; }
        }
        Ok(())
    }
}

/// GEMV kernel per matrix kind. The defaults are the fastest shapes measured on the UHD 630.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ModelTiles {
    pub gdn_in: Tile,
    pub attn_in: Tile,
    pub out: Tile,
    pub gate_up: Tile,
    pub down: Tile,
    pub lm_head: Tile,
}

impl ModelTiles {
    pub fn validate(&self) -> Result<()> {
        for t in [self.gdn_in,self.attn_in,self.out,self.gate_up,self.down,self.lm_head] {
            anyhow::ensure!([1,2,4,8,16,32,64].contains(&t.tpr) && [2,4,8,16,32].contains(&t.rpw)
                && [1,2,4,8].contains(&t.rpt) && [1,2,4,8].contains(&t.splits)
                && t.tpr*t.rpw <= 256 && (t.f16 || (t.rpt == 1 && t.splits == 1)),
                "invalid kernel tile profile: {t:?}");
        }
        Ok(())
    }
    /// The plain f32 kernel everywhere: slower, but closest to an exact reference.
    pub fn f32() -> Self {
        let t = Tile { f16: false, tpr: 16, rpw: 4, rpt: 1, splits: 1 };
        Self { gdn_in: t, attn_in: t, out: t, gate_up: t, down: t, lm_head: t }
    }

    /// Tune the six matrix shapes for this architecture; correctness is checked separately
    /// against the same quantized ONNX model before accepting a profile.
    pub async fn tune(index_json: &str, dxc: Option<&str>, log: &dyn Fn(&str)) -> Result<(Self, Vec<crate::GemvTiming>)> {
        let index: Index = serde_json::from_str(index_json)?;
        let h = index.config.hidden;
        let intermediate = index.intermediate();
        anyhow::ensure!(matches!((h, intermediate), (1024,3584)|(2048,6144)), "unsupported tuning shape");
        let (mut gpu, adapter, backend) = Gpu::new(true, dxc).await?;
        log(&format!("tuning {h}/{intermediate} on {adapter} via {backend}"));
        let mut rng = crate::Rng(0x9E37_79B9_7F4A_7C15);
        let mut report = Vec::new();
        let mut chosen = Vec::new();
        for (name,rows,k,epilogue) in [
            ("gdn_in",GDN_PROJ,h,Epilogue::Store), ("attn_in",5120,h,Epilogue::Store),
            ("out",h,MIXER_OUT,Epilogue::Accumulate), ("gate_up",2*intermediate,h,Epilogue::SwiGlu),
            ("down",h,intermediate,Epilogue::Accumulate), ("lm_head",VOCAB,h,Epilogue::Store),
        ] {
            let tile=crate::tune_gemv(&mut gpu,name,rows,k,epilogue,true,&mut rng,&mut report).await?;
            log(&format!("{name} {rows}x{k}: {tile:?}, {:.3} ms",report.last().unwrap().ms));
            chosen.push(tile);
        }
        Ok((Self {gdn_in:chosen[0],attn_in:chosen[1],out:chosen[2],gate_up:chosen[3],down:chosen[4],lm_head:chosen[5]},report))
    }

    pub fn fast() -> Self {
        let f16 = |tpr, rpw, rpt, splits| Tile { f16: true, tpr, rpw, rpt, splits };
        let f32 = Tile { f16: false, tpr: 16, rpw: 4, rpt: 1, splits: 1 };
        Self { gdn_in: f16(16, 4, 4, 1), attn_in: f16(4, 32, 4, 1), out: f32, gate_up: f16(4, 16, 4, 1), down: f32, lm_head: f16(4, 32, 4, 1) }
    }
}

pub struct Model {
    hidden: u32,
    gpu: Gpu,
    token: Vec<Dispatch>,
    /// Batched prompt graph: `BATCH` tokens per pass, positions and count from `chunk`.
    prefill: Vec<Dispatch>,
    /// Index in `token` where the final norm, LM head and greedy pick start.
    tail: usize,
    h: wgpu::Buffer,
    hb: wgpu::Buffer,
    ids: wgpu::Buffer,
    chunk: wgpu::Buffer,
    tok: wgpu::Buffer,
    logits: wgpu::Buffer,
    staging: wgpu::Buffer,
    logits_staging: wgpu::Buffer,
    /// Recurrent and attention state, cleared between prompts.
    state: Vec<wgpu::Buffer>,
    /// Copies of `state` taken by `snapshot`, created on first use.
    saved: std::sync::OnceLock<Vec<wgpu::Buffer>>,
    pub adapter: String,
}

impl Model {
    pub async fn load(index_json: &str, weights: &[u8], tiles: ModelTiles, allow_f16: bool, dxc: Option<&str>) -> Result<Self> {
        let index: Index = serde_json::from_str(index_json).context("index.json")?;
        index.validate(weights.len())?;
        tiles.validate()?;
        let hidden = index.config.hidden;
        let intermediate = index.intermediate();
        let (mut gpu, adapter, backend) = Gpu::new(allow_f16, dxc).await?;
        let tiles = if gpu.f16 { tiles } else { ModelTiles::f32() };

        let tensor = |name: &str| -> Result<&[u8]> {
            let t = index.tensors.get(name).with_context(|| format!("missing tensor {name}"))?;
            weights.get(t.offset as usize..(t.offset + t.bytes) as usize).with_context(|| format!("{name} is outside weights.bin"))
        };
        let uploads = std::cell::RefCell::new(HashMap::<(u64,u64), wgpu::Buffer>::new());
        let upload = |gpu: &Gpu, name: &str| -> Result<wgpu::Buffer> {
            let bytes = tensor(name)?;
            let entry = &index.tensors[name];
            let key = (entry.offset, entry.bytes);
            if let Some(buffer) = uploads.borrow().get(&key) { return Ok(buffer.clone()); }
            let buffer = gpu.storage(bytes.len() as u64);
            gpu.queue.write_buffer(&buffer, 0, bytes);
            uploads.borrow_mut().insert(key, buffer.clone());
            Ok(buffer)
        };
        let qmat = |gpu: &Gpu, name: &str| -> Result<QMat> {
            let shape = index.matrices.get(name).with_context(|| format!("missing matrix {name}"))?;
            Ok(QMat { w: upload(gpu, &format!("{name}.w"))?, s: upload(gpu, &format!("{name}.s"))?, z: upload(gpu, &format!("{name}.z"))?, rows: shape.rows, k: shape.k })
        };

        let f = |n: u32| u64::from(n) * 4;
        let h = gpu.storage(f(hidden));
        let xn = gpu.storage(f(hidden));
        let proj = gpu.storage(f(GDN_PROJ));
        let mixed = gpu.storage(f(MIXER_OUT));
        let act = gpu.storage(f(intermediate));
        let logits = gpu.storage(f(VOCAB));
        let best_val = gpu.storage(f(ARGMAX_GROUPS));
        let best_idx = gpu.storage(f(ARGMAX_GROUPS));
        let tok = gpu.upload(&[0u32, 0, 0, 0]);
        let mut state = Vec::new();
        // Chunk-sized activations for the batched prompt graph.
        let hb = gpu.storage(f(BATCH * hidden));
        let xnb = gpu.storage(f(BATCH * hidden));
        let projb = gpu.storage(f(BATCH * GDN_PROJ));
        let mixedb = gpu.storage(f(BATCH * MIXER_OUT));
        let actb = gpu.storage(f(BATCH * intermediate));
        let ids = gpu.storage(f(BATCH));
        let chunk = gpu.upload(&[0u32, 0, 0, 0]);

        let embed = qmat(&gpu, "embed")?;
        let mut token = vec![gpu.simple("embed", || shaders::embed_width(hidden), &[&embed.w, &embed.s, &tok, &h, &embed.z], 1)];
        let mut prefill = vec![gpu.simple("embed_b", || batch::embed_width(hidden), &[&embed.w, &embed.s, &ids, &hb, &embed.z, &chunk], BATCH)];
        for i in 0..index.config.layers {
            let input_norm = upload(&gpu, &format!("layer{i}.input_norm"))?;
            let post_norm = upload(&gpu, &format!("layer{i}.post_norm"))?;
            token.push(gpu.simple("rmsnorm", || shaders::rmsnorm_width(hidden), &[&h, &input_norm, &xn], 1));
            prefill.push(gpu.simple("rmsnorm_b", || batch::rmsnorm_width(hidden), &[&hb, &input_norm, &xnb, &chunk], BATCH));
            let mixer_in = qmat(&gpu, &format!("layer{i}.mixer_in"))?;
            prefill.push(gpu.gemm(&mixer_in, &xnb, &projb, Epilogue::Store, &chunk));
            if (i + 1) % index.config.full_attention_every == 0 {
                let params = upload(&gpu, &format!("layer{i}.attn_norms"))?;
                let kc = gpu.storage(f(2 * MAX_CONTEXT * 256));
                let vc = gpu.storage(f(2 * MAX_CONTEXT * 256));
                token.extend(gpu.gemv(&mixer_in, &xn, &proj, Epilogue::Store, tiles.attn_in));
                token.push(gpu.simple("attention", shaders::attention, &[&proj, &kc, &vc, &tok, &mixed, &params], ATTN_HEADS));
                prefill.push(gpu.simple("kv_prep_b", batch::kv_prep, &[&projb, &kc, &vc, &chunk, &params], 2 * BATCH));
                prefill.push(gpu.simple("attention_b", batch::attention, &[&projb, &kc, &vc, &chunk, &mixedb, &params], ATTN_HEADS * BATCH));
                state.extend([kc, vc]);
            } else {
                let conv_w = upload(&gpu, &format!("layer{i}.conv_w"))?;
                let params = upload(&gpu, &format!("layer{i}.gdn_params"))?;
                let conv_state = gpu.storage(f(6144 * 3));
                let recurrent = gpu.storage(f(GDN_HEADS * 16384));
                token.extend(gpu.gemv(&mixer_in, &xn, &proj, Epilogue::Store, tiles.gdn_in));
                token.push(gpu.simple("gdn", shaders::gdn, &[&proj, &conv_state, &conv_w, &recurrent, &mixed, &params], GDN_HEADS));
                prefill.push(gpu.simple("gdn_b", batch::gdn, &[&projb, &conv_state, &conv_w, &recurrent, &mixedb, &params, &chunk], GDN_HEADS));
                state.extend([conv_state, recurrent]);
            }
            let mixer_out = qmat(&gpu, &format!("layer{i}.mixer_out"))?;
            token.extend(gpu.gemv(&mixer_out, &mixed, &h, Epilogue::Accumulate, tiles.out));
            token.push(gpu.simple("rmsnorm", || shaders::rmsnorm_width(hidden), &[&h, &post_norm, &xn], 1));
            let gate_up = qmat(&gpu, &format!("layer{i}.gate_up"))?;
            token.extend(gpu.gemv(&gate_up, &xn, &act, Epilogue::SwiGlu, tiles.gate_up));
            let down = qmat(&gpu, &format!("layer{i}.down"))?;
            token.extend(gpu.gemv(&down, &act, &h, Epilogue::Accumulate, tiles.down));
            prefill.push(gpu.gemm(&mixer_out, &mixedb, &hb, Epilogue::Accumulate, &chunk));
            prefill.push(gpu.simple("rmsnorm_b", || batch::rmsnorm_width(hidden), &[&hb, &post_norm, &xnb, &chunk], BATCH));
            prefill.push(gpu.gemm(&gate_up, &xnb, &actb, Epilogue::SwiGlu, &chunk));
            prefill.push(gpu.gemm(&down, &actb, &hb, Epilogue::Accumulate, &chunk));
        }
        let tail = token.len();
        let final_norm = upload(&gpu, "final_norm")?;
        token.push(gpu.simple("rmsnorm", || shaders::rmsnorm_width(hidden), &[&h, &final_norm, &xn], 1));
        let lm_head = qmat(&gpu, "lm_head")?;
        token.extend(gpu.gemv(&lm_head, &xn, &logits, Epilogue::Store, tiles.lm_head));
        let p1 = gpu.uniform([VOCAB, 0, 0, 0]);
        let p2 = gpu.uniform([ARGMAX_GROUPS, 0, 0, 0]);
        token.push(gpu.simple("argmax_partial", shaders::argmax_partial, &[&logits, &best_val, &best_idx, &p1], ARGMAX_GROUPS));
        token.push(gpu.simple("argmax_final", shaders::argmax_final, &[&best_val, &best_idx, &tok, &p2], 1));
        gpu.idle().await?;

        let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let logits_staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(VOCAB) * 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            hidden,
            gpu,
            token,
            prefill,
            tail,
            h,
            hb,
            ids,
            chunk,
            tok,
            logits,
            staging,
            logits_staging,
            state,
            saved: std::sync::OnceLock::new(),
            adapter: format!("{adapter} via {backend}"),
        })
    }

    /// Consumes one greedy token; intended only for isolated diagnostic runs.
    pub async fn profile_token(&self) -> Result<std::collections::BTreeMap<String, f64>> {
        self.gpu.profile(&self.token).await
    }

    pub fn dispatches_per_token(&self) -> usize {
        self.token.len()
    }

    /// Clears the recurrent states and the KV-cache for a new prompt.
    pub fn reset(&self) {
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        for buffer in &self.state {
            encoder.clear_buffer(buffer, 0, None);
        }
        self.gpu.queue.submit([encoder.finish()]);
    }

    /// Runs the prompt one token at a time from position 0. Afterwards the logits buffer holds
    /// the prediction after the last prompt token, and the token buffer its greedy pick.
    pub async fn prefill(&self, ids: &[u32]) -> Result<()> {
        if ids.is_empty() || ids.len() as u32 >= MAX_CONTEXT {
            return Err(anyhow!("prompt length must be 1..{MAX_CONTEXT}"));
        }
        for (pos, &id) in ids.iter().enumerate() {
            self.gpu.queue.write_buffer(&self.tok, 0, crate::as_bytes(&[id, pos as u32, 0, 0]));
            let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
            self.gpu.encode(&mut encoder, &self.token, 1);
            self.gpu.queue.submit([encoder.finish()]);
        }
        self.gpu.idle().await
    }

    /// Runs the prompt `BATCH` tokens per pass, then the LM head on the last token only. Leaves
    /// the same state as `prefill`.
    pub async fn prefill_batched(&self, ids: &[u32]) -> Result<()> {
        self.prefill_from(ids, 0).await
    }

    /// Like `prefill_batched`, for tokens that continue the current state at position `start`.
    pub async fn prefill_from(&self, ids: &[u32], start: u32) -> Result<()> {
        self.prefill_chunk(ids, start, true).await
    }

    /// Intermediate cancellation checkpoints need the recurrent state, not vocabulary logits.
    pub async fn prefill_chunk(&self, ids: &[u32], start: u32, finish_logits: bool) -> Result<()> {
        if ids.is_empty() || u64::from(start) + ids.len() as u64 >= u64::from(MAX_CONTEXT) {
            return Err(anyhow!("context limit: {} + {} tokens does not fit in {MAX_CONTEXT}", start, ids.len()));
        }
        let mut last_len = 0;
        for (i, part) in ids.chunks(BATCH as usize).enumerate() {
            self.gpu.queue.write_buffer(&self.ids, 0, crate::as_bytes(part));
            self.gpu.queue.write_buffer(&self.chunk, 0, crate::as_bytes(&[start + i as u32 * BATCH, part.len() as u32, 0, 0]));
            let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
            self.gpu.encode(&mut encoder, &self.prefill, 1);
            self.gpu.queue.submit([encoder.finish()]);
            last_len = part.len();
        }
        if finish_logits {
        // The greedy pick increments tok[1], which then holds the next position.
        self.gpu.queue.write_buffer(&self.tok, 0, crate::as_bytes(&[0u32, start + ids.len() as u32 - 1, 0, 0]));
        let row = (last_len as u64 - 1) * u64::from(self.hidden) * 4;
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.hb, row, &self.h, 0, u64::from(self.hidden) * 4);
        self.gpu.encode(&mut encoder, &self.token[self.tail..], 1);
        self.gpu.queue.submit([encoder.finish()]);
        }
        self.gpu.idle().await
    }

    pub async fn logits(&self) -> Result<Vec<f32>> {
        self.gpu.read_f32(&self.logits, VOCAB as usize).await
    }

    /// Feeds one token at position `pos` and returns the logits for the next one. The host then
    /// picks the token (sampling, grammar masks); GPU argmax is skipped on this path.
    pub async fn step(&self, token: u32, pos: u32) -> Result<Vec<f32>> {
        if pos >= MAX_CONTEXT {
            return Err(anyhow!("context limit of {MAX_CONTEXT} tokens reached"));
        }
        self.gpu.queue.write_buffer(&self.tok, 0, crate::as_bytes(&[token, pos, 0, 0]));
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        // Host sampling owns the next token; skip both GPU argmax dispatches.
        self.gpu.encode(&mut encoder, &self.token[..self.token.len()-2], 1);
        encoder.copy_buffer_to_buffer(&self.logits, 0, &self.logits_staging, 0, u64::from(VOCAB) * 4);
        self.gpu.queue.submit([encoder.finish()]);
        let bytes = map_read(&self.gpu, &self.logits_staging).await?;
        Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    /// Saves the recurrent states and KV-cache, e.g. after the system prompt and tool list.
    pub fn snapshot(&self) {
        let saved = self.saved.get_or_init(|| self.state.iter().map(|b| self.gpu.storage(b.size())).collect());
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        for (from, to) in self.state.iter().zip(saved) {
            encoder.copy_buffer_to_buffer(from, 0, to, 0, from.size());
        }
        self.gpu.queue.submit([encoder.finish()]);
    }

    /// Returns to the last `snapshot`. The recurrent state cannot be rolled back token by
    /// token, so this is the only way back to a shorter prefix without recomputing it.
    pub fn restore(&self) -> Result<()> {
        let saved = self.saved.get().ok_or_else(|| anyhow!("no snapshot taken"))?;
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        for (from, to) in saved.iter().zip(&self.state) {
            encoder.copy_buffer_to_buffer(from, 0, to, 0, from.size());
        }
        self.gpu.queue.submit([encoder.finish()]);
        Ok(())
    }

    /// Greedy pick and next position after prefill/decode; host `step` does not update this pick.
    pub async fn last_token(&self) -> Result<(u32, u32)> {
        let v = self.gpu.read_u32(&self.tok, 2).await?;
        Ok((v[0], v[1]))
    }

    /// Greedy decoding from the current state, `window` tokens per submission. The first
    /// returned token is the pick after the prompt; each step feeds back the previous pick.
    /// Fails instead of decoding past the KV-cache, whose last slot the shaders would otherwise
    /// keep overwriting.
    pub async fn decode(&self, n: usize, window: u32, stop: &[u32]) -> Result<Vec<u32>> {
        let (first, mut next_pos) = self.last_token().await?;
        let mut out = vec![first];
        while out.len() < n && !stop.contains(out.last().unwrap()) {
            let steps = window.min((n - out.len()) as u32);
            if u64::from(next_pos) + u64::from(steps) > u64::from(MAX_CONTEXT) {
                return Err(anyhow!("context limit of {MAX_CONTEXT} tokens reached after {} generated tokens", out.len()));
            }
            next_pos += steps;
            let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
            // Each step leaves its pick in `tok`; with a window, only the last one is read back,
            // so the window is used for timing and the token-by-token path for the output.
            self.gpu.encode(&mut encoder, &self.token, steps);
            encoder.copy_buffer_to_buffer(&self.tok, 0, &self.staging, 0, 16);
            self.gpu.queue.submit([encoder.finish()]);
            let bytes = map_read(&self.gpu, &self.staging).await?;
            if steps == 1 {
                out.push(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
            } else {
                out.extend(std::iter::repeat_n(u32::MAX, steps as usize));
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tile_profiles_reject_invalid_dispatch_geometry() {
        assert!(ModelTiles::fast().validate().is_ok());
        assert!(ModelTiles::f32().validate().is_ok());
        let profile: ModelTiles = serde_json::from_str(include_str!("../profiles/uhd630-qwen35-2b.json")).unwrap();
        assert!(profile.validate().is_ok());
        let mut bad=profile; bad.down.splits=0;
        assert!(bad.validate().is_err());
        bad=profile; bad.lm_head.tpr=0;
        assert!(bad.validate().is_err());
    }
}

//! The real model: loads the weights written by `scripts/convert_weights.py` and decodes with
//! the prototype's kernels. Prompt processing runs the same one-token graph once per token.

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;

use crate::{
    batch::{self, BATCH},
    ARGMAX_GROUPS, ATTN_HEADS, Dispatch, GDN_HEADS, GDN_PROJ, Gpu, INTERMEDIATE, MIXER_OUT, QMat, Tile, VOCAB, map_read, shaders,
    shaders::{Epilogue, HIDDEN, MAX_CONTEXT},
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
}

#[derive(Deserialize)]
struct Index {
    config: IndexConfig,
    matrices: HashMap<String, MatrixShape>,
    tensors: HashMap<String, TensorEntry>,
}

/// GEMV kernel per matrix kind. The defaults are the fastest shapes measured on the UHD 630.
#[derive(Clone, Copy)]
pub struct ModelTiles {
    pub gdn_in: Tile,
    pub attn_in: Tile,
    pub out: Tile,
    pub gate_up: Tile,
    pub down: Tile,
    pub lm_head: Tile,
}

impl ModelTiles {
    /// The plain f32 kernel everywhere: slower, but closest to an exact reference.
    pub fn f32() -> Self {
        let t = Tile { f16: false, tpr: 16, rpw: 4, rpt: 1, splits: 1 };
        Self { gdn_in: t, attn_in: t, out: t, gate_up: t, down: t, lm_head: t }
    }

    pub fn fast() -> Self {
        let f16 = |tpr, rpw, rpt, splits| Tile { f16: true, tpr, rpw, rpt, splits };
        let f32 = Tile { f16: false, tpr: 16, rpw: 4, rpt: 1, splits: 1 };
        Self { gdn_in: f16(16, 4, 4, 1), attn_in: f16(4, 32, 4, 1), out: f32, gate_up: f16(4, 16, 4, 1), down: f32, lm_head: f16(4, 32, 4, 1) }
    }
}

pub struct Model {
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
    /// Recurrent and attention state, cleared between prompts.
    state: Vec<wgpu::Buffer>,
    pub adapter: String,
}

impl Model {
    pub async fn load(index_json: &str, weights: &[u8], tiles: ModelTiles, allow_f16: bool, dxc: Option<&str>) -> Result<Self> {
        let index: Index = serde_json::from_str(index_json).context("index.json")?;
        let (mut gpu, adapter, backend) = Gpu::new(allow_f16, dxc).await?;
        if !gpu.f16 && [tiles.gdn_in, tiles.attn_in, tiles.out, tiles.gate_up, tiles.down, tiles.lm_head].iter().any(|t| t.f16) {
            return Err(anyhow!("the f16 kernels need SHADER_F16 (on native D3D12: pass the DXC path)"));
        }

        let tensor = |name: &str| -> Result<&[u8]> {
            let t = index.tensors.get(name).with_context(|| format!("missing tensor {name}"))?;
            weights.get(t.offset as usize..(t.offset + t.bytes) as usize).with_context(|| format!("{name} is outside weights.bin"))
        };
        let upload = |gpu: &Gpu, name: &str| -> Result<wgpu::Buffer> {
            let bytes = tensor(name)?;
            let buffer = gpu.storage(bytes.len() as u64);
            gpu.queue.write_buffer(&buffer, 0, bytes);
            Ok(buffer)
        };
        let qmat = |gpu: &Gpu, name: &str| -> Result<QMat> {
            let shape = index.matrices.get(name).with_context(|| format!("missing matrix {name}"))?;
            Ok(QMat { w: upload(gpu, &format!("{name}.w"))?, s: upload(gpu, &format!("{name}.s"))?, z: upload(gpu, &format!("{name}.z"))?, rows: shape.rows, k: shape.k })
        };

        let f = |n: u32| u64::from(n) * 4;
        let h = gpu.storage(f(HIDDEN));
        let xn = gpu.storage(f(HIDDEN));
        let proj = gpu.storage(f(GDN_PROJ));
        let mixed = gpu.storage(f(MIXER_OUT));
        let act = gpu.storage(f(INTERMEDIATE));
        let logits = gpu.storage(f(VOCAB));
        let best_val = gpu.storage(f(ARGMAX_GROUPS));
        let best_idx = gpu.storage(f(ARGMAX_GROUPS));
        let tok = gpu.upload(&[0u32, 0, 0, 0]);
        let mut state = Vec::new();
        // Chunk-sized activations for the batched prompt graph.
        let hb = gpu.storage(f(BATCH * HIDDEN));
        let xnb = gpu.storage(f(BATCH * HIDDEN));
        let projb = gpu.storage(f(BATCH * GDN_PROJ));
        let mixedb = gpu.storage(f(BATCH * MIXER_OUT));
        let actb = gpu.storage(f(BATCH * INTERMEDIATE));
        let ids = gpu.storage(f(BATCH));
        let chunk = gpu.upload(&[0u32, 0, 0, 0]);

        let embed = qmat(&gpu, "embed")?;
        let mut token = vec![gpu.simple("embed", shaders::embed, &[&embed.w, &embed.s, &tok, &h, &embed.z], 1)];
        let mut prefill = vec![gpu.simple("embed_b", batch::embed, &[&embed.w, &embed.s, &ids, &hb, &embed.z, &chunk], BATCH)];
        for i in 0..index.config.layers {
            let input_norm = upload(&gpu, &format!("layer{i}.input_norm"))?;
            let post_norm = upload(&gpu, &format!("layer{i}.post_norm"))?;
            token.push(gpu.simple("rmsnorm", shaders::rmsnorm, &[&h, &input_norm, &xn], 1));
            prefill.push(gpu.simple("rmsnorm_b", batch::rmsnorm, &[&hb, &input_norm, &xnb, &chunk], BATCH));
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
            token.push(gpu.simple("rmsnorm", shaders::rmsnorm, &[&h, &post_norm, &xn], 1));
            let gate_up = qmat(&gpu, &format!("layer{i}.gate_up"))?;
            token.extend(gpu.gemv(&gate_up, &xn, &act, Epilogue::SwiGlu, tiles.gate_up));
            let down = qmat(&gpu, &format!("layer{i}.down"))?;
            token.extend(gpu.gemv(&down, &act, &h, Epilogue::Accumulate, tiles.down));
            prefill.push(gpu.gemm(&mixer_out, &mixedb, &hb, Epilogue::Accumulate, &chunk));
            prefill.push(gpu.simple("rmsnorm_b", batch::rmsnorm, &[&hb, &post_norm, &xnb, &chunk], BATCH));
            prefill.push(gpu.gemm(&gate_up, &xnb, &actb, Epilogue::SwiGlu, &chunk));
            prefill.push(gpu.gemm(&down, &actb, &hb, Epilogue::Accumulate, &chunk));
        }
        let tail = token.len();
        let final_norm = upload(&gpu, "final_norm")?;
        token.push(gpu.simple("rmsnorm", shaders::rmsnorm, &[&h, &final_norm, &xn], 1));
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
        Ok(Self { gpu, token, prefill, tail, h, hb, ids, chunk, tok, logits, staging, state, adapter: format!("{adapter} via {backend}") })
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
        if ids.is_empty() || ids.len() as u32 >= MAX_CONTEXT {
            return Err(anyhow!("prompt length must be 1..{MAX_CONTEXT}"));
        }
        let mut last_len = 0;
        for (i, part) in ids.chunks(BATCH as usize).enumerate() {
            self.gpu.queue.write_buffer(&self.ids, 0, crate::as_bytes(part));
            self.gpu.queue.write_buffer(&self.chunk, 0, crate::as_bytes(&[i as u32 * BATCH, part.len() as u32, 0, 0]));
            let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
            self.gpu.encode(&mut encoder, &self.prefill, 1);
            self.gpu.queue.submit([encoder.finish()]);
            last_len = part.len();
        }
        // The greedy pick increments tok[1], which then holds the next position.
        self.gpu.queue.write_buffer(&self.tok, 0, crate::as_bytes(&[0u32, ids.len() as u32 - 1, 0, 0]));
        let row = (last_len as u64 - 1) * u64::from(HIDDEN) * 4;
        let mut encoder = self.gpu.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.hb, row, &self.h, 0, u64::from(HIDDEN) * 4);
        self.gpu.encode(&mut encoder, &self.token[self.tail..], 1);
        self.gpu.queue.submit([encoder.finish()]);
        self.gpu.idle().await
    }

    pub async fn logits(&self) -> Result<Vec<f32>> {
        self.gpu.read_f32(&self.logits, VOCAB as usize).await
    }

    /// The token picked by the last step (`tok[0]`) and the next position (`tok[1]`).
    pub async fn last_token(&self) -> Result<(u32, u32)> {
        let v = self.gpu.read_u32(&self.tok, 2).await?;
        Ok((v[0], v[1]))
    }

    /// Greedy decoding from the current state, `window` tokens per submission. The first
    /// returned token is the pick after the prompt; each step feeds back the previous pick.
    pub async fn decode(&self, n: usize, window: u32, stop: &[u32]) -> Result<Vec<u32>> {
        let mut out = vec![self.last_token().await?.0];
        while out.len() < n && !stop.contains(out.last().unwrap()) {
            let steps = window.min((n - out.len()) as u32);
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

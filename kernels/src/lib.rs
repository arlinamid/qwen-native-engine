//! Feasibility prototype for a hand-written WebGPU decoder at Qwen3.5-0.8B shapes.
//!
//! It answers one question before a full engine is written: how fast can this GPU decode a token
//! when every kernel is fused and the whole token (embedding, 24 layers, LM head, greedy pick) is
//! one submission with a 16-byte readback? Weights are random, but every matrix has the model's
//! real shape and 4-bit layout, so the memory traffic per token matches the real model. The
//! kernels are checked against CPU references before anything is timed.

pub mod batch;
pub mod model;
pub mod shaders;

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use shaders::{Epilogue, HIDDEN, MAX_CONTEXT};
use web_time::Instant;
use wgpu::util::DeviceExt;

const VOCAB: u32 = 248_320;
const LAYERS: u32 = 24;
const FULL_ATTENTION_INTERVAL: u32 = 4;
const GDN_PROJ: u32 = 8224; // q, k, v, z (4 x 2048) + a, b (2 x 16)
const ATTN_PROJ: u32 = 5120; // q, gate (2 x 2048) + k, v (2 x 512)
const MIXER_OUT: u32 = 2048;
const INTERMEDIATE: u32 = 3584;
const GDN_HEADS: u32 = 16;
const ATTN_HEADS: u32 = 8;
const ARGMAX_GROUPS: u32 = VOCAB.div_ceil(1024);
const TILES: [(u32, u32); 8] = [(8, 8), (16, 4), (16, 8), (16, 16), (32, 4), (32, 8), (64, 2), (64, 4)];
const TILES_F16: [(u32, u32, u32); 12] = [
    (8, 8, 1), (8, 16, 1), (16, 8, 1), (16, 16, 1),
    (8, 8, 2), (8, 16, 2), (16, 8, 2), (4, 32, 2),
    (8, 8, 4), (8, 16, 4), (16, 4, 4), (4, 32, 4),
];

/// A GEMV kernel choice: the plain f32 kernel or the f16 one, and its tile shape.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Tile {
    pub f16: bool,
    pub tpr: u32,
    pub rpw: u32,
    /// Rows per thread group (f16 kernel only).
    pub rpt: u32,
    /// Split-K factor (f16 kernel only): 1, or the number of workgroups sharing a row group.
    pub splits: u32,
}

const TILES_SPLIT: [(u32, u32, u32); 5] = [(8, 8, 1), (8, 8, 2), (16, 4, 2), (8, 16, 1), (4, 16, 4)];

fn candidates(f16: bool, k: u32) -> Vec<Tile> {
    let plain = TILES.iter().map(|&(tpr, rpw)| Tile { f16: false, tpr, rpw, rpt: 1, splits: 1 });
    let half = TILES_F16.iter().filter(|_| f16).map(|&(tpr, rpw, rpt)| Tile { f16: true, tpr, rpw, rpt, splits: 1 });
    // A split needs whole 32-value blocks per slice and at least one block per lane.
    let split = [2u32, 4, 8].into_iter().filter(|_| f16).flat_map(|splits| {
        TILES_SPLIT
            .iter()
            .filter(move |&&(tpr, _, _)| k % (32 * splits) == 0 && k / 32 / splits >= tpr)
            .map(move |&(tpr, rpw, rpt)| Tile { f16: true, tpr, rpw, rpt, splits })
    });
    plain.chain(half).chain(split).collect()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct Config {
    /// Tokens already in the attention KV-cache when timing starts.
    pub context: u32,
    /// Timed tokens per submission-window setting.
    pub tokens: u32,
    /// Tokens encoded per submission. 1 is the plain loop; more amortizes submission and readback.
    pub windows: Vec<u32>,
    /// Search the GEMV tile shape per matrix shape instead of using (16, 8).
    pub tune: bool,
    pub check: bool,
    /// Use the f16 GEMV when the GPU offers it. Off for drivers that crash on the feature.
    pub f16: bool,
    /// Native D3D12 only: path of `dxcompiler.dll`. wgpu's default FXC compiler cannot emit f16,
    /// DXC (shader model 6.2+) can.
    pub dxc: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self { context: 512, tokens: 32, windows: vec![1, 4], tune: true, check: true, f16: true, dxc: None }
    }
}

#[derive(Serialize)]
pub struct GemvTiming {
    pub name: String,
    pub rows: u32,
    pub k: u32,
    pub tile: Tile,
    pub ms: f64,
    pub gb_per_s: f64,
}

#[derive(Serialize)]
pub struct Check {
    pub name: String,
    pub rel_err: f32,
    pub ok: bool,
}

#[derive(Serialize)]
pub struct TokenTiming {
    pub window: u32,
    pub ms_per_token: f64,
    pub tokens_per_s: f64,
}

#[derive(Serialize, Default)]
pub struct Report {
    pub adapter: String,
    pub backend: String,
    pub shader_f16: bool,
    pub checks: Vec<Check>,
    pub dispatch_us: f64,
    /// Read bandwidth of a pure streaming kernel over a weight-sized buffer.
    pub stream_gb_per_s: f64,
    pub gemv: Vec<GemvTiming>,
    pub gdn_layer_ms: f64,
    pub attention_layer_ms: f64,
    pub weight_mb: f64,
    pub dispatches_per_token: usize,
    pub tokens: Vec<TokenTiming>,
}

struct Gpu {
    /// Kept alive for the device's lifetime: in the browser, dropping it invalidates later maps.
    _instance: wgpu::Instance,
    device: wgpu::Device,
    queue: wgpu::Queue,
    f16: bool,
    pipelines: HashMap<String, wgpu::ComputePipeline>,
}

/// A 4-bit matrix in the layout of `shaders::gemv`.
pub(crate) struct QMat {
    pub(crate) w: wgpu::Buffer,
    pub(crate) s: wgpu::Buffer,
    pub(crate) z: wgpu::Buffer,
    pub(crate) rows: u32,
    pub(crate) k: u32,
}

impl QMat {
    fn bytes(&self) -> u64 {
        let values = u64::from(self.rows) * u64::from(self.k);
        values / 2 + values / 16 + values / 64
    }
}

#[derive(Clone)]
struct Dispatch {
    pipeline: wgpu::ComputePipeline,
    bind: wgpu::BindGroup,
    groups: (u32, u32),
}

pub(crate) fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: only used with u32 and f32, which have no padding or invalid bit patterns.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast(), std::mem::size_of_val(v)) }
}

struct Rng(u64);

impl Rng {
    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }
    fn f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
    }
}

fn groups_for(n: u32) -> (u32, u32) {
    if n <= 65_535 { (n, 1) } else { (65_535, n.div_ceil(65_535)) }
}

impl Gpu {
    async fn new(allow_f16: bool, dxc: Option<&str>) -> Result<(Self, String, String)> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        if let Some(path) = dxc {
            desc.backend_options.dx12.shader_compiler = wgpu::Dx12Compiler::DynamicDxc { dxc_path: path.to_string() };
        }
        let instance = wgpu::Instance::new(desc);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })
            .await
            .context("no GPU adapter")?;
        let info = adapter.get_info();
        let f16 = allow_f16 && adapter.features().contains(wgpu::Features::SHADER_F16);
        let required_features = if f16 { wgpu::Features::SHADER_F16 } else { wgpu::Features::empty() };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor { label: Some("decode"), required_features, required_limits: adapter.limits(), ..Default::default() })
            .await?;
        let name = format!("{} ({:?}, driver {})", info.name, info.device_type, info.driver_info);
        Ok((Self { _instance: instance, device, queue, f16, pipelines: HashMap::new() }, name, format!("{:?}", info.backend)))
    }

    fn pipeline(&mut self, key: &str, source: impl FnOnce() -> String) -> wgpu::ComputePipeline {
        if let Some(p) = self.pipelines.get(key) {
            return p.clone();
        }
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(key),
            source: wgpu::ShaderSource::Wgsl(source().into()),
        });
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(key),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        self.pipelines.insert(key.to_string(), pipeline.clone());
        pipeline
    }

    fn storage(&self, bytes: u64) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes.max(16),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    fn upload<T: Copy>(&self, data: &[T]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: as_bytes(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        })
    }

    fn uniform(&self, data: [u32; 4]) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: as_bytes(&data),
            usage: wgpu::BufferUsages::UNIFORM,
        })
    }

    /// Fills a buffer with random words in chunks, so a 400 MB model never sits in host memory.
    fn fill_random(&self, buffer: &wgpu::Buffer, words: u64, rng: &mut Rng) {
        const CHUNK: u64 = 1 << 20;
        let mut chunk = vec![0u32; CHUNK as usize];
        let mut offset = 0u64;
        while offset < words {
            let n = (words - offset).min(CHUNK) as usize;
            chunk[..n].iter_mut().for_each(|w| *w = rng.next_u32());
            self.queue.write_buffer(buffer, offset * 4, as_bytes(&chunk[..n]));
            offset += n as u64;
        }
    }

    fn fill(&self, buffer: &wgpu::Buffer, words: u64, value: u32) {
        let chunk = vec![value; words.min(1 << 20) as usize];
        let mut offset = 0u64;
        while offset < words {
            let n = (words - offset).min(chunk.len() as u64) as usize;
            self.queue.write_buffer(buffer, offset * 4, as_bytes(&chunk[..n]));
            offset += n as u64;
        }
    }

    /// Random 4-bit weights with a small constant scale (0.02 as f16) and zero point 8, so
    /// activations stay finite.
    fn qmat(&self, rows: u32, k: u32, rng: &mut Rng) -> QMat {
        let words = u64::from(rows) * u64::from(k) / 8;
        let scale_words = u64::from(rows) * u64::from(k) / 64;
        let zp_words = u64::from(rows) * u64::from(k) / 256;
        let w = self.storage(words * 4);
        let s = self.storage(scale_words * 4);
        let z = self.storage(zp_words * 4);
        self.fill_random(&w, words, rng);
        self.fill(&s, scale_words, 0x251F_251F);
        self.fill(&z, zp_words, 0x8888_8888);
        QMat { w, s, z, rows, k }
    }

    fn bind(&self, pipeline: &wgpu::ComputePipeline, buffers: &[&wgpu::Buffer]) -> wgpu::BindGroup {
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
            .collect();
        self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipeline.get_bind_group_layout(0), entries: &entries })
    }

    fn gemv(&mut self, m: &QMat, x: &wgpu::Buffer, y: &wgpu::Buffer, epilogue: Epilogue, tile: Tile) -> Vec<Dispatch> {
        let Tile { f16, tpr, rpw, rpt, splits } = tile;
        let k = m.k;
        if splits > 1 {
            // Partial sums over every row of the matrix (both halves for SwiGLU), then the epilogue.
            let partial_pipeline = self.pipeline(&format!("gemv_split_{tpr}_{rpw}_{rpt}_{k}_{splits}"), || shaders::gemv_split_f16(tpr, rpw, rpt, k, splits));
            let partial = self.storage(u64::from(splits * m.rows) * 4);
            let params = self.uniform([m.rows, k, 0, 0]);
            let bind = self.bind(&partial_pipeline, &[&m.w, &m.s, x, &partial, &params, &m.z]);
            let out_rows = if epilogue == Epilogue::SwiGlu { m.rows / 2 } else { m.rows };
            let reduce = self.pipeline(&format!("split_reduce_{splits}_{epilogue:?}"), || shaders::split_reduce(splits, epilogue));
            let reduce_params = self.uniform([out_rows, m.rows, out_rows, 0]);
            let reduce_bind = self.bind(&reduce, &[&partial, y, &reduce_params]);
            return vec![
                Dispatch { pipeline: partial_pipeline, bind, groups: groups_for(m.rows.div_ceil(rpw * rpt) * splits) },
                Dispatch { pipeline: reduce, bind: reduce_bind, groups: (out_rows.div_ceil(64), 1) },
            ];
        }
        let pipeline = if f16 {
            self.pipeline(&format!("gemv_f16_{tpr}_{rpw}_{rpt}_{k}_{epilogue:?}"), || shaders::gemv_f16(tpr, rpw, rpt, k, epilogue))
        } else {
            self.pipeline(&format!("gemv_{tpr}_{rpw}_{epilogue:?}"), || shaders::gemv(tpr, rpw, epilogue))
        };
        let rows = if epilogue == Epilogue::SwiGlu { m.rows / 2 } else { m.rows };
        let params = self.uniform([rows, m.k, rows, 0]);
        let bind = self.bind(&pipeline, &[&m.w, &m.s, x, y, &params, &m.z]);
        vec![Dispatch { pipeline, bind, groups: groups_for(rows.div_ceil(rpw * rpt)) }]
    }

    /// Batched 4-bit matrix product over the chunk described by `chunk` ([pos0, count]).
    fn gemm(&mut self, m: &QMat, x: &wgpu::Buffer, y: &wgpu::Buffer, epilogue: Epilogue, chunk: &wgpu::Buffer) -> Dispatch {
        let pipeline = self.pipeline(&format!("gemm_{epilogue:?}"), || batch::gemm(epilogue));
        let rows = if epilogue == Epilogue::SwiGlu { m.rows / 2 } else { m.rows };
        let params = self.uniform([rows, m.k, rows, 0]);
        let bind = self.bind(&pipeline, &[&m.w, &m.s, x, y, &params, &m.z, chunk]);
        Dispatch { pipeline, bind, groups: groups_for(rows.div_ceil(64)) }
    }

    fn simple(&mut self, key: &str, source: fn() -> String, buffers: &[&wgpu::Buffer], groups: u32) -> Dispatch {
        let pipeline = self.pipeline(key, source);
        let bind = self.bind(&pipeline, buffers);
        Dispatch { pipeline, bind, groups: (groups, 1) }
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder, dispatches: &[Dispatch], repeat: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
        for _ in 0..repeat {
            for d in dispatches {
                pass.set_pipeline(&d.pipeline);
                pass.set_bind_group(0, &d.bind, &[]);
                pass.dispatch_workgroups(d.groups.0, d.groups.1, 1);
            }
        }
    }

    fn poll(&self) {
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }

    async fn idle(&self) -> Result<()> {
        let (tx, rx) = futures_channel::oneshot::channel();
        self.queue.on_submitted_work_done(move || {
            let _ = tx.send(());
        });
        self.poll();
        rx.await.map_err(|_| anyhow!("GPU queue closed"))
    }

    /// Runs `dispatches` `repeat` times in one submission and returns the wall time in ms.
    async fn time(&self, dispatches: &[Dispatch], repeat: u32) -> Result<f64> {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.encode(&mut encoder, dispatches, repeat);
        let start = Instant::now();
        self.queue.submit([encoder.finish()]);
        self.idle().await?;
        Ok(start.elapsed().as_secs_f64() * 1000.0)
    }

    async fn read(&self, buffer: &wgpu::Buffer, bytes: u64) -> Result<Vec<u8>> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
        self.queue.submit([encoder.finish()]);
        map_read(self, &staging).await
    }

    async fn read_f32(&self, buffer: &wgpu::Buffer, n: usize) -> Result<Vec<f32>> {
        let bytes = self.read(buffer, n as u64 * 4).await?;
        Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    async fn read_u32(&self, buffer: &wgpu::Buffer, n: usize) -> Result<Vec<u32>> {
        let bytes = self.read(buffer, n as u64 * 4).await?;
        Ok(bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    fn submit(&self, dispatches: &[Dispatch]) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.encode(&mut encoder, dispatches, 1);
        self.queue.submit([encoder.finish()]);
    }
}

async fn map_read(gpu: &Gpu, staging: &wgpu::Buffer) -> Result<Vec<u8>> {
    let slice = staging.slice(..);
    let (tx, rx) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    gpu.poll();
    rx.await.map_err(|_| anyhow!("map cancelled"))??;
    let data = staging.get_mapped_range(..)?.to_vec();
    staging.unmap();
    Ok(data)
}

// ---------------------------------------------------------------------------------------------
// CPU references
// ---------------------------------------------------------------------------------------------

fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((h >> 10) & 0x1f);
    let frac = f32::from(h & 0x3ff);
    match exp {
        0 => sign * frac * 2f32.powi(-24),
        31 => f32::NAN,
        _ => sign * (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
    }
}

fn dequant_row(w: &[u32], s: &[u32], z: &[u32], row: usize, k: usize) -> Vec<f32> {
    let words = k / 8;
    (0..k)
        .map(|c| {
            let i = c / 8;
            let blk = c / 32;
            let q = (w[row * words + i] >> (4 * (c % 8))) & 15;
            let sw = s[row * (k / 64) + blk / 2];
            let scale = f16_to_f32(if blk % 2 == 0 { sw as u16 } else { (sw >> 16) as u16 });
            let zp = (z[row * (k / 256) + blk / 8] >> (4 * (blk % 8))) & 15;
            (q as f32 - zp as f32) * scale
        })
        .collect()
}

/// Largest absolute error relative to the largest reference value. NaN and infinity count as an
/// infinite error: `f32::max` skips NaN, so an all-NaN output would otherwise score zero.
fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    if got.len() != want.len() || got.iter().chain(want).any(|v| !v.is_finite()) {
        return f32::INFINITY;
    }
    let scale = want.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
    got.iter().zip(want).fold(0f32, |m, (g, w)| m.max((g - w).abs())) / scale
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

async fn check_gemv(gpu: &mut Gpu, epilogue: Epilogue, f16: bool, splits: u32, rng: &mut Rng) -> Result<Check> {
    let (out_rows, k) = (37u32, 1024u32);
    let rows = if epilogue == Epilogue::SwiGlu { out_rows * 2 } else { out_rows };
    let w: Vec<u32> = (0..rows * k / 8).map(|_| rng.next_u32()).collect();
    let s: Vec<u32> = (0..rows * k / 64).map(|_| 0x2000_2400 + (rng.next_u32() & 0x00ff_00ff)).collect();
    let z: Vec<u32> = (0..rows * k / 256).map(|_| rng.next_u32()).collect();
    let x: Vec<f32> = (0..k).map(|_| rng.f32()).collect();
    let y0: Vec<f32> = (0..out_rows).map(|_| rng.f32()).collect();
    let m = QMat { w: gpu.upload(&w), s: gpu.upload(&s), z: gpu.upload(&z), rows, k };
    let (xb, yb) = (gpu.upload(&x), gpu.upload(&y0));
    let tile = if splits > 1 { Tile { f16, tpr: 8, rpw: 4, rpt: 2, splits } } else { Tile { f16, tpr: 16, rpw: 4, rpt: if f16 { 4 } else { 1 }, splits: 1 } };
    let dispatches = gpu.gemv(&m, &xb, &yb, epilogue, tile);
    gpu.submit(&dispatches);
    let got = gpu.read_f32(&yb, out_rows as usize).await?;

    let dot = |r: usize| dequant_row(&w, &s, &z, r, k as usize).iter().zip(&x).map(|(a, b)| a * b).sum::<f32>();
    let want: Vec<f32> = (0..out_rows as usize)
        .map(|r| match epilogue {
            Epilogue::Store => dot(r),
            Epilogue::Accumulate => y0[r] + dot(r),
            Epilogue::SwiGlu => silu(dot(r)) * dot(r + out_rows as usize),
        })
        .collect();
    let err = rel_err(&got, &want);
    // f16 partial sums keep about three significant digits.
    let tolerance = if f16 { 5e-3 } else { 1e-4 };
    let kind = match (f16, splits) {
        (false, _) => String::new(),
        (true, 1) => " f16".into(),
        (true, n) => format!(" f16 split{n}"),
    };
    Ok(Check { name: format!("gemv{kind} {epilogue:?}"), rel_err: err, ok: err < tolerance })
}

async fn check_gemm(gpu: &mut Gpu, epilogue: Epilogue, rng: &mut Rng) -> Result<Check> {
    let (out_rows, k, count) = (70u32, 1024u32, 11u32);
    let rows = if epilogue == Epilogue::SwiGlu { out_rows * 2 } else { out_rows };
    let w: Vec<u32> = (0..rows * k / 8).map(|_| rng.next_u32()).collect();
    let s: Vec<u32> = (0..rows * k / 64).map(|_| 0x2000_2400 + (rng.next_u32() & 0x00ff_00ff)).collect();
    let z: Vec<u32> = (0..rows * k / 256).map(|_| rng.next_u32()).collect();
    let x: Vec<f32> = (0..batch::BATCH * k).map(|_| rng.f32()).collect();
    let y0: Vec<f32> = (0..batch::BATCH * out_rows).map(|_| rng.f32()).collect();
    let m = QMat { w: gpu.upload(&w), s: gpu.upload(&s), z: gpu.upload(&z), rows, k };
    let (xb, yb, chunk) = (gpu.upload(&x), gpu.upload(&y0), gpu.upload(&[0u32, count, 0, 0]));
    let d = gpu.gemm(&m, &xb, &yb, epilogue, &chunk);
    gpu.submit(&[d]);
    let got = gpu.read_f32(&yb, (count * out_rows) as usize).await?;

    let rows_f: Vec<Vec<f32>> = (0..rows as usize).map(|r| dequant_row(&w, &s, &z, r, k as usize)).collect();
    let mut want = Vec::new();
    for t in 0..count as usize {
        let xt = &x[t * k as usize..(t + 1) * k as usize];
        let dot = |r: usize| rows_f[r].iter().zip(xt).map(|(a, b)| a * b).sum::<f32>();
        for r in 0..out_rows as usize {
            want.push(match epilogue {
                Epilogue::Store => dot(r),
                Epilogue::Accumulate => y0[t * out_rows as usize + r] + dot(r),
                Epilogue::SwiGlu => silu(dot(r)) * dot(r + out_rows as usize),
            });
        }
    }
    let err = rel_err(&got, &want);
    Ok(Check { name: format!("gemm batch {epilogue:?}"), rel_err: err, ok: err < 1e-4 })
}

async fn check_gdn(gpu: &mut Gpu, rng: &mut Rng) -> Result<Vec<Check>> {
    let proj: Vec<f32> = (0..GDN_PROJ).map(|_| rng.f32()).collect();
    let conv0: Vec<f32> = (0..6144 * 3).map(|_| rng.f32()).collect();
    let conv_w: Vec<f32> = (0..6144 * 4).map(|_| rng.f32() * 0.5).collect();
    let state0: Vec<f32> = (0..GDN_HEADS * 16384).map(|_| rng.f32() * 0.1).collect();
    let prm: Vec<f32> = (0..160).map(|i| if i < 16 { rng.f32() * 4.0 } else if i < 32 { -0.5 + rng.f32() * 0.4 } else { 1.0 + rng.f32() * 0.3 }).collect();
    let bufs = [gpu.upload(&proj), gpu.upload(&conv0), gpu.upload(&conv_w), gpu.upload(&state0), gpu.storage(u64::from(MIXER_OUT) * 4), gpu.upload(&prm)];
    let d = gpu.simple("gdn", shaders::gdn, &[&bufs[0], &bufs[1], &bufs[2], &bufs[3], &bufs[4], &bufs[5]], GDN_HEADS);
    gpu.submit(&[d]);
    let got_o = gpu.read_f32(&bufs[4], MIXER_OUT as usize).await?;
    let got_state = gpu.read_f32(&bufs[3], state0.len()).await?;

    let conv = |c: usize| {
        let y = conv0[c * 3] * conv_w[c * 4] + conv0[c * 3 + 1] * conv_w[c * 4 + 1] + conv0[c * 3 + 2] * conv_w[c * 4 + 2] + proj[c] * conv_w[c * 4 + 3];
        silu(y)
    };
    let mut want_o = vec![0f32; MIXER_OUT as usize];
    let mut want_state = state0.clone();
    for h in 0..GDN_HEADS as usize {
        let q: Vec<f32> = (0..128).map(|t| conv(h * 128 + t)).collect();
        let k: Vec<f32> = (0..128).map(|t| conv(2048 + h * 128 + t)).collect();
        let v: Vec<f32> = (0..128).map(|t| conv(4096 + h * 128 + t)).collect();
        let nq = 1.0 / (q.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
        let nk = 1.0 / (k.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt();
        let qs: Vec<f32> = q.iter().map(|x| x * nq * 0.088_388_35).collect();
        let ks: Vec<f32> = k.iter().map(|x| x * nk).collect();
        let g = (prm[16 + h] * (1.0 + (proj[8192 + h] + prm[h]).exp()).ln()).exp();
        let beta = 1.0 / (1.0 + (-proj[8208 + h]).exp());
        let kq: f32 = ks.iter().zip(&qs).map(|(a, b)| a * b).sum();
        let mut out = vec![0f32; 128];
        for (t, out_t) in out.iter_mut().enumerate() {
            let at = |j: usize| h * 16384 + j * 128 + t;
            let sk: f32 = (0..128).map(|j| state0[at(j)] * ks[j]).sum();
            let sq: f32 = (0..128).map(|j| state0[at(j)] * qs[j]).sum();
            let dv = beta * (v[t] - g * sk);
            for (j, kj) in ks.iter().enumerate() {
                want_state[at(j)] = g * state0[at(j)] + dv * kj;
            }
            *out_t = g * sq + dv * kq;
        }
        let inv = 1.0 / (out.iter().map(|x| x * x).sum::<f32>() / 128.0 + 1e-6).sqrt();
        for t in 0..128 {
            want_o[h * 128 + t] = out[t] * inv * silu(proj[6144 + h * 128 + t]) * prm[32 + t];
        }
    }
    let (eo, es) = (rel_err(&got_o, &want_o), rel_err(&got_state, &want_state));
    Ok(vec![
        Check { name: "gdn output".into(), rel_err: eo, ok: eo < 1e-3 },
        Check { name: "gdn state".into(), rel_err: es, ok: es < 1e-4 },
    ])
}

async fn check_attention(gpu: &mut Gpu, rng: &mut Rng) -> Result<Check> {
    let pos = 5usize;
    let proj: Vec<f32> = (0..ATTN_PROJ).map(|_| rng.f32()).collect();
    let cache_len = 2 * MAX_CONTEXT as usize * 256;
    let kc: Vec<f32> = (0..cache_len).map(|_| rng.f32()).collect();
    let vc: Vec<f32> = (0..cache_len).map(|_| rng.f32()).collect();
    let tok = [0u32, pos as u32, 0, 0];
    let prm: Vec<f32> = (0..512).map(|_| 1.0 + rng.f32() * 0.5).collect();
    let bufs = [gpu.upload(&proj), gpu.upload(&kc), gpu.upload(&vc), gpu.upload(&tok), gpu.storage(u64::from(MIXER_OUT) * 4), gpu.upload(&prm)];
    let d = gpu.simple("attention", shaders::attention, &[&bufs[0], &bufs[1], &bufs[2], &bufs[3], &bufs[4], &bufs[5]], ATTN_HEADS);
    gpu.submit(&[d]);
    let got = gpu.read_f32(&bufs[4], MIXER_OUT as usize).await?;

    let norm_rope = |x: &[f32], w: &[f32]| -> Vec<f32> {
        let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() / 256.0 + 1e-6).sqrt();
        let mut y: Vec<f32> = x.iter().zip(w).map(|(v, w)| v * inv * w).collect();
        for i in 0..32 {
            let angle = pos as f32 * 10_000_000f32.powf(-((2 * i) as f32) / 64.0);
            let (s, c) = angle.sin_cos();
            let (a, b) = (y[i], y[i + 32]);
            y[i] = a * c - b * s;
            y[i + 32] = b * c + a * s;
        }
        y
    };
    let mut want = vec![0f32; MIXER_OUT as usize];
    for h in 0..ATTN_HEADS as usize {
        let kvh = h / 4;
        let q = norm_rope(&proj[h * 256..h * 256 + 256], &prm[..256]);
        let kcur = norm_rope(&proj[4096 + kvh * 256..4096 + kvh * 256 + 256], &prm[256..]);
        let key = |l: usize, j: usize| if l == pos { kcur[j] } else { kc[(kvh * MAX_CONTEXT as usize + l) * 256 + j] };
        let val = |l: usize, j: usize| if l == pos { proj[4608 + kvh * 256 + j] } else { vc[(kvh * MAX_CONTEXT as usize + l) * 256 + j] };
        let scores: Vec<f32> = (0..=pos).map(|l| (0..256).map(|j| q[j] * key(l, j)).sum::<f32>() * 0.0625).collect();
        let m = scores.iter().cloned().fold(f32::MIN, f32::max);
        let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
        let total: f32 = e.iter().sum();
        for t in 0..256 {
            let acc: f32 = (0..=pos).map(|l| e[l] * val(l, t)).sum();
            want[h * 256 + t] = acc / total / (1.0 + (-proj[2048 + h * 256 + t]).exp());
        }
    }
    let err = rel_err(&got, &want);
    Ok(Check { name: "attention".into(), rel_err: err, ok: err < 1e-3 })
}

async fn check_embed_and_argmax(gpu: &mut Gpu, rng: &mut Rng) -> Result<Vec<Check>> {
    let rows = 300u32;
    let w: Vec<u32> = (0..rows * HIDDEN / 8).map(|_| rng.next_u32()).collect();
    let s: Vec<u32> = (0..rows * HIDDEN / 64).map(|_| 0x2000_2400 + (rng.next_u32() & 0x00ff_00ff)).collect();
    let z: Vec<u32> = (0..rows * HIDDEN / 256).map(|_| rng.next_u32()).collect();
    let mut logits: Vec<f32> = (0..VOCAB).map(|_| rng.f32()).collect();
    logits[123_457] = 9.0;
    let bufs = [
        gpu.upload(&w),
        gpu.upload(&s),
        gpu.upload(&[7u32, 41, 0, 0]),
        gpu.storage(u64::from(HIDDEN) * 4),
        gpu.upload(&logits),
        gpu.storage(u64::from(ARGMAX_GROUPS) * 4),
        gpu.storage(u64::from(ARGMAX_GROUPS) * 4),
        gpu.upload(&z),
    ];
    let embed = gpu.simple("embed", shaders::embed, &[&bufs[0], &bufs[1], &bufs[2], &bufs[3], &bufs[7]], 1);
    gpu.submit(&[embed]);
    let got = gpu.read_f32(&bufs[3], HIDDEN as usize).await?;
    let want = dequant_row(&w, &s, &z, 7, HIDDEN as usize);
    let e_embed = rel_err(&got, &want);

    let p1 = gpu.uniform([VOCAB, 0, 0, 0]);
    let p2 = gpu.uniform([ARGMAX_GROUPS, 0, 0, 0]);
    let a1 = gpu.simple("argmax_partial", shaders::argmax_partial, &[&bufs[4], &bufs[5], &bufs[6], &p1], ARGMAX_GROUPS);
    let a2 = gpu.simple("argmax_final", shaders::argmax_final, &[&bufs[5], &bufs[6], &bufs[2], &p2], 1);
    gpu.submit(&[a1, a2]);
    let tok = gpu.read_u32(&bufs[2], 2).await?;
    Ok(vec![
        Check { name: "embed".into(), rel_err: e_embed, ok: e_embed < 1e-6 },
        Check { name: format!("argmax -> {} pos {}", tok[0], tok[1]), rel_err: 0.0, ok: tok == [123_457, 42] },
    ])
}

// ---------------------------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------------------------

async fn dispatch_cost(gpu: &mut Gpu) -> Result<f64> {
    let y = gpu.storage(256);
    let d = gpu.simple("noop", shaders::noop, &[&y], 1);
    gpu.time(&[d.clone()], 10).await?;
    let n = 2000;
    Ok(gpu.time(&[d], n).await? * 1000.0 / f64::from(n))
}

/// Times one GEMV shape for each candidate tile and returns the fastest.
async fn tune_gemv(gpu: &mut Gpu, name: &str, rows: u32, k: u32, epilogue: Epilogue, tune: bool, rng: &mut Rng, out: &mut Vec<GemvTiming>) -> Result<Tile> {
    let m = gpu.qmat(rows, k, rng);
    let x = gpu.storage(u64::from(k) * 4);
    let y = gpu.storage(u64::from(rows) * 4);
    let fallback = Tile { f16: gpu.f16, tpr: 16, rpw: 8, rpt: 1, splits: 1 };
    let candidates = if tune { candidates(gpu.f16, k) } else { vec![fallback] };
    let mut best = (f64::MAX, fallback);
    for tile in candidates {
        let d = gpu.gemv(&m, &x, &y, epilogue, tile);
        gpu.time(&d, 3).await?;
        let repeat = if rows > 100_000 { 5 } else { 30 };
        let ms = gpu.time(&d, repeat).await? / f64::from(repeat);
        if ms < best.0 {
            best = (ms, tile);
        }
    }
    let (ms, tile) = best;
    out.push(GemvTiming { name: name.into(), rows, k, tile, ms, gb_per_s: m.bytes() as f64 / (ms / 1000.0) / 1e9 });
    Ok(tile)
}

struct Tiles {
    gdn_in: Tile,
    attn_in: Tile,
    out: Tile,
    gate_up: Tile,
    down: Tile,
    lm_head: Tile,
}

/// Builds every buffer and dispatch of one decode token at the model's real shapes.
fn build_token(gpu: &mut Gpu, tiles: &Tiles, context: u32, rng: &mut Rng) -> (Vec<Dispatch>, wgpu::Buffer, u64) {
    let f = |n: u32| u64::from(n) * 4;
    let h = gpu.storage(f(HIDDEN));
    let xn = gpu.storage(f(HIDDEN));
    let proj = gpu.storage(f(GDN_PROJ));
    let mixed = gpu.storage(f(MIXER_OUT));
    let act = gpu.storage(f(INTERMEDIATE));
    let logits = gpu.storage(f(VOCAB));
    let best_val = gpu.storage(f(ARGMAX_GROUPS));
    let best_idx = gpu.storage(f(ARGMAX_GROUPS));
    let tok = gpu.upload(&[1u32, context, 0, 0]);
    let norm = gpu.upload(&vec![1.0f32; HIDDEN as usize]);
    let conv_w = gpu.upload(&(0..6144 * 4).map(|_| rng.f32() * 0.5).collect::<Vec<f32>>());
    let embedding = gpu.qmat(VOCAB, HIDDEN, rng);
    let mut weight_bytes = embedding.bytes();

    let gdn_params = gpu.upload(&(0..160).map(|i| if i < 16 { 0.0 } else if i < 32 { -0.5 } else { 1.0 }).collect::<Vec<f32>>());
    let attn_params = gpu.upload(&vec![1.0f32; 512]);
    let mut list = vec![gpu.simple("embed", shaders::embed, &[&embedding.w, &embedding.s, &tok, &h, &embedding.z], 1)];
    for layer in 0..LAYERS {
        list.push(gpu.simple("rmsnorm", shaders::rmsnorm, &[&h, &norm, &xn], 1));
        if (layer + 1) % FULL_ATTENTION_INTERVAL == 0 {
            let w_in = gpu.qmat(ATTN_PROJ, HIDDEN, rng);
            let kc = gpu.storage(f(2 * MAX_CONTEXT * 256));
            let vc = gpu.storage(f(2 * MAX_CONTEXT * 256));
            list.extend(gpu.gemv(&w_in, &xn, &proj, Epilogue::Store, tiles.attn_in));
            list.push(gpu.simple("attention", shaders::attention, &[&proj, &kc, &vc, &tok, &mixed, &attn_params], ATTN_HEADS));
            weight_bytes += w_in.bytes();
        } else {
            let w_in = gpu.qmat(GDN_PROJ, HIDDEN, rng);
            let conv_state = gpu.storage(f(6144 * 3));
            let state = gpu.storage(f(GDN_HEADS * 16384));
            list.extend(gpu.gemv(&w_in, &xn, &proj, Epilogue::Store, tiles.gdn_in));
            list.push(gpu.simple("gdn", shaders::gdn, &[&proj, &conv_state, &conv_w, &state, &mixed, &gdn_params], GDN_HEADS));
            weight_bytes += w_in.bytes();
        }
        let w_out = gpu.qmat(HIDDEN, MIXER_OUT, rng);
        list.extend(gpu.gemv(&w_out, &mixed, &h, Epilogue::Accumulate, tiles.out));
        list.push(gpu.simple("rmsnorm", shaders::rmsnorm, &[&h, &norm, &xn], 1));
        let w_gate_up = gpu.qmat(2 * INTERMEDIATE, HIDDEN, rng);
        list.extend(gpu.gemv(&w_gate_up, &xn, &act, Epilogue::SwiGlu, tiles.gate_up));
        let w_down = gpu.qmat(HIDDEN, INTERMEDIATE, rng);
        list.extend(gpu.gemv(&w_down, &act, &h, Epilogue::Accumulate, tiles.down));
        weight_bytes += w_out.bytes() + w_gate_up.bytes() + w_down.bytes();
    }
    list.push(gpu.simple("rmsnorm", shaders::rmsnorm, &[&h, &norm, &xn], 1));
    // Tied embeddings: the LM head reads the same matrix as the embedding lookup.
    list.extend(gpu.gemv(&embedding, &xn, &logits, Epilogue::Store, tiles.lm_head));
    let p1 = gpu.uniform([VOCAB, 0, 0, 0]);
    let p2 = gpu.uniform([ARGMAX_GROUPS, 0, 0, 0]);
    list.push(gpu.simple("argmax_partial", shaders::argmax_partial, &[&logits, &best_val, &best_idx, &p1], ARGMAX_GROUPS));
    list.push(gpu.simple("argmax_final", shaders::argmax_final, &[&best_val, &best_idx, &tok, &p2], 1));
    (list, tok, weight_bytes)
}

/// Decodes `tokens` tokens, `window` per submission, each submission ending in a 16-byte readback
/// of the token buffer, exactly as an interactive loop would.
async fn time_tokens(gpu: &Gpu, token: &[Dispatch], tok: &wgpu::Buffer, tokens: u32, window: u32) -> Result<f64> {
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let submissions = tokens.div_ceil(window);
    let start = Instant::now();
    for _ in 0..submissions {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        gpu.encode(&mut encoder, token, window);
        encoder.copy_buffer_to_buffer(tok, 0, &staging, 0, 16);
        gpu.queue.submit([encoder.finish()]);
        map_read(gpu, &staging).await?;
    }
    Ok(start.elapsed().as_secs_f64() * 1000.0 / f64::from(submissions * window))
}

/// Repeats the streaming-read probe over sizes and data patterns, to tell a real bandwidth limit
/// from clock or power-state noise. Returns GB/s per (size MB, random data, run).
pub async fn bandwidth_probe(config: Config, log: &dyn Fn(&str)) -> Result<Vec<(u64, bool, f64)>> {
    let (mut gpu, adapter, backend) = Gpu::new(config.f16, config.dxc.as_deref()).await?;
    log(&format!("adapter: {adapter} via {backend}"));
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let mut out = Vec::new();
    for mb in [64u64, 400] {
        for random in [false, true] {
            let words = mb << 18; // mb * 2^20 bytes / 4
            let data = gpu.storage(words * 4);
            if random { gpu.fill_random(&data, words, &mut rng) } else { gpu.fill(&data, words, 0x1234_5678) }
            let groups = 24 * 64;
            let sink = gpu.storage(u64::from(groups) * 256 * 4);
            let d = gpu.simple("bandwidth", shaders::bandwidth, &[&data, &sink], groups);
            gpu.time(&[d.clone()], 2).await?;
            for run in 0..3 {
                let repeat = if mb > 100 { 5 } else { 20 };
                let ms = gpu.time(&[d.clone()], repeat).await? / f64::from(repeat);
                let gbps = (words * 4) as f64 / (ms / 1000.0) / 1e9;
                log(&format!("stream {mb:>4} MB {:<8} run {run}: {gbps:.1} GB/s", if random { "random" } else { "constant" }));
                out.push((mb, random, gbps));
            }
        }
    }
    Ok(out)
}

pub async fn run(config: Config, log: &dyn Fn(&str)) -> Result<Report> {
    if config.context + 64 + config.tokens * config.windows.len() as u32 >= MAX_CONTEXT {
        return Err(anyhow!("context + timed tokens must stay below {MAX_CONTEXT}"));
    }
    let (mut gpu, adapter, backend) = Gpu::new(config.f16, config.dxc.as_deref()).await?;
    log(&format!("adapter: {adapter} via {backend}"));
    let mut report = Report { adapter, backend, shader_f16: gpu.f16, ..Default::default() };
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);

    if config.check {
        let variants = if gpu.f16 { vec![false, true] } else { vec![false] };
        for f16 in variants {
            for epilogue in [Epilogue::Store, Epilogue::Accumulate, Epilogue::SwiGlu] {
                report.checks.push(check_gemv(&mut gpu, epilogue, f16, 1, &mut rng).await?);
                if f16 {
                    report.checks.push(check_gemv(&mut gpu, epilogue, f16, 4, &mut rng).await?);
                }
            }
        }
        for epilogue in [Epilogue::Store, Epilogue::Accumulate, Epilogue::SwiGlu] {
            report.checks.push(check_gemm(&mut gpu, epilogue, &mut rng).await?);
        }
        report.checks.extend(check_gdn(&mut gpu, &mut rng).await?);
        report.checks.push(check_attention(&mut gpu, &mut rng).await?);
        report.checks.extend(check_embed_and_argmax(&mut gpu, &mut rng).await?);
        for c in &report.checks {
            log(&format!("check {:<28} rel_err {:.2e} {}", c.name, c.rel_err, if c.ok { "ok" } else { "FAILED" }));
        }
        if report.checks.iter().any(|c| !c.ok) {
            return Err(anyhow!("a kernel does not match its CPU reference"));
        }
    }

    report.dispatch_us = dispatch_cost(&mut gpu).await?;
    log(&format!("dispatch: {:.1} us each, back to back in one pass", report.dispatch_us));
    {
        let words = 100u64 << 20; // 400 MB, about the model's weights
        let data = gpu.storage(words * 4);
        gpu.fill(&data, words, 0x1234_5678);
        let groups = 24 * 64;
        let sink = gpu.storage(u64::from(groups) * 256 * 4);
        let d = gpu.simple("bandwidth", shaders::bandwidth, &[&data, &sink], groups);
        gpu.time(&[d.clone()], 1).await?;
        let ms = gpu.time(&[d], 5).await? / 5.0;
        report.stream_gb_per_s = (words * 4) as f64 / (ms / 1000.0) / 1e9;
        log(&format!("streaming read: {:.1} GB/s ({ms:.1} ms per 400 MB)", report.stream_gb_per_s));
    }

    let mut g = Vec::new();
    let tiles = Tiles {
        gdn_in: tune_gemv(&mut gpu, "gdn in-proj", GDN_PROJ, HIDDEN, Epilogue::Store, config.tune, &mut rng, &mut g).await?,
        attn_in: tune_gemv(&mut gpu, "attention in-proj", ATTN_PROJ, HIDDEN, Epilogue::Store, config.tune, &mut rng, &mut g).await?,
        out: tune_gemv(&mut gpu, "out-proj", HIDDEN, MIXER_OUT, Epilogue::Accumulate, config.tune, &mut rng, &mut g).await?,
        gate_up: tune_gemv(&mut gpu, "mlp gate+up (swiglu)", 2 * INTERMEDIATE, HIDDEN, Epilogue::SwiGlu, config.tune, &mut rng, &mut g).await?,
        down: tune_gemv(&mut gpu, "mlp down", HIDDEN, INTERMEDIATE, Epilogue::Accumulate, config.tune, &mut rng, &mut g).await?,
        lm_head: tune_gemv(&mut gpu, "lm head", VOCAB, HIDDEN, Epilogue::Store, config.tune, &mut rng, &mut g).await?,
    };
    for t in &g {
        let kind = if t.tile.f16 { "f16" } else { "f32" };
        log(&format!(
            "gemv {:<22} {:>6}x{:<5} {kind} tile {:>2}x{:<2}x{} split {} {:>7.3} ms {:>6.1} GB/s",
            t.name, t.rows, t.k, t.tile.tpr, t.tile.rpw, t.tile.rpt, t.tile.splits, t.ms, t.gb_per_s
        ));
    }
    report.gemv = g;

    // Mixer kernels alone, at the model's sizes.
    {
        let proj = gpu.storage(u64::from(GDN_PROJ) * 4);
        let o = gpu.storage(u64::from(MIXER_OUT) * 4);
        let conv_state = gpu.storage(6144 * 3 * 4);
        let conv_w = gpu.storage(6144 * 4 * 4);
        let state = gpu.storage(u64::from(GDN_HEADS) * 16384 * 4);
        let prm = gpu.upload(&vec![0.5f32; 160]);
        let d = gpu.simple("gdn", shaders::gdn, &[&proj, &conv_state, &conv_w, &state, &o, &prm], GDN_HEADS);
        gpu.time(&[d.clone()], 3).await?;
        report.gdn_layer_ms = gpu.time(&[d], 30).await? / 30.0;
        let kc = gpu.storage(2 * u64::from(MAX_CONTEXT) * 256 * 4);
        let vc = gpu.storage(2 * u64::from(MAX_CONTEXT) * 256 * 4);
        let tok = gpu.upload(&[0u32, config.context, 0, 0]);
        let d = gpu.simple("attention", shaders::attention, &[&proj, &kc, &vc, &tok, &o, &prm], ATTN_HEADS);
        gpu.time(&[d.clone()], 3).await?;
        report.attention_layer_ms = gpu.time(&[d], 30).await? / 30.0;
        log(&format!("mixers: gdn {:.3} ms/layer, attention {:.3} ms/layer at context {}", report.gdn_layer_ms, report.attention_layer_ms, config.context));
    }

    let (token, tok, weight_bytes) = build_token(&mut gpu, &tiles, config.context, &mut rng);
    gpu.idle().await?;
    report.weight_mb = weight_bytes as f64 / 1e6;
    report.dispatches_per_token = token.len();
    log(&format!("token: {} dispatches, {:.0} MB of weights", token.len(), report.weight_mb));
    time_tokens(&gpu, &token, &tok, 3, 1).await?;
    for &window in &config.windows {
        let ms = time_tokens(&gpu, &token, &tok, config.tokens, window).await?;
        log(&format!("decode, {window} token(s) per submission: {ms:.2} ms/token = {:.1} tok/s", 1000.0 / ms));
        report.tokens.push(TokenTiming { window, ms_per_token: ms, tokens_per_s: 1000.0 / ms });
    }
    Ok(report)
}

#[cfg(target_arch = "wasm32")]
mod web {
    use wasm_bindgen::prelude::*;

    /// `run_bench(config, onLog)`: resolves to the report object.
    #[wasm_bindgen]
    pub async fn run_bench(config: JsValue, on_log: js_sys_function::Function) -> Result<JsValue, JsValue> {
        console_error_panic_hook::set_once();
        let config: super::Config = if config.is_undefined() { Default::default() } else { serde_wasm_bindgen::from_value(config)? };
        let log = |line: &str| {
            let _ = on_log.call1(&JsValue::NULL, &JsValue::from_str(line));
        };
        let report = super::run(config, &log).await.map_err(|e| JsValue::from_str(&format!("{e:#}")))?;
        report
            .serialize(&serde_wasm_bindgen::Serializer::json_compatible())
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    use serde::Serialize;

    mod js_sys_function {
        pub use wasm_bindgen::JsValue;
        #[wasm_bindgen::prelude::wasm_bindgen]
        extern "C" {
            pub type Function;
            #[wasm_bindgen::prelude::wasm_bindgen(method, catch, js_name = call)]
            pub fn call1(this: &Function, ctx: &JsValue, arg: &JsValue) -> Result<JsValue, JsValue>;
        }
    }
}

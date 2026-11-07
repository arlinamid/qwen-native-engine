//! Kernels for batched prompt processing: `BATCH` tokens per pass. They read the chunk's start
//! position and token count from a small `chunk` buffer ([pos0, count]) and write the same
//! recurrent states and KV-cache as the decode kernels, so decoding continues where they stop.

use crate::shaders::{Epilogue, HIDDEN, MAX_CONTEXT};

/// Tokens per batched pass.
pub const BATCH: u32 = 16;
/// Input values staged in shared memory per pass over `k` (4 blocks of 32).
const SLICE: u32 = 128;

/// 4-bit matrix times `BATCH` activation rows. One thread owns one output row and keeps one
/// accumulator per token; a 64-thread workgroup stages a `BATCH x 128` slice of the inputs in
/// shared memory, so each weight word is unpacked once and used for every token.
/// Inputs are `x[t * k + i]`, outputs `y[t * rows + row]`.
pub fn gemm(epilogue: Epilogue) -> String {
    let swiglu = epilogue == Epilogue::SwiGlu;
    let acc2_decl = if swiglu { format!("var acc2: array<f32, {BATCH}>;") } else { String::new() };
    let acc2_word = if swiglu {
        "let q2 = w[(row + p.half) * words + blk * 4u + e]; let s2 = scale_of(row + p.half, blk); let z2 = zp_of(row + p.half, blk);
                    let lo2 = (unpack_lo(q2) - vec4<f32>(z2)) * s2; let hi2 = (unpack_hi(q2) - vec4<f32>(z2)) * s2;"
    } else {
        ""
    };
    let acc2_tok = if swiglu { "acc2[t] = acc2[t] + dot(lo2, xs[xi]) + dot(hi2, xs[xi + 1u]);" } else { "" };
    let store = match epilogue {
        Epilogue::Store => "y[t * p.rows + row] = acc[t];",
        Epilogue::Accumulate => "y[t * p.rows + row] = y[t * p.rows + row] + acc[t];",
        Epilogue::SwiGlu => "let g = acc[t]; y[t * p.rows + row] = g / (1.0 + exp(-g)) * acc2[t];",
    };
    let slice4 = SLICE / 4;
    format!(
        r#"
const T: u32 = {BATCH}u;
const S: u32 = {SLICE}u;
struct Params {{ rows: u32, k: u32, half: u32, pad: u32 }};
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> s: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
@group(0) @binding(5) var<storage, read> z: array<u32>;
@group(0) @binding(6) var<storage, read> chunk: array<u32>;
var<workgroup> xs: array<vec4<f32>, {xs_len}>;

fn scale_of(row: u32, blk: u32) -> f32 {{ return unpack2x16float(s[row * (p.k / 64u) + blk / 2u])[blk & 1u]; }}
fn zp_of(row: u32, blk: u32) -> f32 {{ return f32((z[row * (p.k / 256u) + blk / 8u] >> (4u * (blk % 8u))) & 15u); }}
fn unpack_lo(q: u32) -> vec4<f32> {{ return vec4<f32>(f32(q & 15u), f32((q >> 4u) & 15u), f32((q >> 8u) & 15u), f32((q >> 12u) & 15u)); }}
fn unpack_hi(q: u32) -> vec4<f32> {{ return vec4<f32>(f32((q >> 16u) & 15u), f32((q >> 20u) & 15u), f32((q >> 24u) & 15u), f32(q >> 28u)); }}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {{
    let row = (wid.x + wid.y * nw.x) * 64u + li;
    let count = chunk[1];
    let words = p.k / 8u;
    var acc: array<f32, {BATCH}>;
    {acc2_decl}
    for (var k0 = 0u; k0 < p.k; k0 = k0 + S) {{
        for (var i = li; i < T * {slice4}u; i = i + 64u) {{
            let t = i / {slice4}u;
            let c = i % {slice4}u;
            var v = vec4<f32>(0.0);
            if (t < count) {{
                let base = t * p.k + k0 + c * 4u;
                v = vec4<f32>(x[base], x[base + 1u], x[base + 2u], x[base + 3u]);
            }}
            xs[i] = v;
        }}
        workgroupBarrier();
        if (row < p.rows) {{
            for (var b = 0u; b < S / 32u; b = b + 1u) {{
                let blk = k0 / 32u + b;
                let sc = scale_of(row, blk);
                let zp = zp_of(row, blk);
                for (var e = 0u; e < 4u; e = e + 1u) {{
                    let q = w[row * words + blk * 4u + e];
                    let lo = (unpack_lo(q) - vec4<f32>(zp)) * sc;
                    let hi = (unpack_hi(q) - vec4<f32>(zp)) * sc;
                    {acc2_word}
                    for (var t = 0u; t < T; t = t + 1u) {{
                        let xi = t * {slice4}u + b * 8u + e * 2u;
                        acc[t] = acc[t] + dot(lo, xs[xi]) + dot(hi, xs[xi + 1u]);
                        {acc2_tok}
                    }}
                }}
            }}
        }}
        workgroupBarrier();
    }}
    if (row < p.rows) {{
        for (var t = 0u; t < count; t = t + 1u) {{
            {store}
        }}
    }}
}}
"#,
        xs_len = BATCH * SLICE / 4
    )
}

/// Embedding rows for the chunk's token ids.
pub fn embed() -> String {
    let words = HIDDEN / 8;
    format!(
        r#"
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> s: array<u32>;
@group(0) @binding(2) var<storage, read> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> h: array<f32>;
@group(0) @binding(4) var<storage, read> z: array<u32>;
@group(0) @binding(5) var<storage, read> chunk: array<u32>;

@compute @workgroup_size({words})
fn main(@builtin(local_invocation_index) i: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let n = wid.x;
    if (n >= chunk[1]) {{ return; }}
    let row = ids[n];
    let q = w[row * {words}u + i];
    let blk = i / 4u;
    let scale = unpack2x16float(s[row * {sw}u + i / 8u])[blk & 1u];
    let zp = f32((z[row * {zw}u + blk / 8u] >> (4u * (blk % 8u))) & 15u);
    for (var j = 0u; j < 8u; j = j + 1u) {{
        h[n * {HIDDEN}u + i * 8u + j] = (f32((q >> (4u * j)) & 15u) - zp) * scale;
    }}
}}
"#,
        sw = HIDDEN / 64,
        zw = HIDDEN / 256
    )
}

/// RMSNorm of each token row of the chunk.
pub fn rmsnorm() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> chunk: array<u32>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let n = wid.x;
    if (n >= chunk[1]) {{ return; }}
    let base = n * {HIDDEN}u;
    var ss = 0.0;
    for (var i = li; i < {HIDDEN}u; i = i + 256u) {{ let v = x[base + i]; ss = ss + v * v; }}
    red[li] = ss;
    workgroupBarrier();
    for (var off = 128u; off > 0u; off = off >> 1u) {{
        if (li < off) {{ red[li] = red[li] + red[li + off]; }}
        workgroupBarrier();
    }}
    let inv = inverseSqrt(red[0] / f32({HIDDEN}u) + 1e-6);
    for (var i = li; i < {HIDDEN}u; i = i + 256u) {{ y[base + i] = x[base + i] * inv * g[i]; }}
}}
"#
    )
}

/// Gated DeltaNet over the chunk: one workgroup per head walks the tokens in order, keeping each
/// channel's convolution window in registers. Same math as `shaders::gdn`, row stride 8224.
pub fn gdn() -> String {
    r#"
@group(0) @binding(0) var<storage, read> proj: array<f32>;
@group(0) @binding(1) var<storage, read_write> conv_state: array<f32>;
@group(0) @binding(2) var<storage, read> conv_w: array<f32>;
@group(0) @binding(3) var<storage, read_write> state: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
@group(0) @binding(5) var<storage, read> prm: array<f32>;
@group(0) @binding(6) var<storage, read> chunk: array<u32>;
var<workgroup> qs: array<f32, 128>;
var<workgroup> ks: array<f32, 128>;
var<workgroup> red: array<f32, 128>;

fn sum128(t: u32, v: f32) -> f32 {
    red[t] = v;
    workgroupBarrier();
    for (var off = 64u; off > 0u; off = off >> 1u) {
        if (t < off) { red[t] = red[t] + red[t + off]; }
        workgroupBarrier();
    }
    let total = red[0];
    workgroupBarrier();
    return total;
}

fn silu(y: f32) -> f32 { return y / (1.0 + exp(-y)); }

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let h = wid.x;
    let c = h * 128u + t;
    let count = chunk[1];
    var wq = vec4<f32>(conv_w[c * 4u], conv_w[c * 4u + 1u], conv_w[c * 4u + 2u], conv_w[c * 4u + 3u]);
    let ck = 2048u + c;
    let cv = 4096u + c;
    var wk = vec4<f32>(conv_w[ck * 4u], conv_w[ck * 4u + 1u], conv_w[ck * 4u + 2u], conv_w[ck * 4u + 3u]);
    var wv = vec4<f32>(conv_w[cv * 4u], conv_w[cv * 4u + 1u], conv_w[cv * 4u + 2u], conv_w[cv * 4u + 3u]);
    var sq3 = vec3<f32>(conv_state[c * 3u], conv_state[c * 3u + 1u], conv_state[c * 3u + 2u]);
    var sk3 = vec3<f32>(conv_state[ck * 3u], conv_state[ck * 3u + 1u], conv_state[ck * 3u + 2u]);
    var sv3 = vec3<f32>(conv_state[cv * 3u], conv_state[cv * 3u + 1u], conv_state[cv * 3u + 2u]);
    let base = h * 16384u + t;

    for (var n = 0u; n < count; n = n + 1u) {
        let row = n * 8224u;
        let xq = proj[row + c];
        let xk = proj[row + ck];
        let xv = proj[row + cv];
        let q = silu(dot(vec4<f32>(sq3, xq), wq));
        let k = silu(dot(vec4<f32>(sk3, xk), wk));
        let v = silu(dot(vec4<f32>(sv3, xv), wv));
        sq3 = vec3<f32>(sq3.y, sq3.z, xq);
        sk3 = vec3<f32>(sk3.y, sk3.z, xk);
        sv3 = vec3<f32>(sv3.y, sv3.z, xv);

        let nq = inverseSqrt(sum128(t, q * q) + 1e-6);
        let nk = inverseSqrt(sum128(t, k * k) + 1e-6);
        qs[t] = q * nq * 0.08838834764;
        ks[t] = k * nk;
        workgroupBarrier();

        let g = exp(prm[16u + h] * log(1.0 + exp(proj[row + 8192u + h] + prm[h])));
        let beta = 1.0 / (1.0 + exp(-proj[row + 8208u + h]));
        var sk = 0.0;
        var sq = 0.0;
        for (var j = 0u; j < 128u; j = j + 1u) {
            let s = state[base + j * 128u];
            sk = sk + s * ks[j];
            sq = sq + s * qs[j];
        }
        var kq = 0.0;
        for (var j = 0u; j < 128u; j = j + 1u) { kq = kq + ks[j] * qs[j]; }
        let dv = beta * (v - g * sk);
        for (var j = 0u; j < 128u; j = j + 1u) {
            let i = base + j * 128u;
            state[i] = g * state[i] + dv * ks[j];
        }
        let out = g * sq + dv * kq;
        let inv = inverseSqrt(sum128(t, out * out) / 128.0 + 1e-6);
        let zz = proj[row + 6144u + c];
        o[n * 2048u + c] = out * inv * silu(zz) * prm[32u + t];
        workgroupBarrier();
    }
    conv_state[c * 3u] = sq3.x; conv_state[c * 3u + 1u] = sq3.y; conv_state[c * 3u + 2u] = sq3.z;
    conv_state[ck * 3u] = sk3.x; conv_state[ck * 3u + 1u] = sk3.y; conv_state[ck * 3u + 2u] = sk3.z;
    conv_state[cv * 3u] = sv3.x; conv_state[cv * 3u + 1u] = sv3.y; conv_state[cv * 3u + 2u] = sv3.z;
}
"#
    .to_string()
}

const NORM_ROPE: &str = r#"
fn sum256(t: u32, v: f32) -> f32 {
    red[t] = v;
    workgroupBarrier();
    for (var off = 128u; off > 0u; off = off >> 1u) {
        if (t < off) { red[t] = red[t] + red[t + off]; }
        workgroupBarrier();
    }
    let total = red[0];
    workgroupBarrier();
    return total;
}

/// RMSNorm with weight `wgt`, then rotate-half RoPE on the first 64 dims; result in `vsh`.
fn norm_rope(t: u32, x: f32, wgt: f32, pos: u32) {
    vsh[t] = x * inverseSqrt(sum256(t, x * x) / 256.0 + 1e-6) * wgt;
    workgroupBarrier();
    var r = vsh[t];
    if (t < 64u) {
        let i = t % 32u;
        let angle = f32(pos) * pow(10000000.0, -f32(2u * i) / 64.0);
        let c = cos(angle);
        let s = sin(angle);
        if (t < 32u) { r = vsh[t] * c - vsh[t + 32u] * s; } else { r = vsh[t] * c + vsh[t - 32u] * s; }
    }
    workgroupBarrier();
    vsh[t] = r;
    workgroupBarrier();
}
"#;

/// Normalizes and rotates the chunk's keys and writes keys and values to the cache, one
/// workgroup per (token, KV head). Row stride of `proj` is 5120.
pub fn kv_prep() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> proj: array<f32>;
@group(0) @binding(1) var<storage, read_write> kc: array<f32>;
@group(0) @binding(2) var<storage, read_write> vc: array<f32>;
@group(0) @binding(3) var<storage, read> chunk: array<u32>;
@group(0) @binding(4) var<storage, read> prm: array<f32>;
var<workgroup> vsh: array<f32, 256>;
var<workgroup> red: array<f32, 256>;
{NORM_ROPE}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let n = wid.x / 2u;
    let kvh = wid.x % 2u;
    if (n >= chunk[1]) {{ return; }}
    let pos = min(chunk[0] + n, {MAX_CONTEXT}u - 1u);
    let row = n * 5120u;
    norm_rope(t, proj[row + 4096u + kvh * 256u + t], prm[256u + t], pos);
    kc[(kvh * {MAX_CONTEXT}u + pos) * 256u + t] = vsh[t];
    vc[(kvh * {MAX_CONTEXT}u + pos) * 256u + t] = proj[row + 4608u + kvh * 256u + t];
}}
"#
    )
}

/// Causal attention for each (token, query head) of the chunk over the cache, which already
/// holds the chunk's own keys and values (`kv_prep` runs first).
pub fn attention() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> proj: array<f32>;
@group(0) @binding(1) var<storage, read> kc: array<f32>;
@group(0) @binding(2) var<storage, read> vc: array<f32>;
@group(0) @binding(3) var<storage, read> chunk: array<u32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
@group(0) @binding(5) var<storage, read> prm: array<f32>;
var<workgroup> vsh: array<f32, 256>;
var<workgroup> red: array<f32, 256>;
var<workgroup> sc: array<f32, {MAX_CONTEXT}>;
{NORM_ROPE}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let n = wid.x / 8u;
    let h = wid.x % 8u;
    if (n >= chunk[1]) {{ return; }}
    let kvh = h / 4u;
    let pos = min(chunk[0] + n, {MAX_CONTEXT}u - 1u);
    let len = pos + 1u;
    let row = n * 5120u;
    norm_rope(t, proj[row + h * 256u + t], prm[t], pos);

    var local_max = -3.0e38;
    for (var l = t; l < len; l = l + 256u) {{
        var d = 0.0;
        for (var j = 0u; j < 256u; j = j + 1u) {{ d = d + vsh[j] * kc[(kvh * {MAX_CONTEXT}u + l) * 256u + j]; }}
        sc[l] = d * 0.0625;
        local_max = max(local_max, sc[l]);
    }}
    red[t] = local_max;
    workgroupBarrier();
    for (var off = 128u; off > 0u; off = off >> 1u) {{
        if (t < off) {{ red[t] = max(red[t], red[t + off]); }}
        workgroupBarrier();
    }}
    let m = red[0];
    workgroupBarrier();
    var local_sum = 0.0;
    for (var l = t; l < len; l = l + 256u) {{
        let e = exp(sc[l] - m);
        sc[l] = e;
        local_sum = local_sum + e;
    }}
    let total = sum256(t, local_sum);
    var acc = 0.0;
    for (var l = 0u; l < len; l = l + 1u) {{ acc = acc + sc[l] * vc[(kvh * {MAX_CONTEXT}u + l) * 256u + t]; }}
    let gate = proj[row + 2048u + h * 256u + t];
    o[n * 2048u + h * 256u + t] = acc / total / (1.0 + exp(-gate));
}}
"#
    )
}

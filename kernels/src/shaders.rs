//! WGSL sources. Constants are substituted as text, so the GEMV tile shape can be tuned per
//! device without relying on pipeline-overridable workgroup sizes.

pub const HIDDEN: u32 = 1024;

/// Reads a `u32` as two f16 values. Browsers (Tint) accept the direct bitcast; wgpu's native
/// WGSL front end (naga) rejects it, so native builds go through `unpack2x16float`.
#[cfg(target_arch = "wasm32")]
const PAIR_FN: &str = "fn pair(w: u32) -> vec2<f16> { return bitcast<vec2<f16>>(w); }";
#[cfg(not(target_arch = "wasm32"))]
const PAIR_FN: &str = "fn pair(w: u32) -> vec2<f16> { return vec2<f16>(unpack2x16float(w)); }";
/// Upper bound of the attention context held in workgroup memory (16 kB of scores).
pub const MAX_CONTEXT: u32 = 4096;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Epilogue {
    Store,
    /// Adds into the output, used for the residual stream.
    Accumulate,
    /// Rows `r` and `r + half` are gate and up; the output is `silu(gate) * up`.
    SwiGlu,
}

/// 4-bit weights, eight per `u32`, one f16 scale per 32 inputs (two scales per `u32`) and one
/// 4-bit zero point per 32 inputs (eight per `u32`): the layout of ONNX `MatMulNBits` with
/// block size 32. `tpr` threads share a row and `rpw` rows share a workgroup.
pub fn gemv(tpr: u32, rpw: u32, epilogue: Epilogue) -> String {
    let wg = tpr * rpw;
    let swiglu = epilogue == Epilogue::SwiGlu;
    let second_acc = if swiglu { "b = row_dot(row + p.half, lane);" } else { "" };
    let second_store = if swiglu { "red2[li] = b;" } else { "" };
    let second_reduce = if swiglu { "red2[li] = red2[li] + red2[li + off];" } else { "" };
    let second_decl = if swiglu { format!("var<workgroup> red2: array<f32, {wg}>;") } else { String::new() };
    let store = match epilogue {
        Epilogue::Store => "y[row] = red[li];",
        Epilogue::Accumulate => "y[row] = y[row] + red[li];",
        Epilogue::SwiGlu => "let g = red[li]; y[row] = g / (1.0 + exp(-g)) * red2[li];",
    };
    let half_tpr = tpr / 2;
    format!(
        r#"
struct Params {{ rows: u32, k: u32, half: u32, pad: u32 }};
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> s: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
@group(0) @binding(5) var<storage, read> z: array<u32>;
var<workgroup> red: array<f32, {wg}>;
{second_decl}

fn dot_word(q: u32, i: u32, zp: f32) -> f32 {{
    let x0 = x[i * 2u];
    let x1 = x[i * 2u + 1u];
    let lo = vec4<f32>(f32(q & 15u), f32((q >> 4u) & 15u), f32((q >> 8u) & 15u), f32((q >> 12u) & 15u)) - vec4<f32>(zp);
    let hi = vec4<f32>(f32((q >> 16u) & 15u), f32((q >> 20u) & 15u), f32((q >> 24u) & 15u), f32(q >> 28u)) - vec4<f32>(zp);
    return dot(lo, x0) + dot(hi, x1);
}}

fn row_dot(row: u32, lane: u32) -> f32 {{
    let words = p.k / 8u;
    let base = row * words;
    let sbase = row * (p.k / 64u);
    let zbase = row * (p.k / 256u);
    var acc = 0.0;
    for (var i = lane; i < words; i = i + {tpr}u) {{
        let blk = i / 4u;
        let scale = unpack2x16float(s[sbase + i / 8u])[blk & 1u];
        let zp = f32((z[zbase + blk / 8u] >> (4u * (blk % 8u))) & 15u);
        acc = acc + dot_word(w[base + i], i, zp) * scale;
    }}
    return acc;
}}

@compute @workgroup_size({wg})
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {{
    let lane = li % {tpr}u;
    let row = (wid.x + wid.y * nw.x) * {rpw}u + li / {tpr}u;
    var a = 0.0;
    var b = 0.0;
    if (row < p.rows) {{
        a = row_dot(row, lane);
        {second_acc}
    }}
    red[li] = a;
    {second_store}
    workgroupBarrier();
    for (var off = {half_tpr}u; off > 0u; off = off >> 1u) {{
        if (lane < off) {{
            red[li] = red[li] + red[li + off];
            {second_reduce}
        }}
        workgroupBarrier();
    }}
    if (lane == 0u && row < p.rows) {{
        {store}
    }}
}}
"#
    )
}

/// Same weight layout as `gemv`, built for GPUs whose ALUs, not memory, limit the plain version:
/// - 16-byte loads, one 32-value block (and so one scale) per load;
/// - nibbles become f16 two at a time: `(word & 0x000F000F) | 0x64006400` reads as the pair
///   (1024 + n0, 1024 + n4), so subtracting 1032 gives the signed values without conversions;
/// - the input vector is converted to f16 once per workgroup, into shared memory.
///
/// Partial sums within a 32-value block are f16; they are scaled and accumulated in f32. The
/// shader is specialized for `k`, which sizes the shared copy of the input.
pub fn gemv_f16(tpr: u32, rpw: u32, rpt: u32, k: u32, epilogue: Epilogue) -> String {
    // `rpt` rows per thread group: each 32-value block of the input is loaded into registers once
    // and used for `rpt` rows, which gives the GPU independent work and halves input reads.
    let wg = tpr * rpw;
    let blocks = k / 32;
    let k4 = k / 4;
    let swiglu = epilogue == Epilogue::SwiGlu;
    let lanes = wg * rpt;
    let second_decl = if swiglu {
        format!("var<workgroup> red2: array<f32, {lanes}>;")
    } else {
        String::new()
    };
    let second_acc_decl = if swiglu { format!("var acc2: array<f32, {rpt}>;") } else { String::new() };
    let second_acc = if swiglu {
        "acc2[r] = acc2[r] + f32(block_dot(w[(row + p.half) * BLOCKS + b], xv, zoff(row + p.half, b))) * scale_of(row + p.half, b);"
    } else {
        ""
    };
    let second_store = if swiglu { "red2[slot] = acc2[r];" } else { "" };
    let second_reduce = if swiglu { "red2[i] = red2[i] + red2[i + off];" } else { "" };
    let store = match epilogue {
        Epilogue::Store => "y[row] = red[slot];",
        Epilogue::Accumulate => "y[row] = y[row] + red[slot];",
        Epilogue::SwiGlu => "let g = red[slot]; y[row] = g / (1.0 + exp(-g)) * red2[slot];",
    };
    let half_tpr = tpr / 2;
    let half_blocks = blocks / 2;
    let zwords = blocks / 8;
    let pair_fn = PAIR_FN;
    format!(
        r#"
enable f16;
{pair_fn}
const BLOCKS: u32 = {blocks}u;
struct Params {{ rows: u32, k: u32, half: u32, pad: u32 }};
@group(0) @binding(0) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> s: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
@group(0) @binding(5) var<storage, read> z: array<u32>;
var<workgroup> xs: array<vec4<f16>, {k4}>;
var<workgroup> red: array<f32, {lanes}>;
{second_decl}

fn block_dot(q: vec4<u32>, xv: array<vec4<f16>, 8>, off: vec4<f16>) -> f16 {{
    var acc = vec4<f16>(0.0);
    for (var e = 0u; e < 4u; e = e + 1u) {{
        let word = q[e];
        let p0 = pair((word & 0x000F000Fu) | 0x64006400u);
        let p1 = pair(((word >> 4u) & 0x000F000Fu) | 0x64006400u);
        let p2 = pair(((word >> 8u) & 0x000F000Fu) | 0x64006400u);
        let p3 = pair(((word >> 12u) & 0x000F000Fu) | 0x64006400u);
        let lo = vec4<f16>(p0.x, p1.x, p2.x, p3.x) - off;
        let hi = vec4<f16>(p0.y, p1.y, p2.y, p3.y) - off;
        acc = acc + lo * xv[e * 2u] + hi * xv[e * 2u + 1u];
    }}
    return acc.x + acc.y + acc.z + acc.w;
}}

fn scale_of(row: u32, b: u32) -> f32 {{
    return unpack2x16float(s[row * {half_blocks}u + b / 2u])[b & 1u];
}}

fn zoff(row: u32, b: u32) -> vec4<f16> {{
    let zp = (z[row * {zwords}u + b / 8u] >> (4u * (b % 8u))) & 15u;
    return vec4<f16>(f16(1024u + zp));
}}

@compute @workgroup_size({wg})
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {{
    for (var i = li; i < {k4}u; i = i + {wg}u) {{ xs[i] = vec4<f16>(x[i]); }}
    workgroupBarrier();
    let lane = li % {tpr}u;
    let group = li / {tpr}u;
    let row0 = ((wid.x + wid.y * nw.x) * {rpw}u + group) * {rpt}u;
    var acc: array<f32, {rpt}>;
    {second_acc_decl}
    for (var b = lane; b < BLOCKS; b = b + {tpr}u) {{
        var xv: array<vec4<f16>, 8>;
        for (var e = 0u; e < 8u; e = e + 1u) {{ xv[e] = xs[b * 8u + e]; }}
        for (var r = 0u; r < {rpt}u; r = r + 1u) {{
            let row = row0 + r;
            if (row < p.rows) {{
                acc[r] = acc[r] + f32(block_dot(w[row * BLOCKS + b], xv, zoff(row, b))) * scale_of(row, b);
                {second_acc}
            }}
        }}
    }}
    for (var r = 0u; r < {rpt}u; r = r + 1u) {{
        let slot = (group * {rpt}u + r) * {tpr}u + lane;
        red[slot] = acc[r];
        {second_store}
    }}
    workgroupBarrier();
    for (var off = {half_tpr}u; off > 0u; off = off >> 1u) {{
        if (lane < off) {{
            for (var r = 0u; r < {rpt}u; r = r + 1u) {{
                let i = (group * {rpt}u + r) * {tpr}u + lane;
                red[i] = red[i] + red[i + off];
                {second_reduce}
            }}
        }}
        workgroupBarrier();
    }}
    if (lane == 0u) {{
        for (var r = 0u; r < {rpt}u; r = r + 1u) {{
            let row = row0 + r;
            let slot = (group * {rpt}u + r) * {tpr}u;
            if (row < p.rows) {{
                {store}
            }}
        }}
    }}
}}
"#
    )
}

/// Split-K partial pass: matrices with few rows (1024 for the down and out projections) leave most
/// of the GPU idle when a row is one workgroup's job, and every workgroup would copy the whole
/// input. Here `splits` workgroups share a row group, each over its own slice of `k`, and write
/// partial sums to `partial[split * rows + row]`; `split_reduce` adds them up.
pub fn gemv_split_f16(tpr: u32, rpw: u32, rpt: u32, k: u32, splits: u32) -> String {
    let wg = tpr * rpw;
    let blocks = k / 32;
    let slice_blocks = blocks / splits;
    let slice4 = k / splits / 4;
    let lanes = wg * rpt;
    let half_tpr = tpr / 2;
    let half_blocks = blocks / 2;
    let zwords = blocks / 8;
    let pair_fn = PAIR_FN;
    format!(
        r#"
enable f16;
{pair_fn}
const BLOCKS: u32 = {blocks}u;
struct Params {{ rows: u32, k: u32, half: u32, pad: u32 }};
@group(0) @binding(0) var<storage, read> w: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> s: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> partial: array<f32>;
@group(0) @binding(4) var<uniform> p: Params;
@group(0) @binding(5) var<storage, read> z: array<u32>;
var<workgroup> xs: array<vec4<f16>, {slice4}>;
var<workgroup> red: array<f32, {lanes}>;

fn block_dot(q: vec4<u32>, xv: array<vec4<f16>, 8>, off: vec4<f16>) -> f16 {{
    var acc = vec4<f16>(0.0);
    for (var e = 0u; e < 4u; e = e + 1u) {{
        let word = q[e];
        let p0 = pair((word & 0x000F000Fu) | 0x64006400u);
        let p1 = pair(((word >> 4u) & 0x000F000Fu) | 0x64006400u);
        let p2 = pair(((word >> 8u) & 0x000F000Fu) | 0x64006400u);
        let p3 = pair(((word >> 12u) & 0x000F000Fu) | 0x64006400u);
        let lo = vec4<f16>(p0.x, p1.x, p2.x, p3.x) - off;
        let hi = vec4<f16>(p0.y, p1.y, p2.y, p3.y) - off;
        acc = acc + lo * xv[e * 2u] + hi * xv[e * 2u + 1u];
    }}
    return acc.x + acc.y + acc.z + acc.w;
}}

@compute @workgroup_size({wg})
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {{
    let index = wid.x + wid.y * nw.x;
    let split = index % {splits}u;
    let row_group = index / {splits}u;
    for (var i = li; i < {slice4}u; i = i + {wg}u) {{ xs[i] = vec4<f16>(x[split * {slice4}u + i]); }}
    workgroupBarrier();
    let lane = li % {tpr}u;
    let group = li / {tpr}u;
    let row0 = (row_group * {rpw}u + group) * {rpt}u;
    let first_block = split * {slice_blocks}u;
    var acc: array<f32, {rpt}>;
    for (var b = lane; b < {slice_blocks}u; b = b + {tpr}u) {{
        var xv: array<vec4<f16>, 8>;
        for (var e = 0u; e < 8u; e = e + 1u) {{ xv[e] = xs[b * 8u + e]; }}
        let gb = first_block + b;
        for (var r = 0u; r < {rpt}u; r = r + 1u) {{
            let row = row0 + r;
            if (row < p.rows) {{
                let scale = unpack2x16float(s[row * {half_blocks}u + gb / 2u])[gb & 1u];
                let zp = (z[row * {zwords}u + gb / 8u] >> (4u * (gb % 8u))) & 15u;
                acc[r] = acc[r] + f32(block_dot(w[row * BLOCKS + gb], xv, vec4<f16>(f16(1024u + zp)))) * scale;
            }}
        }}
    }}
    for (var r = 0u; r < {rpt}u; r = r + 1u) {{
        red[(group * {rpt}u + r) * {tpr}u + lane] = acc[r];
    }}
    workgroupBarrier();
    for (var off = {half_tpr}u; off > 0u; off = off >> 1u) {{
        if (lane < off) {{
            for (var r = 0u; r < {rpt}u; r = r + 1u) {{
                let i = (group * {rpt}u + r) * {tpr}u + lane;
                red[i] = red[i] + red[i + off];
            }}
        }}
        workgroupBarrier();
    }}
    if (lane == 0u) {{
        for (var r = 0u; r < {rpt}u; r = r + 1u) {{
            let row = row0 + r;
            if (row < p.rows) {{
                partial[split * p.rows + row] = red[(group * {rpt}u + r) * {tpr}u];
            }}
        }}
    }}
}}
"#
    )
}

/// Split-K second pass: sums the partials of each row and applies the epilogue.
pub fn split_reduce(splits: u32, epilogue: Epilogue) -> String {
    let store = match epilogue {
        Epilogue::Store => "y[row] = total(row);",
        Epilogue::Accumulate => "y[row] = y[row] + total(row);",
        Epilogue::SwiGlu => "let g = total(row); y[row] = g / (1.0 + exp(-g)) * total(row + p.half);",
    };
    format!(
        r#"
struct Params {{ rows: u32, total_rows: u32, half: u32, pad: u32 }};
@group(0) @binding(0) var<storage, read> partial: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: Params;

fn total(row: u32) -> f32 {{
    var acc = 0.0;
    for (var s = 0u; s < {splits}u; s = s + 1u) {{ acc = acc + partial[s * p.total_rows + row]; }}
    return acc;
}}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let row = gid.x;
    if (row < p.rows) {{
        {store}
    }}
}}
"#
    )
}

pub fn rmsnorm() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read> g: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {{
    var ss = 0.0;
    for (var i = li; i < {HIDDEN}u; i = i + 256u) {{ let v = x[i]; ss = ss + v * v; }}
    red[li] = ss;
    workgroupBarrier();
    for (var off = 128u; off > 0u; off = off >> 1u) {{
        if (li < off) {{ red[li] = red[li] + red[li + off]; }}
        workgroupBarrier();
    }}
    let inv = inverseSqrt(red[0] / f32({HIDDEN}u) + 1e-6);
    for (var i = li; i < {HIDDEN}u; i = i + 256u) {{ y[i] = x[i] * inv * g[i]; }}
}}
"#
    )
}

/// Dequantizes the embedding row of the token that the previous step wrote to `tok[0]`, so the
/// decode loop never needs the CPU.
pub fn embed() -> String {
    let words = HIDDEN / 8;
    format!(
        r#"
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> s: array<u32>;
@group(0) @binding(2) var<storage, read> tok: array<u32>;
@group(0) @binding(3) var<storage, read_write> h: array<f32>;
@group(0) @binding(4) var<storage, read> z: array<u32>;

@compute @workgroup_size({words})
fn main(@builtin(local_invocation_index) i: u32) {{
    let row = tok[0];
    let q = w[row * {words}u + i];
    let blk = i / 4u;
    let scale = unpack2x16float(s[row * {sw}u + i / 8u])[blk & 1u];
    let zp = f32((z[row * {zw}u + blk / 8u] >> (4u * (blk % 8u))) & 15u);
    for (var j = 0u; j < 8u; j = j + 1u) {{
        h[i * 8u + j] = (f32((q >> (4u * j)) & 15u) - zp) * scale;
    }}
}}
"#,
        sw = HIDDEN / 64,
        zw = HIDDEN / 256
    )
}

/// One decode step of a Gated DeltaNet layer, following the ONNX export's decode branch: causal
/// conv (kernel 4) and SiLU on q, k, v; L2-normalized q (scaled by 1/sqrt(128)) and k; decay
/// `exp(A_neg_exp * softplus(a + dt_bias))`, write strength `sigmoid(b)`; the gated delta rule on
/// a 128x128 state per head; then RMSNorm of the output, times `silu(z)`, times the norm weight.
/// The state is stored as `S[h][key][value]` and each thread owns one value column, so the
/// threads of a workgroup read consecutive addresses.
///
/// Input layout of `proj`: q[2048] k[2048] v[2048] z[2048] a[16] b[16].
/// `prm`: dt_bias[16] A_neg_exp[16] norm_weight[128].
pub fn gdn() -> String {
    r#"
@group(0) @binding(0) var<storage, read> proj: array<f32>;
@group(0) @binding(1) var<storage, read_write> conv_state: array<f32>;
@group(0) @binding(2) var<storage, read> conv_w: array<f32>;
@group(0) @binding(3) var<storage, read_write> state: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
@group(0) @binding(5) var<storage, read> prm: array<f32>;
var<workgroup> qs: array<f32, 128>;
var<workgroup> ks: array<f32, 128>;
var<workgroup> red: array<f32, 128>;

fn conv(c: u32) -> f32 {
    let x = proj[c];
    let b = c * 3u;
    let s0 = conv_state[b];
    let s1 = conv_state[b + 1u];
    let s2 = conv_state[b + 2u];
    let y = s0 * conv_w[c * 4u] + s1 * conv_w[c * 4u + 1u] + s2 * conv_w[c * 4u + 2u] + x * conv_w[c * 4u + 3u];
    conv_state[b] = s1;
    conv_state[b + 1u] = s2;
    conv_state[b + 2u] = x;
    return y / (1.0 + exp(-y));
}

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

@compute @workgroup_size(128)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let h = wid.x;
    let c = h * 128u + t;
    let q = conv(c);
    let k = conv(2048u + c);
    let v = conv(4096u + c);
    let nq = inverseSqrt(sum128(t, q * q) + 1e-6);
    let nk = inverseSqrt(sum128(t, k * k) + 1e-6);
    qs[t] = q * nq * 0.08838834764;
    ks[t] = k * nk;
    workgroupBarrier();

    let g = exp(prm[16u + h] * log(1.0 + exp(proj[8192u + h] + prm[h])));
    let beta = 1.0 / (1.0 + exp(-proj[8208u + h]));

    let base = h * 16384u + t;
    var sk = 0.0;
    var sq = 0.0;
    for (var j = 0u; j < 128u; j = j + 1u) {
        let sv = state[base + j * 128u];
        sk = sk + sv * ks[j];
        sq = sq + sv * qs[j];
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
    let z = proj[6144u + c];
    o[c] = out * inv * (z / (1.0 + exp(-z))) * prm[32u + t];
}
"#
    .to_string()
}

/// One decode step of a gated full-attention layer (8 query heads of 256, 2 KV heads, so four
/// query heads share a KV head): per-head RMSNorm of q and k, rotary embedding on the first 64
/// dimensions (rotate-half pairing, theta 1e7), attention over the cache, then the output gate.
/// The current position comes from `tok[1]`. The new key (normed and rotated) and value are
/// written to the cache by the first query head of each group; every head computes the current
/// key itself, so none reads that slot in the same dispatch.
///
/// Input layout of `proj`: q[2048] gate[2048] k[512] v[512]. `prm`: q_norm[256] k_norm[256].
pub fn attention() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> proj: array<f32>;
@group(0) @binding(1) var<storage, read_write> kc: array<f32>;
@group(0) @binding(2) var<storage, read_write> vc: array<f32>;
@group(0) @binding(3) var<storage, read> tok: array<u32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
@group(0) @binding(5) var<storage, read> prm: array<f32>;
var<workgroup> qsh: array<f32, 256>;
var<workgroup> ksh: array<f32, 256>;
var<workgroup> sc: array<f32, {MAX_CONTEXT}>;
var<workgroup> red: array<f32, 256>;

fn sum256(t: u32, v: f32) -> f32 {{
    red[t] = v;
    workgroupBarrier();
    for (var off = 128u; off > 0u; off = off >> 1u) {{
        if (t < off) {{ red[t] = red[t] + red[t + off]; }}
        workgroupBarrier();
    }}
    let total = red[0];
    workgroupBarrier();
    return total;
}}

fn key(kvh: u32, l: u32, pos: u32, j: u32) -> f32 {{
    if (l == pos) {{ return ksh[j]; }}
    return kc[(kvh * {MAX_CONTEXT}u + l) * 256u + j];
}}

fn value(kvh: u32, l: u32, pos: u32, j: u32) -> f32 {{
    if (l == pos) {{ return proj[4608u + kvh * 256u + j]; }}
    return vc[(kvh * {MAX_CONTEXT}u + l) * 256u + j];
}}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let h = wid.x;
    let kvh = h / 4u;
    let pos = min(tok[1], {MAX_CONTEXT}u - 1u);
    let len = pos + 1u;

    let q = proj[h * 256u + t];
    let k = proj[4096u + kvh * 256u + t];
    qsh[t] = q * inverseSqrt(sum256(t, q * q) / 256.0 + 1e-6) * prm[t];
    ksh[t] = k * inverseSqrt(sum256(t, k * k) / 256.0 + 1e-6) * prm[256u + t];
    workgroupBarrier();
    var qr = qsh[t];
    var kr = ksh[t];
    if (t < 64u) {{
        let i = t % 32u;
        let angle = f32(pos) * pow(10000000.0, -f32(2u * i) / 64.0);
        let c = cos(angle);
        let s = sin(angle);
        if (t < 32u) {{
            qr = qsh[t] * c - qsh[t + 32u] * s;
            kr = ksh[t] * c - ksh[t + 32u] * s;
        }} else {{
            qr = qsh[t] * c + qsh[t - 32u] * s;
            kr = ksh[t] * c + ksh[t - 32u] * s;
        }}
    }}
    workgroupBarrier();
    qsh[t] = qr;
    ksh[t] = kr;
    workgroupBarrier();
    if (h % 4u == 0u) {{
        kc[(kvh * {MAX_CONTEXT}u + pos) * 256u + t] = kr;
        vc[(kvh * {MAX_CONTEXT}u + pos) * 256u + t] = proj[4608u + kvh * 256u + t];
    }}

    var local_max = -3.0e38;
    for (var l = t; l < len; l = l + 256u) {{
        var d = 0.0;
        for (var j = 0u; j < 256u; j = j + 1u) {{ d = d + qsh[j] * key(kvh, l, pos, j); }}
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
    for (var l = 0u; l < len; l = l + 1u) {{ acc = acc + sc[l] * value(kvh, l, pos, t); }}
    let gate = proj[2048u + h * 256u + t];
    o[h * 256u + t] = acc / total / (1.0 + exp(-gate));
}}
"#
    )
}

/// First pass of the greedy pick: each workgroup reduces 1024 logits.
pub fn argmax_partial() -> String {
    r#"
struct Params { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> best_val: array<f32>;
@group(0) @binding(2) var<storage, read_write> best_idx: array<u32>;
@group(0) @binding(3) var<uniform> p: Params;
var<workgroup> rv: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    var bv = -3.0e38;
    var bi = 0u;
    for (var k = 0u; k < 4u; k = k + 1u) {
        let i = wid.x * 1024u + k * 256u + t;
        if (i < p.n && logits[i] > bv) { bv = logits[i]; bi = i; }
    }
    rv[t] = bv;
    ri[t] = bi;
    workgroupBarrier();
    for (var off = 128u; off > 0u; off = off >> 1u) {
        if (t < off && rv[t + off] > rv[t]) { rv[t] = rv[t + off]; ri[t] = ri[t + off]; }
        workgroupBarrier();
    }
    if (t == 0u) { best_val[wid.x] = rv[0]; best_idx[wid.x] = ri[0]; }
}
"#
    .to_string()
}

/// Second pass: picks the token, stores it for the next step's embedding and advances the position.
pub fn argmax_final() -> String {
    r#"
struct Params { n: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> best_val: array<f32>;
@group(0) @binding(1) var<storage, read> best_idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> tok: array<u32>;
@group(0) @binding(3) var<uniform> p: Params;
var<workgroup> rv: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) t: u32) {
    var bv = -3.0e38;
    var bi = 0u;
    for (var i = t; i < p.n; i = i + 256u) {
        if (best_val[i] > bv) { bv = best_val[i]; bi = best_idx[i]; }
    }
    rv[t] = bv;
    ri[t] = bi;
    workgroupBarrier();
    for (var off = 128u; off > 0u; off = off >> 1u) {
        if (t < off && rv[t + off] > rv[t]) { rv[t] = rv[t + off]; ri[t] = ri[t + off]; }
        workgroupBarrier();
    }
    if (t == 0u) {
        tok[0] = ri[0];
        tok[1] = tok[1] + 1u;
    }
}
"#
    .to_string()
}

/// Streams a large buffer once with 16-byte loads and keeps one sum per thread: the read
/// bandwidth a perfect GEMV could reach on this GPU, with no unpacking work at all.
pub fn bandwidth() -> String {
    r#"
@group(0) @binding(0) var<storage, read> data: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> sink: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>) {
    let threads = nw.x * 256u;
    let n = arrayLength(&data);
    var acc = vec4<u32>(0u);
    for (var i = gid.x; i < n; i = i + threads) { acc = acc ^ data[i]; }
    sink[gid.x] = acc.x ^ acc.y ^ acc.z ^ acc.w;
}
"#
    .to_string()
}

/// A dispatch that does nearly nothing, to measure the fixed cost of one dispatch.
pub fn noop() -> String {
    r#"
@group(0) @binding(0) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) t: u32) {
    if (t == 0u) { y[0] = y[0] + 1.0; }
}
"#
    .to_string()
}

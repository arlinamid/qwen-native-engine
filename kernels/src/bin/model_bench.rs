//! Runs the real Qwen3.5-0.8B weights on the native engine and compares the result with an
//! ONNX Runtime reference written by `qwen_bench --dump`.
//!
//! cargo run --release -p qwen-kernels --bin model_bench -- --reference results/reference_en.json
//!     [--dxc .toolchain/ort-dll/webgpu/dxcompiler.dll] [--f32] [--tokens 32] [--out file.json]

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    native::main()
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::{path::PathBuf, time::Instant};

    use anyhow::{Context, Result, bail};
    use qwen_kernels::model::{Model, ModelTiles};
    use serde::Deserialize;
    use serde_json::json;
    use tokenizers::Tokenizer;

    #[derive(Deserialize)]
    struct Reference {
        prompt: String,
        prompt_ids: Vec<u32>,
        logits: Vec<f32>,
        ids: Vec<u32>,
        text: String,
    }

    fn top(v: &[f32], k: usize) -> Vec<usize> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.select_nth_unstable_by(k, |&a, &b| v[b].total_cmp(&v[a]));
        idx.truncate(k);
        idx.sort_by(|&a, &b| v[b].total_cmp(&v[a]));
        idx
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| f64::from(*x) * f64::from(*y)).sum();
        let na: f64 = a.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        dot / (na * nb)
    }

    pub fn main() -> Result<()> {
        let mut model_dir = PathBuf::from("models/qwen3.5-0.8b");
        let mut references = Vec::new();
        let mut dxc = None;
        let mut tiles = ModelTiles::fast();
        let mut tokens = 32usize;
        let mut out = None;
        let mut token_prefill = false;
        let mut args = std::env::args().skip(1);
        while let Some(flag) = args.next() {
            match flag.as_str() {
                "--model-dir" => model_dir = args.next().context("value")?.into(),
                "--reference" => references.push(PathBuf::from(args.next().context("value")?)),
                "--dxc" => {
                    let path = std::fs::canonicalize(args.next().context("value")?)?;
                    if let Some(dir) = path.parent() {
                        let old = std::env::var_os("PATH").unwrap_or_default();
                        let joined = std::env::join_paths(std::iter::once(dir.to_path_buf()).chain(std::env::split_paths(&old)))?;
                        unsafe { std::env::set_var("PATH", joined) };
                    }
                    dxc = Some(path.to_string_lossy().trim_start_matches(r"\\?\").to_string());
                }
                "--f32" => tiles = ModelTiles::f32(),
                "--token-prefill" => token_prefill = true,
                "--tokens" => tokens = args.next().context("value")?.parse()?,
                "--out" => out = Some(PathBuf::from(args.next().context("value")?)),
                other => bail!("unknown flag {other}"),
            }
        }
        if references.is_empty() {
            bail!("pass at least one --reference written by qwen_bench --dump");
        }

        let t0 = Instant::now();
        let index = std::fs::read_to_string(model_dir.join("native/index.json"))?;
        let weights = std::fs::read(model_dir.join("native/weights.bin"))?;
        let tok = Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(anyhow::Error::msg)?;
        let model = pollster::block_on(Model::load(&index, &weights, tiles, true, dxc.as_deref()))?;
        drop(weights);
        eprintln!("loaded on {} in {:.1} s, {} dispatches per token", model.adapter, t0.elapsed().as_secs_f64(), model.dispatches_per_token());
        let stop = [tok.token_to_id("<|im_end|>").unwrap(), tok.token_to_id("<|endoftext|>").unwrap()];

        let mut reports = Vec::new();
        for path in &references {
            let r: Reference = serde_json::from_str(&std::fs::read_to_string(path)?)?;
            model.reset();
            let t = Instant::now();
            if token_prefill {
                pollster::block_on(model.prefill(&r.prompt_ids))?;
            } else {
                pollster::block_on(model.prefill_batched(&r.prompt_ids))?;
            }
            let prefill_ms = t.elapsed().as_secs_f64() * 1000.0;
            let logits = pollster::block_on(model.logits())?;

            let (ours5, ref5) = (top(&logits, 5), top(&r.logits, 5));
            let overlap = ours5.iter().filter(|i| ref5.contains(i)).count();
            let max_diff = logits.iter().zip(&r.logits).fold(0f32, |m, (a, b)| m.max((a - b).abs()));
            let cos = cosine(&logits, &r.logits);

            let t = Instant::now();
            let ids = pollster::block_on(model.decode(tokens.min(r.ids.len().max(1)), 1, &stop))?;
            let decode_ms = t.elapsed().as_secs_f64() * 1000.0;
            let matching = ids.iter().zip(&r.ids).take_while(|(a, b)| a == b).count();
            let text = tok.decode(&ids, false).map_err(anyhow::Error::msg)?;
            let per_token = decode_ms / (ids.len().saturating_sub(1).max(1)) as f64;

            eprintln!("\n== {:?} ({} prompt tokens)", r.prompt, r.prompt_ids.len());
            eprintln!("logits: cosine {cos:.5}, max |diff| {max_diff:.3}, top-1 {} vs {} ({}), top-5 overlap {overlap}/5", ours5[0], ref5[0], if ours5[0] == ref5[0] { "same" } else { "DIFFERENT" });
            eprintln!("greedy: {matching}/{} tokens identical to the reference", ids.len().min(r.ids.len()));
            eprintln!("  ours: {text:?}");
            eprintln!("  ref:  {:?}", r.text);
            let mode = if token_prefill { "one token per step" } else { "batched" };
            eprintln!("speed: prompt {prefill_ms:.0} ms ({:.1} tok/s, {mode}), decode {per_token:.1} ms/token = {:.1} tok/s", r.prompt_ids.len() as f64 / (prefill_ms / 1000.0), 1000.0 / per_token);
            reports.push(json!({
                "prompt": r.prompt, "cosine": cos, "max_abs_diff": max_diff, "top1_same": ours5[0] == ref5[0], "top5_overlap": overlap,
                "matching_tokens": matching, "compared_tokens": ids.len().min(r.ids.len()), "text": text, "reference_text": r.text,
                "prefill_ms": prefill_ms, "decode_ms_per_token": per_token,
            }));
        }

        // Decode speed with the GPU-resident loop: 4 tokens per submission.
        if let Some(path) = references.first() {
            let r: Reference = serde_json::from_str(&std::fs::read_to_string(path)?)?;
            model.reset();
            pollster::block_on(model.prefill_batched(&r.prompt_ids))?;
            let t = Instant::now();
            let n = 33;
            pollster::block_on(model.decode(n, 4, &[]))?;
            let per_token = t.elapsed().as_secs_f64() * 1000.0 / (n - 1) as f64;
            eprintln!("\ndecode, 4 tokens per submission: {per_token:.1} ms/token = {:.1} tok/s", 1000.0 / per_token);
            reports.push(json!({ "window4_ms_per_token": per_token }));
        }
        if let Some(path) = out {
            std::fs::write(path, serde_json::to_string_pretty(&reports)?)?;
        }
        Ok(())
    }
}

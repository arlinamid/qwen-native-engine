//! Native runner for the decode-kernel prototype.
//!
//! cargo run --release -p qwen-kernels --bin kernel_bench -- [--context 512] [--tokens 32] [--no-tune] [--out file.json]

#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    let mut config = qwen_kernels::Config::default();
    let mut out = None;
    let mut bandwidth_only = false;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--context" => config.context = args.next().unwrap_or_default().parse()?,
            "--tokens" => config.tokens = args.next().unwrap_or_default().parse()?,
            "--windows" => config.windows = args.next().unwrap_or_default().split(',').map(str::parse).collect::<Result<_, _>>()?,
            "--no-tune" => config.tune = false,
            "--no-check" => config.check = false,
            "--no-f16" => config.f16 = false,
            "--bandwidth" => bandwidth_only = true,
            "--dxc" => {
                let path = std::fs::canonicalize(args.next().unwrap_or_default())?;
                // dxcompiler.dll loads dxil.dll (shader signing) from the DLL search path.
                if let Some(dir) = path.parent() {
                    let old = std::env::var_os("PATH").unwrap_or_default();
                    let joined = std::env::join_paths(std::iter::once(dir.to_path_buf()).chain(std::env::split_paths(&old)))?;
                    unsafe { std::env::set_var("PATH", joined) };
                }
                config.dxc = Some(path.to_string_lossy().trim_start_matches(r"\?\").to_string());
            }
            "--out" => out = args.next(),
            other => anyhow::bail!("unknown flag {other}"),
        }
    }
    if bandwidth_only {
        pollster::block_on(qwen_kernels::bandwidth_probe(config, &|line| eprintln!("{line}")))?;
        return Ok(());
    }
    let report = pollster::block_on(qwen_kernels::run(config, &|line| eprintln!("{line}")))?;
    if let Some(path) = out {
        std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
    }
    Ok(())
}

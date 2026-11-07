# Source this before building. Keeps Rust, npm and Hugging Face caches inside the project (D:).
ROOT="D:/tool/qwen_0.6b"
export RUSTUP_HOME="$ROOT/.toolchain/rustup"
export CARGO_HOME="$ROOT/.toolchain/cargo"
export RUSTUP_TOOLCHAIN="stable-x86_64-pc-windows-gnu"
export npm_config_cache="$ROOT/.toolchain/npm"
export HF_HOME="$ROOT/.toolchain/hf"
export PATH="/d/tool/qwen_0.6b/.toolchain/cargo/bin:$PATH"

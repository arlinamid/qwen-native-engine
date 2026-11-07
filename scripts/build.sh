#!/usr/bin/env bash
# Builds the Rust crate to web/pkg. All tool caches stay inside the project (see env.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
cargo build --release --target wasm32-unknown-unknown --lib
wasm-bindgen --target web --out-dir web/pkg --no-typescript target/wasm32-unknown-unknown/release/qwen_wasm.wasm
ls -la web/pkg

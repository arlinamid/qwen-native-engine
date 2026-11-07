#!/usr/bin/env bash
# Builds the decode-kernel prototype to web/kpkg (open /kernels.html on the dev server).
set -euo pipefail
cd "$(dirname "$0")/.."
. scripts/env.sh
cargo build --release -p qwen-kernels --target wasm32-unknown-unknown --lib
wasm-bindgen --target web --out-dir web/kpkg --no-typescript target/wasm32-unknown-unknown/release/qwen_kernels.wasm
ls -la web/kpkg

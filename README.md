# qwen-native-engine

Local **Qwen3.5** stack: a hand-written WebGPU/D3D12 decoder, a WASM/browser engine, and a TUI agent that talks to MCP tools. Everything runs on the machine. Weights stay on disk; they are not in this repository.

| Piece | Crate | What it is |
|---|---|---|
| Native decoder | `kernels` | Fused f16 GEMV, GDN, attention, SwiGLU. Intel UHD 630 tiles for 0.8B and 2B. |
| WASM / prompt / grammar | `qwen-wasm` | Chat template, llguidance tool grammar, sampling. |
| CLI agent | `qwen-agent` | Warm GPU server, TUI, skills, MCP (stdio). |

The agent crate’s Hungarian usage guide is in [`agent/README.md`](agent/README.md).

## Requirements

- Rust (edition 2024) and a GPU that can run wgpu (Windows: D3D12; f16 kernels need DXC).
- Converted weights, not the Hugging Face tree as-is:

```bash
python -I scripts/convert_weights.py
```

Expected layout: `models/qwen3.5-0.8b/native/` and/or `models/qwen3.5-2b/native/` (`weights.bin` + `index.json`). The `models/` directory is gitignored.

## Build and run

```bash
cargo build --release -p qwen-agent
./target/release/qwen-agent.exe
```

First launch starts a background server, loads the model, and opens the TUI. Later launches reuse the warm process. `/exit` stops the TUI; `qwen-agent stop` unloads the GPU.

Switch size in the TUI (`/model 0.8b` or `/model 2b`) or in `.qwen-agent/settings.json` (`model_dir`, `kernel_tiles`). Profiles: `kernels/profiles/uhd630-qwen35-0.8b.json` and `uhd630-qwen35-2b.json`.

One-shot, no TUI:

```bash
./target/release/qwen-agent.exe --prompt "What time is it?" --yes
```

## MCP and skills

Default servers: `mcp_servers/mcp.json` (Wikipedia/news/notes + demo calculator/time). Tool schemas come from the live MCP catalog; the harness does not bake in tool names.

Skills live in `.qwen-agent/skills/*/SKILL.md` (created on first run). Matching uses the skill `triggers` field, not a hardcoded phrase list in Rust. Edit triggers when you add a new MCP. The harness injects the skill body on matching tasks.

## Layout

```
agent/          TUI, server, MCP client, skills
kernels/        native WebGPU decoder + benches
src/            WASM engine, prompt, grammar, sampling
mcp_servers/    knowledge MCP (stdlib Python)
examples/       demo MCP + sandbox
scripts/        weight convert, eval
web/            browser harness
docs/           2B/UHD 630 notes and reviews
```

Context window is 4096 tokens. Decode on UHD 630 is on the order of 10–20 tok/s depending on model and whether logits are read back for sampling/grammar.

## Tests

```bash
cargo test -p qwen-agent
cargo test -p qwen-wasm --lib
```

Live MCP eval (needs a built agent and a running model): `python -I scripts/eval_mcp.py`.

## License

Apache-2.0. Model weights are subject to their own licenses (Qwen).

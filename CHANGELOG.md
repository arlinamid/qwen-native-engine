# Changelog

## Unreleased

## 0.1.0 — 2026-11-08

### Agent

- Warm local server + TUI (`qwen-agent`): MCP over stdio, tool approval, sessions, profile (`IDENTITY` / `SOUL` / `USER`).
- Harness-forced skills from `.qwen-agent/skills/*/SKILL.md`. Matching uses frontmatter `triggers`; skill bodies are injected on the encode view only.
- Chat vs task gate uses the **connected** MCP catalog (names and descriptions), not a hardcoded tool list. Skill hits promote a turn to a tool task.
- Tool-call logit bias on `<tool_call>` (`+4` when a tool is still needed, `-8` after a success). Duplicate successful calls are skipped; empty or chat-emitted tool XML is recovered instead of a blank truncated turn.
- Thinking: auto / on / off. Think tags are not streamed as answer text. Finish gate recovers once if a required tool never ran.
- Knowledge MCP: Wikipedia, news, `read_page`, `search_notes`. Demo MCP: time, calculate, sandbox files.
- Cursor-style command/skill picker, `/create-skill`, `/validate-skill`.

### Native engine

- Hand-written WebGPU/D3D12 decoder for Qwen3.5-0.8B and Qwen3.5-2B (q4f16, fused GEMV, GDN, SwiGLU).
- UHD 630 tile profiles for both sizes.
- Weight converter (`scripts/convert_weights.py`) writes `weights.bin` + `index.json`.
- Grammar-constrained tool calls (llguidance) after the model opens `<tool_call>`.

### WASM

- Browser/ONNX Runtime path with the same prompt template, sampling, and tool-call parser.

### Fixes

- Review pass: seven correctness issues in the early native/WASM engine.
- Classifier no longer treats “what tools do you have?” / “write a haiku” as tool tasks.
- Truncation note in the TUI only claims a token-limit cut when stop is actually `max_tokens`.

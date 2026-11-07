"""Measures how well the agent uses the demo MCP server, with and without the generated MCP
section in the system prompt (--mcp-guide / --no-mcp-guide).

    python -I scripts/eval_mcp.py                      # noguide and guide, greedy
    python -I scripts/eval_mcp.py guide thinking       # chosen variants
    python -I scripts/eval_mcp.py --local --sample guide thinking

Variants: noguide, guide, thinking (guide + thinking mode). Each variant runs its own background
server, or with --local loads the model for every case (slower, but leaves a running server
alone). Every case is a fresh conversation; greedy unless --sample (temperature 0.6, fixed seed).
"""
import json
import os
import subprocess
import sys
import time

EXE = os.path.join("target", "release", "qwen-agent.exe" if os.name == "nt" else "qwen-agent")
SERVER_FILE = os.path.join(".qwen-agent", "server.json")
TOOLS = ["get_time", "calculate", "list_files", "read_file"]

# (prompt, kind, expected tool): "meta" must name at least 3 tools, "task" must call the tool,
# "chat" must call none.
CASES = [
    ("What MCP tools do you have?", "meta", None),
    ("Which MCP servers are you connected to, and what can they do?", "meta", None),
    ("tell me what mcp tools you had?", "meta", None),
    ("Milyen MCP-eszközeid vannak?", "meta", None),
    ("Milyen eszközöket tudsz használni?", "meta", None),
    ("What time is it?", "task", "get_time"),
    ("What is 1234 * 56?", "task", "calculate"),
    ("Which files are in the sandbox?", "task", "list_files"),
    ("Read the file jegyzet.txt.", "task", "read_file"),
    ("Hány óra van most?", "task", "get_time"),
    ("Mennyi 37 * 89?", "task", "calculate"),
    ("Milyen fájlok vannak a sandboxban?", "task", "list_files"),
    ("Hello! Who are you?", "chat", None),
    ("Write a haiku about autumn.", "chat", None),
    ("Szia, hogy vagy?", "chat", None),
]


VARIANTS = {
    "noguide": ["--no-mcp-guide"],
    "guide": ["--mcp-guide"],
    "thinking": ["--mcp-guide", "--thinking", "--max-tokens", "1024"],
}


def start_server(flags):
    if os.path.exists(SERVER_FILE):
        # It may be someone's TUI session; do not stop it from here.
        sys.exit("a qwen-agent server is running; stop it first (qwen-agent stop)")
    log = open(os.path.join(".qwen-agent", "eval_server.log"), "w")
    proc = subprocess.Popen([EXE, "serve", "--mcp", "examples/mcp.json", *flags], stdout=log, stderr=log, stdin=subprocess.DEVNULL)
    for _ in range(600):
        if os.path.exists(SERVER_FILE):
            return proc
        if proc.poll() is not None:
            sys.exit("server exited; see .qwen-agent/eval_server.log")
        time.sleep(0.5)
    sys.exit("server did not start")


def run_case(prompt, flags, local, sample):
    base = [EXE, "--prompt", prompt, "--mcp", "examples/mcp.json", "--yes", "--max-steps", "3", "--max-tokens", "256"]
    base += ["--temperature", "0.6"] if sample else ["--greedy"]
    out = subprocess.run(base + (["--local"] if local else []) + flags,
                         capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=600)
    called = [line.split()[1] for line in out.stderr.splitlines() if line.strip().startswith("-> ")]
    return out.stdout.strip(), called


def score(kind, expected, text, called):
    if kind == "meta":
        named = [t for t in TOOLS if t in text.lower() or t.replace("_", " ") in text.lower()]
        return len(named) >= 3, f"named {len(named)}/4"
    if kind == "task":
        return expected in called, f"called {called or 'nothing'}"
    return not called, f"called {called or 'nothing'}"


def evaluate(flags, local, sample):
    proc = None if local else start_server(flags)
    results = []
    try:
        for prompt, kind, expected in CASES:
            text, called = run_case(prompt, flags, local, sample)
            ok, why = score(kind, expected, text, called)
            results.append({"prompt": prompt, "kind": kind, "ok": ok, "why": why, "answer": text[:300]})
            print(f"  [{'PASS' if ok else 'fail'}] {kind:4} {prompt!r}: {why}", flush=True)
    finally:
        if proc:
            subprocess.run([EXE, "stop"], capture_output=True)
            proc.wait(timeout=30)
    return results


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = sys.argv[1:]
    local, sample = "--local" in args, "--sample" in args
    variants = [a for a in args if not a.startswith("--")] or ["noguide", "guide"]
    summary = {}
    for name in variants:
        print(f"== {name}{' (local)' if local else ''}{' (sampled)' if sample else ' (greedy)'}", flush=True)
        results = evaluate(VARIANTS[name], local, sample)
        summary[name] = results
        by_kind = {}
        for r in results:
            by_kind.setdefault(r["kind"], []).append(r["ok"])
        print("   " + ", ".join(f"{k} {sum(v)}/{len(v)}" for k, v in by_kind.items()) + f", total {sum(r['ok'] for r in results)}/{len(results)}")
    with open(os.path.join(".qwen-agent", f"eval_mcp_{'_'.join(variants)}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

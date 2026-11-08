"""2B model: MCP tools, skill inject, and multi-turn session continuity.

    python -I scripts/eval_skills_mcp_2b.py

Starts its own server (stop any running qwen-agent first). Writes
results/eval_skills_mcp_2b.json.
"""
from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
import time
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from eval_mcp import CASES, score as mcp_score  # noqa: E402

EXE = os.path.join("target", "release", "qwen-agent.exe" if os.name == "nt" else "qwen-agent")
SERVER_FILE = os.path.join(".qwen-agent", "server.json")


def send(sock: socket.socket, obj: dict) -> None:
    sock.sendall((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))


def read_event(sock: socket.socket, buf: bytearray) -> dict:
    while True:
        nl = buf.find(b"\n")
        if nl >= 0:
            line = bytes(buf[:nl])
            del buf[: nl + 1]
            return json.loads(line.decode("utf-8"))
        chunk = sock.recv(65536)
        if not chunk:
            raise RuntimeError("server closed the connection")
        buf.extend(chunk)


class Session:
    def __init__(self, port: int, token: str):
        self.sock = socket.create_connection(("127.0.0.1", port), 5)
        self.sock.settimeout(300)
        self.buf = bytearray()
        send(self.sock, {"type": "hello", "token": token})
        ready = read_event(self.sock, self.buf)
        if ready.get("type") != "ready":
            raise RuntimeError(f"handshake failed: {ready}")
        self.ready = ready
        self.session_id = str(uuid.uuid4())
        self.reset()

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()

    def reset(self) -> None:
        send(self.sock, {"type": "reset", "session_id": self.session_id})
        end = read_event(self.sock, self.buf)
        if end.get("type") != "end":
            raise RuntimeError(f"reset failed: {end}")

    def turn(self, text: str) -> list[dict]:
        send(
            self.sock,
            {
                "type": "turn",
                "text": text,
                "session_id": self.session_id,
                "options": {
                    "auto_approve": True,
                    "thinking_mode": "auto",
                    "temperature": 0.0,
                    "presence_penalty": 1.5,
                    "turn_timeout_secs": 180,
                    "max_steps": 6,
                    "max_tokens": 256,
                },
            },
        )
        events = []
        while True:
            event = read_event(self.sock, self.buf)
            events.append(event)
            if event.get("type") == "confirm":
                send(
                    self.sock,
                    {
                        "type": "approval",
                        "answer": "y",
                        "turn_id": event.get("turn_id"),
                        "call_id": event.get("call_id"),
                    },
                )
            if event.get("type") == "end":
                return events


def summarize(events: list[dict]) -> dict:
    calls = [e.get("name") for e in events if e.get("type") == "tool_call"]
    phases = [e.get("phase") for e in events if e.get("type") == "phase"]
    text = "".join(e.get("text", "") for e in events if e.get("type") == "text")
    status = next((e.get("status") for e in events if e.get("type") == "end"), None)
    return {"calls": calls, "phases": phases, "text": text.strip(), "status": status}


def start_server() -> subprocess.Popen:
    if os.path.exists(SERVER_FILE):
        subprocess.run([EXE, "stop"], cwd=ROOT, capture_output=True)
        for _ in range(80):
            if not os.path.exists(SERVER_FILE):
                break
            time.sleep(0.5)
    os.makedirs(".qwen-agent", exist_ok=True)
    log = open(os.path.join(".qwen-agent", "eval_skills_mcp_2b.log"), "w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            EXE,
            "serve",
            "--model",
            "2b",
            "--mcp",
            "examples/mcp.json",
            "--mcp-guide",
        ],
        cwd=ROOT,
        stdout=log,
        stderr=log,
        stdin=subprocess.DEVNULL,
    )
    for _ in range(600):
        if os.path.exists(SERVER_FILE):
            return proc
        if proc.poll() is not None:
            sys.exit("server exited; see .qwen-agent/eval_skills_mcp_2b.log")
        time.sleep(0.5)
    sys.exit("server did not start")


def judge_skill_mcp_tools(events: list[dict]) -> tuple[bool, str]:
    info = summarize(events)
    if info["status"] != "completed":
        return False, f"status {info['status']}"
    if "calculate" not in info["calls"]:
        return False, f"expected calculate, got {info['calls']}"
    skill = any(isinstance(p, str) and p.startswith("skill: mcp-tools") for p in info["phases"])
    if "69104" not in info["text"].replace(" ", ""):
        return False, "answer missing 69104"
    if not skill:
        return False, "no skill: mcp-tools phase (answer otherwise ok)"
    return True, "skill injected, calculate, 69104"


def judge_skill_notes(events: list[dict]) -> tuple[bool, str]:
    info = summarize(events)
    if info["status"] != "completed":
        return False, f"status {info['status']}"
    if "read_file" not in info["calls"]:
        return False, f"expected read_file, got {info['calls']}"
    skill = any(isinstance(p, str) and p.startswith("skill: notes-files") for p in info["phases"])
    if "14:30" not in info["text"]:
        return False, "answer missing 14:30"
    if not skill:
        return False, "no skill: notes-files phase (answer otherwise ok)"
    return True, "skill injected, read_file, 14:30"


def judge_chat_no_tools(events: list[dict]) -> tuple[bool, str]:
    info = summarize(events)
    if info["status"] != "completed" or not info["text"]:
        return False, f"status {info['status']} empty={not info['text']}"
    if info["calls"]:
        return False, f"chat called tools {info['calls']}"
    if any(isinstance(p, str) and p.startswith("skill:") for p in info["phases"]):
        return False, f"chat loaded a skill {info['phases']}"
    return True, "chat, no tools, no skill"


def judge_followup_name(first: list[dict], second: list[dict]) -> tuple[bool, str]:
    a, b = summarize(first), summarize(second)
    if a["status"] != "completed" or b["status"] != "completed":
        return False, f"status {a['status']}/{b['status']}"
    if a["calls"] or b["calls"]:
        return False, f"tools {a['calls']}/{b['calls']}"
    if "arlin" not in b["text"].casefold():
        return False, f"did not recall name: {b['text'][:180]}"
    return True, "multi-turn name recall"


def judge_followup_calc(first: list[dict], second: list[dict]) -> tuple[bool, str]:
    a, b = summarize(first), summarize(second)
    if a["status"] != "completed" or b["status"] != "completed":
        return False, f"status {a['status']}/{b['status']}"
    if "calculate" not in a["calls"]:
        return False, f"first turn tools {a['calls']}"
    if "69104" not in a["text"].replace(" ", "") and "69104" not in b["text"].replace(" ", ""):
        return False, "69104 never appeared"
    if "69104" not in b["text"].replace(" ", "") and "138208" not in b["text"].replace(" ", ""):
        return False, f"follow-up missing prior result: {b['text'][:180]}"
    return True, "multi-turn kept 1234*56 result"


def judge_chat_then_task(first: list[dict], second: list[dict]) -> tuple[bool, str]:
    a, b = summarize(first), summarize(second)
    if a["calls"]:
        return False, f"chat called {a['calls']}"
    if "calculate" not in b["calls"]:
        return False, f"task tools {b['calls']}"
    if "3293" not in b["text"].replace(" ", ""):
        return False, f"missing 3293: {b['text'][:180]}"
    return True, "chat then calculate in one session"


def run() -> dict:
    proc = start_server()
    info = json.loads(open(SERVER_FILE, encoding="utf-8").read())
    session = Session(info["port"], info["token"])
    report: dict = {
        "ready": {
            "tools": session.ready.get("tools"),
            "system_tokens": session.ready.get("system_tokens"),
            "skills": session.ready.get("skills"),
            "model_dir": (session.ready.get("settings") or {}).get("model_dir"),
        },
        "unit": {},
        "mcp": [],
        "skills": [],
        "multiturn": [],
    }
    print(
        f"server ready: {report['ready']['tools']} tools, "
        f"{report['ready']['system_tokens']} system tokens, "
        f"skills={report['ready']['skills']}",
        flush=True,
    )
    try:
        print("== MCP (fresh session per case, greedy, 2B)", flush=True)
        for prompt, kind, expected in CASES:
            session.reset()
            events = session.turn(prompt)
            ok, why = mcp_score(kind, expected, events, 0, prompt)
            text = summarize(events)["text"]
            report["mcp"].append({"prompt": prompt, "kind": kind, "ok": ok, "why": why, "answer": text[:300], "calls": summarize(events)["calls"]})
            print(f"  [{'PASS' if ok else 'fail'}] {kind:4} {prompt!r}: {why}", flush=True)

        print("== Skills", flush=True)
        skill_cases = [
            ("$mcp-tools Use calculate for 1234*56", judge_skill_mcp_tools),
            ("$notes-files Read the file jegyzet.txt", judge_skill_notes),
            ("Szia, hogy vagy?", judge_chat_no_tools),
        ]
        for prompt, judge in skill_cases:
            session.reset()
            events = session.turn(prompt)
            ok, why = judge(events)
            info_s = summarize(events)
            report["skills"].append({"prompt": prompt, "ok": ok, "why": why, "answer": info_s["text"][:300], "calls": info_s["calls"], "phases": info_s["phases"]})
            print(f"  [{'PASS' if ok else 'fail'}] {prompt!r}: {why}", flush=True)

        print("== Multi-turn", flush=True)
        session.reset()
        t1 = session.turn("Szia, a nevem Arlin.")
        t2 = session.turn("Mi a nevem? Ne hívj eszközt.")
        ok, why = judge_followup_name(t1, t2)
        report["multiturn"].append({"name": "name-recall", "ok": ok, "why": why, "t1": summarize(t1), "t2": summarize(t2)})
        print(f"  [{'PASS' if ok else 'fail'}] name-recall: {why}", flush=True)

        session.reset()
        t1 = session.turn("Use the calculate tool for 1234*56")
        t2 = session.turn("What number did the tool return? Answer with that number.")
        ok, why = judge_followup_calc(t1, t2)
        report["multiturn"].append({"name": "calc-followup", "ok": ok, "why": why, "t1": summarize(t1), "t2": summarize(t2)})
        print(f"  [{'PASS' if ok else 'fail'}] calc-followup: {why}", flush=True)

        session.reset()
        t1 = session.turn("Hello!")
        t2 = session.turn("Use calculate for 37*89")
        ok, why = judge_chat_then_task(t1, t2)
        report["multiturn"].append({"name": "chat-then-task", "ok": ok, "why": why, "t1": summarize(t1), "t2": summarize(t2)})
        print(f"  [{'PASS' if ok else 'fail'}] chat-then-task: {why}", flush=True)
    finally:
        session.close()
        subprocess.run([EXE, "stop"], cwd=ROOT, capture_output=True)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
    return report


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    report = run()
    groups = {
        "mcp": report["mcp"],
        "skills": report["skills"],
        "multiturn": report["multiturn"],
    }
    print("== totals", flush=True)
    for name, rows in groups.items():
        n = sum(1 for r in rows if r["ok"])
        print(f"  {name} {n}/{len(rows)}", flush=True)
    total_ok = sum(1 for rows in groups.values() for r in rows if r["ok"])
    total = sum(len(rows) for rows in groups.values())
    print(f"  all {total_ok}/{total}", flush=True)
    os.makedirs("results", exist_ok=True)
    path = os.path.join("results", "eval_skills_mcp_2b.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"wrote {path}", flush=True)
    if total_ok < total:
        sys.exit(1)


if __name__ == "__main__":
    main()

"""A small, read-only MCP server for trying qwen-agent (stdio transport, standard library only).

Tools: get_time, calculate, list_files and read_file. The file tools only see the
examples/sandbox folder. It speaks both protocol eras: it answers `server/discover`
(revision 2026-07-28, per-request `_meta`) and the `initialize` handshake (2025-11-25).
Run with --legacy to drop `server/discover` and test the client's fallback.
"""
import ast
import datetime
import json
import operator
import sys
from pathlib import Path

SANDBOX = (Path(__file__).resolve().parent / "sandbox").resolve()
LEGACY_ONLY = "--legacy" in sys.argv
MODERN = "2026-07-28"
LEGACY_VERSIONS = ["2025-11-25", "2025-06-18", "2025-03-26"]

TOOLS = [
    {
        "name": "get_time",
        "description": "Current local date and time of this computer, or UTC.",
        "inputSchema": {"type": "object", "properties": {"utc": {"type": "boolean", "description": "Return UTC instead of local time"}}},
    },
    {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression with + - * / ** and parentheses.",
        "inputSchema": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]},
    },
    {
        "name": "list_files",
        "description": "List the files in a folder of the sandbox.",
        "inputSchema": {"type": "object", "properties": {"path": {"type": "string", "description": "Folder relative to the sandbox, '.' for its root"}}},
    },
    {
        "name": "read_file",
        "description": "Read a text file from the sandbox (first 4000 characters).",
        "inputSchema": {"type": "object", "properties": {"path": {"type": "string", "description": "File relative to the sandbox"}}, "required": ["path"]},
    },
]

OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def evaluate(node):
    if isinstance(node, ast.Expression):
        return evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        if isinstance(node.op, ast.Pow) and abs(evaluate(node.right)) > 100:
            raise ValueError("exponent too large")
        return OPS[type(node.op)](evaluate(node.left), evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](evaluate(node.operand))
    raise ValueError("only numbers and + - * / ** ( ) are allowed")


def inside_sandbox(relative):
    path = (SANDBOX / (relative or ".")).resolve()
    if path != SANDBOX and SANDBOX not in path.parents:
        raise ValueError("path is outside the sandbox")
    return path


def run_tool(name, args):
    if name == "get_time":
        now = datetime.datetime.now(datetime.timezone.utc) if args.get("utc") else datetime.datetime.now().astimezone()
        return now.strftime("%Y-%m-%d %H:%M:%S %Z")
    if name == "calculate":
        return str(evaluate(ast.parse(str(args["expression"]), mode="eval")))
    if name == "list_files":
        folder = inside_sandbox(args.get("path"))
        return "\n".join(sorted(p.name + ("/" if p.is_dir() else "") for p in folder.iterdir())) or "(empty)"
    if name == "read_file":
        return inside_sandbox(args["path"]).read_text(encoding="utf-8")[:4000]
    raise KeyError(name)


INSTRUCTIONS = "Read-only demo tools: time, arithmetic and a sandbox folder."


def result_for(message):
    method, params = message.get("method"), message.get("params") or {}
    if method == "server/discover":
        if LEGACY_ONLY:
            return None, {"code": -32601, "message": "Method not found"}
        return {"resultType": "complete", "supportedVersions": [MODERN], "capabilities": {"tools": {}}, "ttlMs": 0, "cacheScope": "public",
                "instructions": INSTRUCTIONS}, None
    if method == "initialize":
        asked = params.get("protocolVersion")
        version = asked if asked in LEGACY_VERSIONS else LEGACY_VERSIONS[0]
        return {"protocolVersion": version, "capabilities": {"tools": {}}, "serverInfo": {"name": "qwen-demo", "version": "1.0.0"},
                "instructions": INSTRUCTIONS}, None
    if method == "ping":
        return {}, None
    if method == "tools/list":
        return {"tools": TOOLS}, None
    if method == "tools/call":
        name = params.get("name")
        if name not in {t["name"] for t in TOOLS}:
            return None, {"code": -32602, "message": f"Unknown tool: {name}"}
        try:
            return {"content": [{"type": "text", "text": run_tool(name, params.get("arguments") or {})}], "isError": False}, None
        except Exception as error:  # reported to the model, which may retry
            return {"content": [{"type": "text", "text": str(error)}], "isError": True}, None
    return None, {"code": -32601, "message": "Method not found"}


def main():
    # MCP messages are UTF-8; Windows consoles would otherwise default to a legacy code page.
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        message = json.loads(line)
        if "id" not in message:
            continue  # notifications need no answer
        result, error = result_for(message)
        reply = {"jsonrpc": "2.0", "id": message["id"]}
        if error is not None:
            reply["error"] = error
        else:
            if "_meta" in (message.get("params") or {}) and "resultType" not in result:
                result["resultType"] = "complete"
            reply["result"] = result
        sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
        sys.stdout.flush()


main()

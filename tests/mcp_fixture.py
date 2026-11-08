"""Adversarial stdio peer for the Rust transport tests; no network or model needed."""
import json
import sys
import time

mode = sys.argv[1]
for line in sys.stdin:
    request = json.loads(line)
    if "id" not in request:
        continue
    method = request["method"]
    reply = {"jsonrpc": "2.0", "id": request["id"]}
    if method == "server/discover":
        if mode == "version":
            reply["error"] = {"code": -32022, "message": "version", "data": {"supported": ["2099-01-01"]}}
        elif mode == "legacy":
            reply["error"] = {"code": -32601, "message": "legacy"}
        else:
            reply["result"] = {"supportedVersions": ["2026-07-28"]}
    elif method == "initialize":
        reply["result"] = {"protocolVersion": "2025-11-25", "capabilities": {"tools": {}}}
    elif method == "tools/list":
        reply["result"] = {"tools": []}
        if mode == "cursor":
            reply["result"]["nextCursor"] = "same"
    elif method == "tools/call":
        if mode == "hang":
            time.sleep(30)
        if mode == "oversize":
            print("x" * (1024 * 1024 + 10), flush=True)
            continue
        if mode == "exit":
            print("fixture failed", file=sys.stderr, flush=True)
            sys.exit(3)
        reply["result"] = {"resultType": "complete", "content": [{"type": "text", "text": "42"}], "isError": False}
    else:
        reply["error"] = {"code": -32601, "message": "unknown"}
    print(json.dumps(reply), flush=True)

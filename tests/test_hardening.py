"""GPU-free regressions for MCP peers and the evaluator. Run with unittest discovery."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
def module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

knowledge = module("knowledge", "mcp_servers/knowledge.py")
eval_mcp = module("eval_mcp", "scripts/eval_mcp.py")
demo = module("demo", "examples/mcp_demo_server.py")

class Hardening(unittest.TestCase):
    def test_empty_notes_cache_invalidation_and_limits(self):
        with tempfile.TemporaryDirectory() as folder:
            knowledge.NOTES = Path(folder).resolve()
            path = knowledge.NOTES / "note.txt"
            path.write_text("", encoding="utf-8")
            self.assertIn("no searchable text", knowledge.search_notes({"query":"Budapest"}))
            path.write_text("Budapest meeting on Friday.", encoding="utf-8")
            self.assertIn("Friday", knowledge.search_notes({"query":"Budapest"}))
            cached = knowledge.note_passages()
            self.assertIs(cached, knowledge.note_passages())
            path.write_text("x" * 262145, encoding="utf-8")
            with self.assertRaises(ValueError): knowledge.note_passages()

    def test_private_urls_and_redirects_are_checked(self):
        answers = [(2, 1, 6, "", ("127.0.0.1", 80))]
        with patch.object(knowledge.socket, "getaddrinfo", return_value=answers):
            with self.assertRaises(ValueError): knowledge.check_url("http://localhost/")
            with self.assertRaises(ValueError): knowledge.CheckedRedirect().redirect_request(None,None,302,"",{},"http://localhost/")
        with self.assertRaises(ValueError): knowledge.check_url("file:///secret")

    def test_peers_survive_bad_json(self):
        for script in ("examples/mcp_demo_server.py", "mcp_servers/knowledge.py"):
            data = b'{broken}\n' + json.dumps({"jsonrpc":"2.0","id":1,"method":"ping"}).encode() + b'\n'
            run = subprocess.run([sys.executable,"-I",str(ROOT / script)],input=data,capture_output=True,timeout=5)
            replies = [json.loads(line) for line in run.stdout.splitlines()]
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertIn("error",replies[0])
            self.assertEqual(replies[1]["id"],1)
            self.assertEqual(replies[1]["result"],{})

    def test_demo_arithmetic_is_bounded(self):
        self.assertEqual(demo.run_tool("calculate",{"expression":"1234 * 56"}),"69104")
        with self.assertRaises(ValueError): demo.run_tool("calculate",{"expression":"((2**100)**100)**100"})
        with self.assertRaises(ValueError): demo.inside_sandbox("../outside")

    def test_eval_rejects_name_only_cutoff_failure_and_wrong_answer(self):
        events = [{"type":"tool_call","call_id":"1","name":"calculate","arguments":{"expression":"1234*56"}},
                  {"type":"tool_result","call_id":"1","output":"69104","is_error":False},
                  {"type":"text","text":"69104"}, {"type":"finished","status":"completed"}]
        def score(rows,code=0): return eval_mcp.score("task","calculate",rows,code,"1234 * 56")[0]
        self.assertTrue(score(events))
        self.assertFalse(score(events[:1]))
        self.assertFalse(score(events[:-1]))
        self.assertFalse(score(events,1))
        for row, key, value in ((3,"status","truncated"),(1,"is_error",True),(2,"text","123"),(1,"call_id","wrong")):
            modified = [dict(e) for e in events]
            modified[row][key]=value
            self.assertFalse(score(modified))

if __name__ == "__main__": unittest.main()

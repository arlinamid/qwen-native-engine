"""Writes tests/golden.json: prompts and token ids from Hugging Face's own chat template.

Run with: python -I scripts/gen_golden.py
"""
import json
from pathlib import Path

from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parent.parent
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Időjárás lekérdezése \"városra\"\nmásodik sor",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["city"],
            },
        },
    },
    {"type": "function", "function": {"name": "now", "parameters": {"type": "object", "properties": {}}}},
]
CALL = {
    "name": "get_weather",
    "arguments": {"city": "Győr", "n": 5, "x": 2.5, "ok": True, "off": False, "none": None, "list": [1, "ő", None], "obj": {"a": True}},
}

CASES = {
    "plain_user": ([{"role": "user", "content": "Szia!"}], None, True),
    "no_thinking": ([{"role": "user", "content": "Szia!"}], None, False),
    "content_is_trimmed": ([{"role": "system", "content": "  Légy tömör.\n"}, {"role": "user", "content": "\n Hi \n"}], None, False),
    "tools_no_system": ([{"role": "user", "content": "Milyen idő van?"}], TOOLS, False),
    "tools_and_system": ([{"role": "system", "content": "Légy tömör."}, {"role": "user", "content": "Hi"}], TOOLS, True),
    "tool_round_trip": (
        [
            {"role": "user", "content": "Idő Győrben?"},
            {"role": "assistant", "content": "", "tool_calls": [CALL]},
            {"role": "tool", "content": "{\"temp_c\": 18}"},
        ],
        TOOLS,
        False,
    ),
    "parallel_tools_with_text": (
        [
            {"role": "user", "content": "Kettő"},
            {"role": "assistant", "content": "Nézem.", "tool_calls": [CALL, {"name": "now", "arguments": {}}]},
            {"role": "tool", "content": "a"},
            {"role": "tool", "content": "b"},
            {"role": "assistant", "content": "Kész."},
        ],
        TOOLS,
        False,
    ),
    "tool_then_new_query": (
        [
            {"role": "user", "content": "Első"},
            {"role": "assistant", "content": "", "tool_calls": [{"name": "now", "arguments": {}}]},
            {"role": "tool", "content": "t"},
            {"role": "assistant", "content": "Kész."},
            {"role": "user", "content": "Második"},
        ],
        TOOLS,
        True,
    ),
    "history_with_reasoning": (
        [
            {"role": "user", "content": "Első"},
            {"role": "assistant", "content": "<think>\nregi\n</think>\n\nVálasz"},
            {"role": "user", "content": "Második"},
            {"role": "assistant", "content": "<think>\nuj\n</think>\n\nVége"},
        ],
        None,
        True,
    ),
    "user_looks_like_tool_response": (
        [
            {"role": "user", "content": "Valódi"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "<tool_response>x</tool_response>"},
            {"role": "assistant", "content": "<think>\nr\n</think>\n\nb"},
            {"role": "user", "content": "<tool_response>y</tool_response>"},
        ],
        None,
        False,
    ),
}


def main():
    # Built by hand: this tokenizer_config.json names a class that only exists in transformers 5.
    model = ROOT / "models" / "qwen3.5-0.8b"
    tok = PreTrainedTokenizerFast(
        tokenizer_file=str(model / "tokenizer.json"),
        chat_template=(model / "chat_template.jinja").read_text(encoding="utf-8"),
    )
    out = []
    for name, (messages, tools, thinking) in CASES.items():
        text = tok.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
        )
        out.append(
            {
                "name": name,
                "messages": messages,
                "tools": tools or [],
                "enable_thinking": thinking,
                "text": text,
                "ids": tok(text, add_special_tokens=False)["input_ids"],
            }
        )
    path = ROOT / "tests" / "golden.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(out)} cases to {path}")


main()

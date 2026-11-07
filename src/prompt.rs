//! Rust rendering of the Qwen3.5 chat template (`chat_template.jinja`), kept byte-identical to
//! the Jinja original by golden tests in `tests/golden.rs`. Text only: no image or video parts.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::toolcall::ToolCall;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    #[serde(default)]
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}

impl Message {
    pub fn new(role: &str, content: impl Into<String>) -> Self {
        Self { role: role.into(), content: content.into(), tool_calls: Vec::new() }
    }
}

/// Serialises like Python's `json.dumps(x, ensure_ascii=False)`, which is what the template's
/// `tojson` filter produces: `", "` and `": "` separators, key order preserved.
pub fn py_json(value: &Value) -> String {
    use serde::Serialize as _;
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, PyFormatter);
    value.serialize(&mut ser).expect("JSON values always serialise");
    String::from_utf8(out).expect("serde_json emits UTF-8")
}

struct PyFormatter;

impl serde_json::ser::Formatter for PyFormatter {
    fn begin_array_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W, first: bool) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_key<W: ?Sized + std::io::Write>(&mut self, w: &mut W, first: bool) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b": ")
    }
}

const TOOLS_HEADER: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";
const TOOLS_FOOTER: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

/// The part of the prompt that does not change between turns: the system block with the tools.
/// It is the unit the engine caches as a KV-cache snapshot. Empty when there is nothing to cache.
pub fn render_system_block(messages: &[Message], tools: &[Value]) -> String {
    let system = messages.first().filter(|m| m.role == "system").map(|m| m.content.trim());
    let mut out = String::new();
    if !tools.is_empty() {
        out.push_str("<|im_start|>system\n");
        out.push_str(TOOLS_HEADER);
        for tool in tools {
            out.push('\n');
            out.push_str(&py_json(tool));
        }
        out.push_str(TOOLS_FOOTER);
        if let Some(system) = system.filter(|s| !s.is_empty()) {
            out.push_str("\n\n");
            out.push_str(system);
        }
        out.push_str("<|im_end|>\n");
    } else if let Some(system) = system {
        out.push_str("<|im_start|>system\n");
        out.push_str(system);
        out.push_str("<|im_end|>\n");
    }
    out
}

fn is_real_query(m: &Message) -> bool {
    let content = m.content.trim();
    m.role == "user" && !(content.starts_with("<tool_response>") && content.ends_with("</tool_response>"))
}

/// Mirrors the template's extraction of reasoning from an assistant message's trimmed content.
fn split_reasoning(content: &str) -> (&str, &str) {
    let Some((_, after)) = content.rsplit_once("</think>") else { return (content, "") };
    let first = content.split("</think>").next().unwrap_or("").trim_end_matches('\n');
    let reasoning = first.rsplit("<think>").next().unwrap_or(first).trim_start_matches('\n');
    (after.trim_start_matches('\n'), reasoning.trim())
}

/// The template prints scalars with Jinja's `string` filter, so Python's spelling applies.
fn parameter_text(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Null => "None".into(),
        Value::Number(n) => n.to_string(),
        other => py_json(other),
    }
}

pub fn render(messages: &[Message], tools: &[Value], add_generation_prompt: bool, enable_thinking: bool) -> String {
    let mut out = render_system_block(messages, tools);
    let last_query = messages.iter().rposition(is_real_query).unwrap_or(0);

    for (i, m) in messages.iter().enumerate() {
        let content = m.content.trim();
        match m.role.as_str() {
            "user" => {
                out.push_str("<|im_start|>user\n");
                out.push_str(content);
                out.push_str("<|im_end|>\n");
            }
            "assistant" => render_assistant(&mut out, m, content, i > last_query),
            "tool" => {
                if i > 0 && messages[i - 1].role != "tool" {
                    out.push_str("<|im_start|>user");
                }
                out.push_str("\n<tool_response>\n");
                out.push_str(content);
                out.push_str("\n</tool_response>");
                if i + 1 == messages.len() || messages[i + 1].role != "tool" {
                    out.push_str("<|im_end|>\n");
                }
            }
            _ => {}
        }
    }

    if add_generation_prompt {
        out.push_str("<|im_start|>assistant\n");
        out.push_str(if enable_thinking { "<think>\n" } else { "<think>\n\n</think>\n\n" });
    }
    out
}

fn render_assistant(out: &mut String, m: &Message, content: &str, after_last_query: bool) {
    let (content, reasoning) = split_reasoning(content);
    out.push_str("<|im_start|>assistant\n");
    if after_last_query {
        out.push_str("<think>\n");
        out.push_str(reasoning);
        out.push_str("\n</think>\n\n");
    }
    out.push_str(content);
    for (j, call) in m.tool_calls.iter().enumerate() {
        if j > 0 {
            out.push('\n');
        } else if !content.trim().is_empty() {
            out.push_str("\n\n");
        }
        out.push_str("<tool_call>\n<function=");
        out.push_str(&call.name);
        out.push_str(">\n");
        if let Some(args) = call.arguments.as_object() {
            for (key, value) in args {
                out.push_str("<parameter=");
                out.push_str(key);
                out.push_str(">\n");
                out.push_str(&parameter_text(value));
                out.push_str("\n</parameter>\n");
            }
        }
        out.push_str("</function>\n</tool_call>");
    }
    out.push_str("<|im_end|>\n");
}

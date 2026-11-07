//! Extracts Qwen3.5's XML-style tool calls from model output and checks them against the tools.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

const OPEN: &str = "<tool_call>";
const CLOSE: &str = "</tool_call>";
const FUNCTION: &str = "<function=";
const PARAMETER: &str = "<parameter=";
const PARAMETER_END: &str = "</parameter>";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    #[serde(default = "empty_object")]
    pub arguments: Value,
}

fn empty_object() -> Value {
    Value::Object(Default::default())
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct ParsedOutput {
    pub reasoning: String,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    /// Malformed or unterminated call blocks. The model sees these as a tool error.
    pub errors: Vec<String>,
}

pub fn parse_output(text: &str, tools: &[Value]) -> ParsedOutput {
    let mut parsed = ParsedOutput::default();
    let mut rest = text;

    if let Some((before, after)) = rest.rsplit_once("</think>") {
        parsed.reasoning = before.replace("<think>", "").trim().to_string();
        rest = after;
    }

    let mut content = String::new();
    while let Some(start) = rest.find(OPEN) {
        content.push_str(&rest[..start]);
        let body = &rest[start + OPEN.len()..];
        match body.find(CLOSE) {
            Some(end) => {
                match parse_call(&body[..end], tools) {
                    Ok(call) => parsed.tool_calls.push(call),
                    Err(e) => parsed.errors.push(format!("invalid tool call: {e}")),
                }
                rest = &body[end + CLOSE.len()..];
            }
            None => {
                parsed.errors.push("unterminated <tool_call> block".to_string());
                rest = "";
            }
        }
    }
    content.push_str(rest);
    parsed.content = content.trim().to_string();
    parsed
}

fn parse_call(block: &str, tools: &[Value]) -> Result<ToolCall, String> {
    let start = block.find(FUNCTION).ok_or("missing <function=...>")?;
    let after = &block[start + FUNCTION.len()..];
    let name_end = after.find('>').ok_or("unterminated <function=...>")?;
    let name = after[..name_end].trim().to_string();
    let schema = tools.iter().find(|t| tool_name(t) == name).and_then(tool_parameters);

    let mut args = Map::new();
    let mut rest = &after[name_end + 1..];
    while let Some(p) = rest.find(PARAMETER) {
        let tail = &rest[p + PARAMETER.len()..];
        let key_end = tail.find('>').ok_or("unterminated <parameter=...>")?;
        let key = tail[..key_end].trim();
        let value_start = &tail[key_end + 1..];
        let value_end = value_start.find(PARAMETER_END).ok_or_else(|| format!("unterminated parameter '{key}'"))?;
        let raw = value_start[..value_end].strip_prefix('\n').unwrap_or(&value_start[..value_end]);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        args.insert(key.to_string(), coerce(raw, schema, key));
        rest = &value_start[value_end + PARAMETER_END.len()..];
    }
    Ok(ToolCall { name, arguments: Value::Object(args) })
}

/// Parameter values are plain text in this format; the tool's schema decides what they mean.
/// A value that does not fit stays a string, and `validate_call` then reports it.
fn coerce(raw: &str, schema: Option<&Value>, key: &str) -> Value {
    let declared = schema
        .and_then(|s| s.get("properties"))
        .and_then(|p| p.get(key))
        .and_then(|p| p.get("type"))
        .and_then(Value::as_str);
    let text = raw.trim();
    let parsed = match declared {
        Some("number" | "integer" | "boolean" | "array" | "object") => serde_json::from_str::<Value>(text).ok(),
        _ => None,
    };
    parsed.filter(|v| declared.is_some_and(|d| type_matches(d, v))).unwrap_or_else(|| Value::String(raw.to_string()))
}

pub fn tool_function(tool: &Value) -> &Value {
    tool.get("function").unwrap_or(tool)
}

pub fn tool_name(tool: &Value) -> &str {
    tool_function(tool).get("name").and_then(Value::as_str).unwrap_or("")
}

pub fn tool_parameters(tool: &Value) -> Option<&Value> {
    tool_function(tool).get("parameters")
}

fn type_matches(expected: &str, v: &Value) -> bool {
    match expected {
        "string" => v.is_string(),
        "number" => v.is_number(),
        "integer" => v.is_i64() || v.is_u64(),
        "boolean" => v.is_boolean(),
        "array" => v.is_array(),
        "object" => v.is_object(),
        "null" => v.is_null(),
        _ => true,
    }
}

/// Checks tool existence, required arguments, unknown arguments, and top-level types and enums.
pub fn validate_call(call: &ToolCall, tools: &[Value]) -> Result<(), String> {
    let tool = tools
        .iter()
        .find(|t| tool_name(t) == call.name)
        .ok_or_else(|| format!("unknown tool '{}'", call.name))?;
    let Some(params) = tool_parameters(tool) else { return Ok(()) };
    let args = call.arguments.as_object().ok_or("arguments must be an object")?;
    let props = params.get("properties").and_then(Value::as_object);

    if let Some(required) = params.get("required").and_then(Value::as_array) {
        for key in required.iter().filter_map(Value::as_str) {
            if !args.contains_key(key) {
                return Err(format!("missing required argument '{key}'"));
            }
        }
    }
    let Some(props) = props else { return Ok(()) };
    for (key, value) in args {
        let schema = props.get(key).ok_or_else(|| format!("unknown argument '{key}'"))?;
        if let Some(ty) = schema.get("type").and_then(Value::as_str) {
            if !type_matches(ty, value) {
                return Err(format!("argument '{key}' must be of type {ty}"));
            }
        }
        if let Some(allowed) = schema.get("enum").and_then(Value::as_array) {
            if !allowed.contains(value) {
                return Err(format!("argument '{key}' must be one of {allowed:?}"));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tools() -> Vec<Value> {
        vec![json!({"type":"function","function":{"name":"set_timer","parameters":{
            "type":"object",
            "properties":{"minutes":{"type":"number"},"unit":{"type":"string","enum":["a","b"]},"note":{"type":"string"}},
            "required":["minutes"]}}})]
    }

    #[test]
    fn parses_reasoning_content_and_calls() {
        let text = "hmm\n</think>\n\nOk.\n\n<tool_call>\n<function=set_timer>\n<parameter=minutes>\n5\n</parameter>\n<parameter=note>\ntwo\nlines\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=other>\n</function>\n</tool_call>";
        let p = parse_output(text, &tools());
        assert_eq!(p.reasoning, "hmm");
        assert_eq!(p.content, "Ok.");
        assert_eq!(p.tool_calls.len(), 2);
        assert_eq!(p.tool_calls[0].arguments, json!({"minutes": 5, "note": "two\nlines"}));
        assert_eq!(p.tool_calls[1], ToolCall { name: "other".into(), arguments: json!({}) });
        assert!(p.errors.is_empty());
    }

    #[test]
    fn plain_text_has_no_calls() {
        let p = parse_output("Hello!", &tools());
        assert_eq!(p.content, "Hello!");
        assert!(p.tool_calls.is_empty() && p.errors.is_empty());
    }

    #[test]
    fn reports_malformed_and_unterminated_blocks() {
        assert!(parse_output("<tool_call>{\"name\": \"x\"}</tool_call>", &[]).errors[0].contains("missing <function"));
        assert!(parse_output("a<tool_call>\n<function=x>", &[]).errors[0].contains("unterminated"));
        let open_param = "<tool_call><function=x><parameter=a>\n1</tool_call>";
        assert!(parse_output(open_param, &[]).errors[0].contains("unterminated parameter"));
    }

    #[test]
    fn a_value_that_does_not_fit_stays_a_string_and_fails_validation() {
        let text = "<tool_call>\n<function=set_timer>\n<parameter=minutes>\nsoon\n</parameter>\n</function>\n</tool_call>";
        let p = parse_output(text, &tools());
        assert_eq!(p.tool_calls[0].arguments["minutes"], "soon");
        assert!(validate_call(&p.tool_calls[0], &tools()).unwrap_err().contains("type number"));
    }

    #[test]
    fn validates_against_schema() {
        let t = tools();
        let call = |name: &str, args: Value| ToolCall { name: name.into(), arguments: args };
        assert!(validate_call(&call("set_timer", json!({"minutes": 5})), &t).is_ok());
        assert!(validate_call(&call("nope", json!({})), &t).unwrap_err().contains("unknown tool"));
        assert!(validate_call(&call("set_timer", json!({})), &t).unwrap_err().contains("missing"));
        assert!(validate_call(&call("set_timer", json!({"minutes": 5, "x": 1})), &t).unwrap_err().contains("unknown argument"));
        assert!(validate_call(&call("set_timer", json!({"minutes": 5, "unit": "c"})), &t).unwrap_err().contains("one of"));
    }
}

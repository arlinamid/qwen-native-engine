//! Cheap local rules for a small-model agent: chat vs task, plan-step thinking, and finish checks.
//! No second model call; the harness decides.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// How thinking is used for a message.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingMode {
    #[default]
    Auto,
    On,
    Off,
}

impl ThinkingMode {
    pub fn cycle(self) -> Self {
        match self {
            Self::Auto => Self::On,
            Self::On => Self::Off,
            Self::Off => Self::Auto,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::On => "on",
            Self::Off => "off",
        }
    }

    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "on" | "true" | "1" => Some(Self::On),
            "off" | "false" | "0" => Some(Self::Off),
            _ => None,
        }
    }
}

impl<'de> Deserialize<'de> for ThinkingMode {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        match value {
            serde_json::Value::Bool(true) => Ok(Self::On),
            // Old settings wrote `thinking: false` as the factory default; that means Auto now.
            serde_json::Value::Bool(false) => Ok(Self::Auto),
            serde_json::Value::String(s) => Self::parse(&s)
                .ok_or_else(|| serde::de::Error::custom(format!("unknown thinking mode {s}"))),
            _ => Ok(Self::Auto),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestKind {
    Chat,
    Task,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestPlan {
    pub text: String,
    pub kind: RequestKind,
    /// Explicit /think or /no_think on this message, if any.
    pub thinking_override: Option<bool>,
}

/// Sequence language. Domain phrases (weather, time, files) live in skill `triggers`, not here.
const SEQUENCE_CUES: &[&str] = &[
    " then ", " and then ", " first ", " after that ", " next ", " finally ", " step ", " először ",
    " aztán ", " majd ", " végül ", " utána ", " előbb ",
];
const USE_CUES: &[&str] = &[" use the ", " használd "];
/// Tokens too common to match a tool description against the user text.
const GENERIC_TOKENS: &[&str] = &[
    "the", "and", "for", "you", "your", "this", "that", "with", "from", "have", "has", "had",
    "what", "when", "which", "who", "how", "why", "does", "did", "are", "was", "not", "use",
    "used", "using", "call", "calls", "tool", "tools", "mcp", "function", "functions", "please",
    "just", "some", "more", "into", "about", "then", "them", "they", "will", "can", "get", "set",
    "add", "new", "name", "type", "text", "data", "input", "output", "return", "returns", "value",
    "values", "string", "number", "object", "true", "false", "null", "current", "local", "available",
    "connected", "server", "servers", "help", "need", "needs", "want", "make", "give", "tell",
    "show", "know", "one", "two", "any", "all", "also", "only", "now", "here", "there",
];
/// Shape of the catalog (not a fixed MCP name list): arithmetic vs file tools.
const MATH_HINTS: &[&str] = &["calculate", "calculator", "math", "arithmetic", "expression", "eval", "számol"];
const FILE_HINTS: &[&str] = &["file", "files", "path", "folder", "directory", "filename", "fájl"];

/// Strips a trailing `/think` or `/no_think` and classifies the rest against connected tools.
pub fn plan_request(raw: &str, tools: &[Value]) -> RequestPlan {
    let trimmed = raw.trim();
    let (text, thinking_override) = if let Some(rest) = strip_suffix_ci(trimmed, "/no_think") {
        (rest.trim_end().to_string(), Some(false))
    } else if let Some(rest) = strip_suffix_ci(trimmed, "/think") {
        (rest.trim_end().to_string(), Some(true))
    } else {
        (trimmed.to_string(), None)
    };
    let kind = if is_task(&text, tools) { RequestKind::Task } else { RequestKind::Chat };
    RequestPlan { text, kind, thinking_override }
}

fn strip_suffix_ci<'a>(text: &'a str, suffix: &str) -> Option<&'a str> {
    let lower = text.to_ascii_lowercase();
    let needle = suffix.to_ascii_lowercase();
    lower.strip_suffix(&needle).map(|_rest| &text[..text.len() - suffix.len()])
}

/// Chat is a greeting or a question the model can answer from the prompt. Tasks need a connected
/// tool, arithmetic/path the catalog can serve, or an explicit sequence of steps.
/// Language cues (weather, clock, notes) live in skill `triggers`, applied by the session.
pub fn is_task(text: &str, tools: &[Value]) -> bool {
    let t = text.trim();
    if t.is_empty() {
        return false;
    }
    if mentions_catalog(t, tools) {
        return true;
    }
    let blob = catalog_blob(tools);
    if (looks_like_math(t) && catalog_covers(&blob, MATH_HINTS))
        || (looks_like_path(t) && catalog_covers(&blob, FILE_HINTS))
    {
        return true;
    }
    let lower = format!(" {} ", t.to_ascii_lowercase());
    if USE_CUES.iter().any(|cue| lower.contains(cue)) {
        return true;
    }
    if SEQUENCE_CUES.iter().any(|cue| lower.contains(cue)) {
        return true;
    }
    if numbered_steps(t) {
        return true;
    }
    false
}

/// `1234*56` / `37 * 89` are tasks; `2+2` stays chat.
fn looks_like_math(text: &str) -> bool {
    let mut prev_digits = 0usize;
    let mut run = 0usize;
    let mut after_mul = false;
    for c in text.chars() {
        if c.is_ascii_digit() {
            run += 1;
            if after_mul && run > 0 {
                return true;
            }
            continue;
        }
        if run > 0 {
            prev_digits = run;
            run = 0;
        }
        if c.is_whitespace() || c == ',' {
            continue;
        }
        if matches!(c, '*' | '×' | '/' | '%') && prev_digits > 0 {
            after_mul = true;
            continue;
        }
        after_mul = false;
        prev_digits = 0;
    }
    after_mul && run > 0
}

fn looks_like_path(text: &str) -> bool {
    text.contains('\\')
        || text.contains('/') && text.split_whitespace().any(|w| w.contains('.') || w.contains('/') || w.contains('\\'))
        || text.split_whitespace().any(|w| {
            let w = w.trim_matches(|c: char| !c.is_alphanumeric() && c != '.' && c != '_' && c != '-' && c != '/' && c != '\\');
            w.contains('.') && (w.ends_with(".txt") || w.ends_with(".md") || w.ends_with(".json") || w.ends_with(".rs") || w.ends_with(".py"))
        })
}

fn numbered_steps(text: &str) -> bool {
    let mut n = 0;
    for line in text.lines() {
        let s = line.trim_start();
        if s.starts_with("1.") || s.starts_with("2.") || s.starts_with("1)") || s.starts_with("2)") || s.starts_with("- ") || s.starts_with("* ") {
            n += 1;
        }
    }
    n >= 2
}

/// The user asked for a connected tool, a file, the time, or arithmetic; a text-only answer is not enough.
pub fn requires_tool(text: &str, tools: &[Value]) -> bool {
    if mentions_catalog(text, tools) {
        return true;
    }
    let blob = catalog_blob(tools);
    if (looks_like_math(text) && catalog_covers(&blob, MATH_HINTS))
        || (looks_like_path(text) && catalog_covers(&blob, FILE_HINTS))
    {
        return true;
    }
    let lower = format!(" {} ", text.to_ascii_lowercase());
    USE_CUES.iter().any(|cue| lower.contains(cue))
}

fn catalog_blob(tools: &[Value]) -> String {
    let mut blob = String::new();
    for tool in tools {
        let function = tool.get("function").unwrap_or(tool);
        if let Some(name) = function.get("name").and_then(Value::as_str) {
            blob.push(' ');
            blob.push_str(&name.replace(['_', '-'], " "));
        }
        if let Some(description) = function.get("description").and_then(Value::as_str) {
            blob.push(' ');
            blob.push_str(description);
        }
    }
    blob.to_ascii_lowercase()
}

fn catalog_covers(blob: &str, hints: &[&str]) -> bool {
    hints.iter().any(|hint| contains_ident(blob, hint))
}

/// True when the user named a connected tool, or distinctive catalog tokens overlap the request.
fn mentions_catalog(text: &str, tools: &[Value]) -> bool {
    if tools.is_empty() {
        return false;
    }
    let lower = text.to_ascii_lowercase();
    for tool in tools {
        let function = tool.get("function").unwrap_or(tool);
        let name = function.get("name").and_then(Value::as_str).unwrap_or("");
        if name.len() >= 3 && contains_ident(&lower, name) {
            return true;
        }
        let spaced = name.replace(['_', '-'], " ");
        if spaced != name && spaced.len() >= 3 && contains_ident(&lower, &spaced) {
            return true;
        }
    }
    let query = distinctive_tokens(text);
    if query.is_empty() {
        return false;
    }
    for tool in tools {
        let function = tool.get("function").unwrap_or(tool);
        let name = function.get("name").and_then(Value::as_str).unwrap_or("");
        let description = function.get("description").and_then(Value::as_str).unwrap_or("");
        let hay = distinctive_tokens(&format!("{} {description}", name.replace(['_', '-'], " ")));
        if query.iter().any(|token| hay.iter().any(|h| h == token)) {
            return true;
        }
    }
    false
}

fn distinctive_tokens(text: &str) -> Vec<String> {
    text.to_ascii_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|token| token.len() >= 4 && !GENERIC_TOKENS.contains(token))
        .map(str::to_string)
        .collect()
}

fn contains_ident(haystack: &str, ident: &str) -> bool {
    let needle = ident.to_ascii_lowercase();
    if needle.len() < 3 {
        return false;
    }
    let hay = haystack.as_bytes();
    let n = needle.as_bytes();
    let mut i = 0;
    while i + n.len() <= hay.len() {
        if hay[i..i + n.len()].eq_ignore_ascii_case(n) {
            let before = i == 0 || !hay[i - 1].is_ascii_alphanumeric();
            let after = i + n.len() == hay.len() || !hay[i + n.len()].is_ascii_alphanumeric();
            if before && after {
                return true;
            }
        }
        i += 1;
    }
    false
}

/// Whether thinking runs on this hop under the saved mode and request plan.
pub fn thinking_for_hop(mode: ThinkingMode, plan: &RequestPlan, hop: usize, is_task: bool) -> bool {
    if let Some(force) = plan.thinking_override {
        return force && (hop == 0 || mode == ThinkingMode::On);
    }
    match mode {
        ThinkingMode::Off => false,
        ThinkingMode::On => true,
        ThinkingMode::Auto => is_task && hop == 0,
    }
}

pub const PLAN_THINKING_TOKENS: usize = 256;
pub const FULL_THINKING_TOKENS: usize = 1024;
pub const TASK_STEP_CAP: usize = 6;
pub const TOOL_RESULTS_KEPT: usize = 2;
pub const STUB_TOOL_RESULT: &str = "[earlier tool result omitted]";
pub const RECOVERY_NOTE: &str =
    "No tool ran for this request. Call the required tool now, then answer from its result. Do not claim success without a tool result.";
pub const ANSWER_HINT: &str =
    "The tool already returned a result. Answer from that result now. Do not call a tool.";
pub const BLANK_HINT: &str =
    "Your last reply was empty. Answer the user now. Call a connected tool if you need live data.";
/// A tool-call block does not fit in a handful of tokens.
pub const MIN_TASK_TOKENS: usize = 128;
pub const PLAN_HINT: &str =
    " Plan in at most three short lines inside <think>, then call one tool or answer.";
/// Added to the `<tool_call>` logit on the first hop that still needs a tool.
pub const TOOL_CALL_BIAS_NEED: f32 = 4.0;
/// Subtracted from the `<tool_call>` logit after a tool has already succeeded this turn.
pub const TOOL_CALL_BIAS_AFTER: f32 = -8.0;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn demo_tools() -> Vec<Value> {
        vec![
            json!({"type":"function","function":{"name":"calculate","description":"Evaluates arithmetic expressions"}}),
            json!({"type":"function","function":{"name":"get_time","description":"Returns current local time or UTC"}}),
            json!({"type":"function","function":{"name":"list_files","description":"Lists files in a folder"}}),
            json!({"type":"function","function":{"name":"read_file","description":"Reads text files from a path"}}),
        ]
    }

    fn weather_tools() -> Vec<Value> {
        vec![json!({"type":"function","function":{"name":"get_weather","description":"Forecast and current weather for a city"}})]
    }

    #[test]
    fn chat_stays_chat_and_tasks_are_detected() {
        let tools = demo_tools();
        assert_eq!(plan_request("hi", &tools).kind, RequestKind::Chat);
        assert_eq!(plan_request("What is 2+2?", &tools).kind, RequestKind::Chat);
        assert_eq!(plan_request("What MCP tools do you have?", &tools).kind, RequestKind::Chat);
        assert_eq!(plan_request("Write a haiku about autumn.", &tools).kind, RequestKind::Chat);
        assert_eq!(plan_request("Mi a nevem? Ne hívj eszközt.", &tools).kind, RequestKind::Chat);
        assert_eq!(plan_request("What time is it?", &tools).kind, RequestKind::Task);
        assert_eq!(plan_request("Hány óra van most?", &tools).kind, RequestKind::Chat);
        assert_eq!(plan_request("What is 1234 * 56?", &tools).kind, RequestKind::Task);
        assert_eq!(plan_request("Mennyi 37 * 89?", &tools).kind, RequestKind::Task);
        assert_eq!(plan_request("Számold ki a calculate eszközzel az 1234*56 értékét.", &tools).kind, RequestKind::Task);
        assert_eq!(plan_request("Read notes.txt then summarize it", &tools).kind, RequestKind::Task);
        assert_eq!(plan_request("1. open the file\n2. fix the bug", &tools).kind, RequestKind::Task);
        assert!(requires_tool("What time is it?", &tools));
        assert!(requires_tool("What is 1234 * 56?", &tools));
        assert!(!requires_tool("Mi a nevem? Ne hívj eszközt.", &tools));
    }

    #[test]
    fn classification_follows_the_connected_catalog() {
        assert_eq!(plan_request("What time is it?", &[]).kind, RequestKind::Chat);
        assert_eq!(plan_request("What is 1234 * 56?", &[]).kind, RequestKind::Chat);
        assert_eq!(plan_request("What's the weather in Budapest?", &demo_tools()).kind, RequestKind::Chat);
        assert_eq!(plan_request("What's the weather in Budapest?", &weather_tools()).kind, RequestKind::Task);
        let wiki = vec![json!({"type":"function","function":{"name":"wikipedia","description":"Look a topic up on Wikipedia"}})];
        assert_eq!(plan_request("whats the weather in New York?", &wiki).kind, RequestKind::Chat);
        assert_eq!(plan_request("Milyen az idő dunaújvárosba?", &wiki).kind, RequestKind::Chat);
        assert!(requires_tool("Use get_weather for Budapest", &weather_tools()));
        assert!(!requires_tool("What's the weather in Budapest?", &demo_tools()));
    }

    #[test]
    fn think_suffix_overrides_and_is_stripped() {
        let p = plan_request("hello /think", &[]);
        assert_eq!(p.text, "hello");
        assert_eq!(p.thinking_override, Some(true));
        assert_eq!(p.kind, RequestKind::Chat);
        let p = plan_request("Read a.txt /no_think", &demo_tools());
        assert_eq!(p.text, "Read a.txt");
        assert_eq!(p.thinking_override, Some(false));
        assert_eq!(p.kind, RequestKind::Task);
    }

    #[test]
    fn auto_thinks_only_on_first_task_hop() {
        let task = plan_request("read the file and list tools", &demo_tools());
        assert!(thinking_for_hop(ThinkingMode::Auto, &task, 0, true));
        assert!(!thinking_for_hop(ThinkingMode::Auto, &task, 1, true));
        let chat = plan_request("hi", &[]);
        assert!(!thinking_for_hop(ThinkingMode::Auto, &chat, 0, false));
        assert!(thinking_for_hop(ThinkingMode::On, &chat, 3, false));
        assert!(!thinking_for_hop(ThinkingMode::Off, &task, 0, true));
    }

    #[test]
    fn old_bool_settings_map_to_modes() {
        assert_eq!(serde_json::from_value::<ThinkingMode>(serde_json::json!(true)).unwrap(), ThinkingMode::On);
        assert_eq!(serde_json::from_value::<ThinkingMode>(serde_json::json!(false)).unwrap(), ThinkingMode::Auto);
        assert_eq!(serde_json::from_value::<ThinkingMode>(serde_json::json!("off")).unwrap(), ThinkingMode::Off);
    }
}

//! Constrained decoding of a Qwen3.5 tool call, using llguidance.
//!
//! Decoding is free until the model emits the `<tool_call>` token, and only then is this grammar
//! applied. Plain-text answers therefore pay nothing, and any started call names a declared tool,
//! has its required parameters, and carries values of the declared types.

use anyhow::{Result, anyhow};
use llguidance::{Constraint, ParserFactory, api::TopLevelGrammar};
use serde_json::Value;
use tokenizers::Tokenizer;
use toktrie_hf_tokenizers::ByteTokenizer;

use crate::toolcall::{tool_name, tool_parameters};

const END_LITERAL: &str = "\\n</parameter>\\n";
const END_REGEX: &str = "\\n<\\/parameter>\\n";

/// A value and the parameter's closing tag form one terminal. As separate terminals the lexer
/// would let a string's greedy match swallow the newline that belongs to the closing tag.
fn value_rule(schema: &Value) -> String {
    if let Some(options) = schema.get("enum").and_then(Value::as_array) {
        let literals: Vec<String> = options
            .iter()
            .map(|o| {
                let text = o.as_str().map_or_else(|| o.to_string(), str::to_string);
                let quoted = serde_json::to_string(&text).expect("strings serialise");
                format!("{}{END_LITERAL}\"", &quoted[..quoted.len() - 1])
            })
            .collect();
        return format!("({})", literals.join(" | "));
    }
    match schema.get("type").and_then(Value::as_str) {
        // JSON number syntax (no leading zeros), so that every value the grammar allows parses.
        Some("number") => format!("/-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?{END_REGEX}/"),
        Some("integer") => format!("/-?(0|[1-9][0-9]*){END_REGEX}/"),
        Some("boolean") => format!("/(true|false){END_REGEX}/"),
        Some("array" | "object") => format!("%json {schema} \"{END_LITERAL}\""),
        _ => format!("/[^<\\n][^<]*{END_REGEX}/"),
    }
}

/// Grammar for what follows `<tool_call>`. Parameters come in schema order, in the layout the
/// chat template renders. llguidance treats `</tool_call>` as a special token, so it is matched
/// by id, not as text.
pub fn tool_call_lark(tools: &[Value], close_token: u32) -> String {
    let mut rules = vec![
        format!("start: \"\\n\" call \"\\n\" <[{close_token}]>"),
        format!("call: {}", (0..tools.len()).map(|i| format!("tool_{i}")).collect::<Vec<_>>().join(" | ")),
    ];
    for (i, tool) in tools.iter().enumerate() {
        let params = tool_parameters(tool);
        let required: Vec<&str> = params
            .and_then(|p| p.get("required"))
            .and_then(Value::as_array)
            .map(|r| r.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();

        let mut body = String::new();
        if let Some(props) = params.and_then(|p| p.get("properties")).and_then(Value::as_object) {
            for (j, (key, schema)) in props.iter().enumerate() {
                let open = serde_json::to_string(&format!("<parameter={key}>\n")).expect("strings serialise");
                rules.push(format!("tool_{i}_p{j}: {open} {}", value_rule(schema)));
                body.push_str(&format!(" tool_{i}_p{j}{}", if required.contains(&key.as_str()) { "" } else { "?" }));
            }
        }
        let open = serde_json::to_string(&format!("<function={}>\n", tool_name(tool))).expect("strings serialise");
        rules.push(format!("tool_{i}: {open}{body} \"</function>\""));
    }
    rules.join("\n") + "\n"
}

pub struct ToolGrammar {
    factory: ParserFactory,
    close_token: u32,
}

impl ToolGrammar {
    /// Indexes the whole vocabulary, so it is built once and reused.
    pub fn new(tokenizer: &Tokenizer) -> Result<Self> {
        let close_token = tokenizer.token_to_id("</tool_call>").ok_or_else(|| anyhow!("tokenizer has no </tool_call> token"))?;
        let byte_tokenizer = ByteTokenizer::from_tokenizer(tokenizer.clone())?;
        let tok_env = byte_tokenizer.into_tok_env(None)?;
        let mut factory = ParserFactory::new_simple(&tok_env)?;
        factory.quiet();
        Ok(Self { factory, close_token })
    }

    pub fn constraint(&self, tools: &[Value]) -> Result<Constraint> {
        if tools.is_empty() {
            return Err(anyhow!("a tool-call grammar needs at least one tool"));
        }
        let grammar = TopLevelGrammar::from_lark(tool_call_lark(tools, self.close_token));
        Ok(Constraint::new(self.factory.create_parser(grammar)?))
    }
}

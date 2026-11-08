//! The TUI's look: colours, icons with plain-text fallbacks, and a small markdown renderer for
//! the model's answers. Only emoji that terminals draw two cells wide by default are used (none
//! that need a variation selector), so borders stay aligned.

use ratatui::{
    style::{Color, Modifier, Style},
    text::{Line, Span},
};
use serde_json::Value;

pub const AGENT: Color = Color::LightMagenta;
pub const USER: Color = Color::LightCyan;
pub const TOOL: Color = Color::Yellow;
pub const OK: Color = Color::LightGreen;
pub const ERR: Color = Color::LightRed;
pub const NOTE: Color = Color::LightBlue;
pub const CODE: Color = Color::Green;
pub const HEAD: Color = Color::LightYellow;
pub const VALUE: Color = Color::Cyan;
pub const DIM: Color = Color::DarkGray;

pub fn dim() -> Style {
    Style::new().fg(DIM)
}

pub fn bold(color: Color) -> Style {
    Style::new().fg(color).add_modifier(Modifier::BOLD)
}

/// Emoji when on, plain characters when off (for terminals without emoji fonts).
#[derive(Clone, Copy)]
pub struct Icons(pub bool);

impl Icons {
    pub fn pick(self, emoji: &'static str, plain: &'static str) -> &'static str {
        if self.0 { emoji } else { plain }
    }

    /// "💬 text", or just "text" without icons.
    pub fn tagged(self, emoji: &str, text: &str) -> String {
        if self.0 { format!("{emoji} {text}") } else { text.to_string() }
    }

    pub fn user(self) -> &'static str {
        self.pick("🧑", ">")
    }
    pub fn thinking(self) -> &'static str {
        self.pick("💭", "~")
    }
    pub fn ok(self) -> &'static str {
        self.pick("✅", "+")
    }
    pub fn fail(self) -> &'static str {
        self.pick("❌", "x")
    }
    pub fn note(self) -> &'static str {
        self.pick("💡", "i")
    }
    pub fn error(self) -> &'static str {
        self.pick("❗", "!")
    }
    pub fn speed(self) -> &'static str {
        self.pick("⚡", "~")
    }

    /// An icon for a tool, guessed from its name; display only.
    pub fn tool(self, name: &str) -> &'static str {
        if !self.0 {
            return ">";
        }
        let name = name.to_lowercase();
        let has = |words: &[&str]| words.iter().any(|w| name.contains(w));
        if has(&["wiki"]) {
            "📖"
        } else if has(&["news"]) {
            "📰"
        } else if has(&["page", "fetch", "url", "web", "http"]) {
            "🌐"
        } else if has(&["note", "search", "find"]) {
            "🔍"
        } else if has(&["time", "date", "clock"]) {
            "🕐"
        } else if has(&["calc", "math"]) {
            "🧮"
        } else if has(&["file", "read", "list", "dir"]) {
            "📄"
        } else {
            "🔧"
        }
    }

    pub fn setting(self, key: &str) -> &'static str {
        if !self.0 {
            return "-";
        }
        match key {
            "mcp" => "🔌",
            "model_dir" => "🧠",
            "dxc" => "🚀",
            "system" => "📜",
            "mcp_guide" => "🧭",
            "port" => "🔢",
            "auto_approve" => "🔓",
            "thinking" | "thinking_mode" => "💭",
            "temperature" => "🎲",
            "presence_penalty" => "🔁",
            "max_steps" => "👣",
            "max_tokens" => "📏",
            "icons" => "🎨",
            _ => "🔹",
        }
    }
}

/// Tool arguments as (key, value) pairs, strings unquoted.
pub fn tool_arg_pairs(arguments: &str) -> Vec<(String, String)> {
    match serde_json::from_str::<Value>(arguments) {
        Ok(Value::Object(map)) => map
            .into_iter()
            .map(|(k, v)| {
                let v = match v {
                    Value::String(s) => s,
                    other => other.to_string(),
                };
                (k, v)
            })
            .collect(),
        _ => vec![(String::new(), arguments.to_string())],
    }
}

/// `query: Petőfi · lang: hu`, each value cut to keep it on one line.
pub fn tool_args(arguments: &str) -> String {
    tool_arg_pairs(arguments)
        .into_iter()
        .map(|(k, v)| {
            let v: String = if v.chars().count() > 60 { v.chars().take(60).collect::<String>() + "…" } else { v };
            if k.is_empty() { v } else { format!("{k}: {v}") }
        })
        .collect::<Vec<_>>()
        .join(" · ")
}

/// Markdown as the model writes it: headings, bullet and numbered lists, quotes, rules, code
/// blocks, **bold** and `code`. Each line starts with `indent`.
pub fn markdown(text: &str, indent: &str) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    let mut in_code = false;
    for raw in text.trim().lines() {
        let line = raw.trim_end();
        let t = line.trim_start();
        if t.starts_with("```") {
            in_code = !in_code;
            continue;
        }
        if in_code {
            out.push(Line::from(vec![Span::raw(indent.to_string()), Span::styled(format!("│ {line}"), Style::new().fg(CODE))]));
            continue;
        }
        let pad = " ".repeat(line.len() - t.len());
        let mut spans = vec![Span::raw(format!("{indent}{pad}"))];
        if let Some(h) = t.strip_prefix("### ").or_else(|| t.strip_prefix("## ")).or_else(|| t.strip_prefix("# ")) {
            spans.extend(inline(h, bold(HEAD)));
        } else if let Some(item) = t.strip_prefix("- ").or_else(|| t.strip_prefix("* ")).or_else(|| t.strip_prefix("• ")) {
            spans.push(Span::styled("• ", Style::new().fg(AGENT)));
            spans.extend(inline(item, Style::new()));
        } else if let Some((num, rest)) = numbered(t) {
            spans.push(Span::styled(format!("{num}. "), Style::new().fg(AGENT)));
            spans.extend(inline(rest, Style::new()));
        } else if let Some(quote) = t.strip_prefix("> ") {
            spans.push(Span::styled("▎ ", dim()));
            spans.extend(inline(quote, Style::new().add_modifier(Modifier::ITALIC)));
        } else if t.len() >= 3 && t.chars().all(|c| matches!(c, '-' | '*' | '_')) {
            spans.push(Span::styled("─".repeat(24), dim()));
        } else {
            spans.extend(inline(t, Style::new()));
        }
        out.push(Line::from(spans));
    }
    out
}

fn numbered(t: &str) -> Option<(&str, &str)> {
    let (num, rest) = t.split_once(". ")?;
    (!num.is_empty() && num.len() <= 3 && num.chars().all(|c| c.is_ascii_digit())).then_some((num, rest))
}

/// `**bold**` and `` `code` `` inside a line; an unclosed marker runs to the end (while streaming).
fn inline(s: &str, base: Style) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let (mut strong, mut code) = (false, false);
    let mut buf = String::new();
    let style = |strong: bool, code: bool| {
        if code {
            Style::new().fg(CODE)
        } else if strong {
            base.add_modifier(Modifier::BOLD)
        } else {
            base
        }
    };
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        let toggle_code = c == '`';
        let toggle_strong = !code && c == '*' && chars.peek() == Some(&'*');
        if toggle_code || toggle_strong {
            if !buf.is_empty() {
                spans.push(Span::styled(std::mem::take(&mut buf), style(strong, code)));
            }
            if toggle_code {
                code = !code;
            } else {
                chars.next();
                strong = !strong;
            }
            continue;
        }
        buf.push(c);
    }
    if !buf.is_empty() {
        spans.push(Span::styled(buf, style(strong, code)));
    }
    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(line: &Line) -> String {
        line.spans.iter().map(|s| s.content.as_ref()).collect()
    }

    #[test]
    fn markdown_strips_markers_and_keeps_text() {
        let lines = markdown("# Title\n- **a** b\n2. `x` y\n```\ncode\n```", "");
        let texts: Vec<String> = lines.iter().map(text).collect();
        assert_eq!(texts, vec!["Title", "• a b", "2. x y", "│ code"]);
        assert!(lines[1].spans.iter().any(|s| s.content == "a" && s.style.add_modifier.contains(Modifier::BOLD)));
    }

    #[test]
    fn tool_args_are_compact() {
        assert_eq!(tool_args(r#"{"query":"Petőfi","lang":"hu"}"#), "query: Petőfi · lang: hu");
    }
}

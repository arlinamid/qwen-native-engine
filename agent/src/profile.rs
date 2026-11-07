//! The agent's profile, in the OpenClaw workspace format: `IDENTITY.md` (who the agent is),
//! `SOUL.md` (how it behaves) and `USER.md` (who the user is), under `.qwen-agent/profile/`.
//! Fields are `- **Label:** value` lines; everything else in a file is free text. The filled-in
//! parts go into the system prompt, so they are kept short.

use std::path::PathBuf;

use anyhow::{Context, Result};

pub const PROFILE_DIR: &str = ".qwen-agent/profile";
/// Longest profile text put into the prompt; the context is 4096 tokens.
const MAX_PROMPT_CHARS: usize = 1500;

pub struct Field {
    pub file: &'static str,
    pub label: &'static str,
    pub hint: &'static str,
}

pub const FIELDS: &[Field] = &[
    Field { file: "IDENTITY.md", label: "Name", hint: "the agent's name" },
    Field { file: "IDENTITY.md", label: "Creature", hint: "what it is: assistant, robot, familiar…" },
    Field { file: "IDENTITY.md", label: "Vibe", hint: "how it comes across: warm, sharp, calm…" },
    Field { file: "IDENTITY.md", label: "Emoji", hint: "its signature emoji (optional)" },
    Field { file: "USER.md", label: "Name", hint: "your name" },
    Field { file: "USER.md", label: "What to call them", hint: "how the agent should address you" },
    Field { file: "USER.md", label: "Pronouns", hint: "optional" },
    Field { file: "USER.md", label: "Timezone", hint: "e.g. Europe/Budapest" },
    Field { file: "USER.md", label: "Language", hint: "the language you usually write in" },
    Field { file: "USER.md", label: "Notes", hint: "work, interests, what matters to you" },
];

const IDENTITY_TEMPLATE: &str = "# IDENTITY.md - Who the agent is\n\n- **Name:**\n- **Creature:**\n- **Vibe:**\n- **Emoji:**\n";
const USER_TEMPLATE: &str = "# USER.md - About the user\n\n- **Name:**\n- **What to call them:**\n- **Pronouns:**\n- **Timezone:**\n- **Language:**\n- **Notes:**\n\n## Context\n\n<!-- Free text: projects, interests, preferences. It all goes into the prompt, so keep it short. -->\n";
const SOUL_TEMPLATE: &str = "# SOUL.md - How the agent behaves\n\n<!-- Tone, brevity, opinions, boundaries. Short beats long: every line goes into the prompt. -->\n\n- Answer directly; skip filler such as \"Great question!\".\n- Keep answers short unless the user asks for detail.\n- When you are not sure, say so, and look facts up with your tools instead of guessing.\n";

pub fn path(file: &str) -> PathBuf {
    // Forward slashes everywhere, so messages show one consistent path.
    PathBuf::from(format!("{PROFILE_DIR}/{file}"))
}

/// Writes the template of every missing profile file; returns whether it wrote any.
pub fn ensure_templates() -> Result<bool> {
    std::fs::create_dir_all(PROFILE_DIR).with_context(|| format!("creating {PROFILE_DIR}"))?;
    let mut created = false;
    for (file, template) in [("IDENTITY.md", IDENTITY_TEMPLATE), ("SOUL.md", SOUL_TEMPLATE), ("USER.md", USER_TEMPLATE)] {
        let p = path(file);
        if !p.exists() {
            std::fs::write(&p, template).with_context(|| format!("writing {}", p.display()))?;
            created = true;
        }
    }
    Ok(created)
}

fn read(file: &str) -> String {
    std::fs::read_to_string(path(file)).unwrap_or_default()
}

/// `- **Label:** value` (or `- Label: value`) -> (label, value), with template placeholders such
/// as `_(pick something)_` treated as empty.
fn parse_field(line: &str) -> Option<(String, String)> {
    let rest = line.trim_start().strip_prefix("- ")?;
    let rest = rest.trim_start_matches('*');
    let (label, value) = rest.split_once(':')?;
    if label.is_empty() || label.len() > 40 {
        return None;
    }
    let value = value.trim_start_matches('*').trim();
    let value = if value.starts_with("_(") && value.ends_with(")_") { "" } else { value };
    Some((label.trim_end_matches('*').trim().to_string(), value.to_string()))
}

pub fn get(file: &str, label: &str) -> String {
    read(file).lines().filter_map(parse_field).find(|(l, _)| l.eq_ignore_ascii_case(label)).map(|(_, v)| v).unwrap_or_default()
}

/// Sets one field, keeping the rest of the file; a missing field is added after the others.
pub fn set(file: &str, label: &str, value: &str) -> Result<()> {
    ensure_templates()?;
    let value = value.replace('\n', " ");
    let new_line = format!("- **{label}:** {}", value.trim()).trim_end().to_string();
    let text = read(file);
    let mut lines: Vec<String> = text.lines().map(String::from).collect();
    match lines.iter().position(|l| parse_field(l).is_some_and(|(l, _)| l.eq_ignore_ascii_case(label))) {
        Some(i) => lines[i] = new_line,
        None => {
            let after = lines.iter().rposition(|l| parse_field(l).is_some()).map_or(lines.len(), |i| i + 1);
            lines.insert(after, new_line);
        }
    }
    let p = path(file);
    std::fs::write(&p, lines.join("\n") + "\n").with_context(|| format!("writing {}", p.display()))
}

/// The prompt-worthy lines of a file: filled fields and free text, without headings, comments,
/// placeholders and empty fields.
fn content_lines(file: &str) -> Vec<String> {
    let text = read(file);
    let mut out = Vec::new();
    let mut in_comment = false;
    for line in text.lines() {
        let mut line = line.to_string();
        // Drop <!-- ... --> comments, also across lines.
        loop {
            if in_comment {
                match line.find("-->") {
                    Some(end) => {
                        line = line[end + 3..].to_string();
                        in_comment = false;
                    }
                    None => {
                        line.clear();
                        break;
                    }
                }
            }
            match line.find("<!--") {
                Some(start) => {
                    let rest = line[start..].to_string();
                    line.truncate(start);
                    match rest.find("-->") {
                        Some(end) => line.push_str(&rest[end + 3..]),
                        None => in_comment = true,
                    }
                    if in_comment {
                        break;
                    }
                }
                None => break,
            }
        }
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || (trimmed.starts_with("_(") && trimmed.ends_with(")_")) {
            continue;
        }
        match parse_field(trimmed) {
            Some((_, value)) if value.is_empty() => {}
            Some((label, value)) => out.push(format!("- {label}: {value}")),
            None => out.push(trimmed.to_string()),
        }
    }
    out
}

/// The profile as a system-prompt section; empty when nothing is filled in.
pub fn render() -> String {
    let mut text = String::new();
    let agent: Vec<String> = content_lines("IDENTITY.md").into_iter().chain(content_lines("SOUL.md")).collect();
    if !agent.is_empty() {
        text.push_str("## Who you are\n");
        text.push_str(&agent.join("\n"));
    }
    let user = content_lines("USER.md");
    if !user.is_empty() {
        if !text.is_empty() {
            text.push_str("\n\n");
        }
        text.push_str("## Who the user is\n");
        text.push_str(&user.join("\n"));
    }
    if text.chars().count() > MAX_PROMPT_CHARS {
        let cut: String = text.chars().take(MAX_PROMPT_CHARS).collect();
        text = cut + "\n[profile shortened: keep the profile files short]";
    }
    text
}

/// Whether neither the agent's nor the user's name has been set yet.
pub fn is_blank() -> bool {
    get("IDENTITY.md", "Name").is_empty() && get("USER.md", "Name").is_empty()
}

/// Line-by-line setup on the terminal (`qwen-agent setup`); Enter keeps the current value.
pub fn wizard() -> Result<()> {
    ensure_templates()?;
    eprintln!("Set up the agent's profile (Enter keeps the current value, \"-\" clears it).\n");
    let mut last_file = "";
    for field in FIELDS {
        if field.file != last_file {
            eprintln!("{}", if field.file == "IDENTITY.md" { "About the agent" } else { "\nAbout you" });
            last_file = field.file;
        }
        let current = get(field.file, field.label);
        let shown = if current.is_empty() { String::new() } else { format!(" [{current}]") };
        let answer = crate::ask_raw(&format!("  {} ({}){shown}: ", field.label, field.hint));
        match answer.trim() {
            "" => {}
            "-" => set(field.file, field.label, "")?,
            value => set(field.file, field.label, value)?,
        }
    }
    eprintln!("\nSaved in {PROFILE_DIR}/ (IDENTITY.md, USER.md). Behaviour rules are in SOUL.md; edit it freely.");
    eprintln!("A running server keeps the old profile until it restarts (`qwen-agent stop`, or /restart in the TUI).");
    let prompt = render();
    if !prompt.is_empty() {
        eprintln!("\nWhat the model will see:\n{prompt}");
    }
    Ok(())
}

//! On-demand agent skills: catalog in the system prompt, body injected by the harness on tasks.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

pub const AGENTS_FILE: &str = ".qwen-agent/AGENTS.md";
pub const SKILLS_DIR: &str = ".qwen-agent/skills";
pub const MAX_CATALOG: usize = 8;
pub const MAX_DESCRIPTION_CHARS: usize = 80;
pub const MAX_BODY_CHARS: usize = 600;
pub const MAX_AGENTS_CHARS: usize = 600;

const AGENTS_TEMPLATE: &str = "\
# AGENTS.md\n\n\
- Prefer tools over guessing when MCP can answer.\n\
- Never claim a tool ran without a result.\n\
- Keep answers short; outcome first; do not narrate every step.\n\
- Follow the user's language.\n\
";

/// Writes default AGENTS.md if missing. Returns whether it was created.
pub fn ensure_agents_md() -> Result<bool> {
    let path = Path::new(AGENTS_FILE);
    if path.exists() {
        return Ok(false);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, AGENTS_TEMPLATE).with_context(|| format!("writing {AGENTS_FILE}"))?;
    Ok(true)
}

/// Always-on ops rules for the system prompt (capped).
pub fn render_agents_md() -> String {
    let text = std::fs::read_to_string(AGENTS_FILE).unwrap_or_default();
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let body = truncate_chars(trimmed, MAX_AGENTS_CHARS);
    format!("## Agents\n{body}")
}

pub fn done_when_recovery(missing: &str) -> String {
    format!("Still missing: {missing}. Finish that before you stop.")
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkillMeta {
    pub name: String,
    pub description: String,
    /// Extra match keywords (HU+EN). Not shown in the system catalog.
    pub triggers: String,
    pub path: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActiveSkill {
    pub name: String,
    pub body: String,
    pub done_when: Vec<String>,
}

/// Ensures the skills directory and two starter skills exist. Returns whether anything was created.
pub fn ensure_starters() -> Result<bool> {
    std::fs::create_dir_all(SKILLS_DIR).with_context(|| format!("creating {SKILLS_DIR}"))?;
    let mut created = false;
    for (name, body) in [
        ("mcp-tools", MCP_TOOLS_SKILL),
        ("notes-files", NOTES_FILES_SKILL),
    ] {
        let dir = Path::new(SKILLS_DIR).join(name);
        let path = dir.join("SKILL.md");
        if !path.exists() {
            std::fs::create_dir_all(&dir)?;
            std::fs::write(&path, body).with_context(|| format!("writing {}", path.display()))?;
            created = true;
        }
    }
    Ok(created)
}

/// Scans `.qwen-agent/skills/*/SKILL.md`, at most [`MAX_CATALOG`] entries, sorted by name.
pub fn load_catalog(root: &Path) -> Result<Vec<SkillMeta>> {
    let dir = root.join(SKILLS_DIR);
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut skills = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if !is_skill_name(&name) {
            continue;
        }
        let path = entry.path().join("SKILL.md");
        if !path.is_file() {
            continue;
        }
        match parse_skill_file(&path) {
            Ok(parsed) => {
                if parsed.name != name {
                    eprintln!("warning: skill folder {name} has name {} in frontmatter; using folder name", parsed.name);
                }
                skills.push(SkillMeta {
                    name,
                    description: truncate_chars(&parsed.description, MAX_DESCRIPTION_CHARS),
                    triggers: parsed.triggers,
                    path,
                });
            }
            Err(e) => eprintln!("warning: skipping skill {}: {e:#}", path.display()),
        }
        if skills.len() >= MAX_CATALOG {
            break;
        }
    }
    Ok(skills)
}

pub fn catalog_prompt(skills: &[SkillMeta]) -> String {
    if skills.is_empty() {
        return String::new();
    }
    let mut text = String::from(
        "## Skills\nOn a matching task the harness injects the skill body; follow it.\nAvailable:",
    );
    for skill in skills {
        text.push_str(&format!("\n- {}: {}", skill.name, skill.description));
    }
    text
}

/// Strips a leading `$skill-name` and returns (remaining text, explicit name).
pub fn strip_explicit_skill(text: &str) -> (String, Option<String>) {
    let trimmed = text.trim_start();
    if let Some(rest) = trimmed.strip_prefix('$') {
        let (name, after) = rest
            .find(|c: char| c.is_whitespace())
            .map(|i| (&rest[..i], rest[i..].trim_start()))
            .unwrap_or((rest, ""));
        if is_skill_name(name) {
            return (after.to_string(), Some(name.to_ascii_lowercase()));
        }
    }
    (text.to_string(), None)
}

/// Pick at most one skill: explicit name wins, else best token overlap with description/name.
pub fn match_skill<'a>(catalog: &'a [SkillMeta], user_text: &str, forced: Option<&str>) -> Option<&'a SkillMeta> {
    if let Some(name) = forced {
        return catalog.iter().find(|s| s.name.eq_ignore_ascii_case(name));
    }
    let (text, explicit) = strip_explicit_skill(user_text);
    if let Some(name) = explicit {
        return catalog.iter().find(|s| s.name.eq_ignore_ascii_case(&name));
    }
    let query = match_tokens(&text);
    if query.is_empty() {
        return None;
    }
    let mut best: Option<(&SkillMeta, usize)> = None;
    for skill in catalog {
        let hay = match_tokens(&format!("{} {} {}", skill.name, skill.description, skill.triggers));
        let overlap = query.iter().filter(|t| hay.contains(t)).map(|t| t.len()).sum::<usize>();
        if overlap == 0 {
            continue;
        }
        match best {
            None => best = Some((skill, overlap)),
            Some((_, score)) if overlap > score => best = Some((skill, overlap)),
            Some((prev, score)) if overlap == score && skill.name < prev.name => best = Some((skill, overlap)),
            _ => {}
        }
    }
    best.map(|(s, _)| s)
}

pub fn load_active(meta: &SkillMeta) -> Result<ActiveSkill> {
    let parsed = parse_skill_file(&meta.path)?;
    let done_when = parse_done_when(&parsed.body);
    let body = truncate_chars(parsed.body.trim(), MAX_BODY_CHARS);
    Ok(ActiveSkill { name: meta.name.clone(), body, done_when })
}

pub fn inject_block(name: &str, body: &str) -> String {
    format!("## Skill: {name}\n{body}")
}

struct ParsedSkill {
    name: String,
    description: String,
    triggers: String,
    body: String,
}

fn parse_skill_file(path: &Path) -> Result<ParsedSkill> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let (front, body) = split_frontmatter(&text).ok_or_else(|| anyhow::anyhow!("missing YAML frontmatter"))?;
    let mut name = String::new();
    let mut description = String::new();
    let mut triggers = String::new();
    for line in front.lines() {
        let line = line.trim();
        if let Some(v) = line.strip_prefix("name:") {
            name = unquote(v.trim());
        } else if let Some(v) = line.strip_prefix("description:") {
            description = unquote(v.trim());
        } else if let Some(v) = line.strip_prefix("triggers:") {
            triggers = unquote(v.trim());
        }
    }
    if name.is_empty() || description.is_empty() {
        bail!("SKILL.md needs name and description in frontmatter");
    }
    if !is_skill_name(&name) {
        bail!("invalid skill name {name}");
    }
    Ok(ParsedSkill { name, description, triggers, body: body.to_string() })
}

fn split_frontmatter(text: &str) -> Option<(&str, &str)> {
    let text = text.strip_prefix("---").or_else(|| text.strip_prefix("\u{feff}---"))?;
    let text = text.strip_prefix('\n').unwrap_or(text);
    let end = text.find("\n---")?;
    let front = &text[..end];
    let rest = text[end + 4..].trim_start_matches('\r').trim_start_matches('\n');
    Some((front, rest))
}

fn parse_done_when(body: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut in_section = false;
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.eq_ignore_ascii_case("## Done when") || trimmed.eq_ignore_ascii_case("## Done when:") {
            in_section = true;
            continue;
        }
        if in_section && trimmed.starts_with("## ") {
            break;
        }
        if !in_section {
            continue;
        }
        let item = trimmed
            .trim_start_matches('-')
            .trim_start_matches('*')
            .trim_start_matches(['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '.', ')', ' '])
            .trim();
        if !item.is_empty() {
            lines.push(item.to_string());
        }
        if lines.len() == 3 {
            break;
        }
    }
    lines
}

/// Checklist line missing from tool outputs, tool names, and answer.
/// Distinctive tokens (length ≥ 6) avoid false hits on words like "tool".
pub fn missing_done_when(done_when: &[String], evidence: &str) -> Option<String> {
    let hay = evidence.to_ascii_lowercase();
    for line in done_when {
        let needle = line.to_ascii_lowercase();
        if needle.chars().count() < 3 {
            continue;
        }
        let tokens: Vec<_> = tokenize(line).into_iter().filter(|t| t.len() >= 6).collect();
        let hit = if tokens.is_empty() {
            hay.contains(&needle)
        } else {
            tokens.iter().any(|t| hay.contains(t.as_str()))
        };
        if !hit {
            return Some(line.clone());
        }
    }
    None
}

pub fn plan_checklist_from_think(reasoning: &str) -> Vec<String> {
    reasoning
        .lines()
        .map(|l| l.trim().trim_start_matches(['-', '*', '1', '2', '3', '.', ')', ' ']).trim())
        .filter(|l| l.chars().count() >= 4)
        .take(3)
        .map(str::to_string)
        .collect()
}

pub fn is_skill_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
        && !name.starts_with('-')
        && !name.ends_with('-')
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IssueLevel {
    Error,
    Warning,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkillIssue {
    pub level: IssueLevel,
    pub message: String,
}

impl SkillIssue {
    fn err(message: impl Into<String>) -> Self {
        Self { level: IssueLevel::Error, message: message.into() }
    }
    fn warn(message: impl Into<String>) -> Self {
        Self { level: IssueLevel::Warning, message: message.into() }
    }
}

/// Validates one `SKILL.md` (or a skill folder containing it).
pub fn validate_skill_path(path: &Path) -> Vec<SkillIssue> {
    let path = if path.is_dir() { path.join("SKILL.md") } else { path.to_path_buf() };
    let mut issues = Vec::new();
    if !path.is_file() {
        issues.push(SkillIssue::err(format!("missing {}", path.display())));
        return issues;
    }
    let folder_name = path
        .parent()
        .and_then(|p| p.file_name())
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    match parse_skill_file(&path) {
        Ok(parsed) => {
            if !folder_name.is_empty() && folder_name != parsed.name {
                issues.push(SkillIssue::err(format!(
                    "folder name `{folder_name}` must match frontmatter name `{}`", parsed.name
                )));
            }
            if !is_skill_name(&parsed.name) {
                issues.push(SkillIssue::err(format!("invalid name `{}` (use a-z, 0-9, hyphens)", parsed.name)));
            }
            let desc_chars = parsed.description.chars().count();
            if parsed.description.trim().is_empty() {
                issues.push(SkillIssue::err("description is empty"));
            } else if desc_chars > MAX_DESCRIPTION_CHARS {
                issues.push(SkillIssue::warn(format!(
                    "description is {desc_chars} chars (catalog shows first {MAX_DESCRIPTION_CHARS})"
                )));
            }
            let body_trim = parsed.body.trim();
            if body_trim.is_empty() {
                issues.push(SkillIssue::err("skill body is empty"));
            } else {
                let body_chars = body_trim.chars().count();
                if body_chars > MAX_BODY_CHARS {
                    issues.push(SkillIssue::warn(format!(
                        "body is {body_chars} chars (harness injects first {MAX_BODY_CHARS})"
                    )));
                }
            }
            let done = parse_done_when(&parsed.body);
            if done.is_empty() {
                issues.push(SkillIssue::warn("no ## Done when checklist (finish gate will use the plan instead)"));
            } else if done.len() > 3 {
                issues.push(SkillIssue::warn("## Done when has more than 3 items; only the first three are used"));
            }
        }
        Err(e) => issues.push(SkillIssue::err(format!("{e:#}"))),
    }
    issues
}

/// Validates every skill folder under `.qwen-agent/skills/` (including ones past the catalog cap).
pub fn validate_all(root: &Path) -> Result<Vec<(String, PathBuf, Vec<SkillIssue>)>> {
    let dir = root.join(SKILLS_DIR);
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok())
        .collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        let path = entry.path().join("SKILL.md");
        let mut issues = validate_skill_path(&path);
        if !is_skill_name(&name) {
            issues.insert(0, SkillIssue::err(format!("folder `{name}` is not a valid skill name")));
        }
        out.push((name, path, issues));
    }
    let skill_count = out.len();
    if skill_count > MAX_CATALOG {
        out.push((
            "_catalog".into(),
            dir,
            vec![SkillIssue::warn(format!(
                "{skill_count} skill folders on disk; system prompt catalogs at most {MAX_CATALOG} (sorted by name)"
            ))],
        ));
    }
    Ok(out)
}

/// Creates `.qwen-agent/skills/{name}/SKILL.md` from a template. Fails if it already exists.
pub fn create_skill(root: &Path, name: &str, description: Option<&str>) -> Result<PathBuf> {
    let name = name.trim().to_ascii_lowercase();
    if !is_skill_name(&name) {
        bail!("invalid skill name `{name}` (use lowercase letters, digits, hyphens)");
    }
    let dir = root.join(SKILLS_DIR).join(&name);
    let path = dir.join("SKILL.md");
    if path.exists() {
        bail!("skill already exists at {}", path.display());
    }
    let desc = description
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or("When to use this skill; add HU+EN trigger keywords.");
    let desc = truncate_chars(desc, MAX_DESCRIPTION_CHARS);
    let body = format!(
        "---\nname: {name}\ndescription: {desc}\n\
         triggers: keyword, másik-szó, when-to-use\n---\n\n\
         Instructions the agent should follow when this skill is active. Keep it short.\n\n\
         ## Done when\n\
         - First success check\n\
         - Second success check\n"
    );
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    std::fs::write(&path, body).with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

/// Formats a validation report for the TUI note line.
pub fn format_validation_report(results: &[(String, PathBuf, Vec<SkillIssue>)]) -> String {
    if results.is_empty() {
        return "no skills under .qwen-agent/skills/".into();
    }
    let mut lines = Vec::new();
    let mut errors = 0usize;
    let mut warnings = 0usize;
    for (name, path, issues) in results {
        if name == "_catalog" {
            for issue in issues {
                lines.push(format!("! {}", issue.message));
                warnings += 1;
            }
            continue;
        }
        if issues.is_empty() {
            lines.push(format!("ok  {name}  ({})", path.display()));
            continue;
        }
        let bad = issues.iter().any(|i| i.level == IssueLevel::Error);
        if bad {
            errors += 1;
            lines.push(format!("FAIL  {name}"));
        } else {
            warnings += 1;
            lines.push(format!("warn  {name}"));
        }
        for issue in issues {
            let mark = match issue.level {
                IssueLevel::Error => "  x",
                IssueLevel::Warning => "  ~",
            };
            lines.push(format!("{mark} {}", issue.message));
        }
    }
    lines.push(format!("{errors} error(s), {warnings} warning group(s) · /restart after fixes"));
    lines.join("\n")
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_ascii_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '-')
        .filter(|t| t.len() >= 3)
        .map(str::to_string)
        .collect()
}

/// Catalog matching ignores glue words so "what tools do you have?" does not fire mcp-tools.
const MATCH_SKIP: &[&str] = &[
    "the", "and", "for", "you", "your", "this", "that", "with", "from", "have", "has", "had",
    "what", "when", "which", "who", "how", "why", "does", "did", "are", "was", "not", "use",
    "used", "using", "call", "calls", "tool", "tools", "mcp", "please", "just", "some", "more",
    "into", "about", "then", "them", "they", "will", "can", "get", "set", "add", "new", "name",
    "type", "text", "data", "live", "connected", "available", "server", "help", "need", "want",
    "make", "give", "tell", "show", "know", "one", "two", "any", "all", "also", "only", "now",
    "here", "there", "before", "after", "answer", "result",
];

fn match_tokens(text: &str) -> Vec<String> {
    tokenize(text).into_iter().filter(|t| t.len() >= 4 && !MATCH_SKIP.contains(&t.as_str())).collect()
}

fn truncate_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

fn unquote(value: &str) -> String {
    let v = value.trim();
    if (v.starts_with('"') && v.ends_with('"')) || (v.starts_with('\'') && v.ends_with('\'')) {
        v[1..v.len() - 1].to_string()
    } else {
        v.to_string()
    }
}

const MCP_TOOLS_SKILL: &str = r#"---
name: mcp-tools
description: Call a connected MCP tool for live data or a tool result; számolás, idő, fájl.
triggers: weather, forecast, időjárás, hőmérséklet, wikipedia, news, time, óra, calculate, számolás
---

Call whichever connected tool answers the request. Wait for the tool result before the final answer. Never invent a tool result. Do not assume a fixed tool list.

## Done when
- tool-result
"#;

const NOTES_FILES_SKILL: &str = r#"---
name: notes-files
description: List or read notes and files before answering; jegyzet, fájl, folder, path.
triggers: jegyzet, fájl, notes, folder, path, sandbox
---

Before answering about files or notes, list or read with the connected file tools. Quote only what the tool returned.

## Done when
- tool-result
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_frontmatter_and_done_when() {
        let dir = std::env::temp_dir().join(crate::persistence::unique_id());
        let skill_dir = dir.join("mcp-tools");
        std::fs::create_dir_all(&skill_dir).unwrap();
        let path = skill_dir.join("SKILL.md");
        std::fs::write(&path, MCP_TOOLS_SKILL).unwrap();
        let parsed = parse_skill_file(&path).unwrap();
        assert_eq!(parsed.name, "mcp-tools");
        assert!(parsed.description.contains("MCP"));
        assert!(parsed.triggers.contains("weather"));
        let done = parse_done_when(&parsed.body);
        assert_eq!(done.len(), 1);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn explicit_dollar_and_overlap_match() {
        let catalog = vec![
            SkillMeta { name: "mcp-tools".into(), description: "calculate time tool".into(), triggers: "weather időjárás idő".into(), path: "a".into() },
            SkillMeta { name: "notes-files".into(), description: "list_files read_file jegyzet".into(), triggers: "jegyzet".into(), path: "b".into() },
        ];
        assert_eq!(match_skill(&catalog, "$mcp-tools what is 2+2", None).unwrap().name, "mcp-tools");
        assert_eq!(match_skill(&catalog, "Read the jegyzet file please", None).unwrap().name, "notes-files");
        assert_eq!(match_skill(&catalog, "whats the weather in New York?", None).unwrap().name, "mcp-tools");
        assert_eq!(match_skill(&catalog, "Milyen az idő dunaújvárosba?", None).unwrap().name, "mcp-tools");
        assert!(match_skill(&catalog, "hello there", None).is_none());
        assert!(match_skill(&catalog, "What MCP tools do you have?", None).is_none());
        assert_eq!(match_skill(&catalog, "anything", Some("notes-files")).unwrap().name, "notes-files");
    }

    #[test]
    fn missing_done_when_detects_gaps() {
        let vague = vec!["A required tool was called".into()];
        assert!(missing_done_when(&vague, "I guessed 42").is_some());
        assert!(missing_done_when(&vague, "I used a tool").is_some());
        let named = vec!["calculate, get_time, list_files".into()];
        assert!(missing_done_when(&named, "calculate returned 42").is_none());
    }

    #[test]
    fn body_and_description_are_capped() {
        let long = "x".repeat(1000);
        assert_eq!(truncate_chars(&long, MAX_BODY_CHARS).chars().count(), MAX_BODY_CHARS);
    }

    #[test]
    fn create_and_validate_skill_round_trip() {
        let root = std::env::temp_dir().join(crate::persistence::unique_id());
        std::fs::create_dir_all(root.join(SKILLS_DIR)).unwrap();
        let path = create_skill(&root, "demo-skill", Some("demo trigger words")).unwrap();
        assert!(path.is_file());
        let issues = validate_skill_path(&path);
        assert!(issues.iter().all(|i| i.level != IssueLevel::Error), "{issues:?}");
        assert!(create_skill(&root, "demo-skill", None).is_err());
        assert!(create_skill(&root, "Bad Name", None).is_err());
        std::fs::write(&path, "no frontmatter").unwrap();
        let issues = validate_skill_path(&path);
        assert!(issues.iter().any(|i| i.level == IssueLevel::Error));
        std::fs::remove_dir_all(root).unwrap();
    }
}

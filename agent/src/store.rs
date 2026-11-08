//! Saved conversations in `.qwen-agent/sessions/`: what the model saw (`messages`, sent back to
//! the server to continue) and what the TUI showed (`entries`).

use std::{
    path::PathBuf,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Result, bail};
use crate::persistence::{atomic_write, valid_id, unique_id};
use qwen_wasm::prompt::Message;
use serde::{Deserialize, Serialize};
use serde_json::Value;

const SESSIONS_DIR: &str = ".qwen-agent/sessions";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Entry {
    User { text: String },
    Assistant { text: String },
    Thinking { text: String },
    ToolCall { name: String, arguments: String },
    ToolResult { output: String, is_error: bool },
    Rejected { name: String, reason: String },
    Note { text: String },
    Error { text: String },
    Stats { text: String },
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SavedSession {
    #[serde(default)]
    pub pending_turn: bool,
    #[serde(default)]
    pub version: u32,
    pub id: String,
    pub title: String,
    pub created: u64,
    pub updated: u64,
    pub messages: Vec<Value>,
    pub entries: Vec<Entry>,
}

pub fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

impl SavedSession {
    pub fn new() -> Self {
        Self { id: unique_id(), version: 1, created: now(), updated: now(), ..Default::default() }
    }

    fn path(id: &str) -> Result<PathBuf> {
        if !valid_id(id) { bail!("invalid session ID"); }
        Ok(PathBuf::from(SESSIONS_DIR).join(format!("{id}.json")))
    }

    /// Whether there is anything worth keeping: at least one user message.
    pub fn has_content(&self) -> bool {
        self.entries.iter().any(|e| matches!(e, Entry::User { .. }))
    }

    pub fn save(&mut self) -> Result<()> {
        if self.title.is_empty() {
            if let Some(Entry::User { text }) = self.entries.iter().find(|e| matches!(e, Entry::User { .. })) {
                let line = text.lines().next().unwrap_or("");
                self.title = line.chars().take(60).collect();
            }
        }
        self.updated = now();
        std::fs::create_dir_all(SESSIONS_DIR)?;
        atomic_write(&Self::path(&self.id)?, serde_json::to_string_pretty(self)?.as_bytes())?;
        Ok(())
    }
}

/// Saved sessions, most recently used first. Unreadable files are skipped.
pub fn list() -> Vec<SavedSession> {
    let mut sessions: Vec<SavedSession> = std::fs::read_dir(SESSIONS_DIR)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .filter_map(|e| serde_json::from_str(&std::fs::read_to_string(e.path()).ok()?).ok())
        .collect();
    sessions.sort_by(|a, b| b.updated.cmp(&a.updated));
    sessions
}

pub fn delete(id: &str) -> Result<()> {
    Ok(std::fs::remove_file(SavedSession::path(id)?)?)
}

pub fn age(secs: u64) -> String {
    let d = now().saturating_sub(secs);
    match d {
        0..60 => "just now".into(),
        60..3600 => format!("{} min ago", d / 60),
        3600..86400 => format!("{} h ago", d / 3600),
        _ => format!("{} days ago", d / 86400),
    }
}

pub fn checkpoint(id: &str, turn_id: &str, status: &str, messages: &[Message]) -> Result<()> {
    if !valid_id(id) { bail!("invalid recovery ID"); }
    let path = PathBuf::from(".qwen-agent/recovery").join(format!("{id}.json"));
    atomic_write(&path, serde_json::to_vec(&serde_json::json!({"version":1,"turn_id":turn_id,"status":status,"messages":messages}))?.as_slice())
}
pub fn recover(id: &str) -> Result<Option<Vec<Message>>> {
    if !valid_id(id) { bail!("invalid recovery ID"); }
    let path = PathBuf::from(".qwen-agent/recovery").join(format!("{id}.json"));
    match std::fs::read(&path) {
        Ok(bytes) => {
            let value: Value = serde_json::from_slice(&bytes)?;
            if value["version"] != 1 { bail!("unsupported recovery version"); }
            Ok(Some(serde_json::from_value(value["messages"].clone())?))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

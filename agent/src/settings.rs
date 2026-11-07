//! Default settings, kept in `.qwen-agent/settings.json`. Command-line flags override them for
//! one run; the TUI edits and saves them.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub const SETTINGS_FILE: &str = ".qwen-agent/settings.json";
pub const DEFAULT_SYSTEM: &str = "You are a helpful assistant running locally. Use a tool when it is needed to answer; otherwise answer directly. Answer in the language of the user.";
const DEFAULT_DXC: &str = ".toolchain/ort-dll/webgpu/dxcompiler.dll";
const DEFAULT_MCP: &str = "mcp_servers/mcp.json";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// MCP configuration file (`mcpServers`); empty for no tools.
    pub mcp: String,
    pub model_dir: String,
    /// DXC compiler for the f16 kernels on D3D12; empty for the slower f32 kernels.
    pub dxc: String,
    pub system: String,
    /// Add an MCP section to the system prompt, generated from the connected servers.
    pub mcp_guide: bool,
    pub port: u16,
    /// Run tool calls without asking.
    pub auto_approve: bool,
    pub thinking: bool,
    /// 0 decodes greedily.
    pub temperature: f32,
    /// Against repetition (Qwen suggests 1.5); 0 turns it off.
    pub presence_penalty: f32,
    pub max_steps: usize,
    pub max_tokens: usize,
    /// Emoji icons in the TUI; off for terminals without an emoji font.
    pub icons: bool,
}

impl Default for Settings {
    fn default() -> Self {
        let if_exists = |p: &str| if Path::new(p).exists() { p.to_string() } else { String::new() };
        Self {
            mcp: if_exists(DEFAULT_MCP),
            model_dir: "models/qwen3.5-0.8b".into(),
            dxc: if_exists(DEFAULT_DXC),
            system: DEFAULT_SYSTEM.into(),
            mcp_guide: true,
            port: 7878,
            auto_approve: false,
            thinking: false,
            temperature: 0.6,
            presence_penalty: 1.5,
            max_steps: 6,
            max_tokens: 512,
            icons: true,
        }
    }
}

impl Settings {
    /// Reads the settings file, writing the defaults there first if it does not exist.
    pub fn load_or_create(path: &Path) -> Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str(&text).with_context(|| format!("{} is not a valid settings file", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let settings = Self::default();
                settings.save(path)?;
                Ok(settings)
            }
            Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
        }
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(self)? + "\n").with_context(|| format!("writing {}", path.display()))
    }

    pub fn check(&self) -> Result<()> {
        if !(self.temperature >= 0.0 && self.temperature <= 2.0) {
            bail!("temperature must be between 0 and 2");
        }
        if !(0.0..=2.0).contains(&self.presence_penalty) {
            bail!("presence_penalty must be between 0 and 2");
        }
        if self.max_steps == 0 || self.max_tokens == 0 {
            bail!("max_steps and max_tokens must be at least 1");
        }
        if self.port == 0 {
            bail!("port must not be 0");
        }
        Ok(())
    }

    /// The settings a server applies when it starts; changing them needs a restart.
    pub fn server_part(&self) -> Value {
        json!({ "mcp": self.mcp, "model_dir": self.model_dir, "dxc": self.dxc, "system": self.system, "mcp_guide": self.mcp_guide, "port": self.port })
    }

    /// The settings a client sends with every message.
    pub fn turn_options(&self) -> Value {
        json!({
            "auto_approve": self.auto_approve,
            "thinking": self.thinking,
            "temperature": self.temperature,
            "presence_penalty": self.presence_penalty,
            "max_steps": self.max_steps,
            "max_tokens": self.max_tokens,
        })
    }

    /// Returns a copy with one field set from text; booleans toggle and ignore the text.
    pub fn with_field(&self, key: &str, raw: &str) -> Result<Self> {
        let mut obj = serde_json::to_value(self)?;
        let new = match obj.get(key) {
            Some(Value::Bool(b)) => Value::Bool(!b),
            Some(Value::String(_)) => Value::String(raw.trim().to_string()),
            Some(Value::Number(_)) => serde_json::from_str::<Value>(raw.trim()).ok().filter(Value::is_number).context("not a number")?,
            _ => bail!("unknown setting {key}"),
        };
        obj[key] = new;
        let settings: Self = serde_json::from_value(obj).map_err(|e| anyhow::anyhow!("invalid value: {e}"))?;
        settings.check()?;
        Ok(settings)
    }
}

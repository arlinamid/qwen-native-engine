//! Default settings, kept in `.qwen-agent/settings.json`. Command-line flags override them for
//! one run; the TUI edits and saves them.

use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::agentic::ThinkingMode;

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
    /// Optional measured GEMV tile profile; empty uses the built-in baseline.
    pub kernel_tiles: String,
    /// DXC compiler for the f16 kernels on D3D12; empty for the slower f32 kernels.
    pub dxc: String,
    pub system: String,
    /// Add an MCP section to the system prompt, generated from the connected servers.
    pub mcp_guide: bool,
    pub port: u16,
    /// Run tool calls without asking.
    pub auto_approve: bool,
    /// Auto thinks on the first hop of a task; On thinks every hop; Off never thinks.
    #[serde(alias = "thinking")]
    pub thinking_mode: ThinkingMode,
    /// 0 decodes greedily.
    pub temperature: f32,
    /// Against repetition (Qwen suggests 1.5); 0 turns it off.
    pub presence_penalty: f32,
    pub turn_timeout_secs: u64,
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
            kernel_tiles: String::new(),
            dxc: if_exists(DEFAULT_DXC),
            system: DEFAULT_SYSTEM.into(),
            mcp_guide: true,
            port: 7878,
            auto_approve: false,
            thinking_mode: ThinkingMode::Auto,
            temperature: 0.6,
            presence_penalty: 1.5,
            turn_timeout_secs: 180,
            max_steps: 6,
            max_tokens: 512,
            icons: true,
        }
    }
}

impl Settings {
    pub fn model_choice(&self) -> &'static str {
        match Path::new(&self.model_dir).file_name().and_then(|s| s.to_str()) {
            Some("qwen3.5-0.8b") => "0.8b", Some("qwen3.5-2b") => "2b", _ => "custom",
        }
    }
    pub fn matches_model(&self, actual: &str) -> bool {
        let normalize = |p: &str| std::fs::canonicalize(p).unwrap_or_else(|_| std::path::PathBuf::from(p));
        normalize(&self.model_dir) == normalize(actual)
    }
    pub fn select_model(&self, choice: &str) -> Result<Self> {
        let (folder, profile) = match choice.trim().to_lowercase().as_str() {
            "0.8b" | "0.8" => ("models/qwen3.5-0.8b", "kernels/profiles/uhd630-qwen35-0.8b.json"),
            "2b" | "2" => ("models/qwen3.5-2b", "kernels/profiles/uhd630-qwen35-2b.json"),
            _ => bail!("choose 0.8b or 2b"),
        };
        let mut settings = self.clone();
        settings.model_dir = folder.into();
        settings.kernel_tiles = profile.into();
        Ok(settings)
    }
    pub fn check_model_files(&self) -> Result<()> {
        for name in ["tokenizer.json", "native/index.json", "native/weights.bin"] {
            let path = Path::new(&self.model_dir).join(name);
            if !path.is_file() { bail!("model is not installed: {}",path.display()); }
        }
        if !self.kernel_tiles.is_empty() {
            let tiles: qwen_kernels::model::ModelTiles = serde_json::from_str(
                &std::fs::read_to_string(&self.kernel_tiles).context("reading model kernel profile")?
            ).context("invalid model kernel profile")?;
            tiles.validate()?;
        }
        Ok(())
    }

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
        crate::persistence::atomic_write(path, (serde_json::to_string_pretty(self)? + "\n").as_bytes()).with_context(|| format!("writing {}", path.display()))
    }

    pub fn check(&self) -> Result<()> {
        if !(self.temperature >= 0.0 && self.temperature <= 2.0) {
            bail!("temperature must be between 0 and 2");
        }
        if !(0.0..=2.0).contains(&self.presence_penalty) {
            bail!("presence_penalty must be between 0 and 2");
        }
        if !(1..=16).contains(&self.max_steps) || !(1..=2048).contains(&self.max_tokens) || !(1..=600).contains(&self.turn_timeout_secs) {
            bail!("max_steps must be 1..16, max_tokens 1..2048 and turn_timeout_secs 1..600");
        }
        if self.port == 0 {
            bail!("port must not be 0");
        }
        Ok(())
    }

    /// The settings a server applies when it starts; changing them needs a restart.
    pub fn server_part(&self) -> Value {
        json!({ "mcp": self.mcp, "model_dir": self.model_dir, "kernel_tiles": self.kernel_tiles, "dxc": self.dxc, "system": self.system, "mcp_guide": self.mcp_guide, "port": self.port })
    }

    /// The settings a client sends with every message.
    pub fn turn_options(&self) -> Value {
        json!({
            "auto_approve": self.auto_approve,
            "thinking_mode": self.thinking_mode.label(),
            "temperature": self.temperature,
            "presence_penalty": self.presence_penalty,
            "turn_timeout_secs": self.turn_timeout_secs,
            "max_steps": self.max_steps,
            "max_tokens": self.max_tokens,
        })
    }

    /// Returns a copy with one field set from text; booleans toggle and ignore the text.
    pub fn with_field(&self, key: &str, raw: &str) -> Result<Self> {
        let key = if key == "thinking" { "thinking_mode" } else { key };
        if key == "thinking_mode" {
            let mut settings = self.clone();
            settings.thinking_mode = if raw.trim().is_empty() {
                self.thinking_mode.cycle()
            } else {
                ThinkingMode::parse(raw).ok_or_else(|| anyhow::anyhow!("thinking mode must be auto, on, or off"))?
            };
            settings.check()?;
            return Ok(settings);
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_switch_preserves_turn_and_mcp_preferences() {
        let original = Settings { max_tokens: 123, mcp: "custom-mcp.json".into(), temperature: 0.2, ..Settings::default() };
        let large = original.select_model("2B").unwrap();
        assert_eq!(large.model_choice(), "2b");
        assert!(large.kernel_tiles.ends_with("uhd630-qwen35-2b.json"));
        assert_eq!(large.max_tokens,123);
        assert_eq!(large.mcp,original.mcp);
        assert_eq!(large.temperature,0.2);
        assert_eq!(large.select_model("0.8B").unwrap().model_choice(),"0.8b");
        assert!(original.select_model("8b").is_err());
        let missing = Settings { model_dir: std::env::temp_dir().join(crate::persistence::unique_id()).to_string_lossy().into_owned(), ..large };
        assert!(missing.check_model_files().is_err());
    }
    #[test]
    fn model_and_profile_survive_atomic_settings_replacement() {
        let dir=std::env::temp_dir().join(crate::persistence::unique_id());
        let path=dir.join("settings.json");
        let first=Settings::default().select_model("0.8b").unwrap();
        first.save(&path).unwrap();
        let second=first.select_model("2b").unwrap();
        second.save(&path).unwrap();
        assert_eq!(Settings::load_or_create(&path).unwrap(),second);
        std::fs::remove_file(&path).unwrap();
        std::fs::remove_dir(&dir).unwrap();
    }
    #[test]
    fn old_thinking_bool_loads_as_mode_and_cycles() {
        let dir = std::env::temp_dir().join(crate::persistence::unique_id());
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("settings.json");
        std::fs::write(&path, r#"{"thinking":false,"port":7878}"#).unwrap();
        let loaded = Settings::load_or_create(&path).unwrap();
        assert_eq!(loaded.thinking_mode, ThinkingMode::Auto);
        std::fs::write(&path, r#"{"thinking":true,"port":7878}"#).unwrap();
        assert_eq!(Settings::load_or_create(&path).unwrap().thinking_mode, ThinkingMode::On);
        let cycled = loaded.with_field("thinking", "").unwrap();
        assert_eq!(cycled.thinking_mode, ThinkingMode::On);
        assert_eq!(cycled.with_field("thinking", "").unwrap().thinking_mode, ThinkingMode::Off);
        std::fs::remove_dir_all(dir).unwrap();
    }
}

//! Terminal UI. It connects to the background server, starting one (detached, logging to
//! `.qwen-agent/server.log`) when none runs, so the model stays loaded after the TUI exits.
//! Conversations are saved after every reply and can be reopened; settings are edited in place
//! and saved to `.qwen-agent/settings.json`.

use std::{
    net::TcpStream,
    path::Path,
    process::{Child, Command, Stdio},
    sync::mpsc::{self, Receiver, SyncSender},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Result, bail};
use ratatui::{
    DefaultTerminal, Frame,
    crossterm::event::{self, Event as TermEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    layout::{Alignment, Constraint, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Clear, List, ListItem, ListState, Paragraph, Wrap},
};
use serde_json::{Value, json};

use crate::{
    SERVER_FILE,
    client::{self, Connection, send_to},
    profile,
    settings::{SETTINGS_FILE, Settings},
    skills,
    store::{self, Entry, SavedSession},
    ui,
};

const SERVER_LOG: &str = ".qwen-agent/server.log";

/// Editable settings: key, label, and whether the server must restart to use it.
const FIELDS: &[(&str, &str, bool)] = &[
    ("mcp", "MCP config file", true),
    ("model", "Model: 0.8B / 2B", true),
    ("model_dir", "Model folder (custom)", true),
    ("kernel_tiles", "Kernel tile profile", true),
    ("dxc", "DXC compiler (f16 kernels)", true),
    ("system", "System prompt", true),
    ("mcp_guide", "Explain MCP to the model", true),
    ("port", "Server port", true),
    ("auto_approve", "Run tools without asking", false),
    ("thinking", "Thinking: auto / on / off", false),
    ("temperature", "Temperature (0 = greedy)", false),
    ("presence_penalty", "Repetition penalty (0-2)", false),
    ("icons", "Icons and emoji", false),
    ("max_steps", "Tool steps per message", false),
    ("max_tokens", "Tokens per step", false),
    ("turn_timeout_secs", "Turn timeout (seconds)", false),
];

const HELP: &str = "/model 0.8b or /model 2b switches model · /cancel cancels work (Ctrl+C / Esc) · Ctrl+O full tool results · /new new conversation · /sessions saved ones (F2) · /settings (F3) · /profile about you and the agent (F4) · /thinking cycles auto/on/off · Ctrl+T shows/hides the reasoning · /tools · /skills lists skills · /skill name forces one for the next message · /create-skill name scaffolds a skill · /validate-skill [name] checks SKILL.md · Tab completes · skills in .qwen-agent/skills/ (restart after edits) · /restart · /retry · /quit · /exit";

/// Slash commands in the picker (label is the insert text).
const COMMAND_ITEMS: &[(&str, &str)] = &[
    ("/help", "List all commands"),
    ("/cancel", "Cancel the current reply"),
    ("/new", "Start a new conversation"),
    ("/sessions", "Open saved chats"),
    ("/settings", "Open settings"),
    ("/profile", "Edit agent and user profile"),
    ("/setup", "Same as /profile"),
    ("/thinking", "Cycle thinking auto / on / off"),
    ("/tools", "List MCP tools"),
    ("/skills", "List available skills"),
    ("/skill", "Force a skill for the next message"),
    ("/create-skill", "Scaffold a new skill folder + SKILL.md"),
    ("/validate-skill", "Validate one skill or all skills"),
    ("/model", "Switch model: 0.8b or 2b"),
    ("/retry", "Reconnect to the server"),
    ("/restart", "Restart the server"),
    ("/quit", "Quit; keep the server warm"),
    ("/exit", "Quit and stop the server"),
    ("/stop", "Same as /exit"),
];
const MODEL_ITEMS: &[(&str, &str)] = &[
    ("0.8b", "Qwen3.5 0.8B — faster on this GPU"),
    ("2b", "Qwen3.5 2B — stronger answers"),
];

#[derive(Clone, Debug, PartialEq, Eq)]
struct CompleteItem {
    /// Text written into the input when accepted.
    insert: String,
    label: String,
    description: String,
}

enum Msg {
    Status(String),
    Connected(TcpStream, Value),
    Server(Value),
    Closed(String),
    Failed(String),
}

#[derive(PartialEq, Clone, Copy)]
enum Link {
    Connecting,
    Ready,
    Down,
}

#[derive(PartialEq, Clone, Copy)]
enum View {
    Chat,
    Sessions,
    Settings,
    Profile,
}

fn spawn_server(settings: &Settings) -> Result<Child> {
    std::fs::create_dir_all(".qwen-agent")?;
    let log = std::fs::File::create(SERVER_LOG)?;
    let mut command = Command::new(std::env::current_exe()?);
    command
        .arg("serve")
        .args(["--mcp", &settings.mcp, "--model-dir", &settings.model_dir, "--dxc", &settings.dxc, "--system", &settings.system])
        .args(["--kernel-tiles", &settings.kernel_tiles])
        .args(["--port", &settings.port.to_string()])
        .arg(if settings.mcp_guide { "--mcp-guide" } else { "--no-mcp-guide" })
        .stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log);
    // Its own hidden console and process group: it outlives this terminal, and the MCP servers
    // it starts do not open console windows.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        command.creation_flags(CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    Ok(command.spawn()?)
}

fn last_log_line() -> Option<String> {
    let text = std::fs::read_to_string(SERVER_LOG).ok()?;
    let line = text.lines().rev().find(|l| !l.trim().is_empty())?;
    Some(line.trim().chars().take(100).collect())
}

fn published_pid() -> Option<u32> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(SERVER_FILE).ok()?).ok()?;
    Some(v.get("pid")?.as_u64()? as u32)
}

/// Connects (starting a server if none runs) on a thread, then forwards the server's messages.
fn connect_in_background(tx: SyncSender<Msg>, settings: Settings, after_shutdown: bool) {
    thread::spawn(move || {
        let server_file = Path::new(SERVER_FILE);
        let result = (|| -> Result<(Connection, Value)> {
            if after_shutdown {
                // The file disappears only after the old process has freed the GPU and the model lock.
                let _ = tx.send(Msg::Status("waiting for the previous model to release the GPU".into()));
                let started = Instant::now();
                while server_file.exists() {
                    if started.elapsed() > Duration::from_secs(120) {
                        bail!("the previous server did not release the model in 2 minutes; see {SERVER_LOG}");
                    }
                    thread::sleep(Duration::from_millis(200));
                }
            }
            if let Some(found) = client::open(server_file)? {
                return Ok(found);
            }
            let _ = tx.send(Msg::Status("starting the server".into()));
            let started = Instant::now();
            let mut lease_waits = 0u32;
            let mut child = spawn_server(&settings)?;
            let mut last = String::new();
            loop {
                if let Some(status) = child.try_wait()? {
                    let line = last_log_line().unwrap_or_default();
                    // Lost the race with the process that is still dropping the previous model.
                    if line.contains("already owns the model") && lease_waits < 150 {
                        lease_waits += 1;
                        let _ = tx.send(Msg::Status("waiting for the previous model to release the GPU".into()));
                        thread::sleep(Duration::from_millis(200));
                        child = spawn_server(&settings)?;
                        continue;
                    }
                    bail!("the server exited ({status}): {line}");
                }
                if published_pid() == Some(child.id()) {
                    if let Some(found) = client::open(server_file)? {
                        return Ok(found);
                    }
                }
                if started.elapsed() > Duration::from_secs(300) {
                    bail!("the server did not get ready in 5 minutes; see {SERVER_LOG}");
                }
                if let Some(line) = last_log_line().filter(|l| *l != last) {
                    last = line.clone();
                    let _ = tx.send(Msg::Status(line));
                }
                thread::sleep(Duration::from_millis(250));
            }
        })();
        let (connection, ready) = match result {
            Ok(found) => found,
            Err(e) => {
                let _ = tx.send(Msg::Failed(format!("{e:#}")));
                return;
            }
        };
        let Connection { writer, mut reader } = connection;
        let _ = tx.send(Msg::Connected(writer, ready));
        loop {
            let msg = match crate::runtime::read_json(&mut reader, crate::runtime::MAX_FRAME) {
                Ok(Some(v)) => Msg::Server(v),
                Ok(None) => Msg::Closed("the server closed the connection".into()),
                Err(e) => Msg::Closed(e.to_string()),
            };
            let closed = matches!(msg, Msg::Closed(_));
            if tx.send(msg).is_err() || closed {
                return;
            }
        }
    });
}

struct App {
    /// Effective settings: the file plus this run's flags.
    settings: Settings,
    view: View,
    session: SavedSession,
    input: String,
    /// Cursor position in `input`, in characters.
    cursor: usize,
    /// Chat scroll position, in lines up from the bottom.
    scroll: usize,
    link: Link,
    link_status: String,
    link_since: Instant,
    writer: Option<TcpStream>,
    tx: SyncSender<Msg>,
    rx: Receiver<Msg>,
    busy: bool,
    busy_since: Instant,
    confirm: Option<(String, String)>,
    confirmation_id: (String, String),
    turn_id: String,
    phase: String,
    context_status: String,
    confirm_scroll: u16,
    show_tools: bool,
    dirty: bool,
    server_settings: Value,
    restart_after_close: bool,
    quit_after_close: bool,
    sessions: Vec<SavedSession>,
    sessions_state: ListState,
    delete_armed: bool,
    settings_state: ListState,
    editing: Option<String>,
    settings_note: String,
    /// Whether thinking blocks are expanded.
    show_thinking: bool,
    profile_state: ListState,
    profile_note: String,
    /// The profile the server built its system prompt from (from its ready message).
    server_profile: Option<String>,
    agent_name: String,
    agent_emoji: String,
    /// How the chat labels the user: what to call them, their name, or "You".
    user_name: String,
    tool_count: u64,
    /// "16.6 tok/s" of the last generation, for the hint line.
    last_speed: String,
    /// A file to open in the external editor; the event loop hands it the terminal.
    pending_editor: Option<std::path::PathBuf>,
    /// Skill forced by `/skill name` for the next send only.
    forced_skill: Option<String>,
    /// Skill catalog from the server ready message (fallback: local disk).
    server_skills: Vec<(String, String)>,
    /// Slash / skill / model picker above the chat input (Cursor-style).
    palette: Vec<CompleteItem>,
    palette_index: usize,
    quit: bool,
}

impl App {
    fn new(settings: Settings, resume: bool) -> Self {
        let (tx, rx) = mpsc::sync_channel(256);
        let session = if resume { store::list().into_iter().next().unwrap_or_else(SavedSession::new) } else { SavedSession::new() };
        Self {
            settings,
            view: View::Chat,
            session,
            input: String::new(),
            cursor: 0,
            scroll: 0,
            link: Link::Connecting,
            link_status: "connecting".into(),
            link_since: Instant::now(),
            writer: None,
            tx,
            rx,
            busy: false,
            busy_since: Instant::now(),
            confirm: None,
            confirmation_id: (String::new(), String::new()),
            turn_id: String::new(),
            phase: String::new(),
            context_status: String::new(),
            confirm_scroll: 0,
            show_tools: false,
            dirty: true,
            server_settings: Value::Null,
            restart_after_close: false,
            quit_after_close: false,
            sessions: Vec::new(),
            sessions_state: ListState::default(),
            delete_armed: false,
            settings_state: ListState::default().with_selected(Some(0)),
            editing: None,
            settings_note: String::new(),
            show_thinking: false,
            profile_state: ListState::default().with_selected(Some(0)),
            profile_note: String::new(),
            server_profile: None,
            agent_name: String::new(),
            agent_emoji: String::new(),
            user_name: "You".into(),
            tool_count: 0,
            last_speed: String::new(),
            pending_editor: None,
            forced_skill: None,
            server_skills: Vec::new(),
            palette: Vec::new(),
            palette_index: 0,
            quit: false,
        }
    }

    fn reconnect(&mut self, after_shutdown: bool) {
        self.link = Link::Connecting;
        self.link_status = "connecting".into();
        self.link_since = Instant::now();
        connect_in_background(self.tx.clone(), self.settings.clone(), after_shutdown);
    }

    fn push(&mut self, entry: Entry) {
        self.dirty = true;
        self.session.entries.push(entry);
    }

    fn note(&mut self, text: impl Into<String>) {
        self.push(Entry::Note { text: text.into() });
    }

    fn error(&mut self, text: impl Into<String>) {
        self.push(Entry::Error { text: text.into() });
    }

    fn send(&mut self, message: Value) -> bool {
        let Some(writer) = self.writer.as_mut() else {
            self.error("not connected to the server (/retry reconnects)");
            return false;
        };
        if let Err(e) = send_to(writer, &message) {
            self.error(format!("sending to the server failed: {e:#}"));
            return false;
        }
        true
    }

    fn save_session(&mut self) {
        if !self.session.has_content() {
            return;
        }
        // Notes describe this run (connection, help), not the conversation.
        let mut saved = self.session.clone();
        saved.entries.retain(|e| !matches!(e, Entry::Note { .. }));
        match saved.save() {
            Ok(()) => (self.session.title, self.session.updated) = (saved.title, saved.updated),
            Err(e) => self.error(format!("saving the session failed: {e:#}")),
        }
    }

    // ----- server messages -----

    fn on_msg(&mut self, msg: Msg) {
        self.dirty = true;
        match msg {
            Msg::Status(status) => self.link_status = status,
            Msg::Connected(writer, ready) => {
                self.writer = Some(writer);
                self.link = Link::Ready;
                self.server_settings = ready.get("settings").cloned().unwrap_or(Value::Null);
                self.tool_count = ready.get("tools").and_then(Value::as_u64).unwrap_or(0);
                self.note(format!(
                    "connected to the server · {} tools · system prompt {} tokens",
                    ready.get("tools").and_then(Value::as_u64).unwrap_or(0),
                    ready.get("system_tokens").and_then(Value::as_u64).unwrap_or(0)
                ));
                if self.server_settings != self.settings.server_part() {
                    self.note("the server runs with other model/MCP/system settings; /restart applies yours");
                }
                self.server_profile = ready.get("profile").and_then(Value::as_str).map(String::from);
                if self.server_profile.as_deref().is_some_and(|p| p != profile::render()) {
                    self.note("the server runs with an older profile; /restart applies yours");
                }
                self.server_skills = ready
                    .get("skills")
                    .and_then(Value::as_array)
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|s| {
                                Some((
                                    s.get("name")?.as_str()?.to_string(),
                                    s.get("description")?.as_str().unwrap_or("").to_string(),
                                ))
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                if !self.session.messages.is_empty() || self.session.pending_turn {
                    let messages = self.session.messages.clone();
                    self.send(json!({ "type": "load", "messages": messages, "session_id":self.session.id, "recover":self.session.pending_turn }));
                }
            }
            Msg::Server(event) => self.on_event(event),
            Msg::Closed(reason) => {
                self.writer = None;
                self.link = Link::Down;
                self.busy = false;
                self.confirm = None;
                self.save_session();
                if self.quit_after_close {
                    self.quit = true;
                } else if self.restart_after_close {
                    self.restart_after_close = false;
                    self.reconnect(true);
                } else {
                    self.error(format!("lost the server: {reason} (/retry reconnects, starting it if needed)"));
                }
            }
            Msg::Failed(reason) => {
                self.link = Link::Down;
                self.error(format!("server: {reason}"));
                self.note("/retry tries again");
            }
        }
    }

    fn on_event(&mut self, event: Value) {
        self.dirty = true;
        let s = |key: &str| event.get(key).and_then(Value::as_str).unwrap_or("").to_string();
        match s("type").as_str() {
            "phase" => self.phase = s("phase"),
            "context" => self.context_status = format!("{} / {} tokens", event["used"], event["limit"]),
            "checkpoint" => {
                if let Some(messages) = event.get("messages").and_then(Value::as_array) { self.session.messages = messages.clone(); }
                self.save_session();
            }
            "finished" => {
                if s("status") != "completed" { self.note(format!("turn ended: {}",s("status"))); }
            }
            "busy" => {
                self.busy = false; self.session.pending_turn = false;
                if let Some(Entry::User { text }) = self.session.entries.last() { self.input = text.clone(); self.cursor = self.input.chars().count(); self.session.entries.pop(); }
                self.note(s("message")); self.save_session();
            }
            "thinking" => {
                let piece = s("text");
                match self.session.entries.last_mut() {
                    Some(Entry::Thinking { text }) => text.push_str(&piece),
                    _ => self.push(Entry::Thinking { text: piece }),
                }
            }
            "text" => {
                let piece = s("text");
                match self.session.entries.last_mut() {
                    Some(Entry::Assistant { text }) => text.push_str(&piece),
                    _ => self.push(Entry::Assistant { text: piece }),
                }
            }
            "tool_call" => {
                let arguments = event.get("arguments").map(Value::to_string).unwrap_or_default();
                self.push(Entry::ToolCall { name: s("name"), arguments });
            }
            "tool_result" => {
                let is_error = event.get("is_error").and_then(Value::as_bool).unwrap_or(false);
                self.push(Entry::ToolResult { output: s("output"), is_error });
            }
            "rejected" => self.push(Entry::Rejected { name: s("name"), reason: s("reason") }),
            "stats" => {
                let n = |key: &str| event.get(key).and_then(Value::as_f64).unwrap_or(0.0);
                let tokens = n("new_tokens");
                let speed = if tokens > 1.0 { (tokens - 1.0) / (n("decode_ms") / 1000.0) } else { 0.0 };
                self.last_speed = format!("{speed:.1} tok/s");
                let text = format!(
                    "{speed:.1} tok/s · {tokens} tokens · prompt {} ({} cached) in {:.1} s",
                    n("prompt_tokens"),
                    n("reused_tokens"),
                    n("prefill_ms") / 1000.0
                );
                self.push(Entry::Stats { text });
            }
            "done" => {
                if event.get("finished").and_then(Value::as_bool) == Some(false) {
                    let note = if event.get("stop").and_then(Value::as_str) == Some("max_tokens") {
                        "the reply was cut off by the token limit (Tokens per step in F3)"
                    } else {
                        "the reply ended without a complete answer"
                    };
                    self.note(note);
                }
            }
            "step_limit" => self.note("stopped: the tool step limit was reached"),
            "confirm" => {
                let arguments = event.get("arguments").map(Value::to_string).unwrap_or_default();
                self.confirmation_id = (s("turn_id"), s("call_id"));
                self.confirm_scroll = 0;
                self.confirm = Some((s("name"), arguments));
            }
            "error" => self.error(s("message")),
            "end" => {
                self.busy = false;
                self.confirm = None;
                self.session.pending_turn = false;
                if let Some(messages) = event.get("messages").and_then(Value::as_array) {
                    self.session.messages = messages.clone();
                }
                self.save_session();
            }
            "tools" => {
                let list: Vec<String> = event
                    .get("list")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .map(|t| format!("{}: {}", t["name"].as_str().unwrap_or(""), t["description"].as_str().unwrap_or("")))
                    .collect();
                self.note(if list.is_empty() { "no tools (set an MCP config file in F3)".to_string() } else { list.join("\n") });
            }
            _ => {}
        }
    }

    // ----- actions -----

    fn submit(&mut self) {
        let text = self.input.trim().to_string();
        if text.is_empty() {
            return;
        }
        self.input.clear();
        self.cursor = 0;
        if text.starts_with('/') {
            self.command(&text);
            return;
        }
        if self.busy {
            self.input = text;
            self.cursor = self.input.chars().count();
            return;
        }
        if self.link != Link::Ready {
            self.input = text;
            self.cursor = self.input.chars().count();
            self.note(if self.link == Link::Connecting { "the server is still starting; send again when it is ready" } else { "not connected (/retry)" });
            return;
        }
        if !self.server_settings.get("model_dir").and_then(Value::as_str).is_some_and(|p|self.settings.matches_model(p)) {
            self.input = text;
            self.cursor = self.input.chars().count();
            self.note("the selected model is not running; /restart applies it, or use /model 0.8b or /model 2b");
            return;
        }
        self.turn_id = crate::persistence::unique_id();
        let send_text = if let Some(skill) = self.forced_skill.take() {
            format!("${skill} {text}")
        } else {
            text.clone()
        };
        if self.send(json!({ "type": "turn", "text": send_text, "options": self.settings.turn_options(), "turn_id":self.turn_id, "session_id":self.session.id })) {
            self.push(Entry::User { text });
            self.session.pending_turn = true;
            self.save_session();
            self.busy = true;
            self.busy_since = Instant::now();
            self.scroll = 0;
        } else if send_text != text {
            // Restore forced skill if the send failed after take().
            if let Some((name, _)) = send_text.strip_prefix('$').and_then(|rest| rest.split_once(' ')) {
                self.forced_skill = Some(name.to_string());
            }
        }
    }

    fn command(&mut self, command: &str) {
        if let Some(choice) = command.strip_prefix("/model ") { self.select_model(choice); return; }
        if let Some(rest) = command_rest(command, "/create-skill") {
            self.create_skill_cmd(rest);
            return;
        }
        if let Some(rest) = command_rest(command, "/validate-skill") {
            self.validate_skill_cmd(rest);
            return;
        }
        if let Some(name) = command.strip_prefix("/skill ") {
            self.force_skill(name.trim());
            return;
        }
        match command {
            "/model" => { self.view = View::Settings; self.settings_state.select(Some(1)); },
            "/help" => self.note(HELP),
            "/cancel" => self.cancel_turn(),
            "/new" => self.new_session(),
            "/sessions" => self.open_sessions(),
            "/settings" => self.view = View::Settings,
            "/profile" | "/setup" => self.view = View::Profile,
            "/thinking" => {
                self.apply_setting("thinking", "", false);
                self.note(format!("thinking mode {} (saved)", self.settings.thinking_mode.label()));
            }
            "/tools" => {
                self.send(json!({ "type": "tools" }));
            }
            "/skills" | "/skill" => self.list_skills(),
            "/retry" => match self.link {
                Link::Down => self.reconnect(false),
                Link::Ready => self.note("already connected"),
                Link::Connecting => self.note("already connecting"),
            },
            "/restart" => self.restart_server(),
            // /exit also stops the server (frees the GPU memory); /quit leaves it running, warm.
            "/exit" | "/stop" => {
                if self.link == Link::Ready && !self.busy && self.send(json!({ "type": "shutdown" })) {
                    self.quit_after_close = true;
                } else if self.busy {
                    self.note("wait for the reply to finish");
                } else {
                    self.quit = true;
                }
            }
            "/quit" => self.quit = true,
            other => self.note(format!("unknown command {other}; {HELP}")),
        }
    }

    fn list_skills(&mut self) {
        let catalog = self.skill_catalog();
        if catalog.is_empty() {
            self.note("no skills (add folders under .qwen-agent/skills/ then /restart)");
            return;
        }
        let list: Vec<_> = catalog
            .iter()
            .map(|(name, desc)| {
                let loaded = self.server_skills.iter().any(|(n, _)| n == name);
                if loaded || self.server_skills.is_empty() {
                    format!("{name}: {desc}")
                } else {
                    format!("{name}: {desc}  (on disk · /restart to load)")
                }
            })
            .collect();
        self.note(format!(
            "{}\n(skills in .qwen-agent/skills/; /skill name forces one next; /restart after edits)",
            list.join("\n")
        ));
    }

    fn force_skill(&mut self, name: &str) {
        if name.is_empty() {
            self.list_skills();
            return;
        }
        let loaded = self.server_skills.iter().any(|(n, _)| n.eq_ignore_ascii_case(name));
        let on_disk = self.skill_catalog().iter().any(|(n, _)| n.eq_ignore_ascii_case(name));
        if loaded || (self.server_skills.is_empty() && on_disk) {
            self.forced_skill = Some(name.to_ascii_lowercase());
            self.note(format!("skill {name} will apply to the next message"));
            return;
        }
        if on_disk {
            self.note(format!("skill {name} is on disk; /restart so the server can inject it"));
            return;
        }
        self.note(format!("unknown skill {name}; /skills lists them"));
    }

    fn create_skill_cmd(&mut self, rest: &str) {
        let mut parts = rest.splitn(2, char::is_whitespace);
        let name = parts.next().unwrap_or("").trim();
        let description = parts.next().map(str::trim).filter(|s| !s.is_empty());
        if name.is_empty() {
            self.note("usage: /create-skill name [description…]  (lowercase-hyphen name)");
            return;
        }
        match skills::create_skill(Path::new("."), name, description) {
            Ok(path) => {
                self.note(format!(
                    "created {} · edit the body, then /validate-skill {name} · /restart to load",
                    path.display()
                ));
                self.pending_editor = Some(path);
            }
            Err(e) => self.note(format!("create-skill failed: {e:#}")),
        }
    }

    fn validate_skill_cmd(&mut self, rest: &str) {
        let name = rest.trim();
        let report = if name.is_empty() {
            match skills::validate_all(Path::new(".")) {
                Ok(results) => skills::format_validation_report(&results),
                Err(e) => format!("validate-skill failed: {e:#}"),
            }
        } else if !skills::is_skill_name(&name.to_ascii_lowercase()) {
            format!("invalid skill name `{name}`")
        } else {
            let path = Path::new(skills::SKILLS_DIR).join(name.to_ascii_lowercase()).join("SKILL.md");
            let issues = skills::validate_skill_path(&path);
            skills::format_validation_report(&[(name.to_ascii_lowercase(), path, issues)])
        };
        self.note(report);
    }

    fn select_model(&mut self, choice: &str) {
        if self.busy || self.link == Link::Connecting { self.note("finish or cancel the current work before switching model"); return; }
        let result = (|| -> Result<Settings> {
            let selected = self.settings.select_model(choice)?;
            selected.check_model_files()?;
            let mut saved = Settings::load_or_create(Path::new(SETTINGS_FILE))?;
            saved.model_dir = selected.model_dir.clone();
            saved.kernel_tiles = selected.kernel_tiles.clone();
            saved.save(Path::new(SETTINGS_FILE))?;
            Ok(selected)
        })();
        match result {
            Ok(settings) => {
                self.settings = settings;
                self.settings_note = format!("{} selected; switching server",self.settings.model_choice());
                self.note(self.settings_note.clone());
                self.restart_server();
            },
            Err(e) => { self.settings_note = format!("model unchanged: {e:#}"); self.error(self.settings_note.clone()); },
        }
        self.dirty = true;
    }

    fn restart_server(&mut self) {
        if self.busy {
            self.note("wait for the reply to finish");
            return;
        }
        match self.link {
            Link::Ready => {
                if self.send(json!({ "type": "shutdown" })) {
                    self.restart_after_close = true;
                    self.link = Link::Connecting;
                    self.link_status = "restarting the server".into();
                    self.link_since = Instant::now();
                    self.settings_note = "restarting the server with the saved settings".into();
                }
            }
            Link::Down => self.reconnect(false),
            Link::Connecting => self.note("the server is already starting"),
        }
    }

    fn new_session(&mut self) {
        if self.busy {
            self.note("wait for the reply to finish");
            return;
        }
        self.session = SavedSession::new();
        if self.link == Link::Ready {
            self.send(json!({ "type": "reset", "session_id":self.session.id }));
        }
        self.view = View::Chat;
        self.scroll = 0;
    }

    fn open_session(&mut self, session: SavedSession) {
        if self.busy {
            self.note("wait for the reply to finish");
            return;
        }
        self.session = session;
        self.view = View::Chat;
        self.scroll = 0;
        if self.link == Link::Ready {
            let messages = self.session.messages.clone();
            self.send(json!({ "type": "load", "messages": messages, "session_id":self.session.id, "recover":self.session.pending_turn }));
        }
        self.note("conversation reopened; it continues where it stopped");
    }

    fn open_sessions(&mut self) {
        self.sessions = store::list();
        let current = self.sessions.iter().position(|s| s.id == self.session.id).unwrap_or(0);
        self.sessions_state.select((!self.sessions.is_empty()).then_some(current));
        self.delete_armed = false;
        self.view = View::Sessions;
    }

    fn apply_setting(&mut self, key: &str, raw: &str, restart: bool) {
        let result = (|| -> Result<Settings> {
            // The file gets only this field, so this run's flags are not saved with it.
            let effective = self.settings.with_field(key, raw)?;
            let path = Path::new(SETTINGS_FILE);
            let storage_key = if key == "thinking" { "thinking_mode" } else { key };
            let mut file = serde_json::to_value(Settings::load_or_create(path)?)?;
            file[storage_key] = serde_json::to_value(&effective)?[storage_key].clone();
            // Drop the legacy bool key so reload does not resurrect it beside thinking_mode.
            if storage_key == "thinking_mode" { file.as_object_mut().map(|o| o.remove("thinking")); }
            serde_json::from_value::<Settings>(file)?.save(path)?;
            Ok(effective)
        })();
        match result {
            Ok(settings) => {
                self.settings = settings;
                self.settings_note =
                    if restart { "saved · the server uses it after a restart (press r)".into() } else { "saved · applies from the next message".into() };
            }
            Err(e) => self.settings_note = format!("not saved: {e:#}"),
        }
    }

    // ----- keys -----

    fn cancel_turn(&mut self) {
        if self.busy { self.send(json!({"type":"cancel","turn_id":self.turn_id})); self.confirm = None; self.phase = "cancelling".into(); self.dirty = true; }
    }

    fn on_key(&mut self, key: KeyEvent) {
        self.dirty = true;
        // AltGr arrives as Ctrl+Alt on Windows; only plain Ctrl is a shortcut.
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL) && !key.modifiers.contains(KeyModifiers::ALT);
        if ctrl && key.code == KeyCode::Char('c') && self.busy { self.cancel_turn(); return; }
        if ctrl && matches!(key.code, KeyCode::Char('q') | KeyCode::Char('c')) {
            self.cancel_turn(); self.save_session(); self.quit = true; return;
        }
        if self.confirm.is_some() {
            match key.code {
                KeyCode::PageDown | KeyCode::Down => { self.confirm_scroll = self.confirm_scroll.saturating_add(3); return; }
                KeyCode::PageUp | KeyCode::Up => { self.confirm_scroll = self.confirm_scroll.saturating_sub(3); return; }
                _ => {}
            }
            let answer = match key.code {
                KeyCode::Char('y' | 'Y' | 'i' | 'I') => "y",
                KeyCode::Char('a' | 'A') => "a",
                KeyCode::Char('n' | 'N') | KeyCode::Esc => "n",
                _ => return,
            };
            self.confirm = None;
            self.send(json!({ "type": "approval", "answer": answer, "turn_id":self.confirmation_id.0, "call_id":self.confirmation_id.1 }));
            return;
        }
        if self.editing.is_none() {
            match key.code {
                KeyCode::F(1) => return self.view = View::Chat,
                KeyCode::F(2) => return self.open_sessions(),
                KeyCode::F(3) => return self.view = View::Settings,
                KeyCode::F(4) => return self.view = View::Profile,
                _ => {}
            }
        }
        match self.view {
            View::Chat => self.chat_key(key, ctrl),
            View::Sessions => self.sessions_key(key),
            View::Settings => self.settings_key(key),
            View::Profile => self.profile_key(key),
        }
    }

    fn byte_at(&self, chars: usize) -> usize {
        self.input.char_indices().nth(chars).map_or(self.input.len(), |(i, _)| i)
    }

    fn chat_key(&mut self, key: KeyEvent, ctrl: bool) {
        let len = self.input.chars().count();
        let palette_open = !self.palette.is_empty();
        match key.code {
            KeyCode::Enter if palette_open => self.accept_palette(),
            KeyCode::Enter => self.submit(),
            KeyCode::Tab if palette_open => self.move_palette(1),
            KeyCode::BackTab if palette_open => self.move_palette(-1),
            KeyCode::Tab => {
                self.refresh_palette();
                if self.palette.len() == 1 {
                    self.accept_palette();
                }
            }
            KeyCode::Up if palette_open => self.move_palette(-1),
            KeyCode::Down if palette_open => self.move_palette(1),
            KeyCode::Char('n') if ctrl => self.new_session(),
            KeyCode::Char('t') if ctrl => self.show_thinking = !self.show_thinking,
            KeyCode::Char('o') if ctrl => self.show_tools = !self.show_tools,
            KeyCode::Char('u') if ctrl => {
                self.clear_palette();
                self.input.clear();
                self.cursor = 0;
            }
            KeyCode::Char(c) if !ctrl => {
                let at = self.byte_at(self.cursor);
                self.input.insert(at, c);
                self.cursor += 1;
                self.refresh_palette();
            }
            KeyCode::Backspace if self.cursor > 0 => {
                self.cursor -= 1;
                let at = self.byte_at(self.cursor);
                self.input.remove(at);
                self.refresh_palette();
            }
            KeyCode::Delete if self.cursor < len => {
                let at = self.byte_at(self.cursor);
                self.input.remove(at);
                self.refresh_palette();
            }
            KeyCode::Left => self.cursor = self.cursor.saturating_sub(1),
            KeyCode::Right => self.cursor = (self.cursor + 1).min(len),
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = len,
            KeyCode::PageUp => self.scroll += 10,
            KeyCode::PageDown => self.scroll = self.scroll.saturating_sub(10),
            KeyCode::Up => self.scroll += 1,
            KeyCode::Down => self.scroll = self.scroll.saturating_sub(1),
            KeyCode::Esc if palette_open => self.clear_palette(),
            KeyCode::Esc if self.busy => self.cancel_turn(),
            KeyCode::Esc => {
                self.clear_palette();
                self.input.clear();
                self.cursor = 0;
            }
            KeyCode::F(2) => self.open_sessions(),
            KeyCode::F(3) => self.view = View::Settings,
            KeyCode::F(4) => self.view = View::Profile,
            _ => {}
        }
    }

    fn clear_palette(&mut self) {
        self.palette.clear();
        self.palette_index = 0;
    }

    fn skill_catalog(&self) -> Vec<(String, String)> {
        let mut catalog: Vec<(String, String)> = skills::load_catalog(Path::new("."))
            .unwrap_or_default()
            .into_iter()
            .map(|s| (s.name, s.description))
            .collect();
        for (name, desc) in &self.server_skills {
            if !catalog.iter().any(|(n, _)| n.eq_ignore_ascii_case(name)) {
                catalog.push((name.clone(), desc.clone()));
            }
        }
        catalog
    }

    fn refresh_palette(&mut self) {
        let prev = self.palette.get(self.palette_index).map(|i| i.insert.clone());
        self.palette = completion_items(&self.input, &self.skill_catalog());
        self.palette_index = prev
            .and_then(|insert| self.palette.iter().position(|i| i.insert == insert))
            .unwrap_or(0)
            .min(self.palette.len().saturating_sub(1));
    }

    fn move_palette(&mut self, delta: i32) {
        if self.palette.is_empty() {
            return;
        }
        let n = self.palette.len() as i32;
        self.palette_index = ((self.palette_index as i32 + delta).rem_euclid(n)) as usize;
    }

    /// Accept the highlighted picker row into the input; run finished slash commands.
    fn accept_palette(&mut self) {
        let Some(item) = self.palette.get(self.palette_index).cloned() else { return };
        let filled = with_completion_space(&item.insert);
        self.input = filled.clone();
        self.cursor = self.input.chars().count();
        // Arg completions that form a finished command line — run them.
        if item.insert.starts_with("/skill ")
            || item.insert.starts_with("/model ")
            || item.insert.starts_with("/validate-skill ")
        {
            self.clear_palette();
            self.submit();
            return;
        }
        // Bare slash command with no trailing arg slot — run it.
        if filled.starts_with('/') && !filled.ends_with(' ') && !filled[1..].contains(' ') {
            self.clear_palette();
            self.submit();
            return;
        }
        // Commands that still need args — keep the picker open when possible.
        self.refresh_palette();
        if self.palette.is_empty() {
            self.clear_palette();
        }
    }

    fn sessions_key(&mut self, key: KeyEvent) {
        let selected = self.sessions_state.selected();
        let count = self.sessions.len();
        match key.code {
            KeyCode::Up => {
                self.sessions_state.select_previous();
                self.delete_armed = false;
            }
            KeyCode::Down => {
                if selected.is_some_and(|i| i + 1 < count) {
                    self.sessions_state.select_next();
                }
                self.delete_armed = false;
            }
            KeyCode::Enter => {
                if let Some(session) = selected.and_then(|i| self.sessions.get(i)).cloned() {
                    self.open_session(session);
                }
            }
            KeyCode::Char('n') => self.new_session(),
            KeyCode::Char('d') | KeyCode::Delete => {
                let Some(i) = selected.filter(|&i| i < count) else { return };
                if !self.delete_armed {
                    self.delete_armed = true;
                    return;
                }
                self.delete_armed = false;
                let id = self.sessions[i].id.clone();
                if let Err(e) = store::delete(&id) {
                    self.error(format!("deleting the session failed: {e:#}"));
                }
                if id == self.session.id {
                    // Keep the open conversation, but as a new, unsaved one.
                    let entries = std::mem::take(&mut self.session.entries);
                    let messages = std::mem::take(&mut self.session.messages);
                    self.session = SavedSession { entries, messages, ..SavedSession::new() };
                }
                self.sessions.remove(i);
                self.sessions_state.select((!self.sessions.is_empty()).then(|| i.min(self.sessions.len() - 1)));
            }
            KeyCode::Esc | KeyCode::F(2) => self.view = View::Chat,
            KeyCode::F(3) => self.view = View::Settings,
            KeyCode::F(4) => self.view = View::Profile,
            _ => {}
        }
    }

    fn settings_key(&mut self, key: KeyEvent) {
        let index = self.settings_state.selected().unwrap_or(0).min(FIELDS.len() - 1);
        let (field, _, restart) = FIELDS[index];
        if let Some(buffer) = self.editing.as_mut() {
            match key.code {
                KeyCode::Enter => {
                    let raw = std::mem::take(buffer);
                    self.editing = None;
                    self.apply_setting(field, &raw, restart);
                }
                KeyCode::Esc => self.editing = None,
                KeyCode::Backspace => {
                    buffer.pop();
                }
                KeyCode::Char(c) => buffer.push(c),
                _ => {}
            }
            return;
        }
        match key.code {
            KeyCode::Up => self.settings_state.select(Some(index.saturating_sub(1))),
            KeyCode::Down => self.settings_state.select(Some((index + 1).min(FIELDS.len() - 1))),
            KeyCode::Enter | KeyCode::Left | KeyCode::Right if field == "model" => {
                let choice = if self.settings.model_choice() == "2b" { "0.8b" } else { "2b" };
                self.select_model(choice);
            },
            KeyCode::Enter | KeyCode::Left | KeyCode::Right if field == "thinking" => {
                self.apply_setting("thinking", "", restart);
            },
            KeyCode::Enter => match field_value(&self.settings, field) {
                Value::Bool(_) => self.apply_setting(field, "", restart),
                Value::String(s) => self.editing = Some(s),
                Value::Number(n) if n.is_f64() => self.editing = Some(format!("{}", n.as_f64().unwrap_or(0.0) as f32)),
                other => self.editing = Some(other.to_string()),
            },
            KeyCode::Char('r') => self.restart_server(),
            KeyCode::Esc | KeyCode::F(3) => self.view = View::Chat,
            KeyCode::F(2) => self.open_sessions(),
            KeyCode::F(4) => self.view = View::Profile,
            _ => {}
        }
    }

    /// Profile list: the fields, then two entries that open free-text files in an editor.
    fn profile_key(&mut self, key: KeyEvent) {
        let count = profile::FIELDS.len() + 2;
        let index = self.profile_state.selected().unwrap_or(0).min(count - 1);
        if let Some(buffer) = self.editing.as_mut() {
            match key.code {
                KeyCode::Enter => {
                    let value = std::mem::take(buffer);
                    self.editing = None;
                    let field = &profile::FIELDS[index];
                    self.profile_note = match profile::set(field.file, field.label, &value) {
                        Ok(()) => format!("saved in {} · the server uses it after a restart (press r)", field.file),
                        Err(e) => format!("not saved: {e:#}"),
                    };
                    self.refresh_profile();
                }
                KeyCode::Esc => self.editing = None,
                KeyCode::Backspace => {
                    buffer.pop();
                }
                KeyCode::Char(c) => buffer.push(c),
                _ => {}
            }
            return;
        }
        match key.code {
            KeyCode::Up => self.profile_state.select(Some(index.saturating_sub(1))),
            KeyCode::Down => self.profile_state.select(Some((index + 1).min(count - 1))),
            KeyCode::Enter if index < profile::FIELDS.len() => {
                let field = &profile::FIELDS[index];
                self.editing = Some(profile::get(field.file, field.label));
            }
            KeyCode::Enter => {
                let file = if index == profile::FIELDS.len() { "SOUL.md" } else { "USER.md" };
                self.pending_editor = Some(profile::path(file));
            }
            KeyCode::Char('r') => self.restart_server(),
            KeyCode::Esc | KeyCode::F(4) => self.view = View::Chat,
            KeyCode::F(2) => self.open_sessions(),
            KeyCode::F(3) => self.view = View::Settings,
            _ => {}
        }
    }

    // ----- drawing -----

    fn icons(&self) -> ui::Icons {
        ui::Icons(self.settings.icons)
    }

    /// "🦊 Pötyi" from the profile, or "🤖 qwen-agent" before one is set.
    fn agent_label(&self) -> String {
        let name = if self.agent_name.is_empty() { "qwen-agent" } else { &self.agent_name };
        let emoji = if self.agent_emoji.is_empty() { "🤖" } else { &self.agent_emoji };
        self.icons().tagged(emoji, name)
    }

    fn refresh_profile(&mut self) {
        self.agent_name = profile::get("IDENTITY.md", "Name");
        self.agent_emoji = profile::get("IDENTITY.md", "Emoji");
        let call = profile::get("USER.md", "What to call them");
        let name = profile::get("USER.md", "Name");
        self.user_name = [call, name].into_iter().find(|s| !s.is_empty()).unwrap_or_else(|| "You".into());
    }

    fn draw(&mut self, f: &mut Frame) {
        let [top, main, input, hints] =
            Layout::vertical([Constraint::Length(1), Constraint::Min(3), Constraint::Length(3), Constraint::Length(1)]).areas(f.area());
        self.draw_top(f, top);
        match self.view {
            View::Chat => self.draw_chat(f, main),
            View::Sessions => self.draw_sessions(f, main),
            View::Settings => self.draw_settings(f, main),
            View::Profile => self.draw_profile(f, main),
        }
        self.draw_input(f, input);
        self.draw_hints(f, hints);
        self.draw_palette(f, input);
        self.draw_confirm(f);
    }

    /// Agent badge, tabs, and the server state on the right.
    fn draw_top(&self, f: &mut Frame, area: Rect) {
        let ic = self.icons();
        let mut spans = vec![
            Span::styled(format!(" {} ", self.agent_label()), Style::new().fg(Color::Black).bg(ui::AGENT).add_modifier(Modifier::BOLD)),
            Span::raw(" "),
        ];
        for (view, icon, label, key) in [
            (View::Chat, "💬", "Chat", "F1"),
            (View::Sessions, "📚", "Sessions", "F2"),
            (View::Settings, "🧰", "Settings", "F3"),
            (View::Profile, "👤", "Profile", "F4"),
        ] {
            let style = if self.view == view { Style::new().fg(Color::Black).bg(ui::USER).add_modifier(Modifier::BOLD) } else { Style::new().fg(Color::Gray) };
            spans.push(Span::styled(format!(" {} ", ic.tagged(icon, label)), style));
            spans.push(Span::styled(format!("{key} "), ui::dim()));
        }
        let (dot, color, text) = match self.link {
            Link::Ready if self.busy => (ic.pick("⏳", "*"), ui::TOOL, format!("{} · {} s · {}", self.phase, self.busy_since.elapsed().as_secs(), self.context_status)),
            Link::Ready => {
                let active = self.server_settings.get("model_dir").and_then(Value::as_str).unwrap_or("?");
                let name = Path::new(active).file_name().and_then(|s| s.to_str()).unwrap_or(active).trim_start_matches("qwen3.5-");
                (ic.pick("🟢", "*"), ui::OK, format!("{} ready", name.to_uppercase()))
            },
            Link::Connecting => {
                let status: String = self.link_status.chars().take(40).collect();
                (ic.pick("🟡", "~"), ui::TOOL, format!("{status} · {} s", self.link_since.elapsed().as_secs()))
            }
            Link::Down => (ic.pick("🔴", "x"), ui::ERR, "offline · /retry".to_string()),
        };
        let status = Line::from(vec![Span::raw(format!("{dot} ")), Span::styled(text, ui::bold(color)), Span::raw(" ")]);
        let width = (status.width() as u16).min(area.width / 2);
        let [left, right] = Layout::horizontal([Constraint::Min(10), Constraint::Length(width)]).areas(area);
        f.render_widget(Paragraph::new(Line::from(spans)), left);
        f.render_widget(Paragraph::new(status).alignment(Alignment::Right), right);
    }

    /// Key hints at the bottom, and the last speed in the chat.
    fn draw_hints(&self, f: &mut Frame, area: Rect) {
        let keys: &[(&str, &str)] = match self.view {
            View::Chat if !self.palette.is_empty() => &[("↑↓", "choose"), ("Enter", "use"), ("Tab", "next"), ("Esc", "close"), ("Ctrl+Q", "quit")],
            View::Chat => &[("Enter", "send"), ("/", "commands"), ("$", "skills"), ("Tab", "picker"), ("Ctrl+N", "new chat"), ("Ctrl+Q", "quit")],
            View::Sessions => &[("↑↓", "choose"), ("Enter", "open"), ("n", "new"), ("d d", "delete"), ("Esc", "back")],
            View::Settings | View::Profile if self.editing.is_some() => &[("Enter", "save"), ("Esc", "cancel")],
            View::Settings | View::Profile => &[("↑↓", "choose"), ("Enter", "edit"), ("r", "restart server"), ("Esc", "back")],
        };
        let mut spans = vec![Span::raw(" ")];
        // The speed first, so a narrow terminal cuts hints rather than it.
        if self.view == View::Chat && !self.last_speed.is_empty() {
            spans.push(Span::raw(format!("{} ", self.icons().speed())));
            spans.push(Span::styled(self.last_speed.clone(), ui::bold(ui::TOOL)));
            spans.push(Span::styled("  │  ", ui::dim()));
        }
        for (key, what) in keys {
            spans.push(Span::styled(*key, ui::bold(ui::USER)));
            spans.push(Span::styled(format!(" {what}   "), ui::dim()));
        }
        f.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    fn welcome_lines(&self) -> Vec<Line<'static>> {
        let ic = self.icons();
        let mut lines = vec![
            Line::default(),
            Line::styled(format!("  {}", self.agent_label()), ui::bold(ui::AGENT)),
            Line::styled(format!("  Qwen3.5-0.8B on your GPU · {} tools over MCP", self.tool_count), ui::dim()),
            Line::default(),
            Line::styled("  Try asking:", ui::bold(ui::HEAD)),
        ];
        for (icon, question) in [
            ("📰", "What's in the news today?"),
            ("📖", "Who was Marie Curie? Look it up."),
            ("🧮", "What is 17% of 2380?"),
            ("🕐", "What time is it?"),
        ] {
            lines.push(Line::from(vec![Span::raw(format!("    {} ", ic.pick(icon, "-"))), Span::styled(format!("\"{question}\""), Style::new().fg(ui::USER))]));
        }
        lines.push(Line::default());
        lines.push(Line::styled("  F1 chat · F2 saved chats · F3 settings · F4 profile · /help commands · /skills skills", ui::dim()));
        lines
    }

    fn chat_lines(&self) -> Vec<Line<'static>> {
        let ic = self.icons();
        let dim = ui::dim();
        let mut lines: Vec<Line<'static>> = Vec::new();
        // Until the first message, the welcome screen stays, with any notes below it.
        if !self.session.entries.iter().any(|e| matches!(e, Entry::User { .. })) {
            lines = self.welcome_lines();
            lines.push(Line::default());
        }
        // The agent's name heads each reply, once.
        let mut in_reply = false;
        for entry in &self.session.entries {
            let from_agent = !matches!(entry, Entry::User { .. } | Entry::Note { .. } | Entry::Error { .. });
            if from_agent && !in_reply {
                if !lines.is_empty() {
                    lines.push(Line::default());
                }
                lines.push(Line::styled(self.agent_label(), ui::bold(ui::AGENT)));
                in_reply = true;
            }
            match entry {
                Entry::User { text } => {
                    in_reply = false;
                    if !lines.is_empty() {
                        lines.push(Line::default());
                    }
                    lines.push(Line::styled(ic.tagged(ic.user(), &self.user_name), ui::bold(ui::USER)));
                    lines.extend(text.lines().map(|l| Line::from(format!("  {l}"))));
                }
                Entry::Assistant { text } => lines.extend(ui::markdown(text, "  ")),
                Entry::Thinking { text } => {
                    let words = text.split_whitespace().count();
                    let italic = dim.add_modifier(Modifier::ITALIC);
                    let action = if self.show_thinking { "Ctrl+T hides" } else { "Ctrl+T shows" };
                    lines.push(Line::styled(format!("  {} thinking · {words} words · {action}", ic.thinking()), italic));
                    if self.show_thinking {
                        lines.extend(text.trim().lines().map(|l| Line::from(vec![Span::styled("  │ ", dim), Span::styled(l.to_string(), italic)])));
                    }
                }
                Entry::ToolCall { name, arguments } => lines.push(Line::from(vec![
                    Span::raw(format!("  {} ", ic.tool(name))),
                    Span::styled(name.clone(), ui::bold(ui::TOOL)),
                    Span::styled(format!("  {}", ui::tool_args(arguments)), dim),
                ])),
                Entry::ToolResult { output, is_error } if self.show_tools => {
                    lines.push(Line::styled(if *is_error { "  Tool error" } else { "  Tool result" }, ui::bold(ui::TOOL)));
                    lines.extend(ui::markdown(output,"  "));
                }
                Entry::ToolResult { output, is_error } => {
                    let first_line = output.lines().find(|l| !l.trim().is_empty()).unwrap_or("").trim();
                    let first: String = first_line.chars().take(160).collect();
                    let more = if output.lines().filter(|l| !l.trim().is_empty()).count() > 1 || first.len() < first_line.len() { " …" } else { "" };
                    let (mark, color) = if *is_error { (ic.fail(), ui::ERR) } else { (ic.ok(), ui::OK) };
                    lines.push(Line::from(vec![Span::raw(format!("  {mark} ")), Span::styled(format!("{first}{more}"), Style::new().fg(color))]));
                }
                Entry::Rejected { name, reason } => {
                    lines.push(Line::styled(format!("  {} {name}: {reason}", ic.fail()), Style::new().fg(ui::ERR)));
                }
                Entry::Note { text } => {
                    let style = Style::new().fg(ui::NOTE).add_modifier(Modifier::ITALIC);
                    for (i, l) in text.lines().enumerate() {
                        let prefix = if i == 0 { format!("{} ", ic.note()) } else { "   ".to_string() };
                        lines.push(Line::styled(format!("{prefix}{l}"), style));
                    }
                }
                Entry::Error { text } => {
                    for (i, l) in text.lines().enumerate() {
                        let prefix = if i == 0 { format!("{} ", ic.error()) } else { "   ".to_string() };
                        lines.push(Line::styled(format!("{prefix}{l}"), Style::new().fg(ui::ERR)));
                    }
                }
                Entry::Stats { text } => lines.push(Line::styled(format!("  {} {text}", ic.speed()), dim)),
            }
        }
        if self.busy {
            const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
            let elapsed = self.busy_since.elapsed();
            let frame = FRAMES[(elapsed.as_millis() / 100) as usize % FRAMES.len()];
            lines.push(Line::from(vec![Span::styled(format!("  {frame} "), ui::bold(ui::AGENT)), Span::styled(format!("working · {} s", elapsed.as_secs()), dim)]));
        }
        lines
    }

    fn panel<'a>(&self, title: String, color: Color) -> Block<'a> {
        let block = Block::bordered().border_type(BorderType::Rounded).border_style(ui::dim());
        if title.is_empty() { block } else { block.title(Span::styled(format!(" {title} "), ui::bold(color))) }
    }

    fn draw_chat(&mut self, f: &mut Frame, area: Rect) {
        let title = if self.session.title.is_empty() { "new conversation".to_string() } else { self.session.title.clone() };
        let mut block = self.panel(self.icons().tagged("💬", &title), ui::USER);
        let paragraph = Paragraph::new(self.chat_lines()).wrap(Wrap { trim: false });
        let total = paragraph.line_count(area.width.saturating_sub(2));
        let height = area.height.saturating_sub(2) as usize;
        let max_scroll = total.saturating_sub(height);
        self.scroll = self.scroll.min(max_scroll);
        if self.scroll > 0 {
            block = block.title_bottom(Line::styled(format!(" ↓ {} more lines · PgDn ", self.scroll), ui::bold(ui::TOOL)).right_aligned());
        }
        let top = (max_scroll - self.scroll).min(u16::MAX as usize) as u16;
        f.render_widget(paragraph.block(block).scroll((top, 0)), area);
    }

    fn list_highlight() -> Style {
        Style::new().bg(Color::DarkGray).add_modifier(Modifier::BOLD)
    }

    fn draw_sessions(&mut self, f: &mut Frame, area: Rect) {
        let ic = self.icons();
        let mut block = self.panel(ic.tagged("📚", "Saved chats"), ui::USER);
        if self.delete_armed {
            block = block.title_bottom(Line::styled(" press d again to delete ", ui::bold(ui::ERR)).right_aligned());
        }
        if self.sessions.is_empty() {
            let text = Line::styled(format!("  {}", ic.tagged("📭", "No saved chats yet; each chat is saved after every reply.")), ui::dim());
            f.render_widget(Paragraph::new(vec![Line::default(), text]).block(block), area);
            return;
        }
        let items: Vec<ListItem> = self
            .sessions
            .iter()
            .map(|s| {
                let current = s.id == self.session.id;
                let title = if s.title.is_empty() { "(untitled)" } else { &s.title };
                let messages = s.entries.iter().filter(|e| matches!(e, Entry::User { .. })).count();
                let title_style = if current { ui::bold(ui::AGENT) } else { Style::new() };
                ListItem::new(Line::from(vec![
                    Span::raw(format!("{} ", if current { ic.pick("📌", "*") } else { ic.pick("💬", " ") })),
                    Span::styled(title.to_string(), title_style),
                    Span::styled(format!("  · {messages} message{} · {}", if messages == 1 { "" } else { "s" }, store::age(s.updated)), ui::dim()),
                ]))
            })
            .collect();
        let list = List::new(items).block(block).highlight_style(Self::list_highlight()).highlight_symbol("▶ ");
        f.render_stateful_widget(list, area, &mut self.sessions_state);
    }

    fn draw_settings(&mut self, f: &mut Frame, area: Rect) {
        let ic = self.icons();
        let [list_area, note_area] = Layout::vertical([Constraint::Min(3), Constraint::Length(5)]).areas(area);
        let items: Vec<ListItem> = FIELDS
            .iter()
            .map(|&(key, label, restart)| {
                let value = match field_value(&self.settings, key) {
                    Value::String(s) if s.is_empty() => Span::styled("(none)", ui::dim()),
                    Value::String(s) if s.chars().count() > 50 => Span::styled(s.chars().take(50).collect::<String>() + "…", Style::new().fg(ui::VALUE)),
                    Value::String(s) => Span::styled(s, Style::new().fg(ui::VALUE)),
                    Value::Bool(true) => Span::styled("● on", ui::bold(ui::OK)),
                    Value::Bool(false) => Span::styled("○ off", ui::dim()),
                    // f32 settings widen to f64 in JSON (0.6 -> 0.6000000238...).
                    Value::Number(n) if n.is_f64() => Span::styled(
                        format!("{:.3}", n.as_f64().unwrap_or(0.0)).trim_end_matches('0').trim_end_matches('.').to_string(),
                        Style::new().fg(ui::VALUE),
                    ),
                    other => Span::styled(other.to_string(), Style::new().fg(ui::VALUE)),
                };
                let mut spans = vec![Span::raw(format!("{} {label:<28}", ic.setting(key))), value];
                if restart {
                    spans.push(Span::styled(format!("  {}", ic.tagged("🔄", "restart")), ui::dim()));
                }
                ListItem::new(Line::from(spans))
            })
            .collect();
        let list = List::new(items).block(self.panel(ic.tagged("🧰", "Settings"), ui::USER)).highlight_style(Self::list_highlight()).highlight_symbol("▶ ");
        f.render_stateful_widget(list, list_area, &mut self.settings_state);

        let mut lines = vec![
            Line::styled(format!("Model row: ←/→/Enter switches 0.8B / 2B. Other restart settings: r. Saved in {SETTINGS_FILE}"), ui::dim()),
            Line::styled("Skills live in .qwen-agent/skills/; /skills lists them; /restart after AGENTS.md or skill edits.", ui::dim()),
        ];
        if self.link == Link::Ready && self.server_settings != self.settings.server_part() {
            lines.push(Line::styled(ic.tagged("🔄", "the server runs with other settings · press r to restart it"), ui::bold(ui::TOOL)));
        }
        if !self.settings_note.is_empty() {
            lines.push(Line::styled(self.settings_note.clone(), Style::new().fg(ui::NOTE)));
        }
        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }).block(self.panel(String::new(), ui::DIM)), note_area);
    }

    fn draw_profile(&mut self, f: &mut Frame, area: Rect) {
        let ic = self.icons();
        let [list_area, note_area] = Layout::vertical([Constraint::Min(3), Constraint::Length(5)]).areas(area);
        let group = |agent: bool| {
            if agent {
                Span::styled(format!("{:<9}", ic.tagged("🤖", "Agent")), ui::bold(ui::AGENT))
            } else {
                Span::styled(format!("{:<9}", ic.tagged("👤", "You")), ui::bold(ui::USER))
            }
        };
        let mut items: Vec<ListItem> = Vec::new();
        for field in profile::FIELDS {
            let value = profile::get(field.file, field.label);
            let shown = if value.is_empty() {
                Span::styled(field.hint, ui::dim().add_modifier(Modifier::ITALIC))
            } else {
                Span::styled(value, Style::new().fg(ui::VALUE))
            };
            items.push(ListItem::new(Line::from(vec![group(field.file == "IDENTITY.md"), Span::raw(format!("{:<20}", field.label)), shown])));
        }
        items.push(ListItem::new(Line::from(vec![
            group(true),
            Span::raw(format!("{:<20}", ic.tagged("📜", "Behaviour"))),
            Span::styled("SOUL.md · opens in your editor", ui::dim()),
        ])));
        items.push(ListItem::new(Line::from(vec![
            group(false),
            Span::raw(format!("{:<20}", ic.tagged("📝", "More about you"))),
            Span::styled("USER.md · opens in your editor", ui::dim()),
        ])));
        let list = List::new(items)
            .block(self.panel(ic.tagged("👤", "Profile · who you are and who the agent is"), ui::USER))
            .highlight_style(Self::list_highlight())
            .highlight_symbol("▶ ");
        f.render_stateful_widget(list, list_area, &mut self.profile_state);

        let rendered = profile::render();
        let mut lines = vec![Line::styled(
            format!("files in {}/ (OpenClaw format) · {} characters go into the prompt", profile::PROFILE_DIR, rendered.chars().count()),
            ui::dim(),
        )];
        if self.link == Link::Ready && self.server_profile.as_deref().is_some_and(|p| p != rendered) {
            lines.push(Line::styled(ic.tagged("🔄", "the server runs with an older profile · press r to restart it"), ui::bold(ui::TOOL)));
        }
        if !self.profile_note.is_empty() {
            lines.push(Line::styled(self.profile_note.clone(), Style::new().fg(ui::NOTE)));
        }
        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }).block(self.panel(String::new(), ui::DIM)), note_area);
    }

    fn draw_input(&self, f: &mut Frame, area: Rect) {
        let ic = self.icons();
        let skill_hint = self.forced_skill.as_deref().map(|s| format!(" · skill {s} next"));
        let (title, text, cursor, color) = match (&self.editing, self.view) {
            (Some(buffer), View::Profile) => {
                let index = self.profile_state.selected().unwrap_or(0).min(profile::FIELDS.len() - 1);
                let field = &profile::FIELDS[index];
                (ic.tagged("📝", &format!("{} · {}", field.label, field.hint)), buffer.as_str(), buffer.chars().count(), ui::AGENT)
            }
            (Some(buffer), View::Settings) => {
                let index = self.settings_state.selected().unwrap_or(0).min(FIELDS.len() - 1);
                (ic.tagged("📝", FIELDS[index].1), buffer.as_str(), buffer.chars().count(), ui::AGENT)
            }
            (_, View::Chat) if self.busy => (
                ic.tagged("⏳", &format!("working · you can type the next message{}", skill_hint.as_deref().unwrap_or(""))),
                self.input.as_str(),
                self.cursor,
                ui::TOOL,
            ),
            (_, View::Chat) => (
                ic.tagged("💬", &format!("message · / commands · $ skills{}", skill_hint.as_deref().unwrap_or(""))),
                self.input.as_str(),
                self.cursor,
                ui::USER,
            ),
            _ => {
                let block = Block::bordered().border_type(BorderType::Rounded).border_style(ui::dim());
                f.render_widget(Paragraph::new(Line::styled(" F1 or Esc: back to the chat", ui::dim())).block(block), area);
                return;
            }
        };
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(Style::new().fg(color))
            .title(Span::styled(format!(" {title} "), ui::bold(color)));
        if text.is_empty() && self.view == View::Chat {
            let placeholder = if let Some(skill) = &self.forced_skill {
                format!("Ask anything…  skill {skill} next · type / or $ for the picker")
            } else {
                "Ask anything…  type / for commands or $ for skills".into()
            };
            f.render_widget(Paragraph::new(Line::styled(placeholder, ui::dim().add_modifier(Modifier::ITALIC))).block(block), area);
            if self.confirm.is_none() {
                f.set_cursor_position((area.x + 1, area.y + 1));
            }
            return;
        }
        let width = area.width.saturating_sub(2) as usize;
        let before: String = text.chars().take(cursor).collect();
        let before_width = Line::from(before.as_str()).width();
        let offset = before_width.saturating_sub(width.saturating_sub(1));
        f.render_widget(Paragraph::new(text.to_string()).scroll((0, offset as u16)).block(block), area);
        if self.confirm.is_none() {
            f.set_cursor_position((area.x + 1 + (before_width - offset) as u16, area.y + 1));
        }
    }

    /// Cursor-style command/skill picker floating above the chat input.
    fn draw_palette(&self, f: &mut Frame, input_area: Rect) {
        if self.palette.is_empty() || self.view != View::Chat {
            return;
        }
        let ic = self.icons();
        let visible = self.palette.len().min(8);
        let height = (visible as u16).saturating_add(2).min(input_area.y.saturating_add(input_area.height));
        if height < 3 || input_area.y < 2 {
            return;
        }
        let width = input_area.width.min(76).max(28);
        let y = input_area.y.saturating_sub(height);
        let rect = Rect { x: input_area.x, y, width, height };
        f.render_widget(Clear, rect);

        let start = self.palette_index.saturating_sub(visible.saturating_sub(1).min(self.palette_index));
        let end = (start + visible).min(self.palette.len());
        let mut lines = Vec::with_capacity(visible);
        for (offset, item) in self.palette[start..end].iter().enumerate() {
            let index = start + offset;
            let selected = index == self.palette_index;
            let icon = if item.insert.starts_with('$')
                || item.insert.starts_with("/skill ")
                || item.insert.starts_with("/validate-skill ")
            {
                ic.pick("📦", "*")
            } else if item.insert.starts_with("/model") {
                ic.pick("🧠", "*")
            } else if item.insert.starts_with("/create-skill") || item.insert.starts_with("/validate-skill") {
                ic.pick("🛠", "*")
            } else {
                ic.pick("⚡", ">")
            };
            let desc_budget = width.saturating_sub(item.label.chars().count() as u16 + 8) as usize;
            let desc = if item.description.chars().count() > desc_budget && desc_budget > 1 {
                let clipped: String = item.description.chars().take(desc_budget.saturating_sub(1)).collect();
                format!("{clipped}…")
            } else {
                item.description.clone()
            };
            let (label_style, desc_style, hint_style) = if selected {
                (
                    Style::new().fg(Color::Black).bg(ui::USER).add_modifier(Modifier::BOLD),
                    Style::new().fg(Color::Black).bg(ui::USER),
                    Style::new().fg(Color::Black).bg(ui::USER),
                )
            } else {
                (Style::new().fg(Color::White), ui::dim(), ui::dim())
            };
            let mut spans = vec![
                Span::styled(format!(" {icon} "), if selected { Style::new().fg(Color::Black).bg(ui::USER) } else { Style::new() }),
                Span::styled(item.label.clone(), label_style),
                Span::styled("  ", if selected { Style::new().bg(ui::USER) } else { Style::new() }),
                Span::styled(desc, desc_style),
            ];
            if selected {
                spans.push(Span::styled("  Enter", hint_style));
            }
            lines.push(Line::from(spans));
        }
        let title = if self.input.starts_with('$')
            || self.input.starts_with("/skill ")
            || self.input.starts_with("/validate-skill ")
        {
            ic.tagged("📦", "Skills")
        } else if self.input.starts_with("/model ") {
            ic.tagged("🧠", "Models")
        } else {
            ic.tagged("⚡", "Commands")
        };
        let block = Block::bordered()
            .border_type(BorderType::Rounded)
            .border_style(Style::new().fg(Color::Gray))
            .title(Span::styled(format!(" {title} "), ui::bold(ui::USER)))
            .title_bottom(Line::styled(" ↑↓ · Enter use · Esc ", ui::dim()).right_aligned());
        f.render_widget(Paragraph::new(lines).block(block), rect);
    }

    fn draw_confirm(&self, f: &mut Frame) {
        let Some((name, arguments)) = &self.confirm else { return };
        let ic = self.icons();
        let pairs = ui::tool_arg_pairs(arguments);
        let area = f.area();
        let width = area.width.saturating_sub(8).min(84);
        let height = (7 + pairs.len() as u16).min(area.height);
        let rect = Rect { x: area.x + (area.width - width) / 2, y: area.y + (area.height - height) / 2, width, height };
        f.render_widget(Clear, rect);
        let mut lines = vec![Line::from(vec![Span::raw(format!("{} ", ic.tool(name))), Span::styled(name.clone(), ui::bold(ui::TOOL))])];
        for (key, value) in pairs {
            let key = if key.is_empty() { String::new() } else { format!("{key}: ") };
            lines.push(Line::from(vec![Span::styled(format!("   {key}"), ui::dim()), Span::styled(value, Style::new().fg(ui::VALUE))]));
        }
        lines.push(Line::default());
        lines.push(Line::from(vec![
            Span::styled(" y ", Style::new().fg(Color::Black).bg(ui::OK).add_modifier(Modifier::BOLD)),
            Span::raw(" run    "),
            Span::styled(" n ", Style::new().fg(Color::Black).bg(ui::ERR).add_modifier(Modifier::BOLD)),
            Span::raw(" decline    "),
            Span::styled(" a ", Style::new().fg(Color::Black).bg(ui::USER).add_modifier(Modifier::BOLD)),
            Span::raw(" always allow this tool"),
        ]));
        let block = Block::bordered()
            .border_type(BorderType::Thick)
            .border_style(Style::new().fg(ui::TOOL))
            .title(Span::styled(format!(" {} ", ic.tagged("🔧", "Run this tool? · PgUp/PgDn details")), ui::bold(ui::TOOL)));
        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }).scroll((self.confirm_scroll,0)).block(block), rect);
    }

    fn event_loop(&mut self, terminal: &mut DefaultTerminal) -> Result<()> {
        let mut last_draw = Instant::now() - Duration::from_secs(1);
        while !self.quit {
            if self.dirty || (self.busy && last_draw.elapsed() >= Duration::from_millis(200)) {
                terminal.draw(|f| self.draw(f))?; self.dirty = false; last_draw = Instant::now();
            }
            let mut wait = Duration::from_millis(50);
            let mut keys = 0;
            // Handle every pending key (a paste arrives as many) before drawing again.
            while keys < 64 && event::poll(wait)? {
                keys += 1;
                let event = event::read()?;
                if matches!(event, TermEvent::Resize(..)) { self.dirty = true; }
                if let TermEvent::Key(key) = event {
                    if key.kind == KeyEventKind::Press {
                        self.on_key(key);
                    }
                    if let Some(path) = self.pending_editor.take() {
                        // The editor needs the terminal (or, like Notepad, at least to be waited for).
                        ratatui::restore();
                        let result = open_editor(&path);
                        *terminal = ratatui::init();
                        terminal.clear()?;
                        self.profile_note = match result {
                            Ok(()) => format!("{} saved · the server uses it after a restart (press r)", path.display()),
                            Err(e) => format!("editor failed: {e:#} (set EDITOR, or edit {} by hand)", path.display()),
                        };
                        self.refresh_profile();
                    }
                }
                wait = Duration::ZERO;
            }
            for _ in 0..128 {
                match self.rx.try_recv() { Ok(msg) => self.on_msg(msg), Err(_) => break }
            }
        }
        self.cancel_turn();
        self.save_session();
        if let Some(socket) = &self.writer { let _ = socket.shutdown(std::net::Shutdown::Both); }
        Ok(())
    }
}

/// Opens `path` in $VISUAL / $EDITOR (which may carry arguments, e.g. "code --wait"), or Notepad
/// on Windows and nano elsewhere, and waits for it to close.
fn open_editor(path: &Path) -> Result<()> {
    let editor = std::env::var("VISUAL")
        .or_else(|_| std::env::var("EDITOR"))
        .unwrap_or_else(|_| if cfg!(windows) { "notepad".into() } else { "nano".into() });
    let mut parts = editor.split_whitespace();
    let program = parts.next().unwrap_or("notepad");
    let status = Command::new(program).args(parts).arg(path).status()?;
    if !status.success() {
        bail!("{program} exited with {status}");
    }
    Ok(())
}

fn field_value(settings: &Settings, key: &str) -> Value {
    if key == "model" { return json!(settings.model_choice().to_uppercase()); }
    if key == "thinking" { return json!(settings.thinking_mode.label().to_uppercase()); }
    serde_json::to_value(settings).ok().and_then(|v| v.get(key).cloned()).unwrap_or(Value::Null)
}

fn completion_items(input: &str, skills: &[(String, String)]) -> Vec<CompleteItem> {
    if let Some(rest) = input.strip_prefix("/skill ") {
        let prefix = rest.to_ascii_lowercase();
        return skills
            .iter()
            .filter(|(n, _)| n.to_ascii_lowercase().starts_with(&prefix))
            .map(|(n, d)| CompleteItem {
                insert: format!("/skill {n}"),
                label: n.clone(),
                description: d.clone(),
            })
            .collect();
    }
    if let Some(rest) = input.strip_prefix("/validate-skill ") {
        let prefix = rest.to_ascii_lowercase();
        return skills
            .iter()
            .filter(|(n, _)| n.to_ascii_lowercase().starts_with(&prefix))
            .map(|(n, d)| CompleteItem {
                insert: format!("/validate-skill {n}"),
                label: n.clone(),
                description: d.clone(),
            })
            .collect();
    }
    if let Some(rest) = input.strip_prefix("/model ") {
        let prefix = rest.to_ascii_lowercase();
        return MODEL_ITEMS
            .iter()
            .filter(|(m, _)| m.starts_with(&prefix))
            .map(|(m, d)| CompleteItem {
                insert: format!("/model {m}"),
                label: (*m).to_string(),
                description: (*d).to_string(),
            })
            .collect();
    }
    if let Some(rest) = input.strip_prefix('$') {
        if !rest.chars().any(char::is_whitespace) {
            let prefix = rest.to_ascii_lowercase();
            return skills
                .iter()
                .filter(|(n, _)| n.to_ascii_lowercase().starts_with(&prefix))
                .map(|(n, d)| CompleteItem {
                    insert: format!("${n}"),
                    label: format!("${n}"),
                    description: d.clone(),
                })
                .collect();
        }
        return Vec::new();
    }
    if input.starts_with('/') && !input[1..].chars().any(char::is_whitespace) {
        let prefix = input.to_ascii_lowercase();
        return COMMAND_ITEMS
            .iter()
            .filter(|(c, _)| c.starts_with(&prefix))
            .map(|(c, d)| CompleteItem {
                insert: (*c).to_string(),
                label: (*c).to_string(),
                description: (*d).to_string(),
            })
            .collect();
    }
    Vec::new()
}

fn with_completion_space(choice: &str) -> String {
    if matches!(choice, "/skill" | "/model" | "/create-skill")
        || (choice.starts_with('$') && !choice.contains(' '))
    {
        format!("{choice} ")
    } else {
        choice.to_string()
    }
}

/// `/name` or `/name args` — not `/namex`.
fn command_rest<'a>(command: &'a str, name: &str) -> Option<&'a str> {
    if command == name {
        Some("")
    } else {
        command.strip_prefix(name)?.strip_prefix(' ').map(str::trim)
    }
}

pub fn run(settings: Settings, resume: bool) -> Result<()> {
    let mut app = App::new(settings, resume);
    app.refresh_profile();
    if profile::is_blank() {
        // First run, like OpenClaw's onboarding: start on the profile.
        app.view = View::Profile;
        app.profile_note = "Welcome! Tell the agent who you are and who it should be. Esc skips; F4 or /profile opens this later.".into();
    }
    app.reconnect(false);
    let mut terminal = ratatui::init();
    let result = app.event_loop(&mut terminal);
    ratatui::restore();
    if app.link != Link::Down && !app.quit_after_close {
        eprintln!("the server keeps the model loaded in the background; `qwen-agent stop` stops it");
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selector_renders_and_refuses_switch_during_a_turn() {
        let mut app=App::new(Settings::default(),false);
        app.command("/model");
        assert!(app.view == View::Settings);
        assert_eq!(FIELDS[app.settings_state.selected().unwrap()].0,"model");
        app.busy=true;
        let before=app.settings.clone();
        app.on_key(KeyEvent::new(KeyCode::Right,KeyModifiers::NONE));
        assert_eq!(app.settings,before);
        assert!(!app.restart_after_close);
        for (w,h) in [(100,28),(30,12),(8,6)] {
            let mut terminal=ratatui::Terminal::new(ratatui::backend::TestBackend::new(w,h)).unwrap();
            terminal.draw(|f|app.draw(f)).unwrap();
        }
    }
    #[test]
    fn selected_model_mismatch_keeps_unsent_message() {
        let mut app=App::new(Settings::default().select_model("2b").unwrap(),false);
        app.link=Link::Ready;
        app.server_settings=Settings::default().select_model("0.8b").unwrap().server_part();
        app.input="keep this message".into();
        app.submit();
        assert_eq!(app.input,"keep this message");
        assert!(!app.busy);
    }
    #[test]
    fn model_selector_is_a_virtual_field_with_both_choices() {
        let s=Settings::default();
        assert_eq!(field_value(&s.select_model("0.8b").unwrap(),"model"),json!("0.8B"));
        assert_eq!(field_value(&s.select_model("2b").unwrap(),"model"),json!("2B"));
        assert_eq!(field_value(&Settings { model_dir: "custom".into(), ..s },"model"),json!("CUSTOM"));
    }
    #[test]
    fn palette_lists_commands_skills_and_models() {
        let skills = vec![("mcp-tools".into(), "calc".into()), ("notes-files".into(), "files".into())];
        let cmds = completion_items("/sk", &skills);
        assert_eq!(cmds.iter().map(|c| c.insert.as_str()).collect::<Vec<_>>(), vec!["/skills", "/skill"]);
        assert!(cmds[0].description.contains("List"));
        let skill_args = completion_items("/skill m", &skills);
        assert_eq!(skill_args[0].insert, "/skill mcp-tools");
        assert_eq!(skill_args[0].label, "mcp-tools");
        let dollar = completion_items("$note", &skills);
        assert_eq!(dollar[0].insert, "$notes-files");
        let models = completion_items("/model ", &skills);
        assert_eq!(models.len(), 2);
        assert_eq!(with_completion_space("/skill"), "/skill ");
        assert_eq!(with_completion_space("/validate-skill"), "/validate-skill");
        assert_eq!(with_completion_space("$mcp-tools"), "$mcp-tools ");
        assert_eq!(command_rest("/create-skill demo", "/create-skill"), Some("demo"));
        assert_eq!(command_rest("/create-skills", "/create-skill"), None);

        let mut app = App::new(Settings::default(), false);
        app.server_skills = skills;
        app.input = "/sk".into();
        app.cursor = 3;
        app.refresh_palette();
        assert_eq!(app.palette.len(), 2);
        app.move_palette(1);
        assert_eq!(app.palette[app.palette_index].insert, "/skill");
        app.accept_palette();
        assert_eq!(app.input, "/skill ");
        assert!(!app.palette.is_empty());
        app.input = "/skill m".into();
        app.cursor = app.input.chars().count();
        app.refresh_palette();
        assert_eq!(app.palette[0].insert, "/skill mcp-tools");
    }
}

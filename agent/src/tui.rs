//! Terminal UI. It connects to the background server, starting one (detached, logging to
//! `.qwen-agent/server.log`) when none runs, so the model stays loaded after the TUI exits.
//! Conversations are saved after every reply and can be reopened; settings are edited in place
//! and saved to `.qwen-agent/settings.json`.

use std::{
    io::BufRead,
    net::TcpStream,
    path::Path,
    process::{Child, Command, Stdio},
    sync::mpsc::{self, Receiver, Sender},
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
    store::{self, Entry, SavedSession},
    ui,
};

const SERVER_LOG: &str = ".qwen-agent/server.log";

/// Editable settings: key, label, and whether the server must restart to use it.
const FIELDS: &[(&str, &str, bool)] = &[
    ("mcp", "MCP config file", true),
    ("model_dir", "Model folder", true),
    ("dxc", "DXC compiler (f16 kernels)", true),
    ("system", "System prompt", true),
    ("mcp_guide", "Explain MCP to the model", true),
    ("port", "Server port", true),
    ("auto_approve", "Run tools without asking", false),
    ("thinking", "Thinking mode", false),
    ("temperature", "Temperature (0 = greedy)", false),
    ("presence_penalty", "Repetition penalty (0-2)", false),
    ("icons", "Icons and emoji", false),
    ("max_steps", "Tool steps per message", false),
    ("max_tokens", "Tokens per step", false),
];

const HELP: &str = "/new new conversation · /sessions saved ones (F2) · /settings (F3) · /profile about you and the agent (F4) · /thinking turns thinking mode on/off · Ctrl+T shows/hides the reasoning · /tools · /restart restarts the server · /retry reconnects · /quit quits, the server stays warm (Ctrl+Q) · /exit quits and stops the server";

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
fn connect_in_background(tx: Sender<Msg>, settings: Settings, after_shutdown: bool) {
    thread::spawn(move || {
        let server_file = Path::new(SERVER_FILE);
        let result = (|| -> Result<(Connection, Value)> {
            if after_shutdown {
                // The old server removes its file just before it exits.
                for _ in 0..50 {
                    if !server_file.exists() {
                        break;
                    }
                    thread::sleep(Duration::from_millis(100));
                }
            }
            if let Some(found) = client::open(server_file)? {
                return Ok(found);
            }
            let _ = tx.send(Msg::Status("starting the server".into()));
            let mut child = spawn_server(&settings)?;
            let started = Instant::now();
            let mut last = String::new();
            loop {
                if let Some(status) = child.try_wait()? {
                    bail!("the server exited ({status}): {}", last_log_line().unwrap_or_default());
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
            let mut line = String::new();
            let msg = match reader.read_line(&mut line) {
                Ok(0) => Msg::Closed("the server closed the connection".into()),
                Ok(_) => match serde_json::from_str(&line) {
                    Ok(v) => Msg::Server(v),
                    Err(_) => continue,
                },
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
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    busy: bool,
    busy_since: Instant,
    confirm: Option<(String, String)>,
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
    quit: bool,
}

impl App {
    fn new(settings: Settings, resume: bool) -> Self {
        let (tx, rx) = mpsc::channel();
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
                if !self.session.messages.is_empty() {
                    let messages = self.session.messages.clone();
                    self.send(json!({ "type": "load", "messages": messages }));
                }
            }
            Msg::Server(event) => self.on_event(event),
            Msg::Closed(reason) => {
                self.writer = None;
                self.link = Link::Down;
                self.busy = false;
                self.confirm = None;
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
        let s = |key: &str| event.get(key).and_then(Value::as_str).unwrap_or("").to_string();
        match s("type").as_str() {
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
                    self.note("the reply was cut off by the token limit (Tokens per step in F3)");
                }
            }
            "step_limit" => self.note("stopped: the tool step limit was reached"),
            "confirm" => {
                let arguments = event.get("arguments").map(Value::to_string).unwrap_or_default();
                self.confirm = Some((s("name"), arguments));
            }
            "error" => self.error(s("message")),
            "end" => {
                self.busy = false;
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
        if self.send(json!({ "type": "turn", "text": text, "options": self.settings.turn_options() })) {
            self.push(Entry::User { text });
            self.busy = true;
            self.busy_since = Instant::now();
            self.scroll = 0;
        }
    }

    fn command(&mut self, command: &str) {
        match command {
            "/help" => self.note(HELP),
            "/new" => self.new_session(),
            "/sessions" => self.open_sessions(),
            "/settings" => self.view = View::Settings,
            "/profile" | "/setup" => self.view = View::Profile,
            "/thinking" => {
                self.apply_setting("thinking", "", false);
                self.note(format!("thinking mode {} (saved)", if self.settings.thinking { "on" } else { "off" }));
            }
            "/tools" => {
                self.send(json!({ "type": "tools" }));
            }
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
            self.send(json!({ "type": "reset" }));
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
            self.send(json!({ "type": "load", "messages": messages }));
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
            let mut file = serde_json::to_value(Settings::load_or_create(path)?)?;
            file[key] = serde_json::to_value(&effective)?[key].clone();
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

    fn on_key(&mut self, key: KeyEvent) {
        // AltGr arrives as Ctrl+Alt on Windows; only plain Ctrl is a shortcut.
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL) && !key.modifiers.contains(KeyModifiers::ALT);
        if ctrl && matches!(key.code, KeyCode::Char('q') | KeyCode::Char('c')) {
            self.quit = true;
            return;
        }
        if self.confirm.is_some() {
            let answer = match key.code {
                KeyCode::Char('y' | 'Y' | 'i' | 'I') => "y",
                KeyCode::Char('a' | 'A') => "a",
                KeyCode::Char('n' | 'N') | KeyCode::Esc => "n",
                _ => return,
            };
            self.confirm = None;
            self.send(json!({ "type": "approval", "answer": answer }));
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
        match key.code {
            KeyCode::Enter => self.submit(),
            KeyCode::Char('n') if ctrl => self.new_session(),
            KeyCode::Char('t') if ctrl => self.show_thinking = !self.show_thinking,
            KeyCode::Char('u') if ctrl => {
                self.input.clear();
                self.cursor = 0;
            }
            KeyCode::Char(c) if !ctrl => {
                let at = self.byte_at(self.cursor);
                self.input.insert(at, c);
                self.cursor += 1;
            }
            KeyCode::Backspace if self.cursor > 0 => {
                self.cursor -= 1;
                let at = self.byte_at(self.cursor);
                self.input.remove(at);
            }
            KeyCode::Delete if self.cursor < len => {
                let at = self.byte_at(self.cursor);
                self.input.remove(at);
            }
            KeyCode::Left => self.cursor = self.cursor.saturating_sub(1),
            KeyCode::Right => self.cursor = (self.cursor + 1).min(len),
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = len,
            KeyCode::PageUp => self.scroll += 10,
            KeyCode::PageDown => self.scroll = self.scroll.saturating_sub(10),
            KeyCode::Up => self.scroll += 1,
            KeyCode::Down => self.scroll = self.scroll.saturating_sub(1),
            KeyCode::Esc => {
                self.input.clear();
                self.cursor = 0;
            }
            KeyCode::F(2) => self.open_sessions(),
            KeyCode::F(3) => self.view = View::Settings,
            KeyCode::F(4) => self.view = View::Profile,
            _ => {}
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
            Link::Ready if self.busy => (ic.pick("⏳", "*"), ui::TOOL, format!("working · {} s", self.busy_since.elapsed().as_secs())),
            Link::Ready => (ic.pick("🟢", "*"), ui::OK, "ready".to_string()),
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
            View::Chat => &[("Enter", "send"), ("Ctrl+N", "new chat"), ("Ctrl+T", "thinking"), ("PgUp/PgDn", "scroll"), ("/help", "commands"), ("Ctrl+Q", "quit")],
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
        lines.push(Line::styled("  F1 chat · F2 saved chats · F3 settings · F4 profile · /help for all commands", ui::dim()));
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
        let [list_area, note_area] = Layout::vertical([Constraint::Min(3), Constraint::Length(4)]).areas(area);
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

        let mut lines = vec![Line::styled(format!("saved in {SETTINGS_FILE}; settings marked restart apply after the server restarts (r)"), ui::dim())];
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
            (_, View::Chat) if self.busy => (ic.tagged("⏳", "working · you can type the next message"), self.input.as_str(), self.cursor, ui::TOOL),
            (_, View::Chat) => (ic.tagged("💬", "message"), self.input.as_str(), self.cursor, ui::USER),
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
            f.render_widget(Paragraph::new(Line::styled("Ask anything… (/help lists the commands)", ui::dim().add_modifier(Modifier::ITALIC))).block(block), area);
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
            .title(Span::styled(format!(" {} ", ic.tagged("🔧", "Run this tool?")), ui::bold(ui::TOOL)));
        f.render_widget(Paragraph::new(lines).wrap(Wrap { trim: false }).block(block), rect);
    }

    fn event_loop(&mut self, terminal: &mut DefaultTerminal) -> Result<()> {
        while !self.quit {
            terminal.draw(|f| self.draw(f))?;
            let mut wait = Duration::from_millis(50);
            // Handle every pending key (a paste arrives as many) before drawing again.
            while event::poll(wait)? {
                if let TermEvent::Key(key) = event::read()? {
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
            while let Ok(msg) = self.rx.try_recv() {
                self.on_msg(msg);
            }
        }
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
    serde_json::to_value(settings).ok().and_then(|v| v.get(key).cloned()).unwrap_or(Value::Null)
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

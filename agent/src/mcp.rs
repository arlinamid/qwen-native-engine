//! A minimal MCP client over the stdio transport (newline-delimited JSON-RPC 2.0).
//!
//! It speaks both protocol eras: it first probes with `server/discover` and, if the server
//! supports revision 2026-07-28, sends the per-request `_meta` fields that revision requires.
//! Otherwise it falls back to the `initialize` handshake of revision 2025-11-25.

use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    io::{BufReader, Read, Write},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex, mpsc::{self, Receiver, RecvTimeoutError, SyncSender}},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use serde_json::{Value, json};
use crate::runtime::{read_json, MAX_FRAME, TurnControl, UnknownToolOutcome};

const MODERN: &str = "2026-07-28";
const LEGACY: &str = "2025-11-25";
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// The `mcpServers` file format used by Claude Desktop and other clients.
#[derive(Deserialize)]
pub struct Config {
    #[serde(rename = "mcpServers")]
    pub servers: BTreeMap<String, ServerConfig>,
}

#[derive(Deserialize)]
pub struct ServerConfig {
    pub command: String,
    #[serde(default)]
    pub optional: bool,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Only these tools are offered to the model (every tool when absent): tool schemas cost
    /// prompt tokens, and the context is small.
    #[serde(default)]
    pub tools: Option<Vec<String>>,
}

fn default_timeout() -> u64 { 30 }

#[derive(Debug)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

pub struct Server {
    pub name: String,
    child: Child,
    outgoing: SyncSender<(Value, mpsc::Sender<std::io::Result<()>>)>,
    diagnostics: Arc<Mutex<VecDeque<String>>>,
    healthy: bool,
    timeout: Duration,
    incoming: Receiver<Value>,
    next_id: u64,
    /// The negotiated modern version, or `None` after a legacy `initialize`.
    modern: Option<String>,
    pub protocol: String,
    pub instructions: Option<String>,
}

fn spawn(command: &str, config: &ServerConfig) -> std::io::Result<Child> {
    let mut cmd = Command::new(command);
    #[cfg(windows)] { use std::os::windows::process::CommandExt; cmd.creation_flags(0x08000000); }
    #[cfg(unix)] { use std::os::unix::process::CommandExt; cmd.process_group(0); }
    cmd
        .args(&config.args)
        .envs(&config.env)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        // Servers log to stderr; it is not part of the protocol.
        .stderr(Stdio::piped())
        .spawn()
}

impl Server {
    pub fn start(name: &str, config: &ServerConfig) -> Result<Self> {
        let mut child = match spawn(&config.command, config) {
            // npx, uvx and similar launchers are .cmd scripts on Windows.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound && cfg!(windows) => spawn(&format!("{}.cmd", config.command), config),
            other => other,
        }
        .with_context(|| format!("starting MCP server '{name}' ({})", config.command))?;
        let mut stdin = child.stdin.take().context("no stdin")?;
        let stdout = child.stdout.take().context("no stdout")?;
        let mut stderr = child.stderr.take().context("no stderr")?;
        let diagnostics = Arc::new(Mutex::new(VecDeque::new()));
        let log = diagnostics.clone();
        thread::spawn(move || {
            let mut buf = [0u8; 1024];
            while let Ok(n) = stderr.read(&mut buf) {
                if n == 0 { break; }
                let mut log = log.lock().unwrap();
                if log.len() == 16 { log.pop_front(); }
                log.push_back(String::from_utf8_lossy(&buf[..n]).into_owned());
            }
        });
        let (outgoing, writes) = mpsc::sync_channel::<(Value, mpsc::Sender<std::io::Result<()>>)>(4);
        thread::spawn(move || {
            while let Ok((message, done)) = writes.recv() {
                let result = writeln!(stdin, "{message}").and_then(|_| stdin.flush());
                let failed = result.is_err();
                let _ = done.send(result);
                if failed { break; }
            }
        });
        let (tx, incoming) = mpsc::sync_channel(64);
        thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            while let Ok(Some(message)) = read_json(&mut reader, MAX_FRAME) {
                // Overflow is a protocol failure, never unbounded buffering.
                if tx.try_send(message).is_err() { break; }
            }
        });
        let mut server = Self { name: name.into(), child, outgoing, incoming, next_id: 1,
            modern: None, protocol: String::new(), instructions: None, diagnostics,
            healthy: true, timeout: Duration::from_secs(config.timeout_secs.clamp(1, 120)) };
        server.handshake()?;
        Ok(server)
    }

    fn send(&mut self, message: &Value) -> Result<()> {
        if message.to_string().len() > MAX_FRAME { bail!("outgoing MCP frame too large"); }
        let (done, result) = mpsc::channel();
        self.outgoing.try_send((message.clone(), done)).map_err(|_| anyhow!("MCP writer unavailable"))?;
        result.recv_timeout(Duration::from_secs(2)).context("MCP write timed out")??;
        Ok(())
    }

    fn terminate(&mut self) {
        self.healthy = false;
        #[cfg(windows)] {
            use std::os::windows::process::CommandExt;
            let _ = Command::new("taskkill").args(["/PID", &self.child.id().to_string(), "/T", "/F"])
                .creation_flags(0x08000000).stdout(Stdio::null()).stderr(Stdio::null()).status();
        }
        #[cfg(unix)] {
            let _ = Command::new("kill").args(["-TERM", "--", &format!("-{}", self.child.id())])
                .stdout(Stdio::null()).stderr(Stdio::null()).status();
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    fn meta(&self, version: &str) -> Value {
        json!({
            "io.modelcontextprotocol/protocolVersion": version,
            "io.modelcontextprotocol/clientInfo": { "name": "qwen-agent", "version": env!("CARGO_PKG_VERSION") },
            "io.modelcontextprotocol/clientCapabilities": {},
        })
    }

    /// Sends a request and waits for its response: `Ok(result)` or `Err(error object)`.
    fn call(&mut self, method: &str, mut params: Value, timeout: Duration, control: Option<&TurnControl>) -> Result<std::result::Result<Value, Value>> {
        if !self.healthy { bail!("MCP server {} is unavailable; restart it before further calls", self.name); }
        let id = self.next_id;
        self.next_id += 1;
        if let Some(version) = self.modern.clone() {
            params["_meta"] = self.meta(&version);
        }
        if let Some(control) = control { control.check()?; }
        self.send(&json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }))?;
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(control) = control { control.check()?; }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() { bail!("{} timed out waiting for {method}", self.name); }
            let message = match self.incoming.recv_timeout(left.min(Duration::from_millis(50))) {
                Ok(m) => m,
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => bail!("{} exited or sent invalid/oversized data: {}", self.name, self.diagnostics.lock().unwrap().iter().cloned().collect::<String>()),
            };
            // A request from the server (ping, roots, sampling): answer so it does not wait.
            if let (Some(method), Some(req_id)) = (message.get("method").and_then(Value::as_str), message.get("id")) {
                let reply = if method == "ping" {
                    json!({ "jsonrpc": "2.0", "id": req_id, "result": {} })
                } else {
                    json!({ "jsonrpc": "2.0", "id": req_id, "error": { "code": -32601, "message": "not supported by this client" } })
                };
                self.send(&reply)?;
                continue;
            }
            if message.get("id").and_then(Value::as_u64) != Some(id) {
                continue; // a notification, or a late answer to an earlier request
            }
            if message.get("jsonrpc").and_then(Value::as_str) != Some("2.0") || message.get("result").is_some() == message.get("error").is_some() {
                bail!("invalid JSON-RPC response from {}", self.name);
            }
            if let Some(error) = message.get("error") {
                return Ok(Err(error.clone()));
            }
            return Ok(Ok(message.get("result").cloned().unwrap_or(Value::Null)));
        }
    }

    fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        self.call(method, params, self.timeout, None)?
            .map_err(|e| anyhow!("'{}' {method} failed: {}", self.name, e.get("message").and_then(Value::as_str).unwrap_or("error")))
    }

    fn handshake(&mut self) -> Result<()> {
        self.modern = Some(MODERN.into());
        let probe = self.call("server/discover", json!({}), PROBE_TIMEOUT, None);
        if let Ok(Ok(result)) = &probe {
            let versions = result.get("supportedVersions").and_then(Value::as_array);
            if versions.is_some_and(|v| v.iter().any(|x| x == MODERN)) {
                self.protocol = MODERN.into();
                self.instructions = result.get("instructions").and_then(Value::as_str).map(String::from);
                return Ok(());
            }
            bail!("{} is a modern MCP server with no supported protocol version", self.name);
        }
        if let Ok(Err(error)) = &probe {
            if error.get("code").and_then(Value::as_i64) == Some(-32022) {
                bail!("{} reports an unsupported modern protocol version: {}", self.name, error);
            }
        }
        // Any other answer, or none, means a server of the initialize era.
        self.modern = None;
        let result = self.request(
            "initialize",
            json!({
                "protocolVersion": LEGACY,
                "capabilities": {},
                "clientInfo": { "name": "qwen-agent", "version": env!("CARGO_PKG_VERSION") },
            }),
        )?;
        self.protocol = result.get("protocolVersion").and_then(Value::as_str).unwrap_or(LEGACY).to_string();
        if ![LEGACY, "2025-06-18", "2025-03-26"].contains(&self.protocol.as_str()) { bail!("unsupported negotiated MCP version {}", self.protocol); }
        self.instructions = result.get("instructions").and_then(Value::as_str).map(String::from);
        self.send(&json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }))
    }

    pub fn list_tools(&mut self) -> Result<Vec<Tool>> {
        let mut tools = Vec::new();
        let mut cursor: Option<String> = None;
        let mut cursors = HashSet::new();
        let mut names = HashSet::new();
        for _ in 0..32 {
            let params = match &cursor {
                Some(c) => json!({ "cursor": c }),
                None => json!({}),
            };
            let result = self.request("tools/list", params)?;
            for tool in result.get("tools").and_then(Value::as_array).into_iter().flatten() {
                let Some(name) = tool.get("name").and_then(Value::as_str) else { continue };
                if name.is_empty() || name.len() > 128 || !names.insert(name.to_owned()) { bail!("invalid or duplicate MCP tool name"); }
                if tools.len() >= 128 || tool.to_string().len() > 32 * 1024 { bail!("MCP tool registry exceeds limits"); }
                tools.push(Tool {
                    name: name.to_string(),
                    description: tool.get("description").and_then(Value::as_str).unwrap_or("").to_string(),
                    input_schema: tool.get("inputSchema").cloned().unwrap_or_else(|| json!({ "type": "object", "properties": {} })),
                });
            }
            match result.get("nextCursor").and_then(Value::as_str) {
                Some(next) if !next.is_empty() => {
                    if !cursors.insert(next.to_owned()) { bail!("MCP server repeated a pagination cursor"); }
                    cursor = Some(next.to_string());
                },
                _ => return Ok(tools),
            }
        }
        bail!("MCP tool list exceeds 32 pages")
    }

    /// Calls a tool and returns its text output and whether the tool reported an error.
    pub fn call_tool(&mut self, name: &str, arguments: &Value, control: &TurnControl) -> Result<(String, bool)> {
        if !self.healthy { bail!("MCP server unavailable; call was not dispatched"); }
        control.check()?;
        let response = self.call("tools/call", json!({ "name": name, "arguments": arguments }), self.timeout, Some(control));
        let result = match response {
            Ok(Ok(result)) => result,
            Ok(Err(error)) => return Ok((error.get("message").and_then(Value::as_str).unwrap_or("tool call failed").to_string(), true)),
            Err(error) => {
                self.terminate();
                return Err(UnknownToolOutcome(format!("{} / {name}: {error:#}", self.name)).into());
            }
        };
        parse_tool_result(&result, self.modern.is_some()).map_err(|e| UnknownToolOutcome(e.to_string()).into())
    }
}

fn parse_tool_result(result: &Value, modern: bool) -> Result<(String, bool)> {
        if !result.is_object() { bail!("invalid MCP tool result"); }
        match result.get("resultType").and_then(Value::as_str) {
            Some("complete") => {},
            None if !modern => {},
            Some("input_required") => return Ok(("Tool requires additional input; this client cannot complete that request. The operation is not confirmed as completed.".into(), true)),
            _ => bail!("unsupported or missing MCP resultType; operation is not confirmed as completed"),
        }
        if let Some(flag) = result.get("isError") { if !flag.is_boolean() { bail!("invalid MCP isError flag"); } }
        if let Some(content) = result.get("content") { if !content.is_array() { bail!("invalid MCP content"); } }
        let mut parts = Vec::new();
        for block in result.get("content").and_then(Value::as_array).into_iter().flatten() {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => parts.push(block.get("text").and_then(Value::as_str).unwrap_or("").to_string()),
                Some("resource") => {
                    let resource = block.get("resource");
                    let text = resource.and_then(|r| r.get("text")).and_then(Value::as_str);
                    let uri = resource.and_then(|r| r.get("uri")).and_then(Value::as_str).unwrap_or("");
                    parts.push(text.map_or_else(|| format!("[resource {uri}]"), String::from));
                }
                Some("resource_link") => parts.push(format!("[link {}]", block.get("uri").and_then(Value::as_str).unwrap_or(""))),
                Some(other) => parts.push(format!("[{other} content omitted]")),
                None => {}
            }
        }
        {
            if let Some(structured) = result.get("structuredContent") {
                parts.push(structured.to_string());
            }
        }
        let is_error = result.get("isError").and_then(Value::as_bool).unwrap_or(false);
        if parts.is_empty() && result.get("content").is_none() { bail!("MCP tool result contains no content"); }
        Ok((parts.join("\n"), is_error))
}

impl Drop for Server {
    fn drop(&mut self) { if self.healthy || self.child.try_wait().ok().flatten().is_none() { self.terminate(); } }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(mode: &str) -> ServerConfig {
        ServerConfig { command:"python".into(), args:vec!["-I".into(), format!("{}/../tests/mcp_fixture.py",env!("CARGO_MANIFEST_DIR")),mode.into()], env:HashMap::new(), tools:None, optional:false, timeout_secs:1 }
    }
    #[test] fn stdio_modern_and_legacy_and_cursor_limits() {
        for mode in ["modern","legacy"] {
            let mut server=Server::start("fixture",&fixture(mode)).unwrap();
            assert_eq!(server.protocol,if mode=="modern" {MODERN} else {LEGACY});
            assert_eq!(server.call_tool("test",&json!({}),&TurnControl::new(Duration::from_secs(5))).unwrap(),("42".into(),false));
        }
        let mut server=Server::start("fixture",&fixture("cursor")).unwrap();
        assert!(server.list_tools().unwrap_err().to_string().contains("cursor"));
        assert!(Server::start("fixture",&fixture("version")).is_err());
    }
    #[test] fn timeout_crash_and_oversized_reply_disable_the_peer_without_retry() {
        for mode in ["hang","exit","oversize"] {
            let mut server=Server::start("fixture",&fixture(mode)).unwrap();
            let started=Instant::now();
            let error=server.call_tool("test",&json!({}),&TurnControl::new(Duration::from_secs(5))).unwrap_err();
            assert!(error.downcast_ref::<UnknownToolOutcome>().is_some(),"{error}");
            assert!(!server.healthy);
            assert!(started.elapsed()<Duration::from_secs(5));
            assert!(server.call_tool("test",&json!({}),&TurnControl::new(Duration::from_secs(5))).is_err());
        }
    }
    #[test] fn incomplete_and_malformed_results_never_report_success() {
        let (text, error) = parse_tool_result(&json!({"resultType":"input_required","inputRequests":{}}), true).unwrap();
        assert!(error && text.contains("not confirmed"));
        for result in [json!(null), json!({}), json!({"resultType":"future"}), json!({"resultType":"complete","isError":"false"})] {
            assert!(parse_tool_result(&result, true).is_err());
        }
    }
    #[test] fn structured_results_and_legacy_text_are_preserved() {
        let (text, error) = parse_tool_result(&json!({"resultType":"complete","content":[{"type":"text","text":"answer"}],"structuredContent":{"value":42}}), true).unwrap();
        assert!(!error && text.contains("42") && text.contains("answer"));
        assert_eq!(parse_tool_result(&json!({"content":[{"type":"text","text":"ok"}]}), false).unwrap(), ("ok".into(),false));
    }
}

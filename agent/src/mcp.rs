//! A minimal MCP client over the stdio transport (newline-delimited JSON-RPC 2.0).
//!
//! It speaks both protocol eras: it first probes with `server/discover` and, if the server
//! supports revision 2026-07-28, sends the per-request `_meta` fields that revision requires.
//! Otherwise it falls back to the `initialize` handshake of revision 2025-11-25.

use std::{
    collections::{BTreeMap, HashMap},
    io::{BufRead, BufReader, Write},
    process::{Child, ChildStdin, Command, Stdio},
    sync::mpsc::{self, Receiver, RecvTimeoutError},
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use serde::Deserialize;
use serde_json::{Value, json};

const MODERN: &str = "2026-07-28";
const LEGACY: &str = "2025-11-25";
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

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
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Only these tools are offered to the model (every tool when absent): tool schemas cost
    /// prompt tokens, and the context is small.
    #[serde(default)]
    pub tools: Option<Vec<String>>,
}

pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

pub struct Server {
    pub name: String,
    child: Child,
    stdin: ChildStdin,
    incoming: Receiver<Value>,
    next_id: u64,
    /// The negotiated modern version, or `None` after a legacy `initialize`.
    modern: Option<String>,
    pub protocol: String,
    pub instructions: Option<String>,
}

fn spawn(command: &str, config: &ServerConfig) -> std::io::Result<Child> {
    Command::new(command)
        .args(&config.args)
        .envs(&config.env)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        // Servers log to stderr; it is not part of the protocol.
        .stderr(Stdio::null())
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
        let stdin = child.stdin.take().context("no stdin")?;
        let stdout = child.stdout.take().context("no stdout")?;
        let (tx, incoming) = mpsc::channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                if let Ok(message) = serde_json::from_str::<Value>(&line) {
                    if tx.send(message).is_err() {
                        break;
                    }
                }
            }
        });
        let mut server = Self { name: name.into(), child, stdin, incoming, next_id: 1, modern: None, protocol: String::new(), instructions: None };
        server.handshake()?;
        Ok(server)
    }

    fn send(&mut self, message: &Value) -> Result<()> {
        writeln!(self.stdin, "{message}")?;
        self.stdin.flush()?;
        Ok(())
    }

    fn meta(&self, version: &str) -> Value {
        json!({
            "io.modelcontextprotocol/protocolVersion": version,
            "io.modelcontextprotocol/clientInfo": { "name": "qwen-agent", "version": env!("CARGO_PKG_VERSION") },
            "io.modelcontextprotocol/clientCapabilities": {},
        })
    }

    /// Sends a request and waits for its response: `Ok(result)` or `Err(error object)`.
    fn call(&mut self, method: &str, mut params: Value, timeout: Duration) -> Result<std::result::Result<Value, Value>> {
        let id = self.next_id;
        self.next_id += 1;
        if let Some(version) = self.modern.clone() {
            params["_meta"] = self.meta(&version);
        }
        self.send(&json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }))?;
        let deadline = Instant::now() + timeout;
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            let message = match self.incoming.recv_timeout(left) {
                Ok(m) => m,
                Err(RecvTimeoutError::Timeout) => bail!("'{}' did not answer {method} within {timeout:?}", self.name),
                Err(RecvTimeoutError::Disconnected) => bail!("'{}' exited", self.name),
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
            if let Some(error) = message.get("error") {
                return Ok(Err(error.clone()));
            }
            return Ok(Ok(message.get("result").cloned().unwrap_or(Value::Null)));
        }
    }

    fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        self.call(method, params, REQUEST_TIMEOUT)?
            .map_err(|e| anyhow!("'{}' {method} failed: {}", self.name, e.get("message").and_then(Value::as_str).unwrap_or("error")))
    }

    fn handshake(&mut self) -> Result<()> {
        self.modern = Some(MODERN.into());
        let probe = self.call("server/discover", json!({}), PROBE_TIMEOUT);
        if let Ok(Ok(result)) = &probe {
            let versions = result.get("supportedVersions").and_then(Value::as_array);
            if versions.is_some_and(|v| v.iter().any(|x| x == MODERN)) {
                self.protocol = MODERN.into();
                self.instructions = result.get("instructions").and_then(Value::as_str).map(String::from);
                return Ok(());
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
        self.instructions = result.get("instructions").and_then(Value::as_str).map(String::from);
        self.send(&json!({ "jsonrpc": "2.0", "method": "notifications/initialized" }))
    }

    pub fn list_tools(&mut self) -> Result<Vec<Tool>> {
        let mut tools = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            let params = match &cursor {
                Some(c) => json!({ "cursor": c }),
                None => json!({}),
            };
            let result = self.request("tools/list", params)?;
            for tool in result.get("tools").and_then(Value::as_array).into_iter().flatten() {
                let Some(name) = tool.get("name").and_then(Value::as_str) else { continue };
                tools.push(Tool {
                    name: name.to_string(),
                    description: tool.get("description").and_then(Value::as_str).unwrap_or("").to_string(),
                    input_schema: tool.get("inputSchema").cloned().unwrap_or_else(|| json!({ "type": "object", "properties": {} })),
                });
            }
            match result.get("nextCursor").and_then(Value::as_str) {
                Some(next) if !next.is_empty() => cursor = Some(next.to_string()),
                _ => return Ok(tools),
            }
        }
    }

    /// Calls a tool and returns its text output and whether the tool reported an error.
    pub fn call_tool(&mut self, name: &str, arguments: &Value) -> Result<(String, bool)> {
        let result = match self.call("tools/call", json!({ "name": name, "arguments": arguments }), REQUEST_TIMEOUT)? {
            Ok(result) => result,
            // Protocol errors (unknown tool, bad params) go back to the model as tool errors.
            Err(error) => return Ok((error.get("message").and_then(Value::as_str).unwrap_or("tool call failed").to_string(), true)),
        };
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
        if parts.is_empty() {
            if let Some(structured) = result.get("structuredContent") {
                parts.push(structured.to_string());
            }
        }
        let is_error = result.get("isError").and_then(Value::as_bool).unwrap_or(false);
        Ok((parts.join("\n"), is_error))
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

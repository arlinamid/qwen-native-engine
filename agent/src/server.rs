//! Background server: keeps the model, its GPU state and the MCP servers alive between CLI runs.
//!
//! It listens on 127.0.0.1 only, one client at a time (there is one GPU engine), and speaks
//! newline-delimited JSON. A client must first send the random token the server writes to
//! `.qwen-agent/server.json`; tool calls are approved by the client, over the connection.
//!
//! Client -> server: hello {token}, turn {text, options}, approval {answer}, load {messages},
//!                   reset, tools, shutdown.
//! Server -> client: ready {tools, system_tokens, settings}, the agent's events (text, tool_call,
//!                   tool_result, rejected, stats, done, step_limit), confirm {name, arguments},
//!                   end {messages}, tools {list}, bye, error.

use std::{
    cell::RefCell,
    hash::{BuildHasher, Hasher},
    io::{BufRead, BufReader, Write},
    net::{TcpListener, TcpStream},
    path::Path,
    time::{Duration, SystemTime},
};

use anyhow::{Context, Result, anyhow};
use qwen_wasm::prompt::Message;
use serde_json::{Value, json};

use crate::{
    McpTools, event_json,
    session::{Agent, Approval, Event},
    settings::Settings,
};

fn random_token() -> String {
    // RandomState is seeded from the OS; two of them give 128 random bits without a crate.
    let mut parts = [0u64; 2];
    for (i, part) in parts.iter_mut().enumerate() {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u128(SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
        h.write_u32(std::process::id());
        h.write_usize(i);
        *part = h.finish();
    }
    format!("{:016x}{:016x}", parts[0], parts[1])
}

pub fn serve(mut agent: Agent, mut runner: McpTools, settings: &Settings, server_file: &Path) -> Result<()> {
    let listener = TcpListener::bind(("127.0.0.1", settings.port)).with_context(|| format!("port {} is in use (another server?)", settings.port))?;
    // What the system prompt was built from, so clients can tell when the profile changed since.
    let profile = crate::profile::render();
    // Warm the cache with the system prompt and tool list, so even the first client starts hot.
    agent.prime()?;
    // Published only now, so a client that finds the file gets a server that answers at once.
    let token = random_token();
    if let Some(dir) = server_file.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(server_file, serde_json::to_string_pretty(&json!({ "port": settings.port, "token": token, "pid": std::process::id() }))?)?;
    eprintln!("qwen-agent server ready on 127.0.0.1:{} (Ctrl+C or `qwen-agent stop` stops it)", settings.port);

    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("connection failed: {e}");
                continue;
            }
        };
        let shutdown = handle(stream, &mut agent, &mut runner, &token, settings, &profile).unwrap_or_else(|e| {
            eprintln!("client: {e:#}");
            false
        });
        // Each client gets a fresh conversation; the model and the cached system prompt stay.
        agent.reset();
        agent.forget_approvals();
        if shutdown {
            break;
        }
    }
    let _ = std::fs::remove_file(server_file);
    eprintln!("server stopped");
    Ok(())
}

fn send(writer: &RefCell<TcpStream>, message: &Value) -> std::io::Result<()> {
    let mut w = writer.borrow_mut();
    writeln!(w, "{message}")?;
    w.flush()
}

fn end(agent: &Agent) -> Value {
    json!({ "type": "end", "messages": agent.messages })
}

/// Serves one client; returns whether it asked the server to shut down.
fn handle(stream: TcpStream, agent: &mut Agent, runner: &mut McpTools, token: &str, settings: &Settings, profile: &str) -> Result<bool> {
    // A connection that never says hello must not hold the only engine.
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    let writer = RefCell::new(stream.try_clone()?);
    let reader = RefCell::new(BufReader::new(stream));
    let read = || -> Result<Option<Value>> {
        let mut line = String::new();
        if reader.borrow_mut().read_line(&mut line)? == 0 {
            return Ok(None);
        }
        Ok(Some(serde_json::from_str(&line).map_err(|_| anyhow!("not a JSON message"))?))
    };

    let hello = read()?.ok_or_else(|| anyhow!("closed before hello"))?;
    if hello.get("type").and_then(Value::as_str) != Some("hello") || hello.get("token").and_then(Value::as_str) != Some(token) {
        send(&writer, &json!({ "type": "error", "message": "bad token" }))?;
        return Err(anyhow!("rejected a client with a wrong token"));
    }
    send(
        &writer,
        &json!({ "type": "ready", "tools": agent.tools.len(), "system_tokens": agent.system_tokens()?, "settings": settings.server_part(), "profile": profile }),
    )?;
    // From here the client may wait on a person (approvals, the next message).
    reader.borrow().get_ref().set_read_timeout(None)?;

    while let Some(message) = read()? {
        match message.get("type").and_then(Value::as_str) {
            Some("turn") => {
                let text = message.get("text").and_then(Value::as_str).unwrap_or("").to_string();
                // The client's settings apply to this turn only.
                let saved = (agent.auto_approve, agent.thinking, agent.sampling.clone(), agent.max_steps, agent.max_new_tokens);
                let options = message.get("options").cloned().unwrap_or(Value::Null);
                let flag = |k: &str, default: bool| options.get(k).and_then(Value::as_bool).unwrap_or(default);
                let count = |k: &str, default: usize| options.get(k).and_then(Value::as_u64).map_or(default, |n| n.clamp(1, 4096) as usize);
                agent.auto_approve = flag("auto_approve", agent.auto_approve);
                agent.thinking = flag("thinking", agent.thinking);
                agent.sampling.temperature = options.get("temperature").and_then(Value::as_f64).map_or(agent.sampling.temperature, |t| t.clamp(0.0, 2.0) as f32);
                agent.sampling.presence_penalty =
                    options.get("presence_penalty").and_then(Value::as_f64).map_or(agent.sampling.presence_penalty, |p| p.clamp(0.0, 2.0) as f32);
                agent.max_steps = count("max_steps", agent.max_steps);
                agent.max_new_tokens = count("max_tokens", agent.max_new_tokens);

                let mut confirm = |call: &qwen_wasm::toolcall::ToolCall| {
                    if send(&writer, &json!({ "type": "confirm", "name": call.name, "arguments": call.arguments })).is_err() {
                        return Approval::No;
                    }
                    match read() {
                        Ok(Some(reply)) => match reply.get("answer").and_then(Value::as_str) {
                            Some("y") => Approval::Yes,
                            Some("a") => Approval::Always,
                            _ => Approval::No,
                        },
                        _ => Approval::No,
                    }
                };
                let mut out = |event: Event| {
                    let _ = send(&writer, &event_json(event));
                };
                if let Err(e) = agent.turn(&text, runner, &mut confirm, &mut out) {
                    send(&writer, &json!({ "type": "error", "message": format!("{e:#}") }))?;
                }
                (agent.auto_approve, agent.thinking, agent.sampling, agent.max_steps, agent.max_new_tokens) = saved;
                send(&writer, &end(agent))?;
            }
            Some("load") => {
                // A saved conversation; the system message is always the server's current one.
                match serde_json::from_value::<Vec<Message>>(message.get("messages").cloned().unwrap_or(json!([]))) {
                    Ok(mut messages) => {
                        if !messages.is_empty() {
                            if messages[0].role == "system" {
                                messages[0].content = agent.system.clone();
                            } else {
                                messages.insert(0, Message::new("system", agent.system.clone()));
                            }
                        }
                        agent.messages = messages;
                    }
                    Err(e) => send(&writer, &json!({ "type": "error", "message": format!("bad saved conversation: {e}") }))?,
                }
                send(&writer, &end(agent))?;
            }
            Some("reset") => {
                agent.reset();
                send(&writer, &end(agent))?;
            }
            Some("tools") => {
                let list: Vec<Value> = agent
                    .tools
                    .iter()
                    .map(|t| json!({ "name": t["function"]["name"], "description": t["function"]["description"] }))
                    .collect();
                send(&writer, &json!({ "type": "tools", "list": list }))?;
            }
            Some("shutdown") => {
                send(&writer, &json!({ "type": "bye" }))?;
                return Ok(true);
            }
            _ => send(&writer, &json!({ "type": "error", "message": "unknown message" }))?,
        }
    }
    Ok(false)
}

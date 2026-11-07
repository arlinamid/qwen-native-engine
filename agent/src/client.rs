//! Client side of the background server: connecting, and the line-mode chat (`--plain`,
//! `--prompt`). The TUI uses `open` and `Connection` too.

use std::{
    io::{BufRead, BufReader, Write},
    net::TcpStream,
    path::Path,
    time::Duration,
};

use anyhow::{Context, Result, anyhow};
use serde_json::{Value, json};

use crate::{Args, ask_approval, print_event, repl, settings::Settings};

pub struct Connection {
    pub writer: TcpStream,
    pub reader: BufReader<TcpStream>,
}

impl Connection {
    pub fn send(&mut self, message: &Value) -> Result<()> {
        send_to(&mut self.writer, message)
    }

    pub fn read(&mut self) -> Result<Value> {
        let mut line = String::new();
        if self.reader.read_line(&mut line)? == 0 {
            return Err(anyhow!("the server closed the connection"));
        }
        Ok(serde_json::from_str(&line)?)
    }
}

pub fn send_to(writer: &mut TcpStream, message: &Value) -> Result<()> {
    writeln!(writer, "{message}")?;
    writer.flush()?;
    Ok(())
}

/// Connects to the server published in `server_file` and returns the connection and its
/// `ready` message. `Ok(None)` when no server runs; a stale file is removed.
pub fn open(server_file: &Path) -> Result<Option<(Connection, Value)>> {
    let Ok(text) = std::fs::read_to_string(server_file) else { return Ok(None) };
    let info: Option<(u16, String)> = serde_json::from_str::<Value>(&text)
        .ok()
        .and_then(|v| Some((v.get("port")?.as_u64()? as u16, v.get("token")?.as_str()?.to_string())));
    let stream = info.as_ref().and_then(|(port, _)| TcpStream::connect_timeout(&([127, 0, 0, 1], *port).into(), Duration::from_secs(2)).ok());
    let (Some((_, token)), Some(stream)) = (info, stream) else {
        let _ = std::fs::remove_file(server_file);
        return Ok(None);
    };
    let mut connection = Connection { writer: stream.try_clone()?, reader: BufReader::new(stream) };
    connection.send(&json!({ "type": "hello", "token": token }))?;
    // The server serves one client at a time; another chat holds it until it exits.
    connection.reader.get_ref().set_read_timeout(Some(Duration::from_secs(5)))?;
    let ready = connection.read().map_err(|_| anyhow!("the server is busy with another client"))?;
    connection.reader.get_ref().set_read_timeout(None)?;
    if ready.get("type").and_then(Value::as_str) != Some("ready") {
        return Err(anyhow!("server refused: {}", ready.get("message").and_then(Value::as_str).unwrap_or("?")));
    }
    Ok(Some((connection, ready)))
}

/// `open`, telling the user on stderr what they got.
pub fn connect(server_file: &Path, settings: &Settings) -> Result<Option<Connection>> {
    let Some((connection, ready)) = open(server_file)? else { return Ok(None) };
    eprintln!(
        "connected to the qwen-agent server: {} tools, system prompt {} tokens (warm)",
        ready.get("tools").and_then(Value::as_u64).unwrap_or(0),
        ready.get("system_tokens").and_then(Value::as_u64).unwrap_or(0)
    );
    if ready.get("settings") != Some(&settings.server_part()) {
        eprintln!("note: the server was started with other model/MCP/system settings; `qwen-agent stop` and restart it to apply yours");
    }
    if ready.get("profile").and_then(Value::as_str).is_some_and(|p| p != crate::profile::render()) {
        eprintln!("note: the profile changed since the server started; `qwen-agent stop` and restart it to apply it");
    }
    Ok(Some(connection))
}

pub fn stop(server_file: &Path) -> Result<()> {
    let Some((mut connection, _)) = open(server_file)? else {
        eprintln!("no server is running");
        return Ok(());
    };
    connection.send(&json!({ "type": "shutdown" }))?;
    connection.read().context("waiting for the server to stop")?;
    eprintln!("server stopped");
    Ok(())
}

fn turn(connection: &mut Connection, text: &str, settings: &Settings) -> Result<()> {
    connection.send(&json!({ "type": "turn", "text": text, "options": settings.turn_options() }))?;
    loop {
        let event = connection.read()?;
        match event.get("type").and_then(Value::as_str) {
            Some("end") => return Ok(()),
            Some("confirm") => {
                let name = event.get("name").and_then(Value::as_str).unwrap_or("?");
                let answer = ask_approval(name, event.get("arguments").unwrap_or(&Value::Null));
                connection.send(&json!({ "type": "approval", "answer": answer }))?;
            }
            _ => print_event(&event),
        }
    }
}

pub fn run(mut connection: Connection, args: &Args) -> Result<()> {
    if let Some(prompt) = &args.prompt {
        turn(&mut connection, prompt, &args.settings)?;
        println!();
        return Ok(());
    }
    repl(|text| match text {
        "/reset" => {
            connection.send(&json!({ "type": "reset" }))?;
            connection.read()?;
            eprintln!("conversation cleared");
            Ok(())
        }
        "/tools" => {
            connection.send(&json!({ "type": "tools" }))?;
            let reply = connection.read()?;
            for t in reply.get("list").and_then(Value::as_array).into_iter().flatten() {
                eprintln!("  {}: {}", t["name"].as_str().unwrap_or(""), t["description"].as_str().unwrap_or(""));
            }
            Ok(())
        }
        text => turn(&mut connection, text, &args.settings),
    })
}

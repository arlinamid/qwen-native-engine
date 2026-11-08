//! qwen-agent: an agent on the native Qwen3.5-0.8B engine, using tools from MCP servers.
//!
//!   qwen-agent                                     the TUI; starts the background server if needed
//!   qwen-agent serve                               keep the model warm in a background server
//!   qwen-agent stop                                stop the background server
//!   qwen-agent --prompt "What time is it?" --yes   one request (through the server if it runs)
//!   qwen-agent --plain / --local                   line chat; --local loads the model here
//!
//! Defaults come from `.qwen-agent/settings.json` (written on first run); flags override them
//! for one run. Every tool call needs approval unless auto-approve is on or the tool was
//! approved with "always" earlier in the session.

mod agentic;
mod runtime;
mod persistence;
mod client;
mod mcp;
mod profile;
mod server;
mod session;
mod settings;
mod skills;
mod store;
mod tui;
mod ui;

use std::{
    collections::HashMap,
    io::{BufRead, IsTerminal, Write},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result, bail};
use qwen_kernels::model::{Model, ModelTiles};
use serde_json::{Value, json};
use session::{Agent, Approval, Engine, Event, ToolRunner};
use settings::{SETTINGS_FILE, Settings};
use tokenizers::Tokenizer;

/// Where a running server publishes its port and access token.
pub const SERVER_FILE: &str = ".qwen-agent/server.json";

#[derive(PartialEq, Clone, Copy)]
enum Mode {
    Auto,
    Tui,
    Serve,
    Stop,
    Local,
    Plain,
    Setup,
}

pub struct Args {
    mode: Mode,
    pub settings: Settings,
    prompt: Option<String>,
    resume: bool,
    json_output: bool,
}

fn usage() -> &'static str {
    "usage: qwen-agent [serve | stop | tui | setup] [--prompt TEXT] [--plain] [--local] [--resume]\n\
     \x20      settings for this run: [--mcp FILE | --no-mcp] [--model 0.8b|2b] [--model-dir DIR] [--kernel-tiles FILE] [--dxc PATH] [--system TEXT]\n\
     \x20      [--port N] [--yes] [--thinking] [--greedy] [--temperature T] [--presence-penalty P] [--max-steps N] [--max-tokens N]\n\
     defaults: .qwen-agent/settings.json"
}

fn parse_args() -> Result<Args> {
    let mut settings = Settings::load_or_create(Path::new(SETTINGS_FILE))?;
    profile::ensure_templates()?;
    skills::ensure_agents_md()?;
    skills::ensure_starters()?;
    let mut mode = Mode::Auto;
    let mut prompt = None;
    let mut resume = false;
    let mut json_output = false;
    let mut it = std::env::args().skip(1).peekable();
    let sub = match it.peek().map(String::as_str) {
        Some("serve") => Some(Mode::Serve),
        Some("stop") => Some(Mode::Stop),
        Some("tui") => Some(Mode::Tui),
        Some("setup") => Some(Mode::Setup),
        _ => None,
    };
    if let Some(sub) = sub {
        mode = sub;
        it.next();
    }
    while let Some(flag) = it.next() {
        let mut value = || it.next().with_context(|| format!("{flag} needs a value\n{}", usage()));
        match flag.as_str() {
            "--mcp" => settings.mcp = value()?,
            "--no-mcp" => settings.mcp = String::new(),
            "--model" => { settings = settings.select_model(&value()?)?; settings.check_model_files()?; },
            "--model-dir" => settings.model_dir = value()?,
            "--kernel-tiles" => settings.kernel_tiles = value()?,
            "--dxc" => settings.dxc = value()?,
            "--system" => settings.system = value()?,
            "--mcp-guide" => settings.mcp_guide = true,
            "--no-mcp-guide" => settings.mcp_guide = false,
            "--port" => settings.port = value()?.parse()?,
            "--temperature" => settings.temperature = value()?.parse()?,
            "--presence-penalty" => settings.presence_penalty = value()?.parse()?,
            "--max-steps" => settings.max_steps = value()?.parse()?,
            "--max-tokens" => settings.max_tokens = value()?.parse()?,
            "--yes" => settings.auto_approve = true,
            "--thinking" => settings.thinking_mode = crate::agentic::ThinkingMode::On,
            "--greedy" => settings.temperature = 0.0,
            "--prompt" => prompt = Some(value()?),
            "--resume" => resume = true,
            "--json" => json_output = true,
            // --local is line mode too, so --plain adds nothing to it; subcommands win over both.
            "--local" => {
                if matches!(mode, Mode::Auto | Mode::Plain) {
                    mode = Mode::Local;
                }
            }
            "--plain" => {
                if mode == Mode::Auto {
                    mode = Mode::Plain;
                }
            }
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            other => bail!("unknown flag {other}\n{}", usage()),
        }
    }
    settings.check()?;
    Ok(Args { mode, settings, prompt, resume, json_output })
}

/// Routes tool names to MCP servers. Names that two servers share get a `server__` prefix.
pub struct McpTools {
    servers: Vec<mcp::Server>,
    routes: HashMap<String, (usize, String)>,
}

impl ToolRunner for McpTools {
    fn run(&mut self, name: &str, arguments: &Value, control: &runtime::TurnControl) -> Result<(String, bool)> {
        let Some((server, original)) = self.routes.get(name).cloned() else {
            return Ok((format!("unknown tool '{name}'"), true));
        };
        self.servers[server].call_tool(&original, arguments, control)
    }
}

fn start_mcp(path: &Path) -> Result<(McpTools, Vec<Value>)> {
    let config: mcp::Config = serde_json::from_str(&std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?)
        .with_context(|| format!("{} is not an mcpServers config", path.display()))?;
    let mut servers = Vec::new();
    let mut listed = Vec::new();
    for (name, server_config) in &config.servers {
        let started = mcp::Server::start(name, server_config).and_then(|mut server| { let tools = server.list_tools()?; Ok((server, tools)) });
        let (server, mut tools) = match started {
            Ok(value) => value,
            Err(e) if server_config.optional => { eprintln!("optional MCP server {name} unavailable: {e:#}"); continue; }
            Err(e) => return Err(e),
        };
        tools.sort_by(|a,b| a.name.cmp(&b.name));
        if let Some(allowed) = &server_config.tools {
            tools.retain(|t| allowed.contains(&t.name));
        }
        eprintln!("mcp: {name} ({}): {} tools", server.protocol, tools.len());
        listed.push((servers.len(), tools));
        servers.push(server);
    }

    let mut counts: HashMap<String, usize> = HashMap::new();
    for (_, tools) in &listed {
        for t in tools {
            *counts.entry(t.name.clone()).or_default() += 1;
        }
    }
    let mut routes = HashMap::new();
    let mut prompt_tools = Vec::new();
    for (index, tools) in listed {
        for tool in tools {
            if let Err(e) = qwen_wasm::schema::check(&tool.input_schema) { eprintln!("MCP tool {} disabled: {e}",tool.name); continue; }
            if tool.input_schema.get("type").and_then(Value::as_str) != Some("object") { eprintln!("MCP tool {} disabled: root schema must be an object",tool.name); continue; }
            let name = if counts[&tool.name] > 1 { format!("{}__{}", servers[index].name, tool.name) } else { tool.name.clone() };
            let mut parameters = tool.input_schema;
            // The schema URL costs prompt tokens and tells the model nothing.
            if let Some(obj) = parameters.as_object_mut() {
                obj.remove("$schema");
            }
            prompt_tools.push(json!({ "type": "function", "function": { "name": name, "description": tool.description, "parameters": parameters } }));
            if routes.insert(name.clone(), (index, tool.name)).is_some() { bail!("MCP tool name collision after namespacing: {name}"); }
        }
    }
    Ok((McpTools { servers, routes }, prompt_tools))
}

/// Explains MCP to the model, generated from the connected servers: their names, their tools and
/// the servers' own `instructions`, plus how to use MCP tools in general. Nothing is specific to
/// one server. The tool schemas themselves are in the template's `<tools>` block.
fn mcp_guide(runner: &McpTools) -> Option<String> {
    if runner.routes.is_empty() {
        return None;
    }
    let mut by_server: Vec<Vec<&str>> = vec![Vec::new(); runner.servers.len()];
    for (name, (index, _)) in &runner.routes {
        by_server[*index].push(name);
    }
    let mut text = String::from(
        "## MCP\nYou are connected to MCP (Model Context Protocol) servers. Every function in <tools> above is an MCP tool you can call, provided by one of these servers:",
    );
    for (server, names) in runner.servers.iter().zip(&mut by_server) {
        if names.is_empty() {
            continue;
        }
        names.sort();
        text.push_str(&format!("\n- {}: {}", server.name, names.join(", ")));
        if let Some(instructions) = server.instructions.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            let short: String = instructions.chars().take(400).collect();
            text.push_str(&format!(". {short}"));
        }
    }
    text.push_str(
        "\nCall an MCP tool whenever the answer needs live data, files or exact computation, and wait for its result before you answer; never make up a tool result. \
         If a call is declined or fails, say so instead of answering as if it ran. \
         When asked which tools or MCP servers you have, name each tool and say what it does.",
    );
    Some(text)
}

/// Loads the model and starts the MCP servers: the slow part a background server does once.
fn build_agent(settings: &Settings) -> Result<(Agent, McpTools)> {
    // Native Vulkan on this class of Intel driver crashes at device creation; D3D12 works.
    if cfg!(windows) && std::env::var_os("WGPU_BACKEND").is_none() {
        unsafe { std::env::set_var("WGPU_BACKEND", "dx12") };
    }
    let dxc = match settings.dxc.as_str() {
        "" => None,
        path => match std::fs::canonicalize(path) {
            Ok(path) => {
                if let Some(dir) = path.parent() {
                    let old = std::env::var_os("PATH").unwrap_or_default();
                    let joined = std::env::join_paths(std::iter::once(dir.to_path_buf()).chain(std::env::split_paths(&old)))?;
                    unsafe { std::env::set_var("PATH", joined) };
                }
                Some(path.to_string_lossy().trim_start_matches(r"\\?\").to_string())
            }
            Err(_) => {
                eprintln!("warning: DXC not found at {path}; using the slower f32 kernels");
                None
            }
        },
    };

    let (runner, tools) = match settings.mcp.as_str() {
        "" => (McpTools { servers: Vec::new(), routes: HashMap::new() }, Vec::new()),
        path => start_mcp(Path::new(path))?,
    };

    let t0 = Instant::now();
    let model_dir = std::path::absolute(&settings.model_dir).unwrap_or_else(|_| PathBuf::from(&settings.model_dir));
    let index = std::fs::read_to_string(model_dir.join("native/index.json")).with_context(|| {
        format!(
            "no converted model in {} (run scripts/convert_weights.py, set model_dir in {SETTINGS_FILE}, or point QWEN_AGENT_HOME at the project folder)",
            model_dir.display()
        )
    })?;
    let weights = std::fs::read(model_dir.join("native/weights.bin")).with_context(|| format!("reading {}", model_dir.join("native/weights.bin").display()))?;
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|e| anyhow::anyhow!("{}: {e}", model_dir.join("tokenizer.json").display()))?;
    // The f16 kernels need DXC on D3D12; without it the f32 kernels still work.
    let tiles = if !settings.kernel_tiles.is_empty() {
        serde_json::from_str(&std::fs::read_to_string(&settings.kernel_tiles).context("reading kernel tile profile")?).context("invalid kernel tile profile")?
    } else if dxc.is_some() { ModelTiles::fast() } else { ModelTiles::f32() };
    let model = pollster::block_on(Model::load(&index, &weights, tiles, true, dxc.as_deref()))?;
    drop(weights);
    let engine = Engine::new(model, tokenizer)?;
    // System prompt: configured text, profile, AGENTS.md, skill catalog, then MCP.
    let mut system = settings.system.trim_end().to_string();
    let profile = profile::render();
    if !profile.is_empty() {
        system.push_str("\n\n");
        system.push_str(&profile);
    }
    let agents = skills::render_agents_md();
    if !agents.is_empty() {
        system.push_str("\n\n");
        system.push_str(&agents);
    }
    let skill_catalog = skills::load_catalog(Path::new(".")).unwrap_or_else(|e| {
        eprintln!("warning: skill catalog: {e:#}");
        Vec::new()
    });
    let skill_prompt = skills::catalog_prompt(&skill_catalog);
    if !skill_prompt.is_empty() {
        system.push_str("\n\n");
        system.push_str(&skill_prompt);
    }
    if let Some(guide) = mcp_guide(&runner).filter(|_| settings.mcp_guide) {
        system.push_str("\n\n");
        system.push_str(&guide);
    }
    let mut agent = Agent::new(engine, tools, system);
    agent.skills = skill_catalog;
    agent.thinking_mode = settings.thinking_mode;
    agent.max_steps = settings.max_steps;
    agent.max_new_tokens = settings.max_tokens;
    agent.auto_approve = settings.auto_approve;
    agent.turn_timeout_secs = settings.turn_timeout_secs;
    agent.sampling.temperature = settings.temperature;
    agent.sampling.presence_penalty = settings.presence_penalty;
    eprintln!("model loaded in {:.1} s; {} tools, system prompt {} tokens", t0.elapsed().as_secs_f64(), agent.tools.len(), agent.system_tokens()?);
    Ok((agent, runner))
}

/// Events as JSON lines: the server's wire format, and what every mode prints or shows.
pub fn event_json(event: Event) -> Value {
    match event {
        Event::Thinking(text) => json!({ "type": "thinking", "text": text }),
        Event::Text(text) => json!({ "type": "text", "text": text }),
        Event::ToolCall { call_id, name, arguments } => json!({ "type": "tool_call", "call_id":call_id, "name": name, "arguments": arguments }),
        Event::ToolResult { call_id, output, is_error } => json!({ "type": "tool_result", "call_id":call_id, "output": output, "is_error": is_error }),
        Event::Rejected { name, reason } => json!({ "type": "rejected", "name": name, "reason": reason }),
        Event::Stats(g) => json!({
            "type": "stats",
            "prompt_tokens": g.prompt_tokens,
            "reused_tokens": g.reused_tokens,
            "new_tokens": g.new_tokens,
            "prefill_ms": g.prefill_ms,
            "decode_ms": g.decode_ms,
        }),
        Event::Done { finished, stop } => json!({ "type": "done", "finished": finished, "stop": stop.as_str() }),
        Event::Phase(phase) => json!({"type":"phase","phase":phase}),
        Event::Context { used, limit, dropped } => json!({"type":"context","used":used,"limit":limit,"dropped_messages":dropped}),
        Event::Checkpoint(messages) => json!({"type":"checkpoint","messages":messages}),
        Event::Finished(status) => json!({"type":"finished","status":status}),
        Event::StepLimit => json!({ "type": "step_limit" }),
    }
}

/// "prompt 534 tokens (505 reused) in 0.9 s, 32 tokens at 16.6 tok/s"
pub fn stats_line(event: &Value) -> String {
    let n = |key: &str| event.get(key).and_then(Value::as_f64).unwrap_or(0.0);
    let tokens = n("new_tokens");
    let tps = if tokens > 1.0 { (tokens - 1.0) / (n("decode_ms") / 1000.0) } else { 0.0 };
    format!(
        "prompt {} tokens ({} reused) in {:.1} s, {} tokens at {tps:.1} tok/s",
        n("prompt_tokens"),
        n("reused_tokens"),
        n("prefill_ms") / 1000.0,
        tokens
    )
}

pub fn emit_event(event: &Value, json_output: bool) {
    if json_output { println!("{event}"); } else { print_event(event); }
}

pub fn print_event(event: &Value) {
    let s = |key: &str| event.get(key).and_then(Value::as_str).unwrap_or("");
    match s("type") {
        "text" => {
            let mut out = std::io::stdout();
            let _ = write!(out, "{}", s("text"));
            let _ = out.flush();
        }
        // Reasoning goes to stderr, dimmed, so stdout keeps only the answer.
        "thinking" if !std::io::stderr().is_terminal() => eprint!("{}", s("text")),
        "thinking" => eprint!("\x1b[2m{}\x1b[0m", s("text")),
        "tool_call" => eprintln!("\n  -> {} {}", s("name"), event.get("arguments").unwrap_or(&Value::Null)),
        "tool_result" => {
            let output = s("output");
            let first: String = output.lines().next().unwrap_or("").chars().take(160).collect();
            let more = if output.lines().count() > 1 { " ..." } else { "" };
            let error = event.get("is_error").and_then(Value::as_bool).unwrap_or(false);
            eprintln!("  <- {}{first}{more}", if error { "error: " } else { "" });
        }
        "rejected" => eprintln!("  x {}: {}", s("name"), s("reason")),
        "stats" => eprintln!("\n  [{}]", stats_line(event)),
        "done" => {
            if event.get("finished").and_then(Value::as_bool) == Some(false) {
                if event.get("stop").and_then(Value::as_str) == Some("max_tokens") {
                    eprintln!("  (reply cut off by the token limit)");
                } else {
                    eprintln!("  (reply ended without a complete answer)");
                }
            }
        }
        "step_limit" => eprintln!("  (stopped: tool step limit reached)"),
        "finished" if s("status") != "completed" => eprintln!("  (turn ended: {})",s("status")),
        "error" => eprintln!("\nerror: {}", s("message")),
        _ => {}
    }
}

pub fn ask(question: &str) -> String {
    eprint!("{question}");
    let _ = std::io::stderr().flush();
    let mut line = String::new();
    let _ = std::io::stdin().lock().read_line(&mut line);
    line.trim().to_lowercase()
}

/// Reads one answer line as typed (not lowercased).
pub fn ask_raw(question: &str) -> String {
    eprint!("{question}");
    let _ = std::io::stderr().flush();
    let mut line = String::new();
    let _ = std::io::stdin().lock().read_line(&mut line);
    line.trim_end_matches(['\r', '\n']).to_string()
}

pub fn ask_approval(name: &str, arguments: &Value) -> &'static str {
    match ask(&format!("  run {name} {arguments}? [y]es / [n]o / [a]lways: ")).as_str() {
        "y" | "yes" | "i" | "igen" => "y",
        "a" | "always" | "mindig" => "a",
        _ => "n",
    }
}

/// Reads user lines until EOF or /exit and hands each to `handle`.
pub fn repl(mut handle: impl FnMut(&str) -> Result<()>) -> Result<()> {
    eprintln!("type a message; /tools lists tools, /reset starts over, /exit quits");
    loop {
        eprint!("\n> ");
        let _ = std::io::stderr().flush();
        let mut line = String::new();
        if std::io::stdin().lock().read_line(&mut line)? == 0 {
            return Ok(());
        }
        match line.trim() {
            "" => continue,
            "/exit" | "/quit" => return Ok(()),
            text => {
                if let Err(e) = handle(text) {
                    eprintln!("\nerror: {e:#}");
                }
                println!();
            }
        }
    }
}

fn run_local(args: &Args) -> Result<()> {
    let _lease = persistence::model_lease(Path::new(".qwen-agent/model.lock"))?;
    let (mut agent, mut runner) = build_agent(&args.settings)?;
    let mut confirm = |_id: &str, call: &qwen_wasm::toolcall::ToolCall| match ask_approval(&call.name, &call.arguments) {
        "y" => Approval::Yes,
        "a" => Approval::Always,
        _ => Approval::No,
    };
    let mut print = |e: Event| emit_event(&event_json(e), args.json_output);
    if let Some(prompt) = &args.prompt {
        agent.turn(prompt, &mut runner, &mut confirm, &mut print)?;
        if agent.last_status != "completed" { bail!("turn did not complete: {}",agent.last_status); }
        println!();
        return Ok(());
    }
    repl(|text| {
        match text {
            "/reset" => {
                agent.reset();
                eprintln!("conversation cleared");
            }
            "/tools" => {
                for t in &agent.tools {
                    eprintln!("  {}: {}", t["function"]["name"].as_str().unwrap_or(""), t["function"]["description"].as_str().unwrap_or(""));
                }
            }
            text => agent.turn(text, &mut runner, &mut confirm, &mut print)?,
        }
        Ok(())
    })
}

/// The project folder, where the model, `.qwen-agent/` and the MCP configs live: QWEN_AGENT_HOME,
/// else the current folder or the nearest folder above the executable from which the model is
/// reachable. Started from Explorer or from `target/release`, the current folder is not it.
fn find_home() -> Option<PathBuf> {
    if let Some(home) = std::env::var_os("QWEN_AGENT_HOME") {
        return Some(home.into());
    }
    let has_model = |dir: &Path| {
        let model_dir = std::fs::read_to_string(dir.join(SETTINGS_FILE))
            .ok()
            .and_then(|text| serde_json::from_str::<Value>(&text).ok())
            .and_then(|v| v.get("model_dir")?.as_str().map(String::from))
            .filter(|m| !m.is_empty())
            .unwrap_or_else(|| "models/qwen3.5-0.8b".into());
        dir.join(model_dir).join("native/index.json").is_file()
    };
    let cwd = std::env::current_dir().ok()?;
    if has_model(&cwd) {
        return Some(cwd);
    }
    let exe = std::env::current_exe().ok()?;
    exe.ancestors().skip(1).find(|dir| has_model(dir)).map(Path::to_path_buf)
}

fn main() -> Result<()> {
    if let Some(home) = find_home() {
        std::env::set_current_dir(&home).with_context(|| format!("entering {}", home.display()))?;
    }
    let args = parse_args()?;
    let server_file = Path::new(SERVER_FILE);
    match args.mode {
        Mode::Serve => {
            let lease = persistence::model_lease(Path::new(".qwen-agent/model.lock"))?;
            let listener = server::bind(args.settings.port)?;
            let (agent, runner) = build_agent(&args.settings)?;
            server::serve(agent, runner, &args.settings, server_file, listener, lease)
        }
        Mode::Stop => client::stop(server_file),
        Mode::Setup => profile::wizard(),
        Mode::Local => run_local(&args),
        Mode::Tui => tui::run(args.settings, args.resume),
        Mode::Auto if args.prompt.is_none() && std::io::stdin().is_terminal() && std::io::stdout().is_terminal() => {
            tui::run(args.settings, args.resume)
        }
        Mode::Auto | Mode::Plain => match client::connect(server_file, &args.settings) {
            Ok(Some(connection)) => client::run(connection, &args),
            Ok(None) => {
                eprintln!("tip: `qwen-agent serve` keeps the model loaded between runs");
                run_local(&args)
            }
            Err(e) => Err(e.context("could not connect to the existing server; retry or stop it before starting another model")),
        },
    }
}

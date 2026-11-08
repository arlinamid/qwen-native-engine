//! A single inference worker with an independent authenticated control plane.
//! Idle clients never own the model. Cancellation and shutdown remain available during a turn.
use std::{
    collections::HashSet,
    io::{BufReader, Write}, net::{TcpListener, TcpStream, Shutdown}, path::Path,
    sync::{Arc, Mutex, atomic::{AtomicBool, AtomicUsize, Ordering}, mpsc::{self, SyncSender, Receiver}},
    thread, time::Duration,
};
use anyhow::{Result, Context, anyhow, bail};
use serde_json::{Value, json};
use qwen_wasm::{prompt::Message, toolcall::ToolCall};
use crate::{McpTools, event_json, session::{Agent, Approval, Event, Backend}, settings::Settings,
    runtime::{read_json, MAX_FRAME, TurnControl}, persistence::{atomic_write, unique_id, valid_id}, store};

type Writer = Arc<Mutex<TcpStream>>;
fn send(writer: &Writer, message: &Value) -> Result<()> {
    let mut stream = writer.lock().map_err(|_| anyhow!("connection writer failed"))?;
    writeln!(stream, "{message}")?;
    stream.flush()?;
    Ok(())
}
struct Active { id: String, control: TurnControl, approval: SyncSender<Value> }
struct ClientState {
    session_id: String, messages: Vec<Message>, always: HashSet<String>, active: Option<Active>,
}
struct Job {
    state: Arc<Mutex<ClientState>>, writer: Writer, user: String, options: Value,
    messages: Vec<Message>, always: HashSet<String>, control: TurnControl,
    approvals: Receiver<Value>, turn_id: String, session_id: String,
}
struct Shared {
    busy: AtomicBool, shutdown: AtomicBool, clients: AtomicUsize,
    active: Mutex<Option<TurnControl>>, sockets: Mutex<Vec<(String, TcpStream)>>,
    ready: Value, token: String,
}
struct SocketGuard { id: String, shared: Arc<Shared> }
impl Drop for SocketGuard {
    fn drop(&mut self) { self.shared.sockets.lock().unwrap().retain(|(id,_)| *id != self.id); }
}

pub fn bind(port: u16) -> Result<TcpListener> {
    TcpListener::bind(("127.0.0.1", port)).with_context(|| format!("port {port} is already in use"))
}

pub fn serve(agent: Agent, runner: McpTools, settings: &Settings, server_file: &Path, listener: TcpListener, lease: std::fs::File) -> Result<()> {
    serve_with(agent, runner, settings, server_file, listener, lease)
}

fn serve_with<B: Backend, R: crate::session::ToolRunner>(mut agent: Agent<B>, mut runner: R,
    settings: &Settings, server_file: &Path, listener: TcpListener, lease: std::fs::File) -> Result<()> {
    agent.prime()?;
    let token = unique_token();
    let shared = Arc::new(Shared {
        busy: AtomicBool::new(false), shutdown: AtomicBool::new(false), clients: AtomicUsize::new(0),
        active: Mutex::new(None), sockets: Mutex::new(Vec::new()), token: token.clone(),
        ready: json!({"type":"ready", "protocol":1, "tools":agent.tools.len(), "system_tokens":agent.system_tokens()?,
            "settings":settings.server_part(), "profile":crate::profile::render(),
            "tool_list":agent.tools.iter().map(|t| json!({"name":t["function"]["name"],"description":t["function"]["description"]})).collect::<Vec<_>>(),
            "skills":agent.skills.iter().map(|s| json!({"name":s.name,"description":s.description})).collect::<Vec<_>>() }),
    });
    let port = listener.local_addr()?.port();
    atomic_write(server_file, serde_json::to_string_pretty(&json!({"port":port,"token":token,"pid":std::process::id()}))?.as_bytes())?;
    listener.set_nonblocking(true)?;
    let (jobs, pending) = mpsc::sync_channel::<Job>(1);
    let accept_state = shared.clone();
    let acceptor = thread::spawn(move || {
        while !accept_state.shutdown.load(Ordering::Acquire) {
            match listener.accept() {
                Ok((stream, _)) => {
                    if accept_state.clients.fetch_add(1, Ordering::AcqRel) >= 32 {
                        accept_state.clients.fetch_sub(1, Ordering::AcqRel); continue;
                    }
                    let shared = accept_state.clone(); let jobs = jobs.clone();
                    thread::spawn(move || {
                        if let Err(e) = connection(stream, &jobs, &shared) { eprintln!("client: {e:#}"); }
                        shared.clients.fetch_sub(1, Ordering::AcqRel);
                    });
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => thread::sleep(Duration::from_millis(20)),
                Err(e) => { eprintln!("accept: {e}"); thread::sleep(Duration::from_millis(100)); }
            }
        }
    });
    eprintln!("qwen-agent server ready on 127.0.0.1:{port}");
    while !shared.shutdown.load(Ordering::Acquire) {
        match pending.recv_timeout(Duration::from_millis(100)) {
            Ok(job) => {
                run_job(&mut agent, &mut runner, &job, &shared);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {},
            Err(_) => break,
        }
    }
    shared.shutdown.store(true, Ordering::Release);
    for (_, socket) in shared.sockets.lock().unwrap().drain(..) { let _ = socket.shutdown(Shutdown::Both); }
    let _ = acceptor.join();
    // The published file is the signal that another process may load a model. Drop the GPU
    // weights and the lock first; removing the file earlier makes a restart fail the lock
    // while this process is still freeing the previous model.
    drop(agent);
    drop(runner);
    drop(lease);
    let _ = std::fs::remove_file(server_file);
    Ok(())
}

fn unique_token() -> String {
    use std::hash::{BuildHasher, Hasher};
    (0..2).map(|_| { let mut h = std::collections::hash_map::RandomState::new().build_hasher(); h.write(unique_id().as_bytes()); format!("{:016x}",h.finish()) }).collect()
}

fn connection(stream: TcpStream, jobs: &SyncSender<Job>, shared: &Arc<Shared>) -> Result<()> {
    // Windows accepted sockets inherit the listener's nonblocking mode.
    stream.set_nonblocking(false)?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;
    stream.set_nodelay(true)?;
    let writer = Arc::new(Mutex::new(stream.try_clone()?));
    let mut reader = BufReader::new(stream);
    let hello = read_json(&mut reader, MAX_FRAME)?.context("closed before hello")?;
    if hello.get("type").and_then(Value::as_str) != Some("hello") || hello.get("token").and_then(Value::as_str) != Some(&shared.token) {
        bail!("invalid client token");
    }
    let connection_id = unique_id();
    let _socket_guard = SocketGuard { id:connection_id.clone(), shared:shared.clone() };
    shared.sockets.lock().unwrap().push((connection_id.clone(), reader.get_ref().try_clone()?));
    send(&writer, &shared.ready)?;
    reader.get_ref().set_read_timeout(None)?;
    let state = Arc::new(Mutex::new(ClientState { session_id:unique_id(), messages:vec![], always:HashSet::new(), active:None }));
    let result = (|| -> Result<()> {
        while let Some(message) = read_json(&mut reader, MAX_FRAME)? {
            let kind = message.get("type").and_then(Value::as_str).unwrap_or("");
            match kind {
                "shutdown" => {
                    shared.shutdown.store(true, Ordering::Release);
                    if let Some(control) = shared.active.lock().unwrap().as_ref() { control.cancel(); }
                    send(&writer, &json!({"type":"bye"}))?;
                    break;
                }
                "status" => send(&writer, &json!({"type":"status","busy":shared.busy.load(Ordering::Acquire)}))?,
                "cancel" => {
                    if let Some(active) = state.lock().unwrap().active.as_ref() {
                        if message.get("turn_id").and_then(Value::as_str).is_none_or(|id| id == active.id) { active.control.cancel(); }
                    }
                }
                "approval" => {
                    if let Some(active) = state.lock().unwrap().active.as_ref() {
                        if message.get("turn_id").and_then(Value::as_str) == Some(&active.id) {
                            let _ = active.approval.try_send(message);
                        }
                    }
                }
                "tools" => send(&writer, &json!({"type":"tools","list":shared.ready["tool_list"]}))?,
                "reset" | "load" => {
                    let mut state = state.lock().unwrap();
                    if state.active.is_some() { send(&writer, &json!({"type":"error","message":"cancel the current turn first"}))?; continue; }
                    let id = message.get("session_id").and_then(Value::as_str).map(str::to_owned).unwrap_or_else(unique_id);
                    if !valid_id(&id) { bail!("invalid session ID"); }
                    let loaded: Vec<Message> = if kind == "load" {
                        if message.get("recover").and_then(Value::as_bool) == Some(true) {
                            store::recover(&id)?.unwrap_or(serde_json::from_value(message.get("messages").cloned().unwrap_or(json!([])))?)
                        } else { serde_json::from_value(message.get("messages").cloned().unwrap_or(json!([])))? }
                    } else { vec![] };
                    if loaded.len() > 4096 || loaded.iter().any(|m| !matches!(m.role.as_str(),"system"|"user"|"assistant"|"tool")) { bail!("invalid saved history"); }
                    state.session_id = id; state.messages = loaded; state.always.clear();
                    send(&writer, &json!({"type":"end","status":"loaded","messages":state.messages}))?;
                }
                "turn" => {
                    let client_state = state.clone();
                    let mut state = state.lock().unwrap();
                    if state.active.is_some() || shared.busy.compare_exchange(false,true,Ordering::AcqRel,Ordering::Acquire).is_err() {
                        send(&writer, &json!({"type":"busy","message":"another turn is running; retry when it finishes"}))?; continue;
                    }
                    let id = message.get("turn_id").and_then(Value::as_str).filter(|id| valid_id(id)).map(str::to_owned).unwrap_or_else(unique_id);
                    if let Some(session_id) = message.get("session_id").and_then(Value::as_str).filter(|id| valid_id(id)) {
                        if session_id != state.session_id { state.always.clear(); state.session_id = session_id.into(); }
                    }
                    let options = message.get("options").cloned().unwrap_or(Value::Null);
                    let seconds = options.get("turn_timeout_secs").and_then(Value::as_u64).unwrap_or(180).clamp(1,600);
                    let control = TurnControl::new(Duration::from_secs(seconds));
                    let (approval, approvals) = mpsc::sync_channel(8);
                    state.active = Some(Active { id:id.clone(), control:control.clone(), approval });
                    *shared.active.lock().unwrap() = Some(control.clone());
                    let job = Job { state:client_state, writer:writer.clone(),
                        user:message.get("text").and_then(Value::as_str).unwrap_or("").into(), options,
                        messages:std::mem::take(&mut state.messages), always:std::mem::take(&mut state.always),
                        control, approvals, turn_id:id, session_id:state.session_id.clone() };
                    // State is supplied below without holding its guard while executing a job.
                    if jobs.send(job).is_err() { state.active = None; shared.busy.store(false,Ordering::Release); bail!("worker stopped"); }
                }
                _ => send(&writer,&json!({"type":"error","message":"unknown command"}))?,
            }
        }
        Ok(())
    })();
    if let Some(active) = state.lock().unwrap().active.as_ref() { active.control.cancel(); }
    let _ = reader.get_ref().shutdown(Shutdown::Both);
    result
}

fn run_job<B: Backend, R: crate::session::ToolRunner>(agent: &mut Agent<B>, runner: &mut R, job: &Job, shared: &Shared) {
    let saved = (agent.auto_approve, agent.thinking_mode, agent.sampling.clone(), agent.max_steps, agent.max_new_tokens);
    let options = &job.options;
    agent.auto_approve = options.get("auto_approve").and_then(Value::as_bool).unwrap_or(false);
    agent.thinking_mode = options.get("thinking_mode").and_then(Value::as_str).and_then(crate::agentic::ThinkingMode::parse)
        .or_else(|| options.get("thinking").and_then(Value::as_bool).map(|on| if on { crate::agentic::ThinkingMode::On } else { crate::agentic::ThinkingMode::Auto }))
        .unwrap_or(crate::agentic::ThinkingMode::Auto);
    agent.max_steps = options.get("max_steps").and_then(Value::as_u64).unwrap_or(6).clamp(1,16) as usize;
    agent.max_new_tokens = options.get("max_tokens").and_then(Value::as_u64).unwrap_or(512).clamp(1,2048) as usize;
    agent.sampling.temperature = options.get("temperature").and_then(Value::as_f64).unwrap_or(0.6).clamp(0.0,2.0) as f32;
    agent.sampling.presence_penalty = options.get("presence_penalty").and_then(Value::as_f64).unwrap_or(1.5).clamp(0.0,2.0) as f32;
    let mut seq = 0u64;
    let mut storage_error = None;
    let result = agent.load_messages(job.messages.clone()).and_then(|_| {
        agent.always = job.always.clone();
        let mut confirm = |call_id: &str, call: &ToolCall| {
            if send(&job.writer, &json!({"type":"confirm", "turn_id":job.turn_id, "call_id":call_id,"name":call.name,"arguments":call.arguments})).is_err() { job.control.cancel(); return Approval::No; }
            loop {
                if job.control.check().is_err() { return Approval::No; }
                match job.approvals.recv_timeout(Duration::from_millis(50)) {
                    Ok(reply) if reply.get("call_id").and_then(Value::as_str) == Some(call_id) => return match reply.get("answer").and_then(Value::as_str) { Some("y")=>Approval::Yes, Some("a")=>Approval::Always,_=>Approval::No },
                    Ok(_) | Err(mpsc::RecvTimeoutError::Timeout) => {},
                    Err(_) => { job.control.cancel(); return Approval::No; }
                }
            }
        };
        let mut out = |event: Event| {
            if let Event::Checkpoint(messages) = &event {
                if let Err(e) = store::checkpoint(&job.session_id, &job.turn_id, "interrupted", messages) {
                    storage_error = Some(format!("checkpoint failed: {e:#}")); job.control.cancel();
                }
            }
            let mut value = event_json(event); seq += 1;
            value["turn_id"] = json!(job.turn_id); value["event_seq"] = json!(seq);
            if send(&job.writer,&value).is_err() { job.control.cancel(); }
        };
        agent.turn_controlled(&job.user,runner,&mut confirm,&mut out,&job.control)
    });
    if let Some(error) = storage_error { let _ = send(&job.writer,&json!({"type":"error","message":error})); }
    if let Err(error) = result { let _ = send(&job.writer,&json!({"type":"error","message":format!("{error:#}")})); }
    if let Err(error) = store::checkpoint(&job.session_id,&job.turn_id,&agent.last_status,&agent.messages) {
        agent.last_status = "failed".into(); let _ = send(&job.writer,&json!({"type":"error","message":format!("checkpoint failed: {error:#}")}));
    }
    { let mut state = job.state.lock().unwrap(); state.messages = agent.messages.clone(); state.always = agent.always.clone(); }
    job.state.lock().unwrap().active = None;
    *shared.active.lock().unwrap() = None;
    shared.busy.store(false, Ordering::Release);
    let _ = send(&job.writer,&json!({"type":"end","turn_id":job.turn_id,"status":agent.last_status,"messages":agent.messages}));
    agent.reset();
    (agent.auto_approve, agent.thinking_mode, agent.sampling, agent.max_steps, agent.max_new_tokens) = saved;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::tests::{Fake, Runner, CALL, tools};
    fn peer(port:u16, token:&str) -> crate::client::Connection {
        let stream=TcpStream::connect(("127.0.0.1",port)).unwrap();
        stream.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let mut c=crate::client::Connection { writer:stream.try_clone().unwrap(),reader:BufReader::new(stream) };
        c.send(&json!({"type":"hello","token":token})).unwrap();
        assert_eq!(c.read().unwrap()["type"],"ready"); c
    }
    fn until(c:&mut crate::client::Connection, kind:&str) -> Value {
        loop { let value=c.read().unwrap(); if value["type"]==kind { return value; } }
    }
    #[test] fn idle_clients_do_not_block_control_and_approval_can_be_cancelled() {
        let dir=std::env::temp_dir().join(unique_id()); std::fs::create_dir_all(&dir).unwrap();
        let file=dir.join("server.json"); let published=file.clone();
        let lock_path=dir.join("model.lock");
        let listener=bind(0).unwrap(); let port=listener.local_addr().unwrap().port();
        let thread=thread::spawn(move || {
            let lease=crate::persistence::model_lease(&lock_path).unwrap();
            let mut agent=Agent::new(Fake::new(&[CALL,"answer",CALL]),tools(),"test".into());
            agent.max_new_tokens=32;
            serve_with(agent,Runner {calls:0,unknown:false},&Settings::default(),&published,listener,lease).unwrap();
        });
        let started=std::time::Instant::now();
        while !file.exists() { assert!(started.elapsed()<Duration::from_secs(5)); thread::sleep(Duration::from_millis(10)); }
        let token=serde_json::from_slice::<Value>(&std::fs::read(&file).unwrap()).unwrap()["token"].as_str().unwrap().to_owned();
        let mut first=peer(port,&token); let mut second=peer(port,&token);
        let session_id=unique_id();
        first.send(&json!({"type":"turn","session_id":session_id,"turn_id":"test-turn","text":"read","options":{"max_tokens":32}})).unwrap();
        let confirm=until(&mut first,"confirm");
        second.send(&json!({"type":"status"})).unwrap(); assert_eq!(second.read().unwrap()["busy"],true);
        second.send(&json!({"type":"turn","text":"busy"})).unwrap(); assert_eq!(second.read().unwrap()["type"],"busy");
        first.send(&json!({"type":"approval","turn_id":"test-turn","call_id":"wrong","answer":"y"})).unwrap();
        assert_eq!(confirm["call_id"],"1");
        first.send(&json!({"type":"cancel","turn_id":"test-turn"})).unwrap();
        assert_eq!(until(&mut first,"end")["status"],"cancelled");
        second.send(&json!({"type":"shutdown"})).unwrap(); assert_eq!(second.read().unwrap()["type"],"bye");
        thread.join().unwrap();
        assert!(store::recover(&session_id).unwrap().is_some());
        std::fs::remove_file(format!(".qwen-agent/recovery/{session_id}.json")).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn shutdown_drops_the_model_lock_before_the_server_file_disappears() {
        let dir = std::env::temp_dir().join(unique_id());
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("server.json");
        let published = file.clone();
        let lock_path = dir.join("model.lock");
        let held = lock_path.clone();
        let listener = bind(0).unwrap();
        let port = listener.local_addr().unwrap().port();
        let thread = thread::spawn(move || {
            let lease = crate::persistence::model_lease(&held).unwrap();
            let agent = Agent::new(Fake::new(&["ok"]), tools(), "test".into());
            serve_with(agent, Runner { calls: 0, unknown: false }, &Settings::default(), &published, listener, lease).unwrap();
        });
        let started = std::time::Instant::now();
        while !file.exists() {
            assert!(started.elapsed() < Duration::from_secs(5), "server file was not published");
            thread::sleep(Duration::from_millis(10));
        }
        let token = serde_json::from_slice::<Value>(&std::fs::read(&file).unwrap()).unwrap()["token"].as_str().unwrap().to_owned();
        let mut client = peer(port, &token);
        client.send(&json!({"type":"shutdown"})).unwrap();
        assert_eq!(client.read().unwrap()["type"], "bye");
        let started = std::time::Instant::now();
        loop {
            if !file.exists() {
                let again = crate::persistence::model_lease(&lock_path);
                assert!(again.is_ok(), "server file was removed while the previous process still held the model lock");
                drop(again);
                break;
            }
            assert!(started.elapsed() < Duration::from_secs(5), "server did not finish shutting down");
            thread::sleep(Duration::from_millis(10));
        }
        thread.join().unwrap();
        std::fs::remove_dir_all(dir).unwrap();
    }
}

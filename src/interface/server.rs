//! Web server + WhatsApp bridge — synchronous, thread-per-connection.
//!
//! Serves chat.html on GET /, dispatches POST /api/generate through the Pipe,
//! and spawns the WhatsApp bridge as a subprocess communicating via JSONL.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use crate::core::router::DispatchContext;
use crate::interface::exec;
use crate::interface::term_stream;

// ── Hybrid embed for chat.html ────────────────────────────────────────────────

pub fn get_chat_html() -> String {
    #[cfg(debug_assertions)]
    {
        std::fs::read_to_string("web/chat.html")
            .unwrap_or_else(|_| "<h1>chat.html missing — run from project root</h1>".to_string())
    }
    #[cfg(not(debug_assertions))]
    {
        include_str!("../../web/chat.html").to_string()
    }
}

// ── Web server ────────────────────────────────────────────────────────────────

/// Start the web server. Blocks until killed.
pub fn run(port: u16, model_arg: Option<&str>) {
    let api_key = std::env::var("ANTHROPIC_API_KEY").ok();
    let ctx = Arc::new(Mutex::new(DispatchContext::new(api_key, model_arg)));

    let addr = format!("0.0.0.0:{port}");
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("[olorin] cannot bind {addr}: {e}");
        std::process::exit(1);
    });
    println!("[Olorin] Web UI at http://0.0.0.0:{port}");

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(10)));
        let ctx = ctx.clone();

        std::thread::spawn(move || {
            handle_connection(&mut stream, ctx);
        });
    }
}

fn handle_connection(stream: &mut std::net::TcpStream, ctx: Arc<Mutex<DispatchContext>>) {
    // Read until \r\n\r\n
    let mut buf = [0u8; 8192];
    let mut n = 0;
    loop {
        let r = match stream.read(&mut buf[n..]) {
            Ok(0) | Err(_) => break,
            Ok(r) => r,
        };
        n += r;
        if buf[..n].windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
        if n >= buf.len() {
            break;
        }
    }

    let req = match std::str::from_utf8(&buf[..n]) {
        Ok(s) => s,
        Err(_) => return,
    };

    let first_line = req.lines().next().unwrap_or("");
    let mut parts = first_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path   = parts.next().unwrap_or("/");

    // Strip query string
    let path = path.split('?').next().unwrap_or("/");

    match (method, path) {
        ("GET", "/") => serve_html(stream),
        ("GET", "/api/model") => {
            serve_json(stream, r#"{"name":"olorin","backend":"pipe"}"#);
        }
        ("GET", "/api/system") => {
            let recall_level = ctx.lock().unwrap().recall_level();
            let body = build_system_json(recall_level);
            serve_json(stream, &body);
        }
        ("POST", "/api/generate") => {
            handle_generate(stream, req, &buf[..n], n, ctx);
        }
        ("POST", "/api/command") => {
            handle_command(stream, req, &buf[..n], n, ctx);
        }
        ("POST", "/api/term/open") => {
            term_stream::handle_term_open(stream);
        }
        ("POST", path) if path.starts_with("/api/term/") && path.ends_with("/input") => {
            let id = term_stream::parse_term_id(path);
            term_stream::handle_term_input(stream, req, &buf[..n], n, id);
        }
        ("POST", path) if path.starts_with("/api/term/") && path.ends_with("/resize") => {
            let id = term_stream::parse_term_id(path);
            term_stream::handle_term_resize(stream, req, &buf[..n], n, id);
        }
        ("POST", path) if path.starts_with("/api/term/") && path.ends_with("/close") => {
            let id = term_stream::parse_term_id(path);
            term_stream::handle_term_close(stream, id);
        }
        ("GET", path) if path.starts_with("/api/term/") && path.ends_with("/stream") => {
            let id = term_stream::parse_term_id(path);
            term_stream::handle_term_stream(stream, id);
        }
        _ => {
            let _ = write!(
                stream,
                "HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\nConnection: close\r\n\r\nNot Found"
            );
        }
    }
}

fn serve_html(stream: &mut std::net::TcpStream) {
    let body = get_chat_html();
    let _ = write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(), body
    );
}

pub(crate) fn serve_json(stream: &mut std::net::TcpStream, body: &str) {
    let _ = write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(), body
    );
}

pub(crate) fn read_body(stream: &mut std::net::TcpStream, req: &str, buf: &[u8], n: usize) -> Vec<u8> {
    let content_len = parse_content_length(req);
    let header_end  = req.find("\r\n\r\n").unwrap_or(n) + 4;
    let already     = n.saturating_sub(header_end);
    let mut body_buf = vec![0u8; content_len];
    if already > 0 && already <= content_len {
        body_buf[..already].copy_from_slice(&buf[header_end..n]);
    }
    if already < content_len {
        let _ = stream.read_exact(&mut body_buf[already..]);
    }
    body_buf
}

fn handle_generate(
    stream: &mut std::net::TcpStream,
    req: &str,
    buf: &[u8],
    n: usize,
    ctx: Arc<Mutex<DispatchContext>>,
) {
    let body_bytes = read_body(stream, req, buf, n);
    let body_str   = std::str::from_utf8(&body_bytes).unwrap_or("");
    let prompt     = extract_json_string(body_str, "prompt").unwrap_or_default();

    // SSE headers
    let _ = write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
         Cache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\n\
         Connection: close\r\n\r\n"
    );
    let _ = stream.flush();

    if prompt.is_empty() {
        let _ = write!(stream, "data: [DONE]\n\n");
        return;
    }

    let (tx, rx) = std::sync::mpsc::channel();

    let ctx_clone = ctx.clone();
    let prompt_owned = prompt.to_string();
    let sender = std::thread::spawn(move || {
        let mut guard = ctx_clone.lock().unwrap();
        guard.dispatch_streaming(&prompt_owned, tx);
    });

    let mut token_count: u64 = 0;
    let mut decode_start: Option<std::time::Instant> = None;

    for event in rx {
        match event {
            crate::core::router::StreamEvent::Token(tok) => {
                token_count += 1;
                let start = *decode_start.get_or_insert_with(std::time::Instant::now);
                let elapsed = start.elapsed().as_secs_f64();
                let tps = if token_count > 1 && elapsed > 0.0 {
                    (token_count - 1) as f64 / elapsed
                } else {
                    0.0
                };
                let escaped = escape_json(&tok);
                let _ = write!(stream, "data: {{\"token\":\"{escaped}\",\"tps\":{tps:.1}}}\n\n");
                let _ = stream.flush();
            }
            crate::core::router::StreamEvent::Error(msg) => {
                let escaped = escape_json(&msg);
                let _ = write!(stream, "data: {{\"error\":\"{escaped}\"}}\n\n");
                let _ = stream.flush();
            }
            crate::core::router::StreamEvent::Done { .. } => break,
        }
    }

    let _ = write!(stream, "data: [DONE]\n\n");
    let _ = stream.flush();
    let _ = sender.join();
}

fn handle_command(
    stream: &mut std::net::TcpStream,
    req: &str,
    buf: &[u8],
    n: usize,
    ctx: Arc<Mutex<DispatchContext>>,
) {
    let body_bytes = read_body(stream, req, buf, n);
    let body_str   = std::str::from_utf8(&body_bytes).unwrap_or("");
    let command    = extract_json_string(body_str, "command").unwrap_or_default();

    let (output, success) = if command.is_empty() {
        ("missing command".to_string(), false)
    } else {
        let mut guard = ctx.lock().unwrap();
        let resp      = guard.dispatch(&command);
        let ok        = !resp.blocked;
        (resp.text, ok)
    };

    let escaped = escape_json(&output);
    let body    = format!("{{\"output\":\"{escaped}\",\"success\":{success}}}");
    serve_json(stream, &body);
}

// ── WhatsApp bridge ───────────────────────────────────────────────────────────

/// Start the WhatsApp bridge subprocess and run the JSONL message loop.
pub fn run_whatsapp(model_arg: Option<&str>) {
    let bridge_path = find_bridge();

    let home        = std::env::var("HOME").unwrap_or_default();
    let session_dir = format!("{home}/.olorin/wa_session");
    std::fs::create_dir_all(&session_dir).ok();

    let child = match exec::spawn(&[&bridge_path, "--session-dir", &session_dir]) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[olorin] failed to start WhatsApp bridge: {e}");
            std::process::exit(1);
        }
    };

    let api_key = std::env::var("ANTHROPIC_API_KEY").ok();
    let ctx     = Arc::new(Mutex::new(DispatchContext::new(api_key, model_arg)));

    eprintln!("[olorin] WhatsApp bridge started (pid={})", child.pid);
    eprintln!("[olorin] Waiting for bridge connection...");

    // Pull fds out before forgetting Child so Drop doesn't close them
    let pid       = child.pid;
    let stdout_fd = child.stdout_fd;
    let stdin_fd  = child.stdin_fd;
    std::mem::forget(child);

    wa_message_loop(stdout_fd, stdin_fd, pid, ctx);
}

fn wa_message_loop(
    stdout_fd: i32,
    stdin_fd:  i32,
    pid:       i32,
    ctx:       Arc<Mutex<DispatchContext>>,
) {
    let mut line_buf = String::new();
    let mut byte     = [0u8; 1];

    loop {
        // Read one line from bridge stdout
        line_buf.clear();
        loop {
            let n = unsafe {
                libc::read(stdout_fd, byte.as_mut_ptr() as *mut libc::c_void, 1)
            };
            if n <= 0 {
                eprintln!("[olorin] Bridge stdout closed.");
                unsafe { libc::waitpid(pid, std::ptr::null_mut(), 0) };
                unsafe { libc::close(stdout_fd); libc::close(stdin_fd); }
                return;
            }
            if byte[0] == b'\n' { break; }
            line_buf.push(byte[0] as char);
        }

        if line_buf.is_empty() { continue; }

        let msg_type = extract_json_string(&line_buf, "type").unwrap_or_default();
        match msg_type.as_str() {
            "connected" => {
                eprintln!("[olorin] WhatsApp connected!");
            }
            "message" => {
                let text = extract_json_string(&line_buf, "text").unwrap_or_default();
                let jid  = extract_json_string(&line_buf, "jid").unwrap_or_default();
                if text.is_empty() || jid.is_empty() { continue; }

                let response = {
                    let mut guard = ctx.lock().unwrap();
                    guard.dispatch(&text)
                };

                let reply_text = escape_json(&response.text);
                let reply_jid  = escape_json(&jid);
                let reply = format!(
                    "{{\"type\":\"send\",\"jid\":\"{reply_jid}\",\"text\":\"{reply_text}\"}}\n"
                );
                let bytes = reply.as_bytes();
                let mut written = 0;
                while written < bytes.len() {
                    let n = unsafe {
                        libc::write(
                            stdin_fd,
                            bytes[written..].as_ptr() as *const libc::c_void,
                            bytes.len() - written,
                        )
                    };
                    if n <= 0 { break; }
                    written += n as usize;
                }
            }
            _ => {}
        }
    }
}

pub fn find_bridge() -> String {
    if let Ok(p) = std::env::var("OLORIN_BRIDGE") {
        return p;
    }
    if let Ok(exe) = std::env::current_exe() {
        let candidate = exe.parent()
            .map(|p| p.join("bridge/wa-bridge"))
            .unwrap_or_default();
        if candidate.exists() {
            return candidate.to_string_lossy().into_owned();
        }
    }
    "bridge/wa-bridge".to_string()
}

// ── System info ───────────────────────────────────────────────────────────────

pub fn build_system_json(recall_level: usize) -> String {
    let (mem_used, mem_total) = read_memory().unwrap_or((0, 0));
    let uptime   = read_uptime().unwrap_or(0);
    let os       = std::env::consts::OS;
    let arch     = std::env::consts::ARCH;
    let cpu_temp = match read_cpu_temp() {
        Some(t) => t.to_string(),
        None    => "null".to_string(),
    };
    let cpu_percent = read_cpu_percent().unwrap_or(0);
    format!(
        "{{\"cpu_percent\":{cpu_percent},\"cpu_temp\":{cpu_temp},\
         \"memory_used_mb\":{mem_used},\"memory_total_mb\":{mem_total},\
         \"os\":\"{os}\",\"arch\":\"{arch}\",\"uptime_seconds\":{uptime},\
         \"recall_level\":{recall_level}}}"
    )
}

fn read_memory() -> Option<(u64, u64)> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total_kb = 0u64;
    let mut avail_kb = 0u64;
    for line in s.lines() {
        if line.starts_with("MemTotal:") {
            total_kb = line.split_whitespace().nth(1)?.parse().ok()?;
        } else if line.starts_with("MemAvailable:") {
            avail_kb = line.split_whitespace().nth(1)?.parse().ok()?;
        }
    }
    Some((total_kb / 1024 - avail_kb / 1024, total_kb / 1024))
}

fn read_cpu_temp() -> Option<u32> {
    let s = std::fs::read_to_string("/sys/class/thermal/thermal_zone0/temp").ok()?;
    Some(s.trim().parse::<u32>().ok()? / 1000)
}

fn parse_proc_stat() -> Option<(u64, u64)> {
    let s = std::fs::read_to_string("/proc/stat").ok()?;
    let line = s.lines().find(|l| l.starts_with("cpu "))?;
    let vals: Vec<u64> = line.split_whitespace().skip(1)
        .filter_map(|v| v.parse().ok()).collect();
    if vals.len() < 4 { return None; }
    let total: u64 = vals.iter().sum();
    let idle = vals[3];
    Some((total, idle))
}

fn read_cpu_percent() -> Option<u32> {
    use std::sync::Mutex;
    static PREV: Mutex<(u64, u64, u32)> = Mutex::new((0, 0, 0));
    let (t2, i2) = parse_proc_stat()?;
    let mut prev = PREV.lock().ok()?;
    let (t1, i1, last_pct) = *prev;
    let dt = t2.saturating_sub(t1);
    let di = i2.saturating_sub(i1);
    *prev = if dt > 0 {
        (t2, i2, (100 * (dt - di) / dt) as u32)
    } else {
        (t2, i2, last_pct)
    };
    Some(prev.2)
}

fn read_uptime() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/uptime").ok()?;
    Some(s.split_whitespace().next()?.parse::<f64>().ok()? as u64)
}

// ── JSON helpers ──────────────────────────────────────────────────────────────

pub fn parse_content_length(req: &str) -> usize {
    for line in req.lines() {
        if line.to_ascii_lowercase().starts_with("content-length:") {
            return line[15..].trim().parse().unwrap_or(0);
        }
    }
    0
}

pub fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern   = format!("\"{}\"", key);
    let start     = json.find(&pattern)?;
    let after_key = &json[start + pattern.len()..];
    let colon     = after_key.find(':')?;
    let rest      = after_key[colon + 1..].trim_start();
    if !rest.starts_with('"') { return None; }
    let mut result = String::new();
    let mut chars  = rest[1..].chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => if let Some(esc) = chars.next() {
                match esc {
                    '"'  => result.push('"'),
                    '\\' => result.push('\\'),
                    'n'  => result.push('\n'),
                    't'  => result.push('\t'),
                    c    => { result.push('\\'); result.push(c); }
                }
            },
            '"' => break,
            _   => result.push(c),
        }
    }
    Some(result)
}

pub(crate) fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"'  => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            _    => out.push(c),
        }
    }
    out
}



//! Web server + WhatsApp bridge — synchronous, thread-per-connection.
//!
//! Serves chat.html on GET /, dispatches POST /api/generate through the Pipe,
//! and spawns the WhatsApp bridge as a subprocess communicating via JSONL.

use std::io::{Read, Write};
use std::fmt::Write as FmtWrite;
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use crate::core::router::DispatchContext;
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
pub fn run(port: u16, model_arg: Option<&str>, draft_arg: Option<&str>, draft_k: Option<usize>) {
    let api_key = std::env::var("ANTHROPIC_API_KEY").ok();
    let ctx = Arc::new(Mutex::new(DispatchContext::new(api_key, model_arg, draft_arg, draft_k)));

    let bind_host = std::env::var("OLORIN_BIND").unwrap_or_else(|_| "127.0.0.1".to_string());
    let addr = format!("{bind_host}:{port}");
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("[olorin] cannot bind {addr}: {e}");
        std::process::exit(1);
    });
    println!("[Olorin] Web UI at http://{addr}");

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(10)));
        let _ = stream.set_nodelay(true);
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
            let c = ctx.lock().unwrap();
            let recall_level = c.recall_level();
            let config_json = c.get_config();
            drop(c);
            let body = build_system_json(recall_level, &config_json);
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
        ("GET", "/api/config") => {
            let body = ctx.lock().unwrap().get_config();
            serve_json(stream, &body);
        }
        ("POST", "/api/config") => {
            handle_config_update(stream, req, &buf[..n], n, ctx);
        }
        ("POST", "/api/config/apikey") => {
            handle_config_apikey(stream, req, &buf[..n], n, ctx);
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

const MAX_BODY_SIZE: usize = 1024 * 1024; // 1 MB

pub(crate) fn read_body(stream: &mut std::net::TcpStream, req: &str, buf: &[u8], n: usize) -> Vec<u8> {
    let content_len = parse_content_length(req);
    if content_len > MAX_BODY_SIZE { return Vec::new(); }
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

    // Apply per-request inference params
    {
        let mut c = ctx.lock().unwrap();
        if let Some(engine) = &mut c.engine {
            if let Some(v) = crate::storage::json::extract_json_float(body_str, "temperature") {
                engine.temperature = v;
            }
            if let Some(v) = crate::storage::json::extract_json_int(body_str, "top_k") {
                engine.top_k = v as usize;
            }
            if let Some(v) = crate::storage::json::extract_json_float(body_str, "top_p") {
                engine.top_p = v;
            }
            if let Some(v) = crate::storage::json::extract_json_int(body_str, "max_tokens") {
                engine.max_tokens = v as usize;
            }
            if let Some(v) = crate::storage::json::extract_json_float(body_str, "repetition_penalty") {
                engine.repetition_penalty = v;
            }
        }
    }

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
    let mut sse_buf = String::with_capacity(256);

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
                sse_buf.clear();
                sse_buf.push_str("data: {\"token\":\"");
                escape_json_into(&tok, &mut sse_buf);
                let _ = write!(sse_buf, "\",\"tps\":{tps:.1}}}\n\n");
                let _ = stream.write_all(sse_buf.as_bytes());
                let _ = stream.flush();
            }
            crate::core::router::StreamEvent::Thinking(active) => {
                let val = if active { "true" } else { "false" };
                let _ = stream.write_all(
                    format!("data: {{\"thinking\":{val}}}\n\n").as_bytes()
                );
                let _ = stream.flush();
            }
            crate::core::router::StreamEvent::Error(msg) => {
                sse_buf.clear();
                sse_buf.push_str("data: {\"error\":\"");
                escape_json_into(&msg, &mut sse_buf);
                sse_buf.push_str("\"}\n\n");
                let _ = stream.write_all(sse_buf.as_bytes());
                let _ = stream.flush();
            }
            crate::core::router::StreamEvent::Done { .. } => break,
        }
    }

    let _ = stream.write_all(b"data: [DONE]\n\n");
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

// ── System info ───────────────────────────────────────────────────────────────

pub fn build_system_json(recall_level: usize, config_json: &str) -> String {
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
         \"recall_level\":{recall_level},\"config\":{config_json}}}"
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
    escape_json_into(s, &mut out);
    out
}

fn escape_json_into(s: &str, out: &mut String) {
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
}

// ── Config handlers ──────────────────────────────────────────────────────────

fn handle_config_update(
    stream: &mut std::net::TcpStream,
    req: &str,
    buf: &[u8],
    n: usize,
    ctx: Arc<Mutex<DispatchContext>>,
) {
    let body_bytes = read_body(stream, req, buf, n);
    let body_str = std::str::from_utf8(&body_bytes).unwrap_or("");
    ctx.lock().unwrap().update_config(body_str);
    let config = ctx.lock().unwrap().get_config();
    serve_json(stream, &config);
}

fn handle_config_apikey(
    stream: &mut std::net::TcpStream,
    req: &str,
    buf: &[u8],
    n: usize,
    ctx: Arc<Mutex<DispatchContext>>,
) {
    let body_bytes = read_body(stream, req, buf, n);
    let key = std::str::from_utf8(&body_bytes).unwrap_or("").trim();
    if key.is_empty() {
        serve_json(stream, r#"{"ok":false,"error":"empty key"}"#);
        return;
    }
    ctx.lock().unwrap().store_api_key(key);
    serve_json(stream, r#"{"ok":true}"#);
}

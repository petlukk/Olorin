//! Web server + WhatsApp bridge — synchronous, thread-per-connection.
//!
//! Serves chat.html on GET /, dispatches POST /api/generate through the Pipe,
//! and spawns the WhatsApp bridge as a subprocess communicating via JSONL.

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use crate::core::router::DispatchContext;
use crate::interface::exec;

// ── Hybrid embed for chat.html ────────────────────────────────────────────────

fn get_chat_html() -> String {
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
pub fn run(port: u16) {
    let api_key = std::env::var("ANTHROPIC_API_KEY").ok();
    let ctx = Arc::new(Mutex::new(DispatchContext::new(api_key)));

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
            let body = build_system_json();
            serve_json(stream, &body);
        }
        ("POST", "/api/generate") => {
            handle_generate(stream, req, &buf[..n], n, ctx);
        }
        ("POST", "/api/command") => {
            handle_command(stream, req, &buf[..n], n, ctx);
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

fn serve_json(stream: &mut std::net::TcpStream, body: &str) {
    let _ = write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
         Access-Control-Allow-Origin: *\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(), body
    );
}

fn read_body(stream: &mut std::net::TcpStream, req: &str, buf: &[u8], n: usize) -> Vec<u8> {
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

    let response = {
        let mut guard = ctx.lock().unwrap();
        guard.dispatch(&prompt)
    };

    let text = escape_json(&response.text);
    let _ = write!(stream, "data: {{\"token\":\"{text}\",\"tps\":0.0}}\n\n");
    let _ = write!(stream, "data: [DONE]\n\n");
    let _ = stream.flush();
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
pub fn run_whatsapp() {
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
    let ctx     = Arc::new(Mutex::new(DispatchContext::new(api_key)));

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

fn find_bridge() -> String {
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

fn build_system_json() -> String {
    let (mem_used, mem_total) = read_memory().unwrap_or((0, 0));
    let uptime   = read_uptime().unwrap_or(0);
    let os       = std::env::consts::OS;
    let arch     = std::env::consts::ARCH;
    let cpu_temp = match read_cpu_temp() {
        Some(t) => t.to_string(),
        None    => "null".to_string(),
    };
    format!(
        "{{\"cpu_percent\":0,\"cpu_temp\":{cpu_temp},\
         \"memory_used_mb\":{mem_used},\"memory_total_mb\":{mem_total},\
         \"os\":\"{os}\",\"arch\":\"{arch}\",\"uptime_seconds\":{uptime}}}"
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

fn read_uptime() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/uptime").ok()?;
    Some(s.split_whitespace().next()?.parse::<f64>().ok()? as u64)
}

// ── JSON helpers ──────────────────────────────────────────────────────────────

fn parse_content_length(req: &str) -> usize {
    for line in req.lines() {
        if line.to_ascii_lowercase().starts_with("content-length:") {
            return line[15..].trim().parse().unwrap_or(0);
        }
    }
    0
}

fn extract_json_string(json: &str, key: &str) -> Option<String> {
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

fn escape_json(s: &str) -> String {
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_content_length() {
        assert_eq!(parse_content_length("POST /x HTTP/1.1\r\nContent-Length: 42\r\n\r\n"), 42);
    }

    #[test]
    fn test_parse_content_length_missing() {
        assert_eq!(parse_content_length("GET / HTTP/1.1\r\n\r\n"), 0);
    }

    #[test]
    fn test_extract_json_string() {
        let json = r#"{"prompt":"hello world"}"#;
        assert_eq!(extract_json_string(json, "prompt"), Some("hello world".into()));
    }

    #[test]
    fn test_extract_json_string_missing() {
        assert!(extract_json_string(r#"{"other":"val"}"#, "prompt").is_none());
    }

    #[test]
    fn test_escape_json_quotes() {
        assert_eq!(escape_json("he\"llo"), "he\\\"llo");
    }

    #[test]
    fn test_escape_json_newline() {
        assert_eq!(escape_json("line\nnew"), "line\\nnew");
    }

    #[test]
    fn test_escape_json_backslash() {
        assert_eq!(escape_json("a\\b"), "a\\\\b");
    }

    #[test]
    fn test_build_system_json_shape() {
        let json = build_system_json();
        assert!(json.contains("\"cpu_temp\""));
        assert!(json.contains("\"memory_used_mb\""));
        assert!(json.contains("\"os\""));
        assert!(json.contains("\"arch\""));
        assert!(json.contains("\"uptime_seconds\""));
    }

    #[test]
    fn test_find_bridge_env_override() {
        std::env::set_var("OLORIN_BRIDGE", "/tmp/fake-bridge");
        assert_eq!(find_bridge(), "/tmp/fake-bridge");
        std::env::remove_var("OLORIN_BRIDGE");
    }

    #[test]
    fn test_find_bridge_default_nonempty() {
        std::env::remove_var("OLORIN_BRIDGE");
        assert!(!find_bridge().is_empty());
    }

    #[test]
    fn test_get_chat_html_nonempty() {
        // Debug build reads from disk; in CI it may not exist — just check no panic
        let html = get_chat_html();
        assert!(!html.is_empty());
    }
}

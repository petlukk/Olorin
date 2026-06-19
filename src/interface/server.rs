//! Web server — synchronous, thread-per-connection. Serves chat.html and
//! dispatches POST /api/generate through the Pipe.

use std::io::{Read, Write};
use std::fmt::Write as FmtWrite;
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use crate::core::router::DispatchContext;
use crate::interface::server_auth::AuthGate;
use crate::interface::term_stream;

// JSON/HTTP helpers live in `server_http` (split out to keep this file under
// the 500-line cap). Re-exported here so existing `interface::server::…`
// call sites across the crate keep resolving.
pub use crate::interface::server_http::{build_system_json, extract_json_string, parse_content_length};
pub(crate) use crate::interface::server_http::{escape_json, escape_json_into, read_body};

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

/// Cap on simultaneously-handled connections (wave-three S2): the accept loop
/// spawns a 16 MB-stack thread per connection, so without a bound a flood
/// exhausts threads. Override with `OLORIN_MAX_CONN` (default 64).
fn max_connections() -> usize {
    std::env::var("OLORIN_MAX_CONN")
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or(64)
}

/// Decrements the in-flight counter on drop, so a connection slot is released
/// even if `handle_connection` panics.
struct ConnGuard(Arc<AtomicUsize>);
impl Drop for ConnGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

pub fn run(port: u16, model_arg: Option<&str>, strict: bool, audit_path: Option<&str>) {
    let api_key = std::env::var("ANTHROPIC_API_KEY").ok();
    // Two-phase init: open the vault first, fail-fast if it didn't open,
    // then load the inference engine.  Skips the ~5–10 second model
    // load when we're going to refuse to start anyway.
    let mut built = if strict {
        DispatchContext::new_strict(model_arg)
    } else {
        DispatchContext::new_no_engine(api_key)
    };
    if !built.has_vault() {
        eprintln!("[olorin] vault unavailable — refusing to start --serve without persistence.");
        eprintln!("[olorin] Set OLORIN_PASSPHRASE, or launch interactively so the tty prompt can run.");
        std::process::exit(1);
    }
    if !strict {
        built.load_engine_now(model_arg);
    }
    if let Some(path) = audit_path {
        match crate::core::audit::AuditLog::open(std::path::Path::new(path)) {
            Ok(log) => { built = built.with_audit(log); }
            Err(e) => eprintln!("[Olorin] audit: failed to open {path}: {e} — continuing without audit"),
        }
    }
    let ctx = Arc::new(Mutex::new(built));
    let teleported = Arc::new(AtomicBool::new(false));
    ctx.lock().unwrap_or_else(|e| e.into_inner()).server_teleported = Some(teleported.clone());

    let bind_host = std::env::var("OLORIN_BIND").unwrap_or_else(|_| "127.0.0.1".to_string());
    // Resolve the network auth policy BEFORE binding: a non-loopback bind
    // without OLORIN_AUTH_TOKEN is refused here, so the socket never opens.
    let auth = match AuthGate::resolve(&bind_host) {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("[olorin] {msg}");
            std::process::exit(1);
        }
    };
    let addr = format!("{bind_host}:{port}");
    let listener = TcpListener::bind(&addr).unwrap_or_else(|e| {
        eprintln!("[olorin] cannot bind {addr}: {e}");
        std::process::exit(1);
    });
    if auth.is_open() {
        println!("[Olorin] Web UI at http://{addr}");
    } else {
        println!("[Olorin] Web UI at http://{addr}  (token required — first visit: http://{addr}/?token=$OLORIN_AUTH_TOKEN)");
    }

    let max_conn = max_connections();
    let active = Arc::new(AtomicUsize::new(0));

    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(10)));
        let _ = stream.set_nodelay(true);

        // S2: refuse beyond the concurrency cap instead of spawning an
        // unbounded number of 16 MB-stack threads. `fetch_add` then re-check is
        // race-safe enough for a soft cap (a small transient overshoot at most).
        if active.fetch_add(1, Ordering::AcqRel) >= max_conn {
            active.fetch_sub(1, Ordering::AcqRel);
            let _ = stream.write_all(
                b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
            );
            continue;
        }
        let guard = ConnGuard(active.clone());

        let ctx = ctx.clone();
        let teleported = teleported.clone();
        let auth = auth.clone();

        // 16 MB matches the main thread's PE stack reserve. The dispatch path
        // can hit the forward pass, which busts std::thread's 2 MB Windows
        // default. `guard` moves into the closure → released on both spawn
        // success (thread end) and failure (closure dropped), so the slot
        // always frees.
        let _ = std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                let _guard = guard;
                handle_connection(&mut stream, ctx, &teleported, &auth);
            });
    }
}

fn handle_connection(stream: &mut std::net::TcpStream, ctx: Arc<Mutex<DispatchContext>>, teleported: &AtomicBool, auth: &AuthGate) {
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

    // Network auth gate. Open (loopback) binds short-circuit to authorized;
    // exposed binds require the token on every request before any dispatch.
    if !auth.authorized(req) {
        serve_401(stream);
        return;
    }

    let path = path.split('?').next().unwrap_or("/");

    match (method, path) {
        ("GET", "/") => serve_html(stream, auth.bootstrap_cookie(req)),
        ("GET", "/api/model") => {
            serve_json(stream, r#"{"name":"olorin","backend":"pipe"}"#);
        }
        ("GET", "/api/system") => {
            // Never block on the ctx mutex here. A long analysis/generation holds
            // it for minutes, which would freeze the live cpu/temp/mem heartbeat
            // exactly when the Pi is working hardest. Refresh recall/config
            // opportunistically (try_lock) and reuse the last-known values when
            // busy; the system telemetry below never needs the lock.
            static LAST: std::sync::Mutex<(usize, String)> =
                std::sync::Mutex::new((0, String::new()));
            let guard = match ctx.try_lock() {
                Ok(c) => Some(c),
                Err(std::sync::TryLockError::Poisoned(p)) => Some(p.into_inner()),
                Err(std::sync::TryLockError::WouldBlock) => None,
            };
            if let Some(c) = guard {
                let r = c.recall_level();
                let cfg = c.get_config();
                drop(c);
                if let Ok(mut last) = LAST.lock() { *last = (r, cfg); }
            }
            let (recall_level, mut config_json) = {
                let last = LAST.lock().unwrap_or_else(|e| e.into_inner());
                last.clone()
            };
            if config_json.is_empty() { config_json = "{}".to_string(); }
            let body = build_system_json(recall_level, &config_json);
            serve_json(stream, &body);
        }
        ("POST", "/api/generate") => {
            handle_generate(stream, req, &buf[..n], n, ctx, teleported);
        }
        ("POST", "/api/command") => {
            handle_command(stream, req, &buf[..n], n, ctx, teleported);
        }
        ("POST", "/api/analyze") => {
            crate::interface::server_analyze::handle_analyze(stream, req, &buf[..n], n, ctx);
        }
        ("POST", "/api/analyze_raw") => {
            crate::interface::server_analyze::handle_analyze_raw(stream, req, &buf[..n], n, ctx);
        }
        ("POST", "/api/report") => {
            crate::interface::server_analyze::handle_report(stream, req, &buf[..n], n);
        }
        ("POST", "/api/report_raw") => {
            crate::interface::server_analyze::handle_report_raw(stream, req, &buf[..n], n);
        }
        ("POST", "/api/term/open") => {
            term_stream::handle_term_open(stream);
        }
        ("POST", path) if path.starts_with("/api/term/") && path.ends_with("/resize") => {
            let id = term_stream::parse_term_id(path);
            term_stream::handle_term_resize(stream, req, &buf[..n], n, id);
        }
        ("POST", path) if path.starts_with("/api/term/") && path.ends_with("/close") => {
            let id = term_stream::parse_term_id(path);
            term_stream::handle_term_close(stream, id);
        }
        ("GET", path) if path.starts_with("/api/term/") && path.ends_with("/ws") => {
            let id = term_stream::parse_term_id(path);
            term_stream::handle_term_ws(stream, req, id);
        }
        ("GET", "/api/config") => {
            let body = ctx.lock().unwrap_or_else(|e| e.into_inner()).get_config();
            serve_json(stream, &body);
        }
        ("POST", "/api/config") => {
            crate::interface::server_config::handle_config_update(stream, req, &buf[..n], n, ctx);
        }
        ("POST", "/api/config/apikey") => {
            crate::interface::server_config::handle_config_apikey(stream, req, &buf[..n], n, ctx);
        }
        #[cfg(unix)]
        ("POST", "/api/palantir/watch") => {
            crate::interface::server_palantir::handle_palantir_watch(stream, req, &buf[..n], n);
        }
        #[cfg(unix)]
        ("POST", "/api/palantir/stop") => {
            crate::interface::server_palantir::handle_palantir_stop(stream, req, &buf[..n], n);
        }
        _ => {
            let _ = write!(
                stream,
                "HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\nConnection: close\r\n\r\nNot Found"
            );
        }
    }
}

fn serve_html(stream: &mut std::net::TcpStream, set_cookie: Option<String>) {
    let body = get_chat_html();
    // When the visitor authenticated via `?token=`, persist it as a cookie so
    // same-origin fetch/EventSource/WebSocket carry it on every later request.
    let cookie_hdr = match set_cookie {
        Some(c) => format!("Set-Cookie: {c}\r\n"),
        None => String::new(),
    };
    // The HTML is embedded in the binary (include_str!), so it changes on every
    // deploy — never let the browser serve a stale copy, or new frontend code
    // (e.g. a new upload path) silently won't load.
    let _ = write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\
         Cache-Control: no-cache, no-store, must-revalidate\r\n\
         {cookie_hdr}\
         Content-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(), body
    );
}

/// 401 for a request that failed the auth gate. `WWW-Authenticate: Bearer`
/// signals the expected scheme to API clients.
fn serve_401(stream: &mut std::net::TcpStream) {
    let body = "Unauthorized: a valid OLORIN_AUTH_TOKEN is required to reach this server.";
    let _ = write!(
        stream,
        "HTTP/1.1 401 Unauthorized\r\nWWW-Authenticate: Bearer\r\n\
         Content-Type: text/plain; charset=utf-8\r\n\
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

fn handle_generate(
    stream: &mut std::net::TcpStream,
    req: &str,
    buf: &[u8],
    n: usize,
    ctx: Arc<Mutex<DispatchContext>>,
    teleported: &AtomicBool,
) {
    if teleported.load(Ordering::Relaxed) {
        let _ = write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\n\
             Cache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\n\
             Connection: close\r\n\r\n"
        );
        let _ = stream.flush();
        let msg = "Olorin is on WhatsApp. Send /teleport there to return.";
        let _ = write!(stream, "data: {{\"token\":\"{msg}\"}}\n\ndata: [DONE]\n\n");
        let _ = stream.flush();
        return;
    }

    let body_bytes = read_body(stream, req, buf, n);
    let body_str   = std::str::from_utf8(&body_bytes).unwrap_or("");
    let prompt     = extract_json_string(body_str, "prompt").unwrap_or_default();

    {
        let mut c = ctx.lock().unwrap_or_else(|e| e.into_inner());
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

    let _ = write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\n\
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
    // Forward-pass-bearing thread — needs the same 16 MB the main
    // thread has (Windows default 2 MB busts on Win64 ABI overhead).
    let sender = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            let mut guard = ctx_clone.lock().unwrap_or_else(|e| e.into_inner());
            guard.dispatch_streaming(&prompt_owned, tx);
        })
        .expect("failed to spawn dispatch thread");

    relay_sse(stream, rx);
    let _ = sender.join();
}

/// Relay a `StreamEvent` channel to the client as Server-Sent Events, ending
/// with `[DONE]`. Shared by `/api/generate` and `/api/analyze`.
pub(crate) fn relay_sse(
    stream: &mut std::net::TcpStream,
    rx: std::sync::mpsc::Receiver<crate::core::router::StreamEvent>,
) {
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
}

fn handle_command(
    stream: &mut std::net::TcpStream,
    req: &str,
    buf: &[u8],
    n: usize,
    ctx: Arc<Mutex<DispatchContext>>,
    teleported: &AtomicBool,
) {
    if teleported.load(Ordering::Relaxed) {
        let msg = "Olorin is on WhatsApp. Send /teleport there to return.";
        let body = format!("{{\"output\":\"{msg}\",\"success\":false}}");
        serve_json(stream, &body);
        return;
    }

    let body_bytes = read_body(stream, req, buf, n);
    let body_str   = std::str::from_utf8(&body_bytes).unwrap_or("");
    let command    = extract_json_string(body_str, "command").unwrap_or_default();

    let (output, success) = if command.is_empty() {
        ("missing command".to_string(), false)
    } else {
        let mut guard = ctx.lock().unwrap_or_else(|e| e.into_inner());
        let resp      = guard.dispatch(&command);
        let ok        = !resp.blocked;
        (resp.text, ok)
    };

    let escaped = escape_json(&output);
    let body    = format!("{{\"output\":\"{escaped}\",\"success\":{success}}}");
    serve_json(stream, &body);
}

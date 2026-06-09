//! Terminal session handlers — PTY open/resize/close + WebSocket stream.

use std::io::Write;
use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::collections::HashMap;
use crate::interface::server::{serve_json, read_body, escape_json};
use crate::interface::pty::PtySession;
use crate::interface::ws;

static NEXT_TERM_ID: AtomicU32 = AtomicU32::new(0);

type TermSessions = Arc<Mutex<HashMap<u32, Arc<Mutex<PtySession>>>>>;

pub fn term_sessions() -> &'static TermSessions {
    static SESSIONS: OnceLock<TermSessions> = OnceLock::new();
    SESSIONS.get_or_init(|| Arc::new(Mutex::new(HashMap::new())))
}

pub fn parse_term_id(path: &str) -> u32 {
    let parts: Vec<&str> = path.split('/').collect();
    parts.get(3).and_then(|s| s.parse().ok()).unwrap_or(0)
}

const MAX_TERM_SESSIONS: usize = 8;

pub fn handle_term_open(stream: &mut std::net::TcpStream) {
    if term_sessions().lock().unwrap().len() >= MAX_TERM_SESSIONS {
        serve_json(stream, r#"{"error":"too many sessions"}"#);
        return;
    }
    let id = NEXT_TERM_ID.fetch_add(1, Ordering::Relaxed);
    match PtySession::new(80, 24) {
        Ok(session) => {
            let session = Arc::new(Mutex::new(session));
            term_sessions().lock().unwrap().insert(id, session);
            let body = format!("{{\"id\":{id}}}");
            serve_json(stream, &body);
        }
        Err(e) => {
            let escaped = escape_json(&format!("{e}"));
            let body = format!("{{\"error\":\"{escaped}\"}}");
            serve_json(stream, &body);
        }
    }
}

pub fn handle_term_resize(stream: &mut std::net::TcpStream, req: &str, buf: &[u8], n: usize, id: u32) {
    let body_bytes = read_body(stream, req, buf, n);
    let body_str = std::str::from_utf8(&body_bytes).unwrap_or("");
    let cols: u16 = extract_json_number(body_str, "cols").unwrap_or(80) as u16;
    let rows: u16 = extract_json_number(body_str, "rows").unwrap_or(24) as u16;

    let sessions = term_sessions().lock().unwrap();
    if let Some(session) = sessions.get(&id) {
        let mut s = session.lock().unwrap();
        s.resize(cols, rows);
    }
    serve_json(stream, r#"{"ok":true}"#);
}

pub fn handle_term_close(stream: &mut std::net::TcpStream, id: u32) {
    term_sessions().lock().unwrap().remove(&id);
    serve_json(stream, r#"{"ok":true}"#);
}

fn extract_json_number(json: &str, key: &str) -> Option<u32> {
    let pattern = format!("\"{}\"", key);
    let start = json.find(&pattern)?;
    let after_key = &json[start + pattern.len()..];
    let colon = after_key.find(':')?;
    let rest = after_key[colon + 1..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Build the inner JSON for a frame event. Used by both SSE and WS senders.
/// Returns `None` if nothing changed since the previous call.
fn poll_frame_json(session: &Arc<Mutex<PtySession>>, prev_cursor: &mut (u16, u16)) -> Option<String> {
    let mut s = session.lock().unwrap();
    let dirty: Vec<u8> = s.read_and_apply().to_vec();
    let dirty_count = dirty.iter().filter(|&&d| d != 0).count();
    let grid = s.grid();
    let cols = grid.cols;
    let (crow, ccol) = grid.cursor();
    let cursor_moved = *prev_cursor != (ccol, crow);
    *prev_cursor = (ccol, crow);

    if dirty_count == 0 && !cursor_moved {
        return None;
    }

    let mut cells_json = String::with_capacity(dirty_count * 64 + 16);
    cells_json.push('[');
    let mut first = true;
    for (i, &d) in dirty.iter().enumerate() {
        if d == 0 { continue; }
        let row = i / cols as usize;
        let col = i % cols as usize;
        let cell = grid.cell(row as u16, col as u16);
        if !first { cells_json.push(','); }
        first = false;
        let ch = if cell.ch == 0 || cell.ch == 32 {
            " ".to_string()
        } else if let Some(c) = char::from_u32(cell.ch) {
            match c {
                '"' => "\\\"".to_string(),
                '\\' => "\\\\".to_string(),
                '\n' => "\\n".to_string(),
                '\r' => "\\r".to_string(),
                '\t' => "\\t".to_string(),
                _ => c.to_string(),
            }
        } else {
            " ".to_string()
        };
        cells_json.push_str(&format!(
            "{{\"r\":{row},\"c\":{col},\"ch\":\"{ch}\",\"fg\":\"#{:06x}\",\"bg\":\"#{:06x}\",\"fl\":{}}}",
            cell.fg, cell.bg, cell.flags
        ));
    }
    cells_json.push(']');
    Some(format!(
        "{{\"type\":\"frame\",\"cursor\":[{ccol},{crow}],\"cells\":{cells_json}}}"
    ))
}

fn get_session(id: u32) -> Option<Arc<Mutex<PtySession>>> {
    term_sessions().lock().unwrap().get(&id).cloned()
}

/// Shared between the WS writer loop (this thread) and the reader thread.
/// Only the writer touches the socket for frames — the reader signals through
/// these flags so we never have two threads writing frames concurrently.
struct WsShared {
    /// Reader sets this on EOF/Close so the writer stops instead of spinning
    /// forever when the client disconnects at an idle prompt.
    disconnected: AtomicBool,
    /// Reader sets this when write_guarded blocks a line; the writer emits a
    /// `blocked` frame so the client can flash the user-visible signal.
    blocked: AtomicBool,
}

/// WebSocket terminal stream. Streams frame JSON to the client and reads input
/// bytes back over the same socket, eliminating the per-keystroke fetch POST.
pub fn handle_term_ws(stream: &mut std::net::TcpStream, req: &str, id: u32) {
    if ws::handshake(stream, req).is_err() {
        return;
    }
    let session = match get_session(id) {
        Some(s) => s,
        None => {
            let _ = ws::write_text(stream, r#"{"type":"error","msg":"no such session"}"#);
            let _ = ws::write_close(stream);
            return;
        }
    };

    // Clear the connection-level 10s read timeout before cloning for the
    // reader thread — otherwise the reader's blocking read_frame errors out
    // on every idle period.
    let _ = stream.set_read_timeout(None);
    let read_stream = match stream.try_clone() {
        Ok(s) => s,
        Err(_) => return,
    };
    let shared = Arc::new(WsShared {
        disconnected: AtomicBool::new(false),
        blocked: AtomicBool::new(false),
    });
    let reader_session = session.clone();
    let reader_shared = shared.clone();
    let reader_handle = std::thread::Builder::new()
        .stack_size(1024 * 1024)
        .spawn(move || ws_reader_loop(read_stream, reader_session, reader_shared))
        .ok();
    let mut prev_cursor: (u16, u16) = (0, 0);
    loop {
        std::thread::sleep(std::time::Duration::from_millis(16));
        // Client gone — stop before spinning forever (the reader detected it).
        if shared.disconnected.load(Ordering::Relaxed) { break; }
        if shared.blocked.swap(false, Ordering::Relaxed) {
            if ws::write_text(stream, r#"{"type":"blocked"}"#).is_err() { break; }
            if stream.flush().is_err() { break; }
        }
        let readable = session.lock().unwrap().wait_readable(0);
        if !readable { continue; }

        #[cfg(windows)]
        std::thread::sleep(std::time::Duration::from_millis(5));

        if let Some(json) = poll_frame_json(&session, &mut prev_cursor) {
            if ws::write_text(stream, &json).is_err() { break; }
            if stream.flush().is_err() { break; }
        }
        if !session.lock().unwrap().child_alive() {
            let _ = ws::write_text(stream, r#"{"type":"exit","code":0}"#);
            let _ = ws::write_close(stream);
            break;
        }
    }
    // Tear down the reader so its blocking read_frame returns.
    let _ = stream.shutdown(std::net::Shutdown::Both);
    if let Some(h) = reader_handle { let _ = h.join(); }
}

fn ws_reader_loop(mut stream: std::net::TcpStream, session: Arc<Mutex<PtySession>>, shared: Arc<WsShared>) {
    loop {
        let frame = match ws::read_frame(&mut stream) {
            Ok(Some(f)) => f,
            Ok(None) | Err(_) => break,
        };
        match frame.opcode {
            ws::Opcode::Text | ws::Opcode::Binary => {
                let mut s = session.lock().unwrap();
                if s.write_guarded(&frame.payload).is_err() {
                    // Signal the writer to emit a `blocked` frame — we can't
                    // write it here without racing the writer on the socket.
                    shared.blocked.store(true, Ordering::Relaxed);
                }
            }
            ws::Opcode::Close => break,
            // Ping/Pong from browsers is rare in practice; ignore for now.
            _ => {}
        }
    }
    // Let the writer loop break out of its 16ms poll instead of spinning.
    shared.disconnected.store(true, Ordering::Relaxed);
    let _ = stream.shutdown(std::net::Shutdown::Both);
}

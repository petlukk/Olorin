//! Terminal session handlers — PTY open/input/resize/close/stream.

use std::io::Write;
use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicU32, Ordering};
use std::collections::HashMap;
use crate::interface::server::{serve_json, read_body, escape_json};
use crate::interface::pty::PtySession;

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

pub fn handle_term_open(stream: &mut std::net::TcpStream) {
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

pub fn handle_term_input(stream: &mut std::net::TcpStream, req: &str, buf: &[u8], n: usize, id: u32) {
    let body_bytes = read_body(stream, req, buf, n);
    let sessions = term_sessions().lock().unwrap();
    if let Some(session) = sessions.get(&id) {
        let mut s = session.lock().unwrap();
        match s.write_guarded(&body_bytes) {
            Ok(()) => serve_json(stream, r#"{"ok":true}"#),
            Err(reason) => {
                let escaped = escape_json(&reason);
                serve_json(stream, &format!("{{\"ok\":false,\"blocked\":\"{escaped}\"}}"));
            }
        }
    } else {
        serve_json(stream, r#"{"ok":false,"error":"no session"}"#);
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

pub fn handle_term_stream(stream: &mut std::net::TcpStream, id: u32) {
    let _ = write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
         Cache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\n\
         Connection: keep-alive\r\n\r\n"
    );
    let _ = stream.flush();

    let session = {
        let sessions = term_sessions().lock().unwrap();
        match sessions.get(&id) {
            Some(s) => s.clone(),
            None => {
                let _ = write!(stream, "data: {{\"type\":\"error\",\"msg\":\"no such session\"}}\n\n");
                return;
            }
        }
    };

    let _ = stream.set_read_timeout(None);

    let mut pollfd = libc::pollfd {
        fd: session.lock().unwrap().master_fd(),
        events: libc::POLLIN,
        revents: 0,
    };

    let mut prev_cursor: (u16, u16) = (0, 0);
    eprintln!("[term-stream] starting SSE loop for session {id}, fd={}", pollfd.fd);
    loop {
        let ret = unsafe { libc::poll(&mut pollfd, 1, 16) };

        if ret > 0 && pollfd.revents & libc::POLLIN != 0 {
            let mut s = session.lock().unwrap();
            let dirty: Vec<u8> = s.read_and_apply().to_vec();
            let dirty_count = dirty.iter().filter(|&&d| d != 0).count();
            let grid = s.grid();
            let cols = grid.cols;
            let (crow, ccol) = grid.cursor();
            let cursor_moved = prev_cursor != (ccol, crow);
            prev_cursor = (ccol, crow);
            eprintln!("[term-stream] read: {dirty_count} dirty cells, cursor_moved={cursor_moved}");

            if dirty_count > 0 || cursor_moved {

                let mut cells_json = String::with_capacity(1024);
                cells_json.push('[');
                let mut first = true;
                for (i, &d) in dirty.iter().enumerate() {
                    if d != 0 {
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
                }
                cells_json.push(']');

                let frame = format!(
                    "data: {{\"type\":\"frame\",\"cursor\":[{ccol},{crow}],\"cells\":{cells_json}}}\n\n"
                );
                if write!(stream, "{frame}").is_err() { break; }
                if stream.flush().is_err() { break; }
            }

            if !s.child_alive() {
                let _ = write!(stream, "data: {{\"type\":\"exit\",\"code\":0}}\n\n");
                let _ = stream.flush();
                break;
            }
        }

        if ret < 0 { break; }

        if pollfd.revents & (libc::POLLHUP | libc::POLLERR) != 0 {
            break;
        }
    }
}

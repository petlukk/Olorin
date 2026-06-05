//! `POST /api/analyze` — the file-drop analyst's HTTP front door.
//!
//! Accepts `{"files":[{"name":"app.log","b64":"..."}]}`, base64-decodes each
//! file, writes it under `/tmp` (the rune path allowlist), and streams the
//! analysis (deterministic rune pick → SIMD kernel → narration) back as SSE.
//! The drop gesture is the "analyze this" intent, so no autonomous model
//! tool-call is involved — which is exactly why this works on the Pi.

use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::core::router::DispatchContext;
use crate::interface::server::{read_body, relay_sse};
use crate::storage::json;

const SSE_HEADERS: &str =
    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\n\
     Cache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\n\
     Connection: close\r\n\r\n";

static DROP_SEQ: AtomicUsize = AtomicUsize::new(0);

pub(crate) fn handle_analyze(
    stream: &mut std::net::TcpStream,
    req: &str,
    buf: &[u8],
    n: usize,
    ctx: Arc<Mutex<DispatchContext>>,
) {
    let body = read_body(stream, req, buf, n);
    let _ = write!(stream, "{SSE_HEADERS}");
    let _ = stream.flush();

    // Parse {"files":[{"name","b64"}]} and stage every file under /tmp.
    let staged = match parse_and_stage(&body) {
        Ok(v) => v,
        Err(msg) => {
            send_error(stream, &msg);
            return;
        }
    };

    let (tx, rx) = std::sync::mpsc::channel();
    let ctx_clone = ctx.clone();
    // Forward-pass-bearing thread (narration) — 16 MB stack like /api/generate.
    let sender = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            let mut guard = ctx_clone.lock().unwrap_or_else(|e| e.into_inner());
            guard.analyze_files_streaming(&staged, &tx);
        })
        .expect("failed to spawn analyze thread");

    relay_sse(stream, rx);
    let _ = sender.join();
}

/// Decode every file in the request body and write each under /tmp.
/// Returns a (display_name, tmp_path) per file, or a user-facing error.
pub fn parse_and_stage(body: &[u8]) -> Result<Vec<(String, String)>, String> {
    let obj = json::parse(body).map_err(|_| "Couldn't read the upload (bad JSON).".to_string())?;
    let files = obj.get_array("files")
        .ok_or_else(|| "No files in the request.".to_string())?;
    if files.is_empty() {
        return Err("No files in the request.".to_string());
    }
    let mut staged = Vec::with_capacity(files.len());
    for entry_val in files {
        let entry = match entry_val {
            json::Value::Object(o) => o.as_ref(),
            _ => return Err("Malformed file entry.".to_string()),
        };
        let name = entry.get_str("name").unwrap_or("dropped-file").to_string();
        let b64 = entry.get_str("b64").ok_or_else(|| "File has no content.".to_string())?;
        let bytes = base64_decode(b64).ok_or_else(|| "File content isn't valid base64.".to_string())?;
        if bytes.is_empty() {
            return Err("A dropped file is empty.".to_string());
        }
        let path = temp_path_for(&name);
        std::fs::write(&path, &bytes).map_err(|e| format!("Couldn't stage a file: {e}"))?;
        staged.push((name, path));
    }
    Ok(staged)
}

/// Build a collision-free `/tmp` path with a sanitized basename.
fn temp_path_for(name: &str) -> String {
    let seq = DROP_SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = format!("/tmp/olorin-drop-{}-{}", std::process::id(), seq);
    let _ = std::fs::create_dir_all(&dir);
    format!("{dir}/{}", sanitize_filename(name))
}

/// Reduce an untrusted filename to a safe basename: strip directories, keep
/// only `[A-Za-z0-9._-]`, and reject empty / all-dots names (no traversal).
pub fn sanitize_filename(name: &str) -> String {
    let base = name.rsplit(['/', '\\']).next().unwrap_or(name);
    let cleaned: String = base
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') { c } else { '_' })
        .collect();
    if cleaned.is_empty() || cleaned.chars().all(|c| c == '.') {
        return "dropped-file".to_string();
    }
    cleaned
}

/// Standard-alphabet base64 decode. Skips whitespace, stops at `=` padding,
/// and tolerates a leading `data:...;base64,` prefix.
pub fn base64_decode(s: &str) -> Option<Vec<u8>> {
    let s = s.rsplit(',').next().unwrap_or(s); // drop any data-URL prefix
    fn val(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let mut acc: u32 = 0;
    let mut bits = 0u32;
    for &c in s.as_bytes() {
        match c {
            b'=' => break,
            b'\n' | b'\r' | b' ' | b'\t' => continue,
            _ => {}
        }
        acc = (acc << 6) | val(c)? as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Some(out)
}

/// Streaming single-file cap (disk-bound, not RAM): matches the rune's
/// `MAX_INPUT_BYTES`. The raw path streams to /tmp in chunks, so this is about
/// disk space, not the 128 MB in-memory base64 cap on `/api/analyze`.
const MAX_STREAM_UPLOAD: u64 = 4 * 1024 * 1024 * 1024;

/// `POST /api/analyze_raw` — one file streamed straight to /tmp (no base64, no
/// in-memory JSON), so multi-GB logs upload without exhausting RAM. The
/// filename comes from the `X-Filename` header.
pub(crate) fn handle_analyze_raw(
    stream: &mut std::net::TcpStream,
    req: &str,
    buf: &[u8],
    n: usize,
    ctx: Arc<Mutex<DispatchContext>>,
) {
    let _ = write!(stream, "{SSE_HEADERS}");
    let _ = stream.flush();

    let name = header_value(req, "x-filename").unwrap_or_else(|| "dropped-file".to_string());
    let content_len = crate::interface::server::parse_content_length(req) as u64;
    if content_len == 0 {
        send_error(stream, "Empty upload.");
        return;
    }
    if content_len > MAX_STREAM_UPLOAD {
        send_error(stream, "File too large (max 4 GB).");
        return;
    }

    let path = temp_path_for(&name);
    let header_end = req.find("\r\n\r\n").map(|i| i + 4).unwrap_or(n).min(n);
    if let Err(e) = drain_to_file(stream, &buf[header_end..n], content_len, &path) {
        send_error(stream, &format!("Upload failed: {e}"));
        return;
    }

    let (tx, rx) = std::sync::mpsc::channel();
    let ctx_clone = ctx.clone();
    let sender = std::thread::Builder::new()
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            let mut guard = ctx_clone.lock().unwrap_or_else(|e| e.into_inner());
            guard.analyze_files_streaming(&[(name, path)], &tx);
        })
        .expect("failed to spawn analyze thread");
    relay_sse(stream, rx);
    let _ = sender.join();
}

/// Write `leftover` (body bytes already read alongside the headers) then stream
/// the remaining `content_len` bytes from `reader` to `path` in 64 KB chunks —
/// never holding the whole body in memory.
pub fn drain_to_file<R: std::io::Read>(
    reader: &mut R,
    leftover: &[u8],
    content_len: u64,
    path: &str,
) -> std::io::Result<u64> {
    let mut file = std::fs::File::create(path)?;
    let mut written = 0u64;
    let take = (leftover.len() as u64).min(content_len) as usize;
    if take > 0 {
        file.write_all(&leftover[..take])?;
        written += take as u64;
    }
    let mut chunk = [0u8; 64 * 1024];
    while written < content_len {
        let r = reader.read(&mut chunk)?;
        if r == 0 {
            break;
        }
        let r = (r as u64).min(content_len - written) as usize;
        file.write_all(&chunk[..r])?;
        written += r as u64;
    }
    file.flush()?;
    Ok(written)
}

/// Case-insensitive lookup of an HTTP header value in the raw request text.
pub fn header_value(req: &str, name_lower: &str) -> Option<String> {
    for line in req.lines() {
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case(name_lower) {
                return Some(v.trim().to_string());
            }
        }
    }
    None
}

fn send_error(stream: &mut std::net::TcpStream, msg: &str) {
    let mut esc = String::with_capacity(msg.len());
    for c in msg.chars() {
        match c {
            '"' => esc.push_str("\\\""),
            '\\' => esc.push_str("\\\\"),
            '\n' => esc.push_str("\\n"),
            _ => esc.push(c),
        }
    }
    let _ = write!(stream, "data: {{\"error\":\"{esc}\"}}\n\ndata: [DONE]\n\n");
    let _ = stream.flush();
}

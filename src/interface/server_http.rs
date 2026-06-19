//! HTTP/JSON helpers for the web server — request-body reading, content-length
//! parsing, minimal JSON string extraction/escaping, and the system-info JSON
//! payload. Split out of `server.rs` to keep that file under the 500-line cap;
//! `server.rs` re-exports these, so existing `interface::server::…` call sites
//! keep working unchanged.

use std::io::Read;

/// Max request body, in bytes. Default 128 MB so the file-drop analyst can
/// accept real logs (base64 inflates ~33%, so ~95 MB of file). Override with
/// `OLORIN_MAX_UPLOAD=<bytes>`.
fn max_body_size() -> usize {
    std::env::var("OLORIN_MAX_UPLOAD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128 * 1024 * 1024)
}

pub fn read_body(stream: &mut std::net::TcpStream, req: &str, buf: &[u8], n: usize) -> Vec<u8> {
    let content_len = parse_content_length(req);
    if content_len > max_body_size() { return Vec::new(); }

    // Robustness wave-three S1: grow the buffer as bytes actually arrive,
    // rather than `vec![0u8; content_len]` up front. A lying Content-Length
    // (declares 128 MB, sends nothing) used to allocate the full declared size
    // before reading — N such connections → OOM. Now a connection only costs
    // the bytes it really sends, and the 10 s socket read timeout bounds how
    // long it can stall.
    let header_end = req.find("\r\n\r\n").map(|p| p + 4).unwrap_or(n);
    let prefilled = if header_end <= n { &buf[header_end..n] } else { &[][..] };
    let take = prefilled.len().min(content_len);

    let mut body = Vec::with_capacity(take.max(1).min(64 * 1024));
    body.extend_from_slice(&prefilled[..take]);

    let mut chunk = [0u8; 64 * 1024];
    while body.len() < content_len {
        let want = (content_len - body.len()).min(chunk.len());
        match stream.read(&mut chunk[..want]) {
            Ok(0) | Err(_) => break, // EOF / timeout / error — return what arrived
            Ok(r) => body.extend_from_slice(&chunk[..r]),
        }
    }
    body
}

// ── System info ───────────────────────────────────────────────────────────────

pub fn build_system_json(recall_level: usize, config_json: &str) -> String {
    use crate::platform::sysinfo;
    let (mem_used, mem_total) = sysinfo::memory_usage_mb().unwrap_or((0, 0));
    let uptime      = sysinfo::uptime_seconds().unwrap_or(0);
    let os          = std::env::consts::OS;
    let arch        = std::env::consts::ARCH;
    let cpu_temp    = match sysinfo::cpu_temp_c() {
        Some(t) => t.to_string(),
        None    => "null".to_string(),
    };
    let cpu_percent = sysinfo::cpu_percent().unwrap_or(0);
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let palantir = crate::palantir::daemon::status_json(now);
    format!(
        "{{\"cpu_percent\":{cpu_percent},\"cpu_temp\":{cpu_temp},\
         \"memory_used_mb\":{mem_used},\"memory_total_mb\":{mem_total},\
         \"os\":\"{os}\",\"arch\":\"{arch}\",\"uptime_seconds\":{uptime},\
         \"recall_level\":{recall_level},\"palantir\":{palantir},\"config\":{config_json}}}"
    )
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

pub(crate) fn escape_json_into(s: &str, out: &mut String) {
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"'  => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            // JSON requires all control chars U+0000–U+001F to be escaped.
            // Without this, a stray control byte (from file-derived data, or
            // an internal marker) produces invalid JSON and the browser's
            // JSON.parse silently drops the whole token.
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            _    => out.push(c),
        }
    }
}

//! HTTP request-body reading and the hand-rolled JSON field helpers used by the
//! web server. Split out of server.rs to keep that file under the 500-line rule;
//! server.rs stays the router + request handlers, this is the parsing layer.

use std::io::Read;
use std::net::TcpStream;

/// Max request body, in bytes. Default 128 MB so the file-drop analyst can
/// accept real logs (base64 inflates ~33%, so ~95 MB of file). Override with
/// `OLORIN_MAX_UPLOAD=<bytes>`.
fn max_body_size() -> usize {
    std::env::var("OLORIN_MAX_UPLOAD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128 * 1024 * 1024)
}

pub fn read_body(stream: &mut TcpStream, req: &str, buf: &[u8], n: usize) -> Vec<u8> {
    let content_len = parse_content_length(req);
    if content_len > max_body_size() { return Vec::new(); }
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

pub fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    escape_json_into(s, &mut out);
    out
}

pub fn escape_json_into(s: &str, out: &mut String) {
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

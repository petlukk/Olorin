//! Web channel — minimal HTTP/SSE server bridging the Olorin agent to a browser chat UI.

use std::cell::Cell;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::Instant;

/// Chat HTML embedded at compile time from web/chat.html.
const CHAT_HTML: &str = include_str!("../../../../web/chat.html");

/// JSON request for POST /api/generate.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub prompt: String,
    pub max_tokens: Option<usize>,
    pub recall_level: Option<i8>, // -1 = auto, 0-10 = explicit
}

/// Model info returned by GET /api/model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub backend: String,
}

/// Minimal HTTP/SSE web server for the Olorin agent.
pub struct WebChannel {
    port: u16,
}

impl WebChannel {
    pub fn new(port: u16) -> Self {
        Self { port }
    }

    /// Start the HTTP server. Blocks until shutdown.
    /// `on_prompt` receives (request, token_callback) and returns the final response.
    pub fn run<F>(&self, on_prompt: F) -> std::io::Result<()>
    where
        F: Fn(&GenerateRequest, &dyn Fn(&str)) -> String + Send + Sync,
    {
        let listener = TcpListener::bind(format!("0.0.0.0:{}", self.port))?;
        println!("[Olorin] Web UI listening on http://0.0.0.0:{}", self.port);

        for stream in listener.incoming() {
            let mut stream = match stream {
                Ok(s) => s,
                Err(_) => continue,
            };
            let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(5)));

            // Read headers into buffer until we see \r\n\r\n
            let mut buf = [0u8; 8192];
            let mut n = 0;
            loop {
                let r = match stream.read(&mut buf[n..]) {
                    Ok(0) => break,
                    Ok(r) => r,
                    Err(_) => break,
                };
                n += r;
                if n >= 4 && buf[..n].windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
                if n >= buf.len() {
                    break;
                }
            }
            let req = match std::str::from_utf8(&buf[..n]) {
                Ok(s) => s,
                Err(_) => continue,
            };

            let first_line = req.lines().next().unwrap_or("");
            let mut parts = first_line.split_whitespace();
            let method = parts.next().unwrap_or("");
            let path = parts.next().unwrap_or("");

            match (method, path) {
                ("GET", "/") => {
                    let body = CHAT_HTML;
                    let _ = write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\
                         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                }
                ("GET", "/api/model") => {
                    let body = r#"{"name":"olorin","backend":"cougar"}"#;
                    let _ = write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                }
                ("POST", "/api/generate") => {
                    let content_len = parse_content_length(req);
                    let header_end = req.find("\r\n\r\n").unwrap_or(n) + 4;
                    let already = n - header_end;
                    let mut body_buf = vec![0u8; content_len];
                    if already > 0 && already <= content_len {
                        body_buf[..already].copy_from_slice(&buf[header_end..n]);
                    }
                    if already < content_len {
                        let _ = stream.read_exact(&mut body_buf[already..]);
                    }
                    let body_str = std::str::from_utf8(&body_buf).unwrap_or("");

                    let req = parse_generate_request(body_str);

                    // SSE headers
                    let _ = write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
                         Cache-Control: no-cache\r\nConnection: close\r\n\r\n"
                    );
                    let _ = stream.flush();

                    if req.prompt.is_empty() {
                        let _ = write!(stream, "data: [DONE]\n\n");
                    } else {
                        let gen_start = Instant::now();
                        let gen_count = Cell::new(0u64);
                        // Use a RefCell so the Fn closure can borrow stream mutably.
                        let stream_cell = std::cell::RefCell::new(&mut stream);

                        let _response = on_prompt(&req, &|token: &str| {
                            gen_count.set(gen_count.get() + 1);
                            let elapsed = gen_start.elapsed().as_secs_f64();
                            let count = gen_count.get();
                            let tps = if elapsed > 0.0 {
                                count as f64 / elapsed
                            } else {
                                0.0
                            };
                            let escaped = escape_json(token);
                            if let Ok(mut s) = stream_cell.try_borrow_mut() {
                                let _ = write!(
                                    *s,
                                    "data: {{\"token\":\"{escaped}\",\"tps\":{tps:.1}}}\n\n"
                                );
                                let _ = s.flush();
                            }
                        });

                        let s = stream_cell.into_inner();
                        let _ = write!(s, "data: [DONE]\n\n");
                        let _ = s.flush();
                    }
                }
                ("POST", "/teleport") => {
                    let content_len = parse_content_length(req);
                    let header_end = req.find("\r\n\r\n").unwrap_or(n) + 4;
                    let already = n - header_end;
                    let mut body_buf = vec![0u8; content_len];
                    if already > 0 && already <= content_len {
                        body_buf[..already].copy_from_slice(&buf[header_end..n]);
                    }
                    if already < content_len {
                        let _ = stream.read_exact(&mut body_buf[already..]);
                    }
                    let body_str = std::str::from_utf8(&body_buf).unwrap_or("");
                    let token = extract_json_string(body_str, "token").unwrap_or_default();

                    let body = if token.is_empty() {
                        r#"{"status":"error","message":"missing token"}"#.to_string()
                    } else {
                        format!(r#"{{"status":"ok","token":"{token}"}}"#)
                    };
                    let _ = write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    );
                }
                _ => {
                    let _ = write!(
                        stream,
                        "HTTP/1.1 404 Not Found\r\nContent-Length: 9\r\n\
                         Connection: close\r\n\r\nNot Found"
                    );
                }
            }
        }
        Ok(())
    }
}

/// Parse a GenerateRequest from JSON body.
pub fn parse_generate_request(body: &str) -> GenerateRequest {
    let prompt = extract_json_string(body, "prompt").unwrap_or_default();
    let max_tokens = extract_json_number(body, "max_tokens").map(|v| v as usize);
    let recall_level = extract_json_number(body, "recall_level").map(|v| v as i8);
    GenerateRequest { prompt, max_tokens, recall_level }
}

fn parse_content_length(req: &str) -> usize {
    for line in req.lines() {
        let lower = line.to_ascii_lowercase();
        if lower.starts_with("content-length:") {
            return line[15..].trim().parse().unwrap_or(0);
        }
    }
    0
}

fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\"", key);
    let start = json.find(&pattern)?;
    let after_key = &json[start + pattern.len()..];
    let colon = after_key.find(':')?;
    let after_colon = after_key[colon + 1..].trim_start();
    if !after_colon.starts_with('"') {
        return None;
    }
    let content = &after_colon[1..];
    let mut result = String::new();
    let mut chars = content.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                if let Some(esc) = chars.next() {
                    match esc {
                        '"' => result.push('"'),
                        '\\' => result.push('\\'),
                        'n' => result.push('\n'),
                        't' => result.push('\t'),
                        _ => {
                            result.push('\\');
                            result.push(esc);
                        }
                    }
                }
            }
            '"' => break,
            _ => result.push(c),
        }
    }
    Some(result)
}

fn extract_json_number(json: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{}\"", key);
    let start = json.find(&pattern)?;
    let after_key = &json[start + pattern.len()..];
    let colon = after_key.find(':')?;
    let rest = after_key[colon + 1..].trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            _ => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_generate_request() {
        let body = r#"{"prompt":"hello world","max_tokens":128}"#;
        let req = parse_generate_request(body);
        assert_eq!(req.prompt, "hello world");
        assert_eq!(req.max_tokens, Some(128));
    }

    #[test]
    fn test_parse_generate_request_no_max_tokens() {
        let body = r#"{"prompt":"test"}"#;
        let req = parse_generate_request(body);
        assert_eq!(req.prompt, "test");
        assert_eq!(req.max_tokens, None);
    }

    #[test]
    fn test_parse_generate_request_escaped_prompt() {
        let body = r#"{"prompt":"line1\nline2"}"#;
        let req = parse_generate_request(body);
        assert_eq!(req.prompt, "line1\nline2");
    }

    #[test]
    fn test_parse_generate_request_empty() {
        let body = r#"{}"#;
        let req = parse_generate_request(body);
        assert_eq!(req.prompt, "");
        assert_eq!(req.max_tokens, None);
    }

    #[test]
    fn test_chat_html_embedded() {
        assert!(!CHAT_HTML.is_empty());
        assert!(CHAT_HTML.contains("<!DOCTYPE html>"));
        assert!(CHAT_HTML.contains("/api/generate"));
    }

    #[test]
    fn test_escape_json() {
        assert_eq!(escape_json("hello"), "hello");
        assert_eq!(escape_json("he\"llo"), "he\\\"llo");
        assert_eq!(escape_json("line\nnew"), "line\\nnew");
        assert_eq!(escape_json("tab\there"), "tab\\there");
    }

    #[test]
    fn test_parse_content_length() {
        let req = "POST /api/generate HTTP/1.1\r\nContent-Length: 42\r\n\r\n";
        assert_eq!(parse_content_length(req), 42);
    }

    #[test]
    fn test_parse_content_length_missing() {
        let req = "GET / HTTP/1.1\r\n\r\n";
        assert_eq!(parse_content_length(req), 0);
    }

    #[test]
    fn test_model_info() {
        let info = ModelInfo {
            name: "olorin".to_string(),
            backend: "cougar".to_string(),
        };
        assert_eq!(info.name, "olorin");
        assert_eq!(info.backend, "cougar");
    }
}

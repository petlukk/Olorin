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
    /// `on_command` receives a REPL command string and returns (output, success).
    pub fn run<F, C>(&self, on_prompt: F, on_command: C) -> std::io::Result<()>
    where
        F: Fn(&GenerateRequest, &dyn Fn(&str)) -> String + Send + Sync,
        C: Fn(&str) -> (String, bool) + Send + Sync,
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
                ("GET", "/api/system") => {
                    let body = build_system_info_json();
                    let _ = write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                         Access-Control-Allow-Origin: *\r\n\
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
                ("POST", "/api/command") => {
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
                    let command = extract_json_string(body_str, "command").unwrap_or_default();

                    let (output, success) = on_command(&command);
                    let escaped_output = escape_json(&output);
                    let body = format!(
                        "{{\"output\":\"{escaped_output}\",\"success\":{success}}}"
                    );
                    let _ = write!(
                        stream,
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                         Access-Control-Allow-Origin: *\r\n\
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

/// Build JSON for GET /api/system by reading procfs/sysfs.
fn build_system_info_json() -> String {
    let cpu_percent = read_cpu_percent().unwrap_or(0);
    let cpu_temp = read_cpu_temp(); // Option<u32>, null if unavailable
    let (mem_used, mem_total) = read_memory().unwrap_or((0, 0));
    let os = std::env::consts::OS;
    let arch = std::env::consts::ARCH;
    let uptime = read_uptime().unwrap_or(0);

    let temp_str = match cpu_temp {
        Some(t) => t.to_string(),
        None => "null".to_string(),
    };
    format!(
        "{{\"cpu_percent\":{cpu_percent},\"cpu_temp\":{temp_str},\
         \"memory_used_mb\":{mem_used},\"memory_total_mb\":{mem_total},\
         \"os\":\"{os}\",\"arch\":\"{arch}\",\"uptime_seconds\":{uptime}}}"
    )
}

fn read_cpu_percent() -> Option<u32> {
    // Read /proc/stat twice with 100ms gap, compute delta
    let read_stat = || -> Option<(u64, u64)> {
        let s = std::fs::read_to_string("/proc/stat").ok()?;
        let line = s.lines().next()?;
        let vals: Vec<u64> = line.split_whitespace().skip(1)
            .filter_map(|v| v.parse().ok()).collect();
        if vals.len() < 4 { return None; }
        let idle = vals[3];
        let total: u64 = vals.iter().sum();
        Some((idle, total))
    };
    let (idle1, total1) = read_stat()?;
    std::thread::sleep(std::time::Duration::from_millis(100));
    let (idle2, total2) = read_stat()?;
    let d_idle = idle2.saturating_sub(idle1);
    let d_total = total2.saturating_sub(total1);
    if d_total == 0 { return Some(0); }
    Some((100 * (d_total - d_idle) / d_total) as u32)
}

fn read_cpu_temp() -> Option<u32> {
    let s = std::fs::read_to_string(
        "/sys/class/thermal/thermal_zone0/temp",
    ).ok()?;
    let millideg: u32 = s.trim().parse().ok()?;
    Some(millideg / 1000)
}

fn read_memory() -> Option<(u64, u64)> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total_kb = 0u64;
    let mut available_kb = 0u64;
    for line in s.lines() {
        if line.starts_with("MemTotal:") {
            total_kb = line.split_whitespace().nth(1)?.parse().ok()?;
        } else if line.starts_with("MemAvailable:") {
            available_kb = line.split_whitespace().nth(1)?.parse().ok()?;
        }
    }
    let total_mb = total_kb / 1024;
    let used_mb = total_mb - (available_kb / 1024);
    Some((used_mb, total_mb))
}

fn read_uptime() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/uptime").ok()?;
    let secs: f64 = s.split_whitespace().next()?.parse().ok()?;
    Some(secs as u64)
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
        // Tiling UI markers
        assert!(CHAT_HTML.contains("hyprbar"));
        assert!(CHAT_HTML.contains("/api/system"));
        assert!(CHAT_HTML.contains("/api/command"));
        assert!(CHAT_HTML.contains("/api/generate"));
        assert!(CHAT_HTML.contains("altKey")); // keybind references
        // Line count gate
        let lines = CHAT_HTML.lines().count();
        assert!(lines <= 500, "chat.html is {lines} lines — max 500");
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
    fn test_system_info_json_shape() {
        let json = build_system_info_json();
        assert!(json.contains("\"cpu_percent\""));
        assert!(json.contains("\"memory_used_mb\""));
        assert!(json.contains("\"memory_total_mb\""));
        assert!(json.contains("\"os\""));
        assert!(json.contains("\"arch\""));
        assert!(json.contains("\"uptime_seconds\""));
        // cpu_temp may be null
        assert!(json.contains("\"cpu_temp\""));
    }

    #[test]
    fn test_parse_command_request() {
        let body = r#"{"command":"/recall 5"}"#;
        let cmd = extract_json_string(body, "command").unwrap();
        assert_eq!(cmd, "/recall 5");
    }

    #[test]
    fn test_parse_command_request_empty() {
        let body = r#"{}"#;
        let cmd = extract_json_string(body, "command");
        assert!(cmd.is_none());
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

use olorin::interface::server::{build_system_json, get_chat_html};
use olorin::interface::server_http::{parse_content_length, extract_json_string};

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
fn test_build_system_json_shape() {
    let json = build_system_json(0, "{}");
    assert!(json.contains("\"cpu_temp\""));
    assert!(json.contains("\"memory_used_mb\""));
    assert!(json.contains("\"os\""));
    assert!(json.contains("\"arch\""));
    assert!(json.contains("\"uptime_seconds\""));
}

#[test]
fn test_get_chat_html_nonempty() {
    // Debug build reads from disk; in CI it may not exist — just check no panic
    let html = get_chat_html();
    assert!(!html.is_empty());
}

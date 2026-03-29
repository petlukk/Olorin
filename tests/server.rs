use olorin::interface::server::{
    parse_content_length, extract_json_string, escape_json,
    build_system_json, find_bridge, get_chat_html,
};

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
    let json = build_system_json(0);
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

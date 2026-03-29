use olorin::storage::json::{self, Object, Value};

#[test]
fn test_parse_object() {
    let input = r#"{"role":"user","text":"hello"}"#;
    let obj = json::parse(input.as_bytes()).unwrap();
    assert_eq!(obj.get_str("role"), Some("user"));
    assert_eq!(obj.get_str("text"), Some("hello"));
}

#[test]
fn test_serialize_object() {
    let mut obj = Object::new();
    obj.set("role", Value::Str("assistant".into()));
    obj.set("text", Value::Str("hi".into()));
    let out = json::serialize(&obj);
    assert!(out.contains(r#""role":"assistant""#));
    assert!(out.contains(r#""text":"hi""#));
}

#[test]
fn test_parse_nested() {
    let input = r#"{"content":[{"type":"text","text":"hi"}]}"#;
    let obj = json::parse(input.as_bytes()).unwrap();
    let arr = obj.get_array("content").unwrap();
    assert_eq!(arr.len(), 1);
}

#[test]
fn test_parse_numbers() {
    let input = r#"{"count":42,"rate":3.14,"neg":-1}"#;
    let obj = json::parse(input.as_bytes()).unwrap();
    assert_eq!(obj.get_i64("count"), Some(42));
    assert_eq!(obj.get_f64("rate"), Some(3.14));
    assert_eq!(obj.get_i64("neg"), Some(-1));
}

#[test]
fn test_parse_bool_null() {
    let input = r#"{"flag":true,"empty":null}"#;
    let obj = json::parse(input.as_bytes()).unwrap();
    assert_eq!(obj.get_bool("flag"), Some(true));
    assert!(obj.is_null("empty"));
}

#[test]
fn test_roundtrip() {
    let input = r#"{"name":"olorin","version":6,"active":true,"data":null}"#;
    let obj = json::parse(input.as_bytes()).unwrap();
    let output = json::serialize(&obj);
    let obj2 = json::parse(output.as_bytes()).unwrap();
    assert_eq!(obj2.get_str("name"), Some("olorin"));
    assert_eq!(obj2.get_i64("version"), Some(6));
    assert_eq!(obj2.get_bool("active"), Some(true));
    assert!(obj2.is_null("data"));
}

#[test]
fn test_string_escapes() {
    let input = r#"{"msg":"hello\nworld\t\"quoted\""}"#;
    let obj = json::parse(input.as_bytes()).unwrap();
    assert_eq!(obj.get_str("msg"), Some("hello\nworld\t\"quoted\""));
}

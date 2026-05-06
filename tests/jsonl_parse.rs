//! Unit tests for `storage::jsonl_parse` helpers.

use olorin::storage::jsonl_parse::{
    build_escaped_quote_set, classify_scalar, decode_byte_array, looks_iso8601, ScalarKind,
};

#[test]
fn no_backslashes_no_escapes() {
    let escaped = build_escaped_quote_set(&[5, 10, 15], &[]);
    assert!(escaped.is_empty());
}

#[test]
fn single_backslash_before_quote_escapes() {
    let escaped = build_escaped_quote_set(&[5, 10], &[9]);
    assert!(escaped.contains(&10), "quote at pos 10 should be escaped");
    assert!(!escaped.contains(&5));
}

#[test]
fn double_backslash_does_not_escape() {
    // `\\"` — two backslashes pair into one literal `\`, then `"` is real.
    let escaped = build_escaped_quote_set(&[10], &[8, 9]);
    assert!(!escaped.contains(&10), "double backslash makes quote real");
}

#[test]
fn triple_backslash_escapes() {
    // `\\\"` — three backslashes: pair + odd → next char escaped.
    let escaped = build_escaped_quote_set(&[10], &[7, 8, 9]);
    assert!(escaped.contains(&10));
}

#[test]
fn iso8601_recognized() {
    assert!(looks_iso8601("2026-05-06T08:00:00"));
    assert!(looks_iso8601("2026-05-06T08:00:00Z"));
    assert!(looks_iso8601("2026-05-06T08:00:00.123456+02:00"));
    assert!(looks_iso8601("2026-05-06 08:00:00"));
    assert!(!looks_iso8601("2026-05-06"));
    assert!(!looks_iso8601("not a date"));
    assert!(!looks_iso8601(""));
}

#[test]
fn byte_array_decodes_simple() {
    // [72,105] = "Hi"
    let s = decode_byte_array(b"[72,105]").expect("decode succeeds");
    assert_eq!(s, "Hi");
}

#[test]
fn byte_array_handles_whitespace() {
    let s = decode_byte_array(b"[ 72 , 105 ]").expect("decode succeeds");
    assert_eq!(s, "Hi");
}

#[test]
fn byte_array_rejects_out_of_range() {
    assert!(decode_byte_array(b"[256]").is_none());
}

#[test]
fn byte_array_rejects_non_array() {
    assert!(decode_byte_array(b"\"hello\"").is_none());
    assert!(decode_byte_array(b"[]").is_none());
}

#[test]
fn classify_scalar_basics() {
    assert_eq!(classify_scalar(b"42"),    ScalarKind::Number);
    assert_eq!(classify_scalar(b"-1.5"),  ScalarKind::Number);
    assert_eq!(classify_scalar(b"true"),  ScalarKind::Bool);
    assert_eq!(classify_scalar(b"false"), ScalarKind::Bool);
    assert_eq!(classify_scalar(b"null"),  ScalarKind::Skip);
    assert_eq!(classify_scalar(b""),      ScalarKind::Skip);
    assert_eq!(classify_scalar(b"abc"),   ScalarKind::Skip);
}

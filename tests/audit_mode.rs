//! Tests for audit mode: the AuditLog module + end-to-end
//! dispatch-emits-events behavior.

use olorin::core::audit::{AuditLog, AuditValue};
use olorin::core::router::DispatchContext;

fn tmp_path(suffix: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "olorin_audit_{}_{}.jsonl", std::process::id(), suffix
    ))
}

#[test]
fn audit_log_writes_valid_jsonl_per_emit() {
    let path = tmp_path("jsonl_per_emit");
    let log = AuditLog::open(&path).expect("open");
    let turn = log.next_turn();
    log.emit(turn, "test", &[
        ("input_len", AuditValue::I64(42)),
        ("blocked", AuditValue::Bool(false)),
        ("name", AuditValue::Str("eajson")),
    ]);
    let content = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 1, "expected one line per emit");
    let line = lines[0];
    // Cheap structural check — full JSON parse would pull a dep; we
    // verify the keys we care about are all present and the line is
    // well-formed JSON-shaped (starts with {, ends with }).
    assert!(line.starts_with('{') && line.ends_with('}'),
        "not JSON-shaped: {line}");
    assert!(line.contains("\"ts_ms\":"), "missing ts_ms: {line}");
    assert!(line.contains("\"turn\":1"), "missing/wrong turn: {line}");
    assert!(line.contains("\"phase\":\"test\""), "missing phase: {line}");
    assert!(line.contains("\"input_len\":42"), "missing input_len: {line}");
    assert!(line.contains("\"blocked\":false"), "missing bool: {line}");
    assert!(line.contains("\"name\":\"eajson\""), "missing string field: {line}");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn audit_log_escapes_string_fields() {
    let path = tmp_path("escape");
    let log = AuditLog::open(&path).expect("open");
    let turn = log.next_turn();
    log.emit(turn, "test", &[
        ("text", AuditValue::Str("he said \"hello\" \\backslash\nnewline")),
    ]);
    let content = std::fs::read_to_string(&path).unwrap();
    // Quotes escaped, backslash escaped, newline escaped. Output must
    // remain on a single line (trailing newline only at end).
    assert!(content.contains("\\\""), "quote not escaped: {content}");
    assert!(content.contains("\\\\"), "backslash not escaped: {content}");
    assert!(content.contains("\\n"), "newline not escaped: {content}");
    // Should be exactly one line.
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 1, "embedded newline broke the JSONL line: {content}");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn audit_log_assigns_turns_monotonically() {
    let path = tmp_path("turns");
    let log = AuditLog::open(&path).expect("open");
    assert_eq!(log.next_turn(), 1);
    assert_eq!(log.next_turn(), 2);
    assert_eq!(log.next_turn(), 3);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn dispatch_emits_input_and_command_events_for_slash() {
    olorin::kernels::ffi::init().expect("kernel init");
    let path = tmp_path("dispatch_slash");
    let log = AuditLog::open(&path).expect("open");
    let mut ctx = DispatchContext::new_strict(None).with_audit(log);
    let _resp = ctx.dispatch("/help");
    let content = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 2, "expected input + result events: {content}");
    assert!(lines[0].contains("\"phase\":\"input\""),
        "first event must be input: {}", lines[0]);
    assert!(lines[0].contains("\"input_len\":5"),
        "input_len wrong: {}", lines[0]);
    assert!(lines[1].contains("\"phase\":\"command\""),
        "second event must be command (/help is a slash cmd): {}", lines[1]);
    assert!(lines[1].contains("\"wall_us\":"),
        "result event must have wall_us: {}", lines[1]);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn dispatch_emits_strict_refused_for_llm_fallback() {
    olorin::kernels::ffi::init().expect("kernel init");
    let path = tmp_path("dispatch_strict");
    let log = AuditLog::open(&path).expect("open");
    let mut ctx = DispatchContext::new_strict(None).with_audit(log);
    let _resp = ctx.dispatch("hello world");
    let content = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    assert_eq!(lines.len(), 2, "expected input + result events: {content}");
    assert!(lines[1].contains("\"phase\":\"strict_refused\""),
        "second event must be strict_refused: {}", lines[1]);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn dispatch_without_audit_emits_nothing() {
    olorin::kernels::ffi::init().expect("kernel init");
    let mut ctx = DispatchContext::new_strict(None);
    // No with_audit — the audit field is None, no file should be
    // touched. Just verify it doesn't crash and produces a normal
    // response.
    let resp = ctx.dispatch("/help");
    assert!(resp.text.contains("Commands:"),
        "help should still work without audit: {}", resp.text);
}

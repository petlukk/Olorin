//! Tests for the eajson rune — JSON Lines summarizer.

use olorin::runes::{run_rune, RUNES, OutputSafety};

fn ensure_kernels() {
    olorin::kernels::ffi::init().expect("kernel init");
}

#[test]
fn eajson_is_registered() {
    let found = RUNES.iter().any(|r| r.name() == "eajson");
    assert!(found, "eajson rune missing from registry");
}

#[test]
fn eajson_output_safety_is_untrusted() {
    let r = RUNES.iter().find(|r| r.name() == "eajson")
        .expect("eajson registered");
    assert_eq!(r.output_safety(), OutputSafety::UntrustedQuoted,
        "JSON values are file-derived; must be wrapped before reaching LLM");
}

#[test]
fn eajson_summarizes_fixture() {
    ensure_kernels();
    // Stage the repo fixture in /tmp (allowlist requires ~ or /tmp).
    let src = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/runes/tiny.jsonl");
    let dst = std::env::temp_dir().join(format!(
        "olorin_eajson_test_{}.jsonl", std::process::id()
    ));
    std::fs::copy(&src, &dst).expect("copy fixture to /tmp");

    let result = run_rune("eajson", dst.to_string_lossy().as_ref())
        .expect("eajson should be registered and runnable");
    assert!(result.success, "rune failed: {}", result.answer);

    let a = &result.answer;
    // 8 data rows in the fixture.
    assert!(a.contains("rows: 8"), "row count missing or wrong: {a}");
    // 6 top-level keys.
    assert!(a.contains("keys: 6"), "key count missing or wrong: {a}");
    // Numeric keys reported with stats.
    assert!(a.contains("status (number):"), "status not classified as number: {a}");
    assert!(a.contains("latency_ms (number):"), "latency_ms not number: {a}");
    // Text key with top values.
    assert!(a.contains("level (text):"), "level not text: {a}");
    assert!(a.contains("src_ip (text):"), "src_ip not text: {a}");
    // Bool key.
    assert!(a.contains("cached (bool):"), "cached not bool: {a}");
    // Bool counts: 4 true (info+cached), 4 false (warn/error+not-cached).
    assert!(a.contains("true=4, false=4"), "bool counts wrong: {a}");

    let _ = std::fs::remove_file(&dst);
}

#[test]
fn eajson_handles_mixed_type_key() {
    ensure_kernels();
    let dst = std::env::temp_dir().join(format!(
        "olorin_eajson_mixed_{}.jsonl", std::process::id()
    ));
    // `id` appears as number on first line, string on second — must be
    // classified Mixed and reported as such, not silently coerced.
    std::fs::write(&dst, b"{\"id\":1,\"name\":\"a\"}\n{\"id\":\"two\",\"name\":\"b\"}\n").unwrap();

    let result = run_rune("eajson", dst.to_string_lossy().as_ref())
        .expect("eajson runnable");
    assert!(result.success, "rune failed: {}", result.answer);
    assert!(result.answer.contains("id (mixed):"),
        "mixed-type key not surfaced as Mixed: {}", result.answer);
    let _ = std::fs::remove_file(&dst);
}

#[test]
fn eajson_rejects_outside_allowlist() {
    let result = run_rune("eajson", "/etc/passwd")
        .expect("eajson runnable");
    assert!(!result.success);
    assert!(result.answer.contains("outside allowlist"),
        "unexpected error: {}", result.answer);
}

// ── v2 feature tests ──────────────────────────────────────────────────────────

/// Helper: copy a repo fixture into /tmp (allowlisted) and return the path.
fn stage_fixture(name: &str) -> std::path::PathBuf {
    let src = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(format!("tests/fixtures/runes/{name}"));
    let dst = std::env::temp_dir().join(format!(
        "olorin_eajson_v2_{}_{}", std::process::id(), name
    ));
    std::fs::copy(&src, &dst).expect("copy fixture to /tmp");
    dst
}

#[test]
fn eajson_handles_escaped_quotes() {
    ensure_kernels();
    let dst = stage_fixture("escaped.jsonl");
    let result = run_rune("eajson", dst.to_string_lossy().as_ref())
        .expect("eajson runnable");
    assert!(result.success, "rune failed: {}", result.answer);
    let a = &result.answer;
    // 5 rows, 2 keys (id, msg). The fixture has repeats so cardinality
    // filter doesn't suppress msg. If escape handling were broken, the
    // quote-pair walk would mis-align and either: msg would be misclassified,
    // row count would be wrong, or extra spurious keys would appear.
    assert!(a.contains("rows: 5"), "wrong row count: {a}");
    assert!(a.contains("id (number):"), "id key missing or misclassified: {a}");
    assert!(a.contains("msg (text):"), "msg key missing or wrong type: {a}");
    // The "he said \"hello\"" value appears 3 times → top of frequency list.
    // If escapes weren't handled, the value-quote-pair walk would slice the
    // string at the inner `\"` and we'd see fragments instead of full strings.
    assert!(a.contains("he said \"hello\""),
        "escaped-quote value not preserved verbatim: {a}");
    let _ = std::fs::remove_file(&dst);
}

#[test]
fn eajson_flattens_nested_objects() {
    ensure_kernels();
    let dst = stage_fixture("nested.jsonl");
    let result = run_rune("eajson", dst.to_string_lossy().as_ref())
        .expect("eajson runnable");
    assert!(result.success, "rune failed: {}", result.answer);
    let a = &result.answer;
    assert!(a.contains("rows: 4"), "wrong row count: {a}");
    // Nested fields should be flattened to parent.child.
    assert!(a.contains("http.status (number):"),
        "http.status not flattened: {a}");
    assert!(a.contains("http.method (text):"),
        "http.method not flattened: {a}");
    // Top-level fields should still appear (fixture has /home & /api each
    // appearing twice so path isn't suppressed by cardinality filter).
    assert!(a.contains("id (number):"), "id top-level missing: {a}");
    assert!(a.contains("path (text):"), "path top-level missing: {a}");
    let _ = std::fs::remove_file(&dst);
}

#[test]
fn eajson_decodes_byte_arrays() {
    ensure_kernels();
    let dst = stage_fixture("bytes.jsonl");
    let result = run_rune("eajson", dst.to_string_lossy().as_ref())
        .expect("eajson runnable");
    assert!(result.success, "rune failed: {}", result.answer);
    let a = &result.answer;
    // MESSAGE is a byte array; v2 decodes it as UTF-8 text and the values
    // should appear as top-3.
    assert!(a.contains("MESSAGE (text):"),
        "MESSAGE not decoded as text: {a}");
    assert!(a.contains("Hello World"),
        "decoded byte-array text missing: {a}");
}

#[test]
fn eajson_suppresses_high_cardinality_text() {
    ensure_kernels();
    let dst = stage_fixture("cursors.jsonl");
    let result = run_rune("eajson", dst.to_string_lossy().as_ref())
        .expect("eajson runnable");
    assert!(result.success, "rune failed: {}", result.answer);
    let a = &result.answer;
    // 4 rows, 4 unique cursor values (every value unique) → suppressed.
    // 3 keys total but only 2 shown.
    assert!(a.contains("rows: 4"), "wrong row count: {a}");
    assert!(a.contains("(+1 high-cardinality keys suppressed)"),
        "cursor-noise suppression notice missing: {a}");
    assert!(!a.contains("cursor (text):"), "cursor field should be suppressed: {a}");
    // level appears 3x info + 1x warn — not suppressed.
    assert!(a.contains("level (text):"), "level should appear: {a}");
    let _ = std::fs::remove_file(&dst);
}

#[test]
fn eajson_detects_iso8601_timestamps() {
    ensure_kernels();
    // Reuse the original tiny.jsonl which has ts as ISO-8601.
    let dst = stage_fixture("tiny.jsonl");
    let result = run_rune("eajson", dst.to_string_lossy().as_ref())
        .expect("eajson runnable");
    assert!(result.success, "rune failed: {}", result.answer);
    let a = &result.answer;
    assert!(a.contains("ts (timestamp):"),
        "ts not classified as timestamp: {a}");
    assert!(a.contains("range:"),
        "timestamp range report missing: {a}");
    assert!(a.contains("2026-05-06T08:00:00Z .. 2026-05-06T08:00:07Z"),
        "timestamp range bounds missing or wrong: {a}");
    let _ = std::fs::remove_file(&dst);
}

#[test]
fn eajson_does_not_byte_decode_numeric_arrays() {
    // A JSON numeric array (latencies, ports) must NOT be reinterpreted as a
    // systemd binary MESSAGE string. Only the MESSAGE key is byte-decoded;
    // ordinary arrays are skipped, and the rest of the line still parses.
    ensure_kernels();
    let dst = std::env::temp_dir().join(format!(
        "olorin_eajson_numarr_{}.jsonl", std::process::id()
    ));
    // `status` repeats (low cardinality) so it isn't high-cardinality-
    // suppressed; it sits AFTER the array, proving the array skip doesn't
    // drop the following key.
    std::fs::write(&dst,
        b"{\"id\":1,\"latencies\":[12,45,78],\"status\":\"ok\"}\n\
          {\"id\":2,\"latencies\":[99,23,11],\"status\":\"ok\"}\n\
          {\"id\":3,\"latencies\":[5,5,5],\"status\":\"err\"}\n\
          {\"id\":4,\"latencies\":[1,2,3],\"status\":\"ok\"}\n").unwrap();
    struct Tmp(std::path::PathBuf);
    impl Drop for Tmp { fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); } }
    let _g = Tmp(dst.clone());

    let result = run_rune("eajson", dst.to_string_lossy().as_ref())
        .expect("eajson runnable");
    assert!(result.success, "rune failed: {}", result.answer);
    let a = &result.answer;
    // The numeric array is skipped, NOT emitted as a garbage byte-string.
    assert!(!a.contains("latencies"), "numeric array byte-decoded: {a}");
    // The scalar fields around it still parse correctly.
    assert!(a.contains("status (text):"), "field after array not parsed: {a}");
    assert!(a.contains("id (number):"), "id not parsed: {a}");
}

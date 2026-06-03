//! Tests for the ealog rune — log severity scanner.

use olorin::runes::{run_rune, RUNES, OutputSafety};

fn ensure_kernels() {
    olorin::kernels::ffi::init().expect("kernel init");
}

fn unique_path(stem: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "olorin_ealog_{stem}_{}.log", std::process::id()
    ))
}

#[test]
fn ealog_is_registered() {
    let found = RUNES.iter().any(|r| r.name() == "ealog");
    assert!(found, "ealog rune missing from registry");
}

#[test]
fn ealog_output_safety_is_untrusted() {
    let r = RUNES.iter().find(|r| r.name() == "ealog")
        .expect("ealog registered");
    assert_eq!(
        r.output_safety(),
        OutputSafety::UntrustedQuoted,
        "log contents include file-derived strings; must be wrapped"
    );
}

#[test]
fn ealog_summarizes_synthetic_log() {
    ensure_kernels();
    let dst = unique_path("synthetic");
    let mut buf = String::new();
    for line in 0..500 {
        let level = ["INFO", "INFO", "INFO", "WARN", "INFO", "DEBUG", "ERROR"][line % 7];
        buf.push_str(&format!("2026-05-11T12:34:{:02} [{level}] line {line}: handler took 12ms\n", line % 60));
    }
    std::fs::write(&dst, &buf).unwrap();
    let result = run_rune("ealog", dst.to_string_lossy().as_ref())
        .expect("ealog runnable");
    assert!(result.success, "rune failed: {}", result.answer);
    let a = &result.answer;

    assert!(a.contains("lines:   500"), "wrong line count: {a}");
    assert!(a.contains("INFO"), "INFO missing: {a}");
    assert!(a.contains("DEBUG"), "DEBUG missing: {a}");
    assert!(a.contains("WARN"), "WARN missing: {a}");
    assert!(a.contains("ERROR"), "ERROR missing: {a}");

    let _ = std::fs::remove_file(&dst);
}

#[test]
fn ealog_detects_jsonl_format() {
    ensure_kernels();
    let dst = unique_path("jsonl");
    let mut buf = String::new();
    for i in 0..40 {
        buf.push_str(&format!(
            "{{\"ts\":{i},\"level\":\"{}\",\"msg\":\"hi\"}}\n",
            if i % 3 == 0 { "ERROR" } else { "INFO" }
        ));
    }
    std::fs::write(&dst, &buf).unwrap();
    let result = run_rune("ealog", dst.to_string_lossy().as_ref())
        .expect("runnable");
    assert!(result.success);
    let a = &result.answer;
    assert!(a.contains("format:  jsonl"), "did not detect jsonl: {a}");
    let _ = std::fs::remove_file(&dst);
}

#[test]
fn ealog_handles_empty_file() {
    ensure_kernels();
    let dst = unique_path("empty");
    std::fs::write(&dst, b"").unwrap();
    let result = run_rune("ealog", dst.to_string_lossy().as_ref())
        .expect("runnable");
    assert!(result.success);
    assert!(result.answer.contains("lines:   0"), "should report 0 lines: {}", result.answer);
    let _ = std::fs::remove_file(&dst);
}

#[test]
fn ealog_handles_non_log_file() {
    ensure_kernels();
    let dst = unique_path("not_a_log");
    std::fs::write(&dst, b"hello world\nthis is not a log file at all\njust some text\n").unwrap();
    let result = run_rune("ealog", dst.to_string_lossy().as_ref())
        .expect("runnable");
    assert!(result.success, "should succeed even on non-log");
    let a = &result.answer;
    assert!(a.contains("no severity keywords found"), "missing zero-severity hint: {a}");
    let _ = std::fs::remove_file(&dst);
}

#[test]
fn ealog_word_boundary_no_false_positives() {
    ensure_kernels();
    let dst = unique_path("boundary");
    let content = b"ERROR_HANDLER fired in INFO_TYPE module\nWARN_THRESHOLD = 42\nDEBUG_MODE = true\n";
    std::fs::write(&dst, content).unwrap();
    let result = run_rune("ealog", dst.to_string_lossy().as_ref())
        .expect("runnable");
    assert!(result.success);
    let a = &result.answer;
    assert!(
        a.contains("no severity keywords found"),
        "ERROR_HANDLER etc. should NOT count: {a}"
    );
    let _ = std::fs::remove_file(&dst);
}

#[test]
fn ealog_rejects_outside_allowlist() {
    let result = run_rune("ealog", "/etc/passwd")
        .expect("runnable");
    assert!(!result.success);
    assert!(
        result.answer.contains("outside allowlist"),
        "unexpected error: {}",
        result.answer
    );
}

#[test]
fn ealog_rejects_missing_path() {
    let result = run_rune("ealog", "")
        .expect("runnable");
    assert!(!result.success);
    assert!(
        result.answer.contains("usage:"),
        "expected usage hint: {}",
        result.answer
    );
}

#[test]
fn ealog_handles_unterminated_last_line() {
    ensure_kernels();
    let dst = unique_path("no_trailing_nl");
    // 3 lines, last one missing trailing \n
    std::fs::write(&dst, b"line one\nline two\nline three no newline").unwrap();
    let result = run_rune("ealog", dst.to_string_lossy().as_ref())
        .expect("runnable");
    assert!(result.success);
    assert!(result.answer.contains("lines:   3"),
        "should count 3 lines even without trailing newline: {}", result.answer);
    let _ = std::fs::remove_file(&dst);
}

#[test]
fn ealog_emits_high_severity_sample() {
    ensure_kernels();
    let dst = unique_path("with_errors");
    let mut buf = String::new();
    for i in 0..40 {
        let level = if i == 7 || i == 21 { "ERROR" } else { "INFO" };
        let msg = if i == 7 {
            "connection reset by peer (upstream timeout 5s)".to_string()
        } else if i == 21 {
            "auth failed user=anonymous token=invalid".to_string()
        } else {
            format!("ok request {i}")
        };
        buf.push_str(&format!("2026-05-11T12:34:{:02} [{level}] {msg}\n", i % 60));
    }
    std::fs::write(&dst, &buf).unwrap();
    let result = run_rune("ealog", dst.to_string_lossy().as_ref())
        .expect("runnable");
    assert!(result.success);
    let a = &result.answer;
    assert!(a.contains("high-severity sample:"), "no sample section: {a}");
    assert!(a.contains("connection reset by peer"),
        "first ERROR line content missing: {a}");
    assert!(a.contains("auth failed user=anonymous"),
        "second ERROR line content missing: {a}");
    // Lines 8 and 22 are the ERROR lines (1-indexed).
    assert!(a.contains("L8:"), "wrong line number for first ERROR: {a}");
    assert!(a.contains("L22:"), "wrong line number for second ERROR: {a}");
    let _ = std::fs::remove_file(&dst);
}

#[test]
fn ealog_no_sample_section_when_no_high_severity() {
    ensure_kernels();
    let dst = unique_path("info_only");
    let mut buf = String::new();
    for i in 0..40 {
        buf.push_str(&format!("2026-05-11T12:34:00 [INFO] line {i}\n"));
    }
    std::fs::write(&dst, &buf).unwrap();
    let result = run_rune("ealog", dst.to_string_lossy().as_ref())
        .expect("runnable");
    assert!(result.success);
    assert!(
        !result.answer.contains("high-severity sample:"),
        "should NOT emit sample section when no ERROR/FATAL: {}",
        result.answer
    );
    let _ = std::fs::remove_file(&dst);
}

#[test]
fn ealog_sample_caps_at_five() {
    ensure_kernels();
    let dst = unique_path("many_errors");
    let mut buf = String::new();
    for i in 0..30 {
        let level = if i % 2 == 0 { "ERROR" } else { "INFO" };
        buf.push_str(&format!("2026-05-11T12:34:00 [{level}] event {i}\n"));
    }
    std::fs::write(&dst, &buf).unwrap();
    let result = run_rune("ealog", dst.to_string_lossy().as_ref())
        .expect("runnable");
    assert!(result.success);
    let a = &result.answer;
    let sample_idx = a.find("high-severity sample:").expect("sample section present");
    let sample_section = &a[sample_idx..];
    let line_count = sample_section.matches("  L").count();
    assert_eq!(line_count, 5, "should show exactly 5 sample lines, got {line_count}: {a}");
    let _ = std::fs::remove_file(&dst);
}

#[test]
fn ealog_sample_includes_fatal_alongside_error() {
    ensure_kernels();
    let dst = unique_path("error_and_fatal");
    let content = b"2026-05-11T01:00 [INFO] starting up\n\
                    2026-05-11T01:01 [ERROR] connection refused\n\
                    2026-05-11T01:02 [INFO] retrying\n\
                    2026-05-11T01:03 [FATAL] aborting after 3 failures\n";
    std::fs::write(&dst, content).unwrap();
    let result = run_rune("ealog", dst.to_string_lossy().as_ref())
        .expect("runnable");
    assert!(result.success);
    let a = &result.answer;
    assert!(a.contains("connection refused"), "ERROR sample missing: {a}");
    assert!(a.contains("aborting after 3 failures"), "FATAL sample missing: {a}");
    let _ = std::fs::remove_file(&dst);
}

#[test]
fn ealog_counts_lowercase_and_mixed_case() {
    // Real logs (Go zap/logrus, Python logging, nginx, journald) emit
    // lowercase or mixed-case severities. They must be counted, not reported
    // as "no severity keywords found".
    ensure_kernels();
    let dst = unique_path("lowercase");
    let content = b"2026-06-03 error: connection refused\n\
                    2026-06-03 warn: disk full\n\
                    2026-06-03 info: started\n\
                    2026-06-03 Error: retrying\n";
    std::fs::write(&dst, content).unwrap();

    let result = run_rune("ealog", &format!("--json {}", dst.to_string_lossy()))
        .expect("ealog runnable");
    assert!(result.success, "rune failed: {}", result.answer);
    let a = &result.answer;
    // lowercase `error` + mixed-case `Error` both count → 2 ERROR.
    assert!(a.contains("\"name\":\"ERROR\",\"count\":2"),
        "lowercase/mixed ERROR not counted: {a}");
    assert!(a.contains("\"name\":\"WARN\",\"count\":1"), "lowercase WARN missing: {a}");
    assert!(a.contains("\"name\":\"INFO\",\"count\":1"), "lowercase INFO missing: {a}");
    assert!(!a.contains("no severity keywords found"), "{a}");
    let _ = std::fs::remove_file(&dst);
}

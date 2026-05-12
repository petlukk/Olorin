//! eatime rune: ISO-8601 timestamp histogram. Covers SIMD kernel
//! correctness + Rust orchestration + edge cases.

use olorin::runes::output::RuneOutput;
use olorin::runes::{run_rune, RUNES, OutputSafety};
use std::io::Write;

fn write_tmp(name: &str, bytes: &[u8]) -> String {
    let path = format!("/tmp/{name}");
    let mut f = std::fs::File::create(&path).expect("tmp create");
    f.write_all(bytes).expect("tmp write");
    path
}

fn parse_answer(answer: &str) -> RuneOutput {
    RuneOutput::from_json(answer.as_bytes())
        .unwrap_or_else(|e| panic!("not parseable JSON: {e}\nanswer={answer}"))
}

#[test]
fn eatime_is_registered() {
    olorin::kernels::ffi::init().unwrap();
    let found = RUNES.iter().any(|r| r.name() == "eatime");
    assert!(found, "eatime missing from registry");
}

#[test]
fn eatime_output_safety_is_untrusted() {
    let r = RUNES.iter().find(|r| r.name() == "eatime")
        .expect("eatime registered");
    assert_eq!(r.output_safety(), OutputSafety::UntrustedQuoted);
}

#[test]
fn eatime_counts_hours_in_synthetic_log() {
    olorin::kernels::ffi::init().unwrap();
    // Twelve log lines, six at 06:xx, three at 07:xx, three at 23:xx.
    let log = b"\
2026-05-11T06:00:00 INFO startup
2026-05-11T06:14:32 WARN cache miss
2026-05-11T06:18:00 ERROR connection reset
2026-05-11T06:42:11 INFO retrying
2026-05-11T06:50:00 ERROR auth failed
2026-05-11T06:59:59 FATAL aborting
2026-05-11T07:01:00 INFO restart
2026-05-11T07:15:00 INFO healthy
2026-05-11T07:30:00 INFO heartbeat
2026-05-11T23:00:00 INFO end of shift
2026-05-11T23:15:00 INFO log rotated
2026-05-11T23:59:59 WARN clock skew
";
    let path = write_tmp("olorin_eatime_synth.log", log);
    let result = run_rune("eatime", &format!("--json {path}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);

    let out = parse_answer(&result.answer);
    assert_eq!(out.totals.rows, 12, "12 ISO timestamps expected");
    let by_hour: std::collections::HashMap<&str, u64> =
        out.categories.iter().map(|c| (c.name.as_str(), c.count)).collect();
    assert_eq!(by_hour.get("06:00"), Some(&6));
    assert_eq!(by_hour.get("07:00"), Some(&3));
    assert_eq!(by_hour.get("23:00"), Some(&3));
    // Every hour-of-day appears even when count is 0 — deterministic
    // output makes downstream eadiff chaining clean.
    assert_eq!(out.categories.len(), 24);
    assert_eq!(by_hour.get("00:00"), Some(&0));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn eatime_finds_timestamps_anywhere_in_line() {
    olorin::kernels::ffi::init().unwrap();
    // Timestamps embedded mid-line (not only at line starts) — verifies
    // the kernel scans the whole buffer, not just per-line.
    let log = b"\
[INFO] event=login user=alice ts=2026-05-11T08:30:00 ok=true
[WARN] event=retry  user=bob   ts=2026-05-11T08:45:12 ok=false
[ERROR] event=login user=carol ts=2026-05-11T15:22:00 ok=false
";
    let path = write_tmp("olorin_eatime_midline.log", log);
    let result = run_rune("eatime", &format!("--json {path}")).unwrap();
    let out = parse_answer(&result.answer);
    assert_eq!(out.totals.rows, 3);
    let by_hour: std::collections::HashMap<&str, u64> =
        out.categories.iter().map(|c| (c.name.as_str(), c.count)).collect();
    assert_eq!(by_hour.get("08:00"), Some(&2));
    assert_eq!(by_hour.get("15:00"), Some(&1));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn eatime_rejects_invalid_hours() {
    olorin::kernels::ffi::init().unwrap();
    // Three timestamps; one has an invalid hour (25). The Rust-side
    // hour-range guard drops it without touching the histogram.
    let log = b"\
2026-05-11T10:00:00 ok
2026-05-11T25:00:00 bogus-hour
2026-05-11T11:00:00 ok
";
    let path = write_tmp("olorin_eatime_badhour.log", log);
    let result = run_rune("eatime", &format!("--json {path}")).unwrap();
    let out = parse_answer(&result.answer);
    assert_eq!(out.totals.rows, 2, "bogus hour 25 should be dropped");
    let by_hour: std::collections::HashMap<&str, u64> =
        out.categories.iter().map(|c| (c.name.as_str(), c.count)).collect();
    assert_eq!(by_hour.get("10:00"), Some(&1));
    assert_eq!(by_hour.get("11:00"), Some(&1));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn eatime_no_timestamps_reports_zero() {
    olorin::kernels::ffi::init().unwrap();
    let log = b"\
no timestamps here just text
another line same story
2026-05-11 missing the T part
2026/05/11T10:00:00 wrong separator
";
    let path = write_tmp("olorin_eatime_empty.log", log);
    let result = run_rune("eatime", &format!("--json {path}")).unwrap();
    let out = parse_answer(&result.answer);
    assert_eq!(out.totals.rows, 0);
    assert!(out.categories.iter().all(|c| c.count == 0));
    assert_eq!(out.categories.len(), 24);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn eatime_handles_tiny_buffer() {
    olorin::kernels::ffi::init().unwrap();
    // <32 bytes triggers the scalar fallback path in the kernel.
    let log = b"2026-05-11T03:14:15";
    let path = write_tmp("olorin_eatime_tiny.log", log);
    let result = run_rune("eatime", &format!("--json {path}")).unwrap();
    let out = parse_answer(&result.answer);
    assert_eq!(out.totals.rows, 1, "scalar fallback must catch the single match");
    let by_hour: std::collections::HashMap<&str, u64> =
        out.categories.iter().map(|c| (c.name.as_str(), c.count)).collect();
    assert_eq!(by_hour.get("03:00"), Some(&1));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn eatime_text_mode_shows_peak_hour() {
    olorin::kernels::ffi::init().unwrap();
    let log = b"\
2026-05-11T09:00:00 a
2026-05-11T09:30:00 b
2026-05-11T15:00:00 c
";
    let path = write_tmp("olorin_eatime_peak.log", log);
    let result = run_rune("eatime", &path).unwrap();
    assert!(result.success);
    let a = &result.answer;
    assert!(a.contains("timestamps:  3"), "wrong total: {a}");
    assert!(a.contains("09:00"), "missing 09:00 row: {a}");
    assert!(a.contains("peak: 09:00"), "missing peak line: {a}");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn eatime_json_mode_error_path_is_structured() {
    olorin::kernels::ffi::init().unwrap();
    let result = run_rune("eatime", "--json /tmp/does_not_exist_xyz_123abc.log").unwrap();
    assert!(!result.success);
    let out = parse_answer(&result.answer);
    assert!(!out.success);
    assert!(out.error.expect("error populated").contains("not found"));
    assert!(out.categories.is_empty());
}

#[test]
fn eatime_rejects_outside_allowlist() {
    olorin::kernels::ffi::init().unwrap();
    let result = run_rune("eatime", "/etc/passwd").unwrap();
    assert!(!result.success);
    assert!(result.answer.contains("outside allowlist"));
}

#[test]
fn eatime_emits_only_categories_not_fields() {
    olorin::kernels::ffi::init().unwrap();
    let log = b"2026-05-11T10:00:00 hello\n";
    let path = write_tmp("olorin_eatime_no_fields.log", log);
    let result = run_rune("eatime", &format!("--json {path}")).unwrap();
    let out = parse_answer(&result.answer);
    assert!(out.fields.is_empty(), "eatime should not emit fields[]");
    assert!(!out.categories.is_empty(), "eatime must emit categories[]");
    let _ = std::fs::remove_file(&path);
}

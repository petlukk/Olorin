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
fn eatime_weekday_bucket_groups_by_day_of_week() {
    olorin::kernels::ffi::init().unwrap();
    // 2026-05-11 is a Monday (calendar verified). Build a log with
    // 3 timestamps on Monday 2026-05-11, 2 on Tuesday 2026-05-12,
    // and 1 on Saturday 2026-05-16.
    let log = b"\
2026-05-11T08:00:00 mon a
2026-05-11T09:00:00 mon b
2026-05-11T10:00:00 mon c
2026-05-12T08:00:00 tue a
2026-05-12T11:00:00 tue b
2026-05-16T14:30:00 sat single
";
    let path = write_tmp("olorin_eatime_weekday.log", log);
    let result = run_rune("eatime", &format!("--json --bucket weekday {path}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);
    let out = parse_answer(&result.answer);
    assert_eq!(out.totals.rows, 6, "6 timestamps in fixture");
    assert_eq!(out.categories.len(), 7,
        "weekday bucket emits all 7 days deterministically");
    let by_name: std::collections::HashMap<&str, u64> =
        out.categories.iter().map(|c| (c.name.as_str(), c.count)).collect();
    assert_eq!(by_name.get("Mon"), Some(&3));
    assert_eq!(by_name.get("Tue"), Some(&2));
    assert_eq!(by_name.get("Wed"), Some(&0));
    assert_eq!(by_name.get("Sat"), Some(&1));
    assert_eq!(by_name.get("Sun"), Some(&0));
    let _ = std::fs::remove_file(&path);
}

#[test]
fn eatime_weekday_bucket_text_mode_uses_weekday_label() {
    olorin::kernels::ffi::init().unwrap();
    let log = b"2026-05-11T10:00:00 mon\n2026-05-12T11:00:00 tue\n";
    let path = write_tmp("olorin_eatime_weekday_text.log", log);
    let result = run_rune("eatime", &format!("--bucket weekday {path}")).unwrap();
    assert!(result.success);
    let a = &result.answer;
    assert!(a.contains("weekday:"),     "missing weekday label: {a}");
    assert!(a.contains("Mon"),          "missing Mon row: {a}");
    assert!(a.contains("Tue"),          "missing Tue row: {a}");
    assert!(a.contains("peak: Mon"),    "missing peak row: {a}");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn eatime_unknown_bucket_emits_usage_error() {
    olorin::kernels::ffi::init().unwrap();
    let result = run_rune("eatime", "--bucket month /tmp/x.log").unwrap();
    assert!(!result.success);
    assert!(result.answer.contains("unknown --bucket"),
        "expected unknown-bucket error: {}", result.answer);
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

#[test]
fn eatime_parses_classic_syslog() {
    olorin::kernels::ffi::init().unwrap();
    // Classic BSD syslog `MMM DD HH:MM:SS` (Linux /var/log, sshd, cron, network
    // gear). Yearless — a fixed reference year is assigned; hour-of-day buckets
    // are unaffected. Covers the space-padded single-digit day (`Jun  4`).
    let log = b"\
Jun 14 15:16:01 host sshd[1]: auth failure
Jun 14 15:16:05 host sshd[2]: accepted
Jun  4 02:13:01 host kernel: boot
Dec 25 23:59:59 host cron[9]: nightly
";
    let path = write_tmp("olorin_eatime_syslog.log", log);
    let out = parse_answer(&run_rune("eatime", &format!("--json {path}")).unwrap().answer);

    assert_eq!(out.totals.rows, 4, "syslog must find all 4 timestamps: {out:?}");
    assert_eq!(out.source.as_ref().unwrap().format, "syslog");
    let count = |hh: &str| out.categories.iter().find(|c| c.name == hh).map(|c| c.count).unwrap_or(0);
    assert_eq!(count("15:00"), 2, "two 15:xx events");
    assert_eq!(count("02:00"), 1, "the space-padded `Jun  4 02:13` event");
    assert_eq!(count("23:00"), 1, "the Dec 25 23:59 event");

    let _ = std::fs::remove_file(&path);
}

#[test]
fn eatime_parses_space_separated_iso() {
    olorin::kernels::ffi::init().unwrap();
    // The space-separated ISO variant Postgres / MySQL / Python-logging /
    // OpenStack emit (`YYYY-MM-DD HH:MM:SS`, often with fractional seconds).
    // It must bucket IDENTICALLY to the 'T'-separated twin — the separator is
    // cosmetic; both decode the same instant.
    let space = b"\
2024-03-01 06:00:00.123 LOG: a
2024-03-01 06:30:00 LOG: b
2024-03-01 07:00:00.9 ERROR: c
2024-03-01 23:59:59 LOG: d
";
    let tee = b"\
2024-03-01T06:00:00 LOG: a
2024-03-01T06:30:00 LOG: b
2024-03-01T07:00:00 ERROR: c
2024-03-01T23:59:59 LOG: d
";
    let ps = write_tmp("olorin_eatime_space.log", space);
    let pt = write_tmp("olorin_eatime_tee.log", tee);
    let os = parse_answer(&run_rune("eatime", &format!("--json {ps}")).unwrap().answer);
    let ot = parse_answer(&run_rune("eatime", &format!("--json {pt}")).unwrap().answer);

    assert_eq!(os.totals.rows, 4, "space-ISO must find all 4 timestamps: {os:?}");
    assert_eq!(os.source.as_ref().unwrap().format, "iso8601");
    // Hour buckets must match the 'T' twin exactly (fractional seconds ignored).
    assert_eq!(os.categories, ot.categories,
        "space-separated and T-separated ISO must bucket identically");

    let _ = std::fs::remove_file(&ps);
    let _ = std::fs::remove_file(&pt);
}

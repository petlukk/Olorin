//! eatime Common Log Format (CLF) support: the `clf_scan` SIMD kernel,
//! textual-month decode, format auto-detection, and all three bucket
//! modes working on `[dd/MMM/yyyy:hh:mm:ss]` Apache/nginx access logs.

use olorin::runes::output::RuneOutput;
use olorin::runes::run_rune;
use olorin::runes::timekey::{clf_bytes_to_seconds, iso_to_seconds};
use std::collections::HashMap;
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

fn clf_line(day: u32, mon: &str, year: u32, hh: u32, mm: u32, ss: u32) -> String {
    format!(
        "127.0.0.1 - - [{day:02}/{mon}/{year}:{hh:02}:{mm:02}:{ss:02} +0000] \"GET / HTTP/1.0\" 200 100\n"
    )
}

fn by_name(out: &RuneOutput) -> HashMap<&str, u64> {
    out.categories.iter().map(|c| (c.name.as_str(), c.count)).collect()
}

#[test]
fn timekey_clf_decodes_and_rejects() {
    olorin::kernels::ffi::init().unwrap();
    // CLF instant equals the same wall-clock ISO instant (zone ignored).
    let clf = clf_bytes_to_seconds(b"[15/Jun/2026:14:30:05 -0700] rest").unwrap();
    let iso = iso_to_seconds("2026-06-15T14:30:05").unwrap();
    assert_eq!(clf, iso, "CLF and ISO decoders must agree on the instant");
    // Case-insensitive month.
    assert_eq!(
        clf_bytes_to_seconds(b"[15/jun/2026:14:30:05 +0000] x"),
        clf_bytes_to_seconds(b"[15/JUN/2026:14:30:05 +0000] x"),
    );
    // Bad month → None.
    assert!(clf_bytes_to_seconds(b"[15/Xyz/2026:14:30:05 +0000] x").is_none());
    // Too short → None.
    assert!(clf_bytes_to_seconds(b"[15/Jun/2026:14").is_none());
}

#[test]
fn clf_format_is_autodetected() {
    olorin::kernels::ffi::init().unwrap();
    let mut log = String::new();
    for _ in 0..10 { log.push_str(&clf_line(15, "Jun", 2026, 6, 0, 0)); }
    let path = write_tmp("olorin_clf_auto.log", log.as_bytes());
    // No --format flag → must auto-detect CLF.
    let result = run_rune("eatime", &format!("--json {path}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);
    let out = parse_answer(&result.answer);
    assert_eq!(out.source.as_ref().unwrap().format, "clf", "should auto-detect CLF");
    assert_eq!(out.totals.rows, 10);
}

#[test]
fn clf_hour_bucketing() {
    olorin::kernels::ffi::init().unwrap();
    // 6 at 06:xx, 3 at 07:xx, 3 at 23:xx — mirrors the ISO hour test.
    let mut log = String::new();
    for _ in 0..6 { log.push_str(&clf_line(15, "Jun", 2026, 6, 14, 0)); }
    for _ in 0..3 { log.push_str(&clf_line(15, "Jun", 2026, 7, 1, 0)); }
    for _ in 0..3 { log.push_str(&clf_line(15, "Jun", 2026, 23, 0, 0)); }
    let path = write_tmp("olorin_clf_hours.log", log.as_bytes());
    let result = run_rune("eatime", &format!("--json --format clf {path}")).unwrap();
    let out = parse_answer(&result.answer);
    assert_eq!(out.totals.rows, 12);
    let h = by_name(&out);
    assert_eq!(h.get("06:00"), Some(&6));
    assert_eq!(h.get("07:00"), Some(&3));
    assert_eq!(h.get("23:00"), Some(&3));
    assert_eq!(out.categories.len(), 24);
}

#[test]
fn clf_weekday_matches_known_day() {
    olorin::kernels::ffi::init().unwrap();
    // 2026-06-15 is a Monday. All hits must land in the Mon bucket.
    let mut log = String::new();
    for _ in 0..5 { log.push_str(&clf_line(15, "Jun", 2026, 12, 0, 0)); }
    let path = write_tmp("olorin_clf_weekday.log", log.as_bytes());
    let result = run_rune("eatime", &format!("--json --bucket weekday --format clf {path}")).unwrap();
    let out = parse_answer(&result.answer);
    let h = by_name(&out);
    assert_eq!(h.get("Mon"), Some(&5), "2026-06-15 is a Monday: {:?}", out.categories);
    assert_eq!(out.categories.iter().map(|c| c.count).sum::<u64>(), 5);
}

#[test]
fn clf_series_detects_spike() {
    olorin::kernels::ffi::init().unwrap();
    // Flat 50/min baseline over 120 minutes (hours 0..1) with a 50× spike
    // at minute 75 (= 01:15). Bucket labels are ISO-normalized.
    let mut log = String::new();
    for m in 0..120usize {
        let hh = (m / 60) as u32;
        let mm = (m % 60) as u32;
        let n = if m == 75 { 2500 } else { 50 };
        for _ in 0..n { log.push_str(&clf_line(4, "Jun", 2026, hh, mm, 0)); }
    }
    let path = write_tmp("olorin_clf_spike.log", log.as_bytes());
    let result = run_rune("eatime", &format!("--json --bucket series --format clf {path}")).unwrap();
    let out = parse_answer(&result.answer);
    assert_eq!(out.anomalies.len(), 1, "one spike expected: {:?}", out.anomalies);
    let a = &out.anomalies[0];
    assert_eq!(a.bucket, "2026-06-04T01:15:00");
    assert_eq!(a.count, 2500);
    assert_eq!(a.baseline, 50.0);
}

#[test]
fn force_iso_on_clf_finds_nothing() {
    olorin::kernels::ffi::init().unwrap();
    // CLF has no YYYY-MM-DDT prefix, so forcing the ISO kernel must find
    // zero timestamps — proves --format actually dispatches the kernel.
    let mut log = String::new();
    for _ in 0..10 { log.push_str(&clf_line(15, "Jun", 2026, 6, 0, 0)); }
    let path = write_tmp("olorin_clf_forceiso.log", log.as_bytes());
    let result = run_rune("eatime", &format!("--json --format iso {path}")).unwrap();
    let out = parse_answer(&result.answer);
    assert_eq!(out.totals.rows, 0, "ISO kernel must not match CLF timestamps");
    assert_eq!(out.source.as_ref().unwrap().format, "iso8601");
}

#[test]
fn unknown_format_is_a_usage_error() {
    olorin::kernels::ffi::init().unwrap();
    let result = run_rune("eatime", "--json --format bogus /tmp/whatever").unwrap();
    assert!(!result.success);
    let out = parse_answer(&result.answer);
    assert!(out.error.unwrap().contains("unknown --format"));
}

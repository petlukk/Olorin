//! eatime `--bucket series`: chronological histogram + robust spike
//! detection. Proves the detector flags an injected spike (both the
//! z-score and the MAD=0 ratio path), stays silent on flat data, and
//! that the additive `anomalies[]` schema field doesn't disturb the
//! other bucket modes or eadiff's consumption of eatime `--json`.

use olorin::runes::output::RuneOutput;
use olorin::runes::run_rune;
use olorin::runes::timekey::{iso_to_seconds, seconds_to_iso};
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

/// Build a log spanning `minutes` one-minute buckets. `count_for(m)`
/// returns the number of events to emit in minute `m`; every event is
/// stamped `2026-06-04THH:MM:00`. Minutes roll into hours so timestamps
/// stay valid past minute 59.
fn synthetic_log(minutes: usize, count_for: impl Fn(usize) -> usize) -> Vec<u8> {
    let mut buf = String::new();
    for m in 0..minutes {
        let hh = m / 60;
        let mm = m % 60;
        for _ in 0..count_for(m) {
            buf.push_str(&format!("2026-06-04T{hh:02}:{mm:02}:00 INFO event\n"));
        }
    }
    buf.into_bytes()
}

#[test]
fn flat_rate_yields_no_anomalies() {
    olorin::kernels::ffi::init().unwrap();
    // 120 minutes, exactly 50 events each → median 50, MAD 0, and no
    // bucket exceeds the median. Must produce zero false positives.
    let log = synthetic_log(120, |_| 50);
    let path = write_tmp("olorin_eatime_flat.log", &log);
    let result = run_rune("eatime", &format!("--json --bucket series {path}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);

    let out = parse_answer(&result.answer);
    assert_eq!(out.totals.rows, 120 * 50, "all timestamps counted");
    assert_eq!(out.categories.len(), 120, "one bucket per minute");
    assert!(
        out.anomalies.is_empty(),
        "flat rate must not flag anomalies, got {:?}",
        out.anomalies
    );
}

#[test]
fn injected_spike_flags_exact_bucket_mad_zero_path() {
    olorin::kernels::ffi::init().unwrap();
    // Perfectly flat 50/min baseline (MAD = 0) with one 50× spike at
    // minute 75 (= 01:15). Exercises the ratio fallback.
    let log = synthetic_log(120, |m| if m == 75 { 2500 } else { 50 });
    let path = write_tmp("olorin_eatime_spike_flat.log", &log);
    let result = run_rune("eatime", &format!("--json --bucket series {path}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);

    let out = parse_answer(&result.answer);
    assert_eq!(out.anomalies.len(), 1, "exactly one spike expected");
    let a = &out.anomalies[0];
    assert_eq!(a.bucket, "2026-06-04T01:15:00", "spike at minute 75");
    assert_eq!(a.count, 2500);
    assert_eq!(a.baseline, 50.0);
    assert!((a.ratio - 50.0).abs() < 1e-9, "50× baseline, got {}", a.ratio);
}

#[test]
fn injected_spike_flags_with_noise_zscore_path() {
    olorin::kernels::ffi::init().unwrap();
    // Deterministic 48..=52 jitter (MAD > 0 → robust z-score path) with a
    // single large spike. The spike must be flagged; jitter must not.
    let log = synthetic_log(120, |m| if m == 30 { 3000 } else { 48 + (m % 5) });
    let path = write_tmp("olorin_eatime_spike_noisy.log", &log);
    let result = run_rune("eatime", &format!("--json --bucket series {path}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);

    let out = parse_answer(&result.answer);
    assert_eq!(out.anomalies.len(), 1, "only the real spike, not jitter: {:?}", out.anomalies);
    let a = &out.anomalies[0];
    assert_eq!(a.bucket, "2026-06-04T00:30:00");
    assert_eq!(a.count, 3000);
    assert!(a.score >= 4.0, "robust z-score above threshold, got {}", a.score);
}

#[test]
fn series_text_mode_surfaces_the_spike() {
    olorin::kernels::ffi::init().unwrap();
    let log = synthetic_log(120, |m| if m == 75 { 2500 } else { 50 });
    let path = write_tmp("olorin_eatime_spike_text.log", &log);
    // No --json → human-readable summary path.
    let result = run_rune("eatime", &format!("--bucket series {path}")).unwrap();
    assert!(result.success);
    assert!(result.answer.contains("anomalies:"), "summary names anomalies");
    assert!(result.answer.contains("2026-06-04T01:15:00"), "summary cites the spike bucket");
    assert!(result.answer.contains("baseline"), "summary reports baseline");
}

#[test]
fn hour_mode_unaffected_by_additive_schema() {
    olorin::kernels::ffi::init().unwrap();
    // The same data under the default hour bucketing must carry no
    // anomalies and serialize without an `anomalies` key.
    let log = synthetic_log(120, |m| if m == 75 { 2500 } else { 50 });
    let path = write_tmp("olorin_eatime_hourmode.log", &log);
    let result = run_rune("eatime", &format!("--json {path}")).unwrap();
    assert!(result.success);
    assert!(out_has_no_anomalies_key(&result.answer), "hour-mode JSON must omit anomalies");
    let out = parse_answer(&result.answer);
    assert!(out.anomalies.is_empty());
}

fn out_has_no_anomalies_key(json: &str) -> bool {
    !json.contains("\"anomalies\"")
}

#[test]
fn eadiff_still_consumes_eatime_series_json() {
    olorin::kernels::ffi::init().unwrap();
    // A series output (carrying anomalies) must remain a valid eadiff
    // input — the additive field cannot break the v1 chaining contract.
    let log = synthetic_log(120, |m| if m == 75 { 2500 } else { 50 });
    let path = write_tmp("olorin_eatime_for_diff.log", &log);
    let a = run_rune("eatime", &format!("--json --bucket series {path}")).unwrap();
    assert!(!a.anomalies_unused());
    let a_path = write_tmp("olorin_eatime_a.json", a.answer.as_bytes());
    let b_path = write_tmp("olorin_eatime_b.json", a.answer.as_bytes());
    let diff = run_rune("eadiff", &format!("--json {a_path} {b_path}")).unwrap();
    assert!(diff.success, "eadiff rejected a series output: {}", diff.answer);
}

#[test]
fn timekey_roundtrips_including_leap_day() {
    for ts in [
        "2000-01-01T00:00:00",
        "2026-06-04T02:11:37",
        "2024-02-29T23:59:59", // leap day
        "1999-12-31T23:59:59", // pre-epoch
    ] {
        let secs = iso_to_seconds(ts).unwrap_or_else(|| panic!("parse {ts}"));
        assert_eq!(seconds_to_iso(secs), ts, "round-trip {ts}");
    }
}

// Small extension trait so the eadiff test reads intention-first.
trait AnomalyProbe { fn anomalies_unused(&self) -> bool; }
impl AnomalyProbe for olorin::runes::RuneResult {
    fn anomalies_unused(&self) -> bool {
        !self.answer.contains("\"anomalies\"")
    }
}

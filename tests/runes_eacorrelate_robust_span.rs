//! eacorrelate — robust grid span (`stream::robust_bounds`).
//!
//! A minority of far-offset timestamps (a 1970 epoch from a missing field, a
//! clock-skew future, or a yearless format parsed to a fixed era while its
//! dated peers sit in the real year) used to stretch `max − min` across months.
//! `auto_width` then snapped to a coarse rung and a genuinely short incident
//! collapsed into one bucket (trivial r=1.00). The grid span is now clipped to a
//! Tukey fence, so short incidents keep fine resolution.

use olorin::runes::output::RuneOutput;
use olorin::runes::run_rune;
use olorin::runes::stream::robust_bounds;
use std::fmt::Write as _;
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

// ── unit: the Tukey-fence math ──────────────────────────────────────────────

#[test]
fn robust_bounds_clips_minority_outliers() {
    // 100 events in a dense band, plus 5 far-below outliers (1970-style).
    let mut e: Vec<i64> = (1000..1100).collect();
    e.extend([0, 1, 2, 3, 4]);
    let (lo, hi) = robust_bounds(&e).expect("non-empty");
    assert!(lo >= 700, "low outliers not clipped: lo={lo}");
    assert_eq!(hi, 1099, "upper bound should stay at the real max");
}

#[test]
fn robust_bounds_is_noop_without_outliers() {
    let e: Vec<i64> = (0..1000).collect();
    assert_eq!(robust_bounds(&e), Some((0, 999)), "clean data must be unchanged");
}

#[test]
fn robust_bounds_zero_iqr_falls_back_to_full_range() {
    // >75% identical → IQR 0 → robust scale undefined → don't clip.
    let mut e = vec![5i64; 100];
    e.push(0);
    e.push(9999);
    assert_eq!(robust_bounds(&e), Some((0, 9999)));
}

// ── end-to-end: a short incident survives an outlier-era stream ──────────────

/// ISO-8601 stamp within hour 06 of the given year: `YYYY-06-18T06:MM:SS`.
fn iso(year: i32, sec_in_hour: i64) -> String {
    format!("{year}-06-18T06:{:02}:{:02}", sec_in_hour / 60, sec_in_hour % 60)
}

/// ISO app log: a heartbeat every 5s across a ~200s window, plus a 20-line
/// ERROR burst `lag` after each deploy.
fn iso_app(year: i32, base: i64, deploys: &[i64], lag: i64) -> Vec<u8> {
    let mut buf = String::new();
    let mut t = base;
    while t <= base + 200 {
        writeln!(buf, "{} INFO heartbeat ok", iso(year, t)).unwrap();
        t += 5;
    }
    for &d in deploys {
        for k in 0..20 {
            writeln!(buf, "{} ERROR upstream timeout #{k}", iso(year, d + lag)).unwrap();
        }
    }
    buf.into_bytes()
}

fn iso_deploys(year: i32, deploys: &[i64]) -> Vec<u8> {
    let mut buf = String::new();
    for &d in deploys {
        writeln!(buf, "{} INFO released sha abc{d}", iso(year, d)).unwrap();
    }
    buf.into_bytes()
}

#[test]
fn short_incident_keeps_fine_bucket_despite_outlier_era_stream() {
    olorin::kernels::ffi::init().unwrap();

    // Real incident: 2026, ~200s window, deploys at 06:10:40/11:40/12:40,
    // error bursts 20s later. The dense window alone wants a 1s bucket.
    let deploys = [640, 700, 760];
    let app = write_tmp("olorin_robust_app.log", &iso_app(2026, 600, &deploys, 20));
    let dep = write_tmp("olorin_robust_deploys.log", &iso_deploys(2026, &deploys));
    // Outlier-era stream: same clock, two years earlier (the syslog/yearless
    // case). A minority of pooled events — must NOT widen the grid.
    let noise = write_tmp("olorin_robust_noise.log", &iso_app(2024, 600, &[], 0));

    let result = run_rune("eacorrelate", &format!("--json {app} {dep} {noise}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);
    let out = parse_answer(&result.answer);

    let f = out.correlations.iter()
        .find(|c| c.stream_a == "olorin_robust_app.log (errors)")
        .unwrap_or_else(|| panic!("no errors-substream finding: {}", result.answer));
    assert_eq!(f.stream_b, "olorin_robust_deploys.log");
    // The fix's payoff: a 1s bucket and the exact 20s lag. Without it the 2024
    // stream stretches the span to ~2 years → a 604800s (1-week) bucket → lag
    // would collapse to 0 and width blow up.
    assert_eq!(f.width_seconds, 1, "coarse bucket — outlier era widened the grid: {f:?}");
    assert_eq!(f.lag_seconds, 20, "wrong lag: {f:?}");
    assert!(f.score > 0.8, "weak score: {f:?}");

    // The 2024 stream is two years off the incident; it can't align and must
    // not appear in any finding.
    assert!(
        !out.correlations.iter().any(|c|
            c.stream_a.contains("noise") || c.stream_b.contains("noise")),
        "outlier-era stream should drop out, got: {:?}", out.correlations
    );

    for p in [&app, &dep, &noise] { let _ = std::fs::remove_file(p); }
}

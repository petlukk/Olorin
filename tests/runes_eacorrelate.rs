//! eacorrelate E2E — the cross-file lag rune must recover a planted
//! deploy→error lag exactly, stay silent on uncorrelated and flat
//! streams, survive its error paths, and round-trip the additive
//! `correlations[]` block through the v1 JSON contract.

use olorin::runes::output::RuneOutput;
use olorin::runes::run_rune;
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

fn stamp(secs: i64) -> String {
    // Day-long tests stay inside 2026-06-11.
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    format!("2026-06-11T{h:02}:{m:02}:{s:02}")
}

/// 8h syslog: one INFO line per minute as baseline, plus a 20-line
/// ERROR burst `lag_secs` after each entry of `deploy_secs`.
fn syslog_with_bursts(deploy_secs: &[i64], lag_secs: i64) -> Vec<u8> {
    let mut buf = String::new();
    for m in 0..=(8 * 60) {
        writeln!(buf, "{} INFO heartbeat", stamp(m * 60)).unwrap();
    }
    for &d in deploy_secs {
        for k in 0..20 {
            writeln!(buf, "{} ERROR upstream timeout #{k}", stamp(d + lag_secs)).unwrap();
        }
    }
    buf.into_bytes()
}

fn deploys_csv(deploy_secs: &[i64]) -> Vec<u8> {
    let mut buf = String::from("time,event,sha\n");
    for &d in deploy_secs {
        writeln!(buf, "{},deploy,abc{d}", stamp(d)).unwrap();
    }
    buf.into_bytes()
}

#[test]
fn recovers_planted_deploy_error_lag() {
    olorin::kernels::ffi::init().unwrap();
    // Deploys at 02:00 / 04:00 / 06:00, error bursts 240s later. Grid:
    // 8h span / 512 target -> 60s buckets, so the lag is exactly 4 buckets.
    let deploys = [2 * 3600, 4 * 3600, 6 * 3600];
    let log_path = write_tmp("olorin_corr_errors.log", &syslog_with_bursts(&deploys, 240));
    let csv_path = write_tmp("olorin_corr_deploys.csv", &deploys_csv(&deploys));

    let result = run_rune("eacorrelate", &format!("--json {log_path} {csv_path}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);
    let out = parse_answer(&result.answer);

    // Streams: log all-events, log (errors), deploys all-events.
    assert_eq!(out.categories.len(), 3, "answer={}", result.answer);
    assert!(!out.correlations.is_empty(), "no findings: {}", result.answer);

    let f = out.correlations.iter()
        .find(|c| c.stream_a == "olorin_corr_errors.log (errors)")
        .unwrap_or_else(|| panic!("no errors-substream finding: {}", result.answer));
    assert_eq!(f.stream_b, "olorin_corr_deploys.csv");
    assert_eq!(f.lag_seconds, 240, "wrong lag: {:?}", f);
    assert_eq!(f.width_seconds, 60);
    assert!(f.score > 0.8, "weak score: {:?}", f);
    assert_eq!(f.events_a, 60); // 3 bursts x 20 lines
    assert_eq!(f.events_b, 3);
    // Peak lands on one of the burst minutes (xx:04:00).
    assert!(f.peak_bucket.ends_with(":04:00"), "peak: {}", f.peak_bucket);

    let _ = std::fs::remove_file(&log_path);
    let _ = std::fs::remove_file(&csv_path);
}

#[test]
fn uncorrelated_streams_yield_no_findings() {
    olorin::kernels::ffi::init().unwrap();
    // Two files whose events are independent LCG scatter over 8h —
    // nothing should cross the 0.5 threshold.
    let mut state: u64 = 0x9e3779b97f4a7c15;
    let mut scatter = |n: usize, salt: u64| -> Vec<u8> {
        let mut buf = String::new();
        let mut secs: Vec<i64> = (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(salt | 1);
                ((state >> 33) % (8 * 3600)) as i64
            })
            .collect();
        secs.sort_unstable();
        for s in secs {
            writeln!(buf, "{} INFO event", stamp(s)).unwrap();
        }
        buf.into_bytes()
    };
    let a = write_tmp("olorin_corr_rand_a.log", &scatter(200, 7));
    let b = write_tmp("olorin_corr_rand_b.log", &scatter(200, 99));

    let result = run_rune("eacorrelate", &format!("--json {a} {b}")).unwrap();
    assert!(result.success);
    let out = parse_answer(&result.answer);
    assert!(
        out.correlations.is_empty(),
        "false positive on independent streams: {}", result.answer
    );

    let _ = std::fs::remove_file(&a);
    let _ = std::fs::remove_file(&b);
}

#[test]
fn disjoint_eras_yield_no_findings() {
    olorin::kernels::ffi::init().unwrap();
    // The wild v2.12.0 bug (found 2026-06-12 with two NASA-log slices):
    // file A covers days 1–2, file B covers days 6–13 — no shared time
    // at all. A lag that shifts A's activity onto B's era compares A's
    // ZERO-EVENT overlap window against B's traffic; global-window
    // cosine scored that constant-vs-curve pairing r=1.00 at +5 days.
    // Per-window Pearson + the active-window gate must report nothing.
    let mut a = String::new();
    for m in 0..(2 * 24 * 60) {
        let burst = if (m / 60) % 24 < 12 { 3 } else { 1 }; // diurnal-ish
        for k in 0..burst {
            writeln!(a, "{} INFO svc-a event {k}", stamp_day(1 + m / (24 * 60), (m % (24 * 60)) * 60)).unwrap();
        }
    }
    let mut b = String::new();
    for m in 0..(7 * 24 * 60) {
        let burst = if (m / 60) % 24 < 12 { 4 } else { 1 };
        for k in 0..burst {
            writeln!(b, "{} INFO svc-b event {k}", stamp_day(6 + m / (24 * 60), (m % (24 * 60)) * 60)).unwrap();
        }
    }
    let pa = write_tmp("olorin_corr_era_a.log", a.as_bytes());
    let pb = write_tmp("olorin_corr_era_b.log", b.as_bytes());

    let result = run_rune("eacorrelate", &format!("--json {pa} {pb}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);
    let out = parse_answer(&result.answer);
    assert!(
        out.correlations.is_empty(),
        "disjoint eras must not correlate: {}", result.answer
    );

    let _ = std::fs::remove_file(&pa);
    let _ = std::fs::remove_file(&pb);
}

/// ISO stamp on a given June day at `secs` into the day.
fn stamp_day(day: i64, secs: i64) -> String {
    format!(
        "2026-06-{:02}T{:02}:{:02}:{:02}",
        day, secs / 3600, (secs % 3600) / 60, secs % 60
    )
}

#[test]
fn flat_stream_is_skipped_not_correlated() {
    olorin::kernels::ffi::init().unwrap();
    // File A is a metronome (every minute, zero variance on the grid) —
    // z-score is undefined, so it must be skipped without panic.
    let mut flat = String::new();
    for m in 0..=(8 * 60) {
        writeln!(flat, "{} INFO tick", stamp(m * 60)).unwrap();
    }
    let deploys = [2 * 3600, 4 * 3600, 6 * 3600];
    let a = write_tmp("olorin_corr_flat.log", flat.as_bytes());
    let b = write_tmp("olorin_corr_flat_deploys.csv", &deploys_csv(&deploys));

    let result = run_rune("eacorrelate", &format!("--json {a} {b}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);
    let out = parse_answer(&result.answer);
    assert!(out.correlations.is_empty(), "flat stream correlated: {}", result.answer);

    let _ = std::fs::remove_file(&a);
    let _ = std::fs::remove_file(&b);
}

#[test]
fn single_usable_file_reports_nothing_to_correlate() {
    olorin::kernels::ffi::init().unwrap();
    // Second file has < 3 events, so only one file survives — success
    // with zero findings, not an error.
    let deploys = [2 * 3600, 4 * 3600, 6 * 3600];
    let a = write_tmp("olorin_corr_only.log", &syslog_with_bursts(&deploys, 240));
    let b = write_tmp("olorin_corr_sparse.log", b"2026-06-11T01:00:00 INFO lonely\n");

    let result = run_rune("eacorrelate", &format!("--json {a} {b}")).unwrap();
    assert!(result.success);
    let out = parse_answer(&result.answer);
    assert!(out.correlations.is_empty());
    assert_eq!(out.categories.len(), 2); // all-events + (errors), both from file A

    let _ = std::fs::remove_file(&a);
    let _ = std::fs::remove_file(&b);
}

#[test]
fn arg_errors_fail_closed() {
    olorin::kernels::ffi::init().unwrap();
    let one = run_rune("eacorrelate", "--json /tmp/only_one.log").unwrap();
    assert!(!one.success, "single path must be a usage error");

    let missing = run_rune(
        "eacorrelate",
        "--json /tmp/olorin_corr_nope_a.log /tmp/olorin_corr_nope_b.log",
    ).unwrap();
    assert!(!missing.success, "missing files must fail");
}

/// Emit `counts[i]` ISO events into 6h bucket `i` (day 1 + i/4, hour (i%4)*6),
/// at distinct seconds so they land in the same bucket.
fn emit_buckets(counts: &[u32]) -> Vec<u8> {
    let mut s = String::new();
    for (i, &c) in counts.iter().enumerate() {
        let day = 1 + (i / 4) as i64;
        let hour = ((i % 4) * 6) as i64;
        for k in 0..c {
            writeln!(s, "{} INFO event", stamp_day(day, hour * 3600 + k as i64)).unwrap();
        }
    }
    s.into_bytes()
}

#[test]
fn long_span_excludes_boundary_lag_artifact() {
    olorin::kernels::ffi::init().unwrap();
    // The NASA real-data bug (found 2026-06-15): a 28-day traffic/errors split
    // scored errors "following" traffic by +654h, r=1.00, on a 4-bucket overlap
    // — a near-boundary lag where Pearson hits 1.00 trivially. Reproduce the
    // shape: a dense stream whose TAIL ramps and a sparse stream whose HEAD
    // ramps, so the max-lag overlap (A-tail vs B-head) is a perfect 4-point
    // correlation. The span/4 lag cap + the overlap-window floor must exclude
    // it; no finding may claim a lag beyond a quarter of the span.
    let span = 28 * 86400i64;
    let mut a = vec![3u32; 112];        // dense, mild variation below
    for (i, c) in a.iter_mut().enumerate() { *c = 3 + (i as u32 % 3); }
    a[108..112].copy_from_slice(&[1, 2, 3, 4]); // ramp tail
    let mut b = vec![0u32; 112];        // sparse
    b[0..4].copy_from_slice(&[1, 2, 3, 4]);     // ramp head — aligns with A-tail at max lag
    b[40] = 3; b[64] = 2; b[88] = 3;            // scatter: keep it active, non-flat

    let pa = write_tmp("olorin_corr_boundary_a.log", &emit_buckets(&a));
    let pb = write_tmp("olorin_corr_boundary_b.log", &emit_buckets(&b));
    let result = run_rune("eacorrelate", &format!("--json {pa} {pb}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);
    let out = parse_answer(&result.answer);

    for c in &out.correlations {
        assert!(c.lag_seconds.abs() <= span / 4 + c.width_seconds,
            "boundary-lag artifact survived: lag {} on {}s span: {:?}", c.lag_seconds, span, c);
        assert!(!(c.score >= 0.999 && c.lag_seconds.abs() > span / 8),
            "degenerate r~1.00 at a large lag: {:?}", c);
    }
    if let Some(inc) = &out.incident {
        for s in &inc.steps {
            assert!(s.lag_seconds <= span / 4 + 21600, "absurd incident cascade lag: {:?}", s);
        }
    }

    let _ = std::fs::remove_file(&pa);
    let _ = std::fs::remove_file(&pb);
}

#[test]
fn correlations_block_round_trips_and_rounds_score() {
    olorin::kernels::ffi::init().unwrap();
    let deploys = [2 * 3600, 4 * 3600, 6 * 3600];
    let a = write_tmp("olorin_corr_rt.log", &syslog_with_bursts(&deploys, 240));
    let b = write_tmp("olorin_corr_rt_deploys.csv", &deploys_csv(&deploys));

    let result = run_rune("eacorrelate", &format!("--json {a} {b}")).unwrap();
    let out = parse_answer(&result.answer);
    assert!(!out.correlations.is_empty());

    // Wire scores carry at most 4 decimals (cross-arch golden defense).
    for c in &out.correlations {
        let rewire = (c.score * 10_000.0).round() / 10_000.0;
        assert_eq!(c.score, rewire, "wire score not 4dp-rounded: {}", c.score);
    }

    // to_json -> from_json must preserve the block exactly.
    let again = parse_answer(&out.to_json());
    assert_eq!(out.correlations, again.correlations);

    let _ = std::fs::remove_file(&a);
    let _ = std::fs::remove_file(&b);
}

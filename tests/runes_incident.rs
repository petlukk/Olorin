//! Incident timeline E2E — eacorrelate must assemble its pairwise lag
//! correlations into a single ordered story: anchor on the sparse trigger
//! (the deploy), order the followers by lag, report a weakest-link
//! confidence, and round-trip the additive `incident` block through the v1
//! JSON contract. Built on the same planted deploy->error fixture the
//! correlation E2E uses, so the cascade is known-strong.

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

fn parse(answer: &str) -> RuneOutput {
    RuneOutput::from_json(answer.as_bytes())
        .unwrap_or_else(|e| panic!("not parseable JSON: {e}\nanswer={answer}"))
}

fn stamp(secs: i64) -> String {
    format!("2026-06-11T{:02}:{:02}:{:02}", secs / 3600, (secs % 3600) / 60, secs % 60)
}

/// 8h log: one INFO heartbeat per minute, plus a 20-line ERROR burst
/// `lag_secs` after each deploy instant — the planted cascade.
fn log_with_error_bursts(deploy_secs: &[i64], lag_secs: i64) -> Vec<u8> {
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
fn incident_anchors_on_trigger_and_orders_cascade() {
    olorin::kernels::ffi::init().unwrap();
    // Deploys at 02:00/04:00/06:00, error bursts 240s later. deploys.csv has 3
    // events vs the log's 500+ — sparse, so it's the trigger/anchor.
    let deploys = [2 * 3600, 4 * 3600, 6 * 3600];
    let log = write_tmp("olorin_inc_app.log", &log_with_error_bursts(&deploys, 240));
    let csv = write_tmp("olorin_inc_deploys.csv", &deploys_csv(&deploys));

    let result = run_rune("eacorrelate", &format!("--json {log} {csv}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);
    let out = parse(&result.answer);

    let inc = out.incident.as_ref()
        .unwrap_or_else(|| panic!("no incident assembled: {}", result.answer));

    // Anchored on the sparse deploy trigger, at the deploy instant (the error
    // peak at xx:04:00 minus the 240s lag = xx:00:00).
    assert_eq!(inc.anchor.kind, "trigger", "expected a discrete trigger anchor");
    assert_eq!(inc.anchor.stream, "olorin_inc_deploys.csv");
    assert!(inc.anchor.time.ends_with(":00:00"), "anchor not on a deploy instant: {}", inc.anchor.time);

    // The error substream is a step, reached at the planted +240s lag, going up.
    let err_step = inc.steps.iter()
        .find(|s| s.stream == "olorin_inc_app.log (errors)")
        .unwrap_or_else(|| panic!("no error step: {:?}", inc.steps));
    assert_eq!(err_step.lag_seconds, 240, "wrong cascade lag: {:?}", err_step);
    assert_eq!(err_step.direction, "increase");
    assert!(err_step.score > 0.8, "weak step score: {:?}", err_step);

    // Confidence is the weakest link across the chain, and never exceeds it.
    let min_step = inc.steps.iter().map(|s| s.score).fold(f64::INFINITY, f64::min);
    assert!((inc.confidence - (min_step * 10_000.0).round() / 10_000.0).abs() < 1e-9,
        "confidence != min(step.score): conf={} min={}", inc.confidence, min_step);
    assert!(inc.confidence > 0.5);

    // Steps are in timeline order (soonest after the anchor first).
    for w in inc.steps.windows(2) {
        assert!(w[0].lag_seconds <= w[1].lag_seconds, "steps out of order: {:?}", inc.steps);
    }

    let _ = std::fs::remove_file(&log);
    let _ = std::fs::remove_file(&csv);
}

#[test]
fn single_deploy_line_anchors_the_incident() {
    olorin::kernels::ffi::init().unwrap();
    // Part B: a real deploy is ONE line — too sparse to be a correlation stream.
    // The cascade (db errors at the deploy -> app errors 240s later) infers its
    // root instant; the lone deploy event sitting there snaps the anchor onto it,
    // so the timeline names the DEPLOY instead of the first error stream.
    let deploy_at = 2 * 3600;
    let db  = write_tmp("olorin_b_db.log",  &log_with_error_bursts(&[deploy_at], 0));
    let app = write_tmp("olorin_b_app.log", &log_with_error_bursts(&[deploy_at], 240));
    let dep = write_tmp("olorin_b_deploy.csv", &deploys_csv(&[deploy_at]));

    let result = run_rune("eacorrelate", &format!("--json {db} {app} {dep}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);
    let out = parse(&result.answer);
    let inc = out.incident.as_ref()
        .unwrap_or_else(|| panic!("no incident: {}", result.answer));

    // The single deploy line became the anchor (not the db error stream).
    assert_eq!(inc.anchor.kind, "trigger", "anchor: {:?}", inc.anchor);
    assert_eq!(inc.anchor.stream, "olorin_b_deploy.csv",
        "anchor did not snap to the deploy: {:?}", inc.anchor);
    assert!(inc.anchor.time.ends_with("02:00:00"), "anchor time: {}", inc.anchor.time);
    assert!(inc.steps.iter().any(|s| s.stream.contains("app")),
        "expected the app follower as a step: {:?}", inc.steps);

    for p in [&db, &app, &dep] { let _ = std::fs::remove_file(p); }
}

#[test]
fn distant_deploy_does_not_hijack_the_anchor() {
    olorin::kernels::ffi::init().unwrap();
    // The match window is the cascade's own span, so a deploy far outside it (an
    // unrelated earlier release) must NOT be snapped onto — the anchor stays on
    // the error stream. Guards the snap against over-eager matching.
    let cascade_at = 4 * 3600;
    let db  = write_tmp("olorin_b2_db.log",  &log_with_error_bursts(&[cascade_at], 0));
    let app = write_tmp("olorin_b2_app.log", &log_with_error_bursts(&[cascade_at], 240));
    let dep = write_tmp("olorin_b2_deploy.csv", &deploys_csv(&[0])); // 00:00, cascade at 04:00

    let result = run_rune("eacorrelate", &format!("--json {db} {app} {dep}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);
    let out = parse(&result.answer);
    let inc = out.incident.as_ref()
        .unwrap_or_else(|| panic!("no incident: {}", result.answer));

    assert_ne!(inc.anchor.stream, "olorin_b2_deploy.csv",
        "a distant deploy wrongly hijacked the anchor: {:?}", inc.anchor);

    for p in [&db, &app, &dep] { let _ = std::fs::remove_file(p); }
}

/// Steady traffic with deterministic variation (so MAD > 0 — the robust-z
/// path, as real traffic behaves), collapsing to near-zero for `drop_dur`
/// seconds starting `drop_lag` after each deploy.
fn traffic_with_drop(deploy_secs: &[i64], drop_lag: i64, drop_dur: i64) -> Vec<u8> {
    let mut buf = String::new();
    for m in 0..=(8 * 60) {
        let t = m * 60;
        let in_drop = deploy_secs.iter().any(|&d| t >= d + drop_lag && t < d + drop_lag + drop_dur);
        let n = if in_drop { 1 } else { 16 + (m % 7) };
        for k in 0..n {
            writeln!(buf, "{} GET /api ok #{k}", stamp(t)).unwrap();
        }
    }
    buf.into_bytes()
}

#[test]
fn incident_detects_traffic_drop_as_decrease_step() {
    olorin::kernels::ffi::init().unwrap();
    // deploy -> errors rise +240s -> traffic DROPS +720s. The drop anti-
    // correlates with the error spike, so it must surface as a signed
    // downward-anomaly observation, NOT a correlation.
    let deploys = [2 * 3600, 4 * 3600, 6 * 3600];
    let app = write_tmp("olorin_inc_s2_app.log", &log_with_error_bursts(&deploys, 240));
    let csv = write_tmp("olorin_inc_s2_deploys.csv", &deploys_csv(&deploys));
    let traf = write_tmp("olorin_inc_s2_traffic.log", &traffic_with_drop(&deploys, 720, 180));

    let result = run_rune("eacorrelate", &format!("--json {csv} {app} {traf}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);
    let out = parse(&result.answer);
    let inc = out.incident.as_ref().unwrap_or_else(|| panic!("no incident: {}", result.answer));

    let drop = inc.steps.iter()
        .find(|s| s.stream == "olorin_inc_s2_traffic.log")
        .unwrap_or_else(|| panic!("no traffic step: {:?}", inc.steps));
    assert_eq!(drop.direction, "decrease", "traffic must be a DROP: {drop:?}");
    assert_eq!(drop.kind, "anomaly", "a drop is an observation, not a correlation: {drop:?}");
    // Drop is ~12 min after the anchor (720s ± a bucket of slack).
    assert!((660..=840).contains(&drop.lag_seconds), "drop lag off: {drop:?}");
    assert!(drop.score > 0.5 && drop.score <= 1.0, "implausible drop score: {drop:?}");

    // The error cascade is still there, and confidence is the weakest link.
    assert!(inc.steps.iter().any(|s| s.direction == "increase" && s.kind == "correlated"),
        "lost the error cascade: {:?}", inc.steps);
    let min_score = inc.steps.iter().map(|s| s.score).fold(f64::INFINITY, f64::min);
    assert!((inc.confidence - (min_score * 10_000.0).round() / 10_000.0).abs() < 1e-9,
        "confidence != weakest link: {} vs {}", inc.confidence, min_score);

    for p in [&app, &csv, &traf] { let _ = std::fs::remove_file(p); }
}

#[test]
fn incident_text_reads_as_a_story() {
    olorin::kernels::ffi::init().unwrap();
    let deploys = [2 * 3600, 4 * 3600, 6 * 3600];
    let log = write_tmp("olorin_inc_txt_app.log", &log_with_error_bursts(&deploys, 240));
    let csv = write_tmp("olorin_inc_txt_deploys.csv", &deploys_csv(&deploys));

    // Text mode (no --json) renders the human timeline.
    let result = run_rune("eacorrelate", &format!("{log} {csv}")).unwrap();
    assert!(result.success);
    let text = &result.answer;
    assert!(text.contains("incident timeline"), "no timeline header:\n{text}");
    // "deploy" in the filename -> "Deployment" label; deploy at 02:00.
    assert!(text.contains("Deployment at 02:00"), "no anchored deployment line:\n{text}");
    assert!(text.contains("minutes later"), "no lag phrasing:\n{text}");

    let _ = std::fs::remove_file(&log);
    let _ = std::fs::remove_file(&csv);
}

#[test]
fn incident_round_trips_through_json() {
    olorin::kernels::ffi::init().unwrap();
    let deploys = [2 * 3600, 4 * 3600, 6 * 3600];
    let log = write_tmp("olorin_inc_rt_app.log", &log_with_error_bursts(&deploys, 240));
    let csv = write_tmp("olorin_inc_rt_deploys.csv", &deploys_csv(&deploys));

    let result = run_rune("eacorrelate", &format!("--json {log} {csv}")).unwrap();
    let out = parse(&result.answer);
    assert!(out.incident.is_some());

    // to_json -> from_json must preserve the incident block exactly.
    let again = parse(&out.to_json());
    assert_eq!(out.incident, again.incident, "incident block not round-trip stable");

    let _ = std::fs::remove_file(&log);
    let _ = std::fs::remove_file(&csv);
}

#[test]
fn no_incident_without_a_cascade() {
    olorin::kernels::ffi::init().unwrap();
    // Independent LCG scatter — no correlation crosses threshold, so no
    // cascade and therefore no incident (silence is the honest answer).
    let mut state: u64 = 0x9e3779b97f4a7c15;
    let mut scatter = |salt: u64| -> Vec<u8> {
        let mut buf = String::new();
        let mut secs: Vec<i64> = (0..200).map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(salt | 1);
            ((state >> 33) % (8 * 3600)) as i64
        }).collect();
        secs.sort_unstable();
        for s in secs { writeln!(buf, "{} INFO event", stamp(s)).unwrap(); }
        buf.into_bytes()
    };
    let a = write_tmp("olorin_inc_rand_a.log", &scatter(7));
    let b = write_tmp("olorin_inc_rand_b.log", &scatter(99));

    let result = run_rune("eacorrelate", &format!("--json {a} {b}")).unwrap();
    let out = parse(&result.answer);
    assert!(out.incident.is_none(), "incident invented without a cascade: {}", result.answer);

    let _ = std::fs::remove_file(&a);
    let _ = std::fs::remove_file(&b);
}

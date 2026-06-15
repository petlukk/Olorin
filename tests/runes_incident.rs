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

//! eacorrelate — pino/bunyan NUMERIC severity levels (`"level":50`).
//!
//! The JSON error sub-stream historically matched only string levels
//! (`"level":"error"`, zap/zerolog) via the keyword scan. pino/bunyan encode
//! severity as a number (error=50, fatal=60), so their errors were folded into
//! baseline traffic. `json_level_scan` (value >= 50) is the numeric twin; these
//! tests prove a planted deploy->error lag recovers, and that warn (40) is not
//! treated as an error.

use olorin::runes::output::RuneOutput;
use olorin::runes::run_rune;
use std::fmt::Write as _;
use std::io::Write;

/// Unix-epoch base (2025-06-11-ish); pino `time` is epoch-millis.
const BASE: i64 = 1_749_600_000;

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

fn ms(secs: i64) -> i64 {
    (BASE + secs) * 1000
}

/// 8h pino ndjson app log: one info (level 30) line per minute, plus a 20-line
/// `level`-burst `lag_secs` after each deploy. `lvl` chooses the burst severity
/// (50=error, 40=warn) so the same shape drives both the positive and the
/// negative-control test.
fn pino_app(deploy_secs: &[i64], lag_secs: i64, lvl: i64) -> Vec<u8> {
    let mut buf = String::new();
    for m in 0..=(8 * 60) {
        writeln!(buf, "{{\"level\":30,\"time\":{},\"msg\":\"heartbeat\"}}", ms(m * 60)).unwrap();
    }
    for &d in deploy_secs {
        for _ in 0..20 {
            writeln!(buf, "{{\"level\":{},\"time\":{},\"msg\":\"upstream timeout\"}}", lvl, ms(d + lag_secs)).unwrap();
        }
    }
    buf.into_bytes()
}

/// pino deploy log: one info line per release, same epoch-millis grid.
fn pino_deploys(deploy_secs: &[i64]) -> Vec<u8> {
    let mut buf = String::new();
    for &d in deploy_secs {
        writeln!(buf, "{{\"level\":30,\"time\":{},\"msg\":\"released\"}}", ms(d)).unwrap();
    }
    buf.into_bytes()
}

#[test]
fn recovers_planted_pino_numeric_error_lag() {
    olorin::kernels::ffi::init().unwrap();
    let deploys = [2 * 3600, 4 * 3600, 6 * 3600];
    let log_path = write_tmp("olorin_corr_pino.jsonl", &pino_app(&deploys, 240, 50));
    let dep_path = write_tmp("olorin_corr_pino_deploys.jsonl", &pino_deploys(&deploys));

    let result = run_rune("eacorrelate", &format!("--json {log_path} {dep_path}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);
    let out = parse_answer(&result.answer);

    // Streams: app all-events, app (errors), deploys all-events.
    assert_eq!(out.categories.len(), 3, "answer={}", result.answer);

    let f = out.correlations.iter()
        .find(|c| c.stream_a == "olorin_corr_pino.jsonl (errors)")
        .unwrap_or_else(|| panic!("no pino numeric errors-substream finding: {}", result.answer));
    assert_eq!(f.stream_b, "olorin_corr_pino_deploys.jsonl");
    assert_eq!(f.lag_seconds, 240, "wrong lag: {:?}", f);
    assert_eq!(f.events_a, 60); // 3 bursts x 20 lines
    assert_eq!(f.events_b, 3);
    assert!(f.score > 0.8, "weak score: {:?}", f);

    let _ = std::fs::remove_file(&log_path);
    let _ = std::fs::remove_file(&dep_path);
}

#[test]
fn pino_warn_level_is_not_an_error_substream() {
    // level 40 (warn) is below the error threshold — no (errors) stream forms,
    // so there is nothing to correlate against the deploys (mirrors the keyword
    // sub-stream, which surfaces ERROR/FATAL but not WARN).
    olorin::kernels::ffi::init().unwrap();
    let deploys = [2 * 3600, 4 * 3600, 6 * 3600];
    let log_path = write_tmp("olorin_corr_pino_warn.jsonl", &pino_app(&deploys, 240, 40));
    let dep_path = write_tmp("olorin_corr_pino_warn_deploys.jsonl", &pino_deploys(&deploys));

    let result = run_rune("eacorrelate", &format!("--json {log_path} {dep_path}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);
    let out = parse_answer(&result.answer);

    assert!(
        !out.categories.iter().any(|c| c.name.ends_with("(errors)")),
        "warn level must not form an error sub-stream: {}", result.answer
    );

    let _ = std::fs::remove_file(&log_path);
    let _ = std::fs::remove_file(&dep_path);
}

//! eacorrelate — classic BSD syslog (`MMM DD HH:MM:SS`) error sub-stream.
//!
//! Split out of `runes_eacorrelate.rs` (500-line cap). Guards the
//! `Format::Syslog` substream wiring: before it the dispatch returned
//! `Vec::new()`, so a syslog app log never split its ERROR burst from its INFO
//! baseline and a deploy lag was invisible. `log_level_scan` is format-agnostic
//! (keyword), so this reuses it with `syslog_bytes_to_seconds` as the decoder.

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

/// Classic BSD syslog stamp (`Jun 14 HH:MM:SS`) for `secs` into an 8h day.
/// Day 14 has no space-padding, so the parser's two-digit path is exercised.
fn syslog_stamp(secs: i64) -> String {
    format!("Jun 14 {:02}:{:02}:{:02}", secs / 3600, (secs % 3600) / 60, secs % 60)
}

/// 8h BSD-syslog app log: one INFO heartbeat per minute, plus a 20-line ERROR
/// burst `lag_secs` after each deploy — real syslog format so `detect_format`
/// classifies it `Format::Syslog`.
fn syslog_app_with_bursts(deploy_secs: &[i64], lag_secs: i64) -> Vec<u8> {
    let mut buf = String::new();
    for m in 0..=(8 * 60) {
        writeln!(buf, "{} host app[42]: INFO heartbeat", syslog_stamp(m * 60)).unwrap();
    }
    for &d in deploy_secs {
        for k in 0..20 {
            writeln!(buf, "{} host app[42]: ERROR upstream timeout #{k}", syslog_stamp(d + lag_secs)).unwrap();
        }
    }
    buf.into_bytes()
}

/// BSD-syslog deploy log: one INFO line per release. Same syslog time grid as
/// the app log so the two correlate (an ISO/CSV trigger would land in a
/// different decoded era — syslog decodes to a fixed reference year).
fn syslog_deploys(deploy_secs: &[i64]) -> Vec<u8> {
    let mut buf = String::new();
    for &d in deploy_secs {
        writeln!(buf, "{} host deploy: INFO released sha abc{d}", syslog_stamp(d)).unwrap();
    }
    buf.into_bytes()
}

#[test]
fn recovers_planted_syslog_error_lag() {
    olorin::kernels::ffi::init().unwrap();
    // Deploys at 02:00 / 04:00 / 06:00, error bursts 240s later. 8h span /
    // 512 target -> 60s buckets, so the lag is exactly 4 buckets.
    let deploys = [2 * 3600, 4 * 3600, 6 * 3600];
    let log_path = write_tmp("olorin_corr_syslog.log", &syslog_app_with_bursts(&deploys, 240));
    let dep_path = write_tmp("olorin_corr_syslog_deploys.log", &syslog_deploys(&deploys));

    let result = run_rune("eacorrelate", &format!("--json {log_path} {dep_path}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);
    let out = parse_answer(&result.answer);

    // Streams: app all-events, app (errors), deploys all-events.
    assert_eq!(out.categories.len(), 3, "answer={}", result.answer);

    let f = out.correlations.iter()
        .find(|c| c.stream_a == "olorin_corr_syslog.log (errors)")
        .unwrap_or_else(|| panic!("no syslog errors-substream finding: {}", result.answer));
    assert_eq!(f.stream_b, "olorin_corr_syslog_deploys.log");
    assert_eq!(f.lag_seconds, 240, "wrong lag: {:?}", f);
    assert_eq!(f.width_seconds, 60);
    assert!(f.score > 0.8, "weak score: {:?}", f);
    assert_eq!(f.events_a, 60); // 3 bursts x 20 lines
    assert_eq!(f.events_b, 3);

    let _ = std::fs::remove_file(&log_path);
    let _ = std::fs::remove_file(&dep_path);
}

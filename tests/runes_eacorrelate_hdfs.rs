//! eacorrelate — HDFS / Hadoop log format (`YYMMDD HHMMSS LEVEL ...`).
//!
//! Two proofs: (1) the real Loghub `HDFS_2k.log` is detected and fully decoded
//! (2000 events), and — since it carries only INFO/WARN — forms NO error
//! sub-stream (we don't fabricate errors); (2) a planted deploy→ERROR lag in
//! synthetic HDFS format recovers via the incident timeline.

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

#[test]
fn hdfs_real_loghub_detects_with_no_error_substream() {
    olorin::kernels::ffi::init().unwrap();
    // Real Loghub sample. Oracle (verified with awk/grep): 2000 timestamped
    // lines, levels only INFO (1920) and WARN (80) — zero ERROR/FATAL, and no
    // line contains the word "error". So no error sub-stream may form. The data
    // is fetched on demand (gitignored), so skip when absent; the planted test
    // covers ERROR detection in CI.
    let src = concat!(env!("CARGO_MANIFEST_DIR"), "/benchmarks/robustness/data/HDFS_2k.log");
    if !std::path::Path::new(src).exists() {
        eprintln!("skip: {src} not fetched");
        return;
    }
    let bytes = std::fs::read(src).expect("read vendored HDFS_2k.log");
    let log = write_tmp("olorin_hdfs_real.log", &bytes);
    let trig = write_tmp("olorin_hdfs_real_trig.log",
        b"251116 020000 9 INFO dfs.DataNode: restart a\n\
          251116 020001 9 INFO dfs.DataNode: restart b\n\
          251116 020002 9 INFO dfs.DataNode: restart c\n");

    let result = run_rune("eacorrelate", &format!("--json {log} {trig}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);
    let out = parse_answer(&result.answer);

    let all = out.categories.iter()
        .find(|c| c.name == "olorin_hdfs_real.log")
        .unwrap_or_else(|| panic!("HDFS log not detected: {}", result.answer));
    assert_eq!(all.count, 2000, "all-events count drifted from the oracle");
    assert!(
        !out.categories.iter().any(|c| c.name == "olorin_hdfs_real.log (errors)"),
        "HDFS_2k has no ERROR lines — an error substream must not form: {}", result.answer
    );

    let _ = std::fs::remove_file(&log);
    let _ = std::fs::remove_file(&trig);
}

/// HDFS stamp `YYMMDD HHMMSS` on a fixed 2025-11-16 day for `secs` into it.
fn hdfs_stamp(secs: i64) -> String {
    format!("251116 {:02}{:02}{:02}", secs / 3600, (secs % 3600) / 60, secs % 60)
}

fn hdfs_app(deploy_secs: &[i64], lag_secs: i64) -> Vec<u8> {
    let mut buf = String::new();
    for m in 0..=(8 * 60) {
        writeln!(buf, "{} 148 INFO dfs.DataNode: heartbeat", hdfs_stamp(m * 60)).unwrap();
    }
    for &d in deploy_secs {
        for k in 0..20 {
            writeln!(buf, "{} 148 ERROR dfs.DataNode: block recovery failed #{k}", hdfs_stamp(d + lag_secs)).unwrap();
        }
    }
    buf.into_bytes()
}

fn hdfs_deploys(deploy_secs: &[i64]) -> Vec<u8> {
    let mut buf = String::new();
    for &d in deploy_secs {
        writeln!(buf, "{} 1 INFO org.apache.hadoop: namenode restart", hdfs_stamp(d)).unwrap();
    }
    buf.into_bytes()
}

#[test]
fn recovers_planted_hdfs_error_lag() {
    olorin::kernels::ffi::init().unwrap();
    let deploys = [2 * 3600, 4 * 3600, 6 * 3600];
    let log = write_tmp("olorin_hdfs_plant.log", &hdfs_app(&deploys, 240));
    let dep = write_tmp("olorin_hdfs_plant_deploys.log", &hdfs_deploys(&deploys));

    let result = run_rune("eacorrelate", &format!("--json {log} {dep}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);
    let out = parse_answer(&result.answer);

    assert_eq!(out.categories.len(), 3, "answer={}", result.answer);
    let f = out.correlations.iter()
        .find(|c| c.stream_a == "olorin_hdfs_plant.log (errors)")
        .unwrap_or_else(|| panic!("no HDFS errors-substream finding: {}", result.answer));
    assert_eq!(f.stream_b, "olorin_hdfs_plant_deploys.log");
    assert_eq!(f.lag_seconds, 240, "wrong lag: {:?}", f);
    assert_eq!(f.events_a, 60); // 3 bursts x 20 lines
    assert_eq!(f.events_b, 3);
    assert!(f.score > 0.8, "weak score: {:?}", f);

    let _ = std::fs::remove_file(&log);
    let _ = std::fs::remove_file(&dep);
}

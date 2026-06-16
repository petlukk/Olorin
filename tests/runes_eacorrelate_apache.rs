//! eacorrelate — Apache error-log format (`[Www Mmm DD HH:MM:SS YYYY] [sev]`).
//!
//! Two proofs: (1) a differential pass on the real Loghub `Apache_2k.log` —
//! detection + decode + the error sub-stream count must match an independent
//! grep of the `[error]` severity; (2) a planted deploy→error lag in synthetic
//! Apache format recovers via the incident timeline.

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
fn apache_error_substream_matches_loghub_oracle() {
    olorin::kernels::ffi::init().unwrap();
    // Real Loghub sample vendored in the repo. Independent oracle (verified with
    // `grep`): 2000 timestamped lines, of which 595 are `[error]` severity (the
    // other 1405 are `[notice]`; no `[notice]` line contains the word "error").
    let src = concat!(env!("CARGO_MANIFEST_DIR"), "/benchmarks/robustness/data/Apache_2k.log");
    let bytes = std::fs::read(src).expect("read vendored Apache_2k.log");
    let log = write_tmp("olorin_apache_real.log", &bytes);
    // A 3-line Apache trigger so eacorrelate retains the log and forms substreams.
    let trig = write_tmp("olorin_apache_real_trig.log",
        b"[Sun Dec 04 04:50:00 2005] [notice] deploy a\n\
          [Sun Dec 04 04:50:01 2005] [notice] deploy b\n\
          [Sun Dec 04 04:50:02 2005] [notice] deploy c\n");

    let result = run_rune("eacorrelate", &format!("--json {log} {trig}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);
    let out = parse_answer(&result.answer);

    let all = out.categories.iter()
        .find(|c| c.name == "olorin_apache_real.log")
        .unwrap_or_else(|| panic!("Apache log not detected: {}", result.answer));
    assert_eq!(all.count, 2000, "all-events count drifted from the oracle");

    let errs = out.categories.iter()
        .find(|c| c.name == "olorin_apache_real.log (errors)")
        .unwrap_or_else(|| panic!("no Apache errors substream: {}", result.answer));
    assert_eq!(errs.count, 595, "error substream != the 595 [error] lines (dedup/severity drift)");

    let _ = std::fs::remove_file(&log);
    let _ = std::fs::remove_file(&trig);
}

/// Apache stamp `[Mon Jun 16 HH:MM:SS 2025]` for `secs` into an 8h day (the
/// weekday is fixed — the decoder ignores it).
fn apache_stamp(secs: i64) -> String {
    format!("[Mon Jun 16 {:02}:{:02}:{:02} 2025]", secs / 3600, (secs % 3600) / 60, secs % 60)
}

fn apache_app(deploy_secs: &[i64], lag_secs: i64) -> Vec<u8> {
    let mut buf = String::new();
    for m in 0..=(8 * 60) {
        writeln!(buf, "{} [notice] heartbeat", apache_stamp(m * 60)).unwrap();
    }
    for &d in deploy_secs {
        for k in 0..20 {
            writeln!(buf, "{} [error] mod_jk child workerEnv failed #{k}", apache_stamp(d + lag_secs)).unwrap();
        }
    }
    buf.into_bytes()
}

fn apache_deploys(deploy_secs: &[i64]) -> Vec<u8> {
    let mut buf = String::new();
    for &d in deploy_secs {
        writeln!(buf, "{} [notice] graceful restart: config v1.1.0", apache_stamp(d)).unwrap();
    }
    buf.into_bytes()
}

/// ISO deploys at the SAME real wall-clock as `apache_stamp` (2025-06-16).
fn iso_deploys_matching(deploy_secs: &[i64]) -> Vec<u8> {
    let mut buf = String::new();
    for &d in deploy_secs {
        writeln!(buf, "2025-06-16T{:02}:{:02}:{:02} INFO deploy v1.1.0",
            d / 3600, (d % 3600) / 60, d % 60).unwrap();
    }
    buf.into_bytes()
}

#[test]
fn apache_shares_real_era_with_iso_not_misdetected_as_syslog() {
    // Regression: an Apache instant `[Www Mmm DD HH:MM:SS YYYY]` contains a valid
    // syslog substring `Mmm DD HH:MM:SS`, so syslog_scan also matches Apache
    // lines. If Apache loses that detection tie it decodes with syslog's fixed
    // reference YEAR and lands in a disjoint era — so it must still correlate
    // with a real-dated ISO trigger at the SAME wall-clock. (An Apache-vs-Apache
    // test can't catch this: both files misdetect identically and still align.)
    olorin::kernels::ffi::init().unwrap();
    let deploys = [2 * 3600, 4 * 3600, 6 * 3600];
    let log = write_tmp("olorin_apache_era.log", &apache_app(&deploys, 240));
    let dep = write_tmp("olorin_apache_era_iso_deploys.log", &iso_deploys_matching(&deploys));

    let result = run_rune("eacorrelate", &format!("--json {log} {dep}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);
    let out = parse_answer(&result.answer);
    let f = out.correlations.iter()
        .find(|c| c.stream_a == "olorin_apache_era.log (errors)")
        .unwrap_or_else(|| panic!("Apache decoded to the wrong era (misdetected as syslog?): {}", result.answer));
    assert_eq!(f.lag_seconds, 240, "wrong lag: {:?}", f);

    let _ = std::fs::remove_file(&log);
    let _ = std::fs::remove_file(&dep);
}

#[test]
fn recovers_planted_apache_error_lag() {
    olorin::kernels::ffi::init().unwrap();
    let deploys = [2 * 3600, 4 * 3600, 6 * 3600];
    let log = write_tmp("olorin_apache_plant.log", &apache_app(&deploys, 240));
    let dep = write_tmp("olorin_apache_plant_deploys.log", &apache_deploys(&deploys));

    let result = run_rune("eacorrelate", &format!("--json {log} {dep}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);
    let out = parse_answer(&result.answer);

    assert_eq!(out.categories.len(), 3, "answer={}", result.answer);
    let f = out.correlations.iter()
        .find(|c| c.stream_a == "olorin_apache_plant.log (errors)")
        .unwrap_or_else(|| panic!("no Apache errors-substream finding: {}", result.answer));
    assert_eq!(f.stream_b, "olorin_apache_plant_deploys.log");
    assert_eq!(f.lag_seconds, 240, "wrong lag: {:?}", f);
    assert_eq!(f.events_a, 60); // 3 bursts x 20 lines, deduped one-per-line
    assert_eq!(f.events_b, 3);
    assert!(f.score > 0.8, "weak score: {:?}", f);

    let _ = std::fs::remove_file(&log);
    let _ = std::fs::remove_file(&dep);
}

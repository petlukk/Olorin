//! ealog: the long severity spellings WARNING and CRITICAL (used by Python
//! `logging` and syslog) must fold into the WARN and FATAL buckets, in both
//! the scalar (small-file / head / tail) and SIMD-body (large-file) paths of
//! `log_level_scan.ea`. Word boundaries must still reject plurals and
//! compounds (WARNINGS, CRITICALLY, UNCRITICAL).
//!
//! Found by the runes robustness pass: a Python/syslog log silently
//! undercounted because only DEBUG/INFO/WARN/ERROR/FATAL were matched.

use olorin::runes::output::RuneOutput;
use olorin::runes::run_rune;

fn ensure_kernels() {
    olorin::kernels::ffi::init().expect("kernel init");
}

/// Run ealog --json on `raw` and return the severity-bucket counts.
fn counts(stem: &str, raw: &str) -> std::collections::HashMap<String, u64> {
    ensure_kernels();
    let path = std::env::temp_dir().join(format!(
        "olorin_ealog_wc_{stem}_{}.log", std::process::id()
    ));
    std::fs::write(&path, raw).unwrap();
    let res = run_rune("ealog", &format!("--json {}", path.display()))
        .expect("ealog runs");
    let _ = std::fs::remove_file(&path);
    let out = RuneOutput::from_json(res.answer.as_bytes()).expect("parse RuneOutput");
    assert!(out.success, "ealog should succeed");
    out.categories.into_iter().map(|c| (c.name, c.count)).collect()
}

fn warn(c: &std::collections::HashMap<String, u64>) -> u64 { c["WARN"] }
fn fatal(c: &std::collections::HashMap<String, u64>) -> u64 { c["FATAL"] }

#[test]
fn warning_spellings_fold_into_warn() {
    // upper / lower / title WARNING, plus the short WARN and Warn.
    let c = counts("warn", "WARNING a\nwarning b\nWarning c\nWARN d\nWarn e\n");
    assert_eq!(warn(&c), 5);
    assert_eq!(fatal(&c), 0);
}

#[test]
fn critical_spellings_fold_into_fatal() {
    let c = counts("crit", "CRITICAL a\ncritical b\nCritical c\nFATAL d\n");
    assert_eq!(fatal(&c), 4);
    assert_eq!(warn(&c), 0);
}

#[test]
fn plurals_and_compounds_do_not_match() {
    let c = counts(
        "neg",
        "WARNINGS x\nCRITICALLY y\nUNCRITICAL z\nERRORS w\nINFORMATION q\nTERROR r\n",
    );
    assert_eq!(warn(&c), 0, "WARNINGS must not match");
    assert_eq!(fatal(&c), 0, "CRITICALLY / UNCRITICAL must not match");
    assert_eq!(c["ERROR"], 0);
    assert_eq!(c["INFO"], 0);
}

#[test]
fn eof_terminated_keywords_match() {
    // No trailing newline: end-of-file is an implicit delimiter.
    assert_eq!(warn(&counts("eofw", "msg WARNING")), 1);
    assert_eq!(fatal(&counts("eofc", "boom CRITICAL")), 1);
}

#[test]
fn simd_body_path_counts_long_spellings() {
    // >37 bytes with the keyword inside the SIMD region (repeated lines).
    let c = counts("simdw", &"WARNING line of text here\n".repeat(50));
    assert_eq!(warn(&c), 50);
    assert_eq!(fatal(&c), 0);

    let c = counts("simdc", &"CRITICAL line of text here\n".repeat(50));
    assert_eq!(fatal(&c), 50);
    assert_eq!(warn(&c), 0);

    // Negative under the SIMD path too.
    let c = counts("simdn", &"WARNINGS not counted here ok\n".repeat(50));
    assert_eq!(warn(&c), 0);
}

#[test]
fn base_five_keywords_unregressed() {
    let c = counts("base", "DEBUG a\nINFO b\nWARN c\nERROR d\nFATAL e\n");
    assert_eq!(c["DEBUG"], 1);
    assert_eq!(c["INFO"], 1);
    assert_eq!(c["WARN"], 1);
    assert_eq!(c["ERROR"], 1);
    assert_eq!(c["FATAL"], 1);
}

#[test]
fn critical_surfaces_a_high_severity_sample() {
    ensure_kernels();
    let path = std::env::temp_dir().join(format!(
        "olorin_ealog_wc_sample_{}.log", std::process::id()
    ));
    std::fs::write(&path, "info ok\nCRITICAL meltdown now\ninfo ok\n").unwrap();
    let res = run_rune("ealog", &format!("--json {}", path.display())).unwrap();
    let _ = std::fs::remove_file(&path);
    let out = RuneOutput::from_json(res.answer.as_bytes()).unwrap();
    assert!(
        out.samples.iter().any(|s| s.text.contains("CRITICAL")),
        "CRITICAL is high-severity and must be recorded as a sample line"
    );
}

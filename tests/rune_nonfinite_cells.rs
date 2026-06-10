//! Regression (runes robustness wave, found via differential vs pandas):
//! a non-finite cell value ("nan"/"inf") must not poison a numeric column's
//! additive stats.
//!
//! Before the fix, eacrunch on `v\n1\n2\nnan\n` reported count=3, min=1, max=2
//! but sum=null, mean=null — Rust's `f64::parse` accepts "nan"/"inf", the NaN
//! propagated through `sum`/`mean` (serialized as null by `finite_f64`) while
//! `min`/`max` survived, leaving an internally inconsistent, silently-wrong
//! column summary that the contract fuzz (valid-JSON-only) couldn't catch.
//! The fix drops non-finite parses so the column summarizes its finite values
//! consistently (matching pandas' skipna behavior).

use olorin::runes::output::RuneOutput;
use std::io::Write;
use std::process::Command;

const OLORIN: &str = env!("CARGO_BIN_EXE_olorin");

fn run_eacrunch(csv: &str, tag: &str) -> RuneOutput {
    let path = format!("/tmp/olorin_nonfinite_{tag}.csv");
    std::fs::File::create(&path).unwrap().write_all(csv.as_bytes()).unwrap();
    let out = Command::new(OLORIN)
        .args(["rune", "eacrunch", "--json", &path])
        .output()
        .expect("spawn olorin");
    let _ = std::fs::remove_file(&path);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .find(|l| l.starts_with("{\"schema_version\""))
        .unwrap_or_else(|| panic!("no RuneOutput JSON on stdout:\n{stdout}"));
    RuneOutput::from_json(line.as_bytes()).expect("parse RuneOutput")
}

/// (count, min, max, sum, mean) of a numeric column.
fn numeric(out: &RuneOutput, col: &str) -> (u64, f64, f64, f64, f64) {
    let f = out.fields.iter().find(|f| f.name == col).expect("column present");
    let n = f.numeric.as_ref().expect("numeric stats present");
    (f.count, n.min, n.max, n.sum, n.mean)
}

#[test]
fn nan_cell_is_excluded_not_poisoning() {
    let out = run_eacrunch("v\n1\n2\nnan\n", "nan");
    assert!(out.success, "should succeed, got error {:?}", out.error);
    let (count, min, max, sum, mean) = numeric(&out, "v");
    assert_eq!(count, 2, "nan must be excluded from the numeric count");
    assert_eq!(sum, 3.0, "sum must be over the finite values, not poisoned to null");
    assert_eq!(mean, 1.5, "mean must be over the finite values");
    assert_eq!(min, 1.0);
    assert_eq!(max, 2.0);
}

#[test]
fn inf_cell_is_excluded_not_poisoning() {
    let out = run_eacrunch("v\n10\n20\n30\ninf\n", "inf");
    assert!(out.success, "should succeed, got error {:?}", out.error);
    let (count, min, max, sum, mean) = numeric(&out, "v");
    assert_eq!(count, 3, "inf must be excluded from the numeric count");
    assert_eq!(sum, 60.0, "sum must be over the finite values");
    assert_eq!(mean, 20.0, "mean must be over the finite values");
    assert_eq!(min, 10.0);
    assert_eq!(max, 30.0);
}

/// A column that is *entirely* non-finite degrades cleanly: still numeric,
/// zero finite values, no panic, valid output.
#[test]
fn all_nonfinite_column_degrades_cleanly() {
    let out = run_eacrunch("v\nnan\ninf\nnan\n", "allnonfinite");
    assert!(out.success, "should succeed, got error {:?}", out.error);
    let f = out.fields.iter().find(|f| f.name == "v").expect("column present");
    assert_eq!(f.count, 0, "no finite values to count");
}

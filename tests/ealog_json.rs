//! ealog `--json` mode: structured RuneOutput on success AND failure.
//!
//! These tests pin the proof-of-migration contract:
//! - Success path emits a parseable RuneOutput with correct categories
//!   and samples.
//! - Failure path (missing file, bad path) ALSO emits a RuneOutput so
//!   chained downstream runes can read the error structurally instead
//!   of choking on free-form text.
//! - The structured form must agree with the legacy text form on the
//!   same input — counts and sample line numbers.

use olorin::runes::output::{RuneOutput, SCHEMA_VERSION};
use olorin::runes::run_rune;
use std::io::Write;

const SYNTHETIC_LOG: &[u8] = b"\
2026-05-11 INFO: starting
2026-05-11 DEBUG: loaded config
2026-05-11 WARN: cache miss
2026-05-11 ERROR: connection reset by peer
2026-05-11 INFO: retrying
2026-05-11 ERROR: auth failed user=anonymous
2026-05-11 FATAL: aborting after 3 failures
";

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
fn json_mode_emits_structured_output() {
    let path = write_tmp("olorin_ealog_json_ok.log", SYNTHETIC_LOG);
    let result = run_rune("ealog", &format!("--json {path}")).expect("rune ran");

    assert!(result.success, "rune should succeed: {}", result.answer);
    let out = parse_answer(&result.answer);

    assert_eq!(out.schema_version, SCHEMA_VERSION);
    assert_eq!(out.rune, "ealog");
    assert!(out.success);
    assert!(out.error.is_none());

    let src = out.source.expect("source populated");
    assert!(src.path.ends_with("olorin_ealog_json_ok.log"), "bad path: {}", src.path);
    assert_eq!(src.bytes, SYNTHETIC_LOG.len() as u64);
    assert_eq!(src.format, "plaintext");

    assert_eq!(out.totals.rows, 7, "7 log lines");

    let counts: std::collections::HashMap<_, _> =
        out.categories.iter().map(|c| (c.name.as_str(), c.count)).collect();
    assert_eq!(counts.get("DEBUG"), Some(&1));
    assert_eq!(counts.get("INFO"),  Some(&2));
    assert_eq!(counts.get("WARN"),  Some(&1));
    assert_eq!(counts.get("ERROR"), Some(&2));
    assert_eq!(counts.get("FATAL"), Some(&1));

    // High-severity samples: 2 ERROR + 1 FATAL = 3, all under the cap of 5.
    assert_eq!(out.samples.len(), 3);
    let texts: Vec<&str> = out.samples.iter().map(|s| s.text.as_str()).collect();
    assert!(texts.iter().any(|t| t.contains("connection reset")));
    assert!(texts.iter().any(|t| t.contains("auth failed")));
    assert!(texts.iter().any(|t| t.contains("aborting after")));
    for s in &out.samples {
        assert!(s.line.is_some() && s.line.unwrap() > 0);
        assert!(s.byte_offset.is_some());
    }
}

#[test]
fn json_mode_accepts_flag_before_or_after_path() {
    let path = write_tmp("olorin_ealog_json_order.log", SYNTHETIC_LOG);

    let prefix = run_rune("ealog", &format!("--json {path}")).unwrap();
    let suffix = run_rune("ealog", &format!("{path} --json")).unwrap();

    // Round-trip both and compare structural fields. Direct string comparison
    // is fragile because scan_us varies per call.
    let a = parse_answer(&prefix.answer);
    let b = parse_answer(&suffix.answer);
    assert_eq!(a.categories, b.categories);
    assert_eq!(a.samples,    b.samples);
    assert_eq!(a.totals.rows, b.totals.rows);
}

#[test]
fn json_mode_error_path_also_emits_json() {
    // Chaining contract: downstream runes get a parseable RuneOutput
    // with success=false even when the rune fails — otherwise the chain
    // would fail with a meaningless "expected JSON, got plain text".
    let result = run_rune("ealog", "--json /tmp/does_not_exist_xyz_12345.log")
        .expect("rune ran");

    assert!(!result.success);
    let out = parse_answer(&result.answer);
    assert!(!out.success);
    let err = out.error.expect("error populated");
    assert!(err.contains("not found"), "unexpected error: {err}");
    assert!(out.categories.is_empty());
    assert!(out.samples.is_empty());
}

#[test]
fn json_and_text_modes_agree_on_counts() {
    // The schema migration's invariant: structured form and legacy text
    // form are built from the same source-of-truth RuneOutput, so they
    // must report the same numbers. If they ever drift, this test fires.
    let path = write_tmp("olorin_ealog_agree.log", SYNTHETIC_LOG);

    let text = run_rune("ealog", &path).unwrap();
    let json = run_rune("ealog", &format!("--json {path}")).unwrap();
    let out = parse_answer(&json.answer);

    // Line count appears as "lines:   N" in text.
    assert!(text.answer.contains(&format!("lines:   {}", out.totals.rows)));

    // Each severity count appears in the text — match the structured form.
    for c in &out.categories {
        // Text format pads the name to width 6 and the count to width 12.
        let needle = format!("  {:<6} {:>12}", c.name, c.count);
        assert!(
            text.answer.contains(&needle),
            "text missing severity row for {} ({}): {}",
            c.name, c.count, text.answer
        );
    }
}

#[test]
fn json_mode_missing_path_emits_usage_error() {
    let result = run_rune("ealog", "--json").expect("rune ran");
    assert!(!result.success);
    let out = parse_answer(&result.answer);
    assert!(!out.success);
    let err = out.error.expect("error populated");
    assert!(err.contains("usage:"));
}

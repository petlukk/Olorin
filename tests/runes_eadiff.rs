//! eadiff: structural delta between two RuneOutput JSON files.
//!
//! Hardest test target of the rune family because it's the only rune
//! whose inputs are *other* runes' outputs — this is the schema's
//! end-to-end chaining proof.

use olorin::runes::output::{
    Category, FieldKind, FieldStats, NumericStats, RuneOutput, Totals,
};
use olorin::runes::{run_rune, RUNES, OutputSafety};
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

fn write_runeoutput(name: &str, out: &RuneOutput) -> String {
    let path = format!("/tmp/{name}");
    let mut f = std::fs::File::create(&path).expect("tmp create");
    f.write_all(out.to_json().as_bytes()).expect("tmp write");
    path
}

fn numeric_field(name: &str, min: f64, max: f64, mean: f64, sum: f64, count: u64) -> FieldStats {
    FieldStats {
        name: name.to_string(),
        kind: FieldKind::Number,
        count,
        null_count: None,
        numeric: Some(NumericStats { min, max, mean, sum }),
        text: None, bool: None, timestamp: None,
    }
}

fn fixture_with_numerics(rune: &str, bearing: f64, motor: f64) -> RuneOutput {
    let mut r = RuneOutput::new(rune, 1);
    r.totals = Totals { rows: 1000, scan_us: 0 };
    r.fields.push(numeric_field("bearing_stock", bearing, bearing, bearing, bearing * 1000.0, 1000));
    r.fields.push(numeric_field("motor_temp",    motor,   motor,   motor,   motor   * 1000.0, 1000));
    r
}

fn fixture_with_hour_categories(rune: &str, hour_counts: &[(usize, u64)]) -> RuneOutput {
    let mut r = RuneOutput::new(rune, 1);
    r.totals = Totals { rows: hour_counts.iter().map(|(_, c)| c).sum(), scan_us: 0 };
    let mut counts = [0u64; 24];
    for &(h, c) in hour_counts { counts[h] = c; }
    r.categories = (0..24).map(|h| Category {
        name:  format!("{h:02}:00"),
        count: counts[h],
    }).collect();
    r
}

#[test]
fn eadiff_is_registered() {
    olorin::kernels::ffi::init().unwrap();
    let found = RUNES.iter().any(|r| r.name() == "eadiff");
    assert!(found, "eadiff missing from registry");
}

#[test]
fn eadiff_output_safety_is_untrusted() {
    let r = RUNES.iter().find(|r| r.name() == "eadiff")
        .expect("eadiff registered");
    assert_eq!(r.output_safety(), OutputSafety::UntrustedQuoted);
}

#[test]
fn eadiff_numeric_fields_delta_b_minus_a() {
    olorin::kernels::ffi::init().unwrap();
    let yesterday = fixture_with_numerics("eacrunch", 100.0, 70.0);
    let today     = fixture_with_numerics("eacrunch",  85.0, 72.0);
    let a = write_runeoutput("olorin_eadiff_a_numeric.json", &yesterday);
    let b = write_runeoutput("olorin_eadiff_b_numeric.json", &today);

    let result = run_rune("eadiff", &format!("--json {a} {b}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);
    let out = parse_answer(&result.answer);
    assert_eq!(out.rune, "eadiff");

    let bearing = out.fields.iter().find(|f| f.name == "bearing_stock")
        .expect("bearing field present");
    let n = bearing.numeric.as_ref().unwrap();
    assert_eq!(n.mean, -15.0, "stock dropped 15");
    assert_eq!(n.min,  -15.0);

    let motor = out.fields.iter().find(|f| f.name == "motor_temp")
        .expect("motor field present");
    assert_eq!(motor.numeric.as_ref().unwrap().mean, 2.0, "temp rose 2");

    let _ = std::fs::remove_file(&a);
    let _ = std::fs::remove_file(&b);
}

#[test]
fn eadiff_categories_use_directional_prefix() {
    olorin::kernels::ffi::init().unwrap();
    // 06:00 grew 6→12; 07:00 shrank 3→0; 23:00 unchanged at 3.
    let yesterday = fixture_with_hour_categories("eatime",
        &[(6,  6), (7, 3), (23, 3)]);
    let today     = fixture_with_hour_categories("eatime",
        &[(6, 12), (7, 0), (23, 3)]);
    let a = write_runeoutput("olorin_eadiff_a_cats.json", &yesterday);
    let b = write_runeoutput("olorin_eadiff_b_cats.json", &today);

    let result = run_rune("eadiff", &format!("--json {a} {b}")).unwrap();
    let out = parse_answer(&result.answer);

    let by_name: std::collections::HashMap<&str, u64> =
        out.categories.iter().map(|c| (c.name.as_str(), c.count)).collect();
    assert_eq!(by_name.get("+06:00"), Some(&6), "06:00 grew by 6");
    assert_eq!(by_name.get("-07:00"), Some(&3), "07:00 shrank by 3");
    assert!(by_name.get("+23:00").is_none(), "23:00 unchanged, must be omitted");
    assert!(by_name.get("-23:00").is_none());
    // All other 0→0 buckets should be omitted too.
    assert_eq!(out.categories.len(), 2);

    let _ = std::fs::remove_file(&a);
    let _ = std::fs::remove_file(&b);
}

#[test]
fn eadiff_skips_text_bool_timestamp_mixed_fields() {
    olorin::kernels::ffi::init().unwrap();
    let mut a = RuneOutput::new("eajson", 1);
    let mut b = RuneOutput::new("eajson", 1);
    // Number — diffed.
    a.fields.push(numeric_field("status", 200.0, 200.0, 200.0, 200000.0, 1000));
    b.fields.push(numeric_field("status", 404.0, 404.0, 404.0, 404000.0, 1000));
    // Text — skipped in v1.
    a.fields.push(FieldStats {
        name: "level".into(), kind: FieldKind::Text, count: 1000,
        null_count: None, numeric: None,
        text: Some(olorin::runes::output::TextStats { unique: 3, top: vec![] }),
        bool: None, timestamp: None,
    });
    b.fields.push(FieldStats {
        name: "level".into(), kind: FieldKind::Text, count: 1000,
        null_count: None, numeric: None,
        text: Some(olorin::runes::output::TextStats { unique: 4, top: vec![] }),
        bool: None, timestamp: None,
    });
    let pa = write_runeoutput("olorin_eadiff_skips_a.json", &a);
    let pb = write_runeoutput("olorin_eadiff_skips_b.json", &b);

    let result = run_rune("eadiff", &format!("--json {pa} {pb}")).unwrap();
    let out = parse_answer(&result.answer);
    assert_eq!(out.fields.len(), 1, "only the Number field is diffed");
    assert_eq!(out.fields[0].name, "status");

    let _ = std::fs::remove_file(&pa);
    let _ = std::fs::remove_file(&pb);
}

#[test]
fn eadiff_asymmetric_keys_are_skipped() {
    olorin::kernels::ffi::init().unwrap();
    let mut a = RuneOutput::new("eacrunch", 1);
    a.fields.push(numeric_field("amount", 100.0, 100.0, 100.0, 100.0, 1));
    let mut b = RuneOutput::new("eacrunch", 1);
    b.fields.push(numeric_field("amount", 150.0, 150.0, 150.0, 150.0, 1));
    b.fields.push(numeric_field("new_col", 5.0, 5.0, 5.0, 5.0, 1));
    let pa = write_runeoutput("olorin_eadiff_asym_a.json", &a);
    let pb = write_runeoutput("olorin_eadiff_asym_b.json", &b);

    let result = run_rune("eadiff", &format!("--json {pa} {pb}")).unwrap();
    let out = parse_answer(&result.answer);
    assert_eq!(out.fields.len(), 1, "only the matched name survives");
    assert_eq!(out.fields[0].name, "amount");

    let _ = std::fs::remove_file(&pa);
    let _ = std::fs::remove_file(&pb);
}

#[test]
fn eadiff_end_to_end_chains_through_eatime() {
    // The killer demo: take two log files, run eatime --json on each,
    // pipe both outputs into eadiff, observe the hour-of-day delta.
    olorin::kernels::ffi::init().unwrap();

    let yesterday_log = b"\
2026-05-10T06:00:00 INFO start
2026-05-10T06:30:00 INFO heartbeat
2026-05-10T06:45:00 INFO heartbeat
2026-05-10T07:00:00 INFO ok
2026-05-10T15:00:00 INFO afternoon
";
    let today_log = b"\
2026-05-11T06:00:00 INFO start
2026-05-11T06:10:00 ERROR spike
2026-05-11T06:15:00 ERROR spike
2026-05-11T06:20:00 ERROR spike
2026-05-11T06:25:00 ERROR spike
2026-05-11T06:30:00 ERROR spike
2026-05-11T15:00:00 INFO afternoon
";
    let y_log = write_tmp("olorin_chain_yesterday.log", yesterday_log);
    let t_log = write_tmp("olorin_chain_today.log",     today_log);

    let y_result = run_rune("eatime", &format!("--json {y_log}")).unwrap();
    let t_result = run_rune("eatime", &format!("--json {t_log}")).unwrap();
    let y_json = write_tmp("olorin_chain_yesterday.json", y_result.answer.as_bytes());
    let t_json = write_tmp("olorin_chain_today.json",     t_result.answer.as_bytes());

    let diff = run_rune("eadiff", &format!("--json {y_json} {t_json}")).unwrap();
    assert!(diff.success, "diff failed: {}", diff.answer);
    let out = parse_answer(&diff.answer);

    // Yesterday: 3 at 06:00, 1 at 07:00, 1 at 15:00.
    // Today:     6 at 06:00, 0 at 07:00, 1 at 15:00.
    // Diff:      +06:00 = 3, -07:00 = 1. 15:00 unchanged, omitted.
    let by_name: std::collections::HashMap<&str, u64> =
        out.categories.iter().map(|c| (c.name.as_str(), c.count)).collect();
    assert_eq!(by_name.get("+06:00"), Some(&3),
        "06:00 should have grown by 3: {:?}", out.categories);
    assert_eq!(by_name.get("-07:00"), Some(&1),
        "07:00 should have shrunk by 1");

    let _ = std::fs::remove_file(&y_log);
    let _ = std::fs::remove_file(&t_log);
    let _ = std::fs::remove_file(&y_json);
    let _ = std::fs::remove_file(&t_json);
}

#[test]
fn eadiff_text_mode_uses_signed_formatting() {
    olorin::kernels::ffi::init().unwrap();
    let yesterday = fixture_with_numerics("eacrunch", 100.0, 70.0);
    let today     = fixture_with_numerics("eacrunch",  85.0, 72.0);
    let a = write_runeoutput("olorin_eadiff_text_a.json", &yesterday);
    let b = write_runeoutput("olorin_eadiff_text_b.json", &today);

    let result = run_rune("eadiff", &format!("{a} {b}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);
    let text = &result.answer;
    assert!(text.contains("bearing_stock: mean=-15.00"),
        "missing signed mean delta: {text}");
    assert!(text.contains("motor_temp: mean=+2.00"),
        "missing positive sign on motor delta: {text}");

    let _ = std::fs::remove_file(&a);
    let _ = std::fs::remove_file(&b);
}

#[test]
fn eadiff_usage_error_for_wrong_arg_count() {
    olorin::kernels::ffi::init().unwrap();
    let result = run_rune("eadiff", "/tmp/only_one.json").unwrap();
    assert!(!result.success);
    assert!(result.answer.contains("usage:"));

    let result = run_rune("eadiff", "--json /tmp/only_one.json").unwrap();
    let out = parse_answer(&result.answer);
    assert!(!out.success);
    assert!(out.error.expect("error populated").contains("usage:"));
}

#[test]
fn eadiff_invalid_json_input_emits_structured_failure() {
    olorin::kernels::ffi::init().unwrap();
    let bad = write_tmp("olorin_eadiff_bad.json", b"this is not JSON at all");
    let yesterday = fixture_with_numerics("eacrunch", 100.0, 70.0);
    let good = write_runeoutput("olorin_eadiff_good.json", &yesterday);

    let result = run_rune("eadiff", &format!("--json {bad} {good}")).unwrap();
    assert!(!result.success);
    let out = parse_answer(&result.answer);
    assert!(!out.success);
    let err = out.error.expect("error populated");
    assert!(err.contains("a:"), "should label which input failed: {err}");

    let _ = std::fs::remove_file(&bad);
    let _ = std::fs::remove_file(&good);
}

#[test]
fn eadiff_no_overlap_emits_empty_result() {
    olorin::kernels::ffi::init().unwrap();
    let mut a = RuneOutput::new("eacrunch", 1);
    a.fields.push(numeric_field("only_in_a", 1.0, 1.0, 1.0, 1.0, 1));
    let mut b = RuneOutput::new("eacrunch", 1);
    b.fields.push(numeric_field("only_in_b", 2.0, 2.0, 2.0, 2.0, 1));
    let pa = write_runeoutput("olorin_eadiff_noov_a.json", &a);
    let pb = write_runeoutput("olorin_eadiff_noov_b.json", &b);

    let result = run_rune("eadiff", &format!("{pa} {pb}")).unwrap();
    assert!(result.success);
    assert!(result.answer.contains("no diffable structure overlap"));

    let _ = std::fs::remove_file(&pa);
    let _ = std::fs::remove_file(&pb);
}

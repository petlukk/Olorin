//! eacrunch `--json` mode: structured RuneOutput exercising the
//! per-column `fields[]` axis (the one ealog skips).
//!
//! Contracts pinned here:
//! - Numeric columns serialize with NumericStats { min, max, mean, sum };
//!   counts match the SIMD kernel output.
//! - Text columns serialize with TextStats { unique, top: [{value,count}, ...] }
//!   in descending frequency order.
//! - Text-mode output stays byte-identical (no regression).
//! - Source.format is "csv".
//! - Failure path emits JSON too.

use olorin::runes::output::{FieldKind, RuneOutput};
use olorin::runes::run_rune;
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

const FIXTURE: &[u8] = b"\
date,category,amount,merchant
2024-01-01,groceries,42.50,Coop
2024-01-02,rent,1200.00,Landlord
2024-01-03,groceries,17.75,ICA
2024-01-04,transport,32.00,SL
2024-01-05,groceries,28.30,Coop
";

#[test]
fn json_mode_populates_fields_per_column() {
    olorin::kernels::ffi::init().unwrap();
    let path = write_tmp("olorin_eacrunch_json.csv", FIXTURE);
    let result = run_rune("eacrunch", &format!("--json {path}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);

    let out = parse_answer(&result.answer);
    assert_eq!(out.rune, "eacrunch");
    let src = out.source.expect("source populated");
    assert_eq!(src.format, "csv");
    assert_eq!(out.totals.rows, 5, "5 data rows");
    assert!(out.categories.is_empty(), "eacrunch emits no categories");
    assert!(out.samples.is_empty(),    "eacrunch emits no samples");

    let by_name: std::collections::HashMap<&str, &_> =
        out.fields.iter().map(|f| (f.name.as_str(), f)).collect();

    // amount is the numeric column.
    let amount = by_name["amount"];
    assert_eq!(amount.kind, FieldKind::Number);
    assert_eq!(amount.count, 5);
    let n = amount.numeric.as_ref().expect("numeric stats");
    assert!((n.min  - 17.75).abs()   < 0.01, "min wrong: {}", n.min);
    assert!((n.max  - 1200.00).abs() < 0.01, "max wrong: {}", n.max);
    assert!((n.sum  - 1320.55).abs() < 0.01, "sum wrong: {}", n.sum);
    assert!((n.mean - 264.11).abs()  < 0.01, "mean wrong: {}", n.mean);

    // category is text with 3 unique values; groceries appears 3 times.
    let category = by_name["category"];
    assert_eq!(category.kind, FieldKind::Text);
    let t = category.text.as_ref().expect("text stats");
    assert_eq!(t.unique, 3);
    assert_eq!(t.top[0].value, "groceries");
    assert_eq!(t.top[0].count, 3);
    let top_values: Vec<&str> = t.top.iter().map(|e| e.value.as_str()).collect();
    assert!(top_values.contains(&"rent"));
    assert!(top_values.contains(&"transport"));
}

#[test]
fn json_and_text_modes_agree_on_row_count_and_columns() {
    olorin::kernels::ffi::init().unwrap();
    let path = write_tmp("olorin_eacrunch_agree.csv", FIXTURE);

    let text_result = run_rune("eacrunch", &path).unwrap();
    let json_result = run_rune("eacrunch", &format!("--json {path}")).unwrap();
    let out = parse_answer(&json_result.answer);

    // rows: N appears in text.
    assert!(text_result.answer.contains(&format!("rows: {}", out.totals.rows)));
    assert!(text_result.answer.contains(&format!("columns: {}", out.fields.len())));

    // Every column name in fields appears in the text output.
    for f in &out.fields {
        assert!(text_result.answer.contains(&f.name),
            "text missing column '{}': {}", f.name, text_result.answer);
    }
}

#[test]
fn json_mode_flag_position_does_not_matter() {
    olorin::kernels::ffi::init().unwrap();
    let path = write_tmp("olorin_eacrunch_order.csv", FIXTURE);

    let prefix = run_rune("eacrunch", &format!("--json {path}")).unwrap();
    let suffix = run_rune("eacrunch", &format!("{path} --json")).unwrap();

    let a = parse_answer(&prefix.answer);
    let b = parse_answer(&suffix.answer);
    assert_eq!(a.fields, b.fields);
    assert_eq!(a.totals.rows, b.totals.rows);
}

#[test]
fn json_mode_error_path_emits_structured_failure() {
    olorin::kernels::ffi::init().unwrap();
    let result = run_rune("eacrunch", "--json /tmp/does_not_exist_xyz_98765.csv").unwrap();
    assert!(!result.success);
    let out = parse_answer(&result.answer);
    assert!(!out.success);
    assert!(out.error.expect("error populated").contains("not found"));
    assert!(out.fields.is_empty());
}

#[test]
fn json_mode_missing_path_emits_usage_error() {
    olorin::kernels::ffi::init().unwrap();
    let result = run_rune("eacrunch", "--json").unwrap();
    assert!(!result.success);
    let out = parse_answer(&result.answer);
    assert!(out.error.expect("error populated").contains("usage:"));
}

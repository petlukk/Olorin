//! eaparquet `--json` mode: structured RuneOutput exercising
//! `null_count` (the only rune that populates it) and the
//! "undecoded values" pattern (Text/Number with sub-stats = None).

use olorin::runes::output::{FieldKind, FieldStats, RuneOutput};
use olorin::runes::run_rune;

fn parse_answer(answer: &str) -> RuneOutput {
    RuneOutput::from_json(answer.as_bytes())
        .unwrap_or_else(|e| panic!("not parseable JSON: {e}\nanswer={answer}"))
}

fn find_field<'a>(out: &'a RuneOutput, name: &str) -> &'a FieldStats {
    out.fields.iter().find(|f| f.name == name)
        .unwrap_or_else(|| panic!("missing field '{name}' in {:?}",
            out.fields.iter().map(|f| f.name.as_str()).collect::<Vec<_>>()))
}

fn stage_fixture() -> std::path::PathBuf {
    let src = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/runes/tiny.parquet");
    let dst = std::env::temp_dir().join(format!(
        "olorin_eaparquet_json_{}.parquet", std::process::id()
    ));
    std::fs::copy(&src, &dst).expect("copy fixture to /tmp");
    dst
}

#[test]
fn json_mode_populates_fields_from_parquet_footer() {
    olorin::kernels::ffi::init().unwrap();
    let path = stage_fixture();
    let result = run_rune("eaparquet", &format!("--json {}", path.display())).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);

    let out = parse_answer(&result.answer);
    assert_eq!(out.rune, "eaparquet");
    let src = out.source.as_ref().expect("source populated");
    assert_eq!(src.format, "parquet");
    assert_eq!(out.totals.rows, 10);
    assert_eq!(out.fields.len(), 4);

    // id is INT64 → Number with min/max from precomputed footer stats.
    let id = find_field(&out, "id");
    assert_eq!(id.kind, FieldKind::Number);
    let n = id.numeric.as_ref().expect("INT64 has min/max");
    assert!((n.min - 1.0).abs()  < 0.01, "id min: {}", n.min);
    assert!((n.max - 10.0).abs() < 0.01, "id max: {}", n.max);

    // amount is DOUBLE → Number with float min/max.
    let amount = find_field(&out, "amount");
    assert_eq!(amount.kind, FieldKind::Number);
    let n = amount.numeric.as_ref().expect("DOUBLE has min/max");
    assert!((n.min - 12.00).abs()   < 0.01, "amount min: {}", n.min);
    assert!((n.max - 1800.00).abs() < 0.01, "amount max: {}", n.max);

    // is_recurring is BOOLEAN — kind=Bool, count populated; true/false
    // breakdown is unavailable from footer stats.
    let recurring = find_field(&out, "is_recurring");
    assert_eq!(recurring.kind, FieldKind::Bool);
    assert!(recurring.count > 0);

    // category is BYTE_ARRAY → Text with text=None signals undecoded.
    let category = find_field(&out, "category");
    assert_eq!(category.kind, FieldKind::Text);
    assert!(category.text.is_none(),
        "byte-array column should NOT populate TextStats (no decode): {:?}", category);
    assert!(category.count > 0);

    let _ = std::fs::remove_file(&path);
}

#[test]
fn json_and_text_modes_agree_on_visible_data() {
    olorin::kernels::ffi::init().unwrap();
    let path = stage_fixture();
    let text = run_rune("eaparquet", path.to_str().unwrap()).unwrap();
    let json = run_rune("eaparquet", &format!("--json {}", path.display())).unwrap();
    let out = parse_answer(&json.answer);

    assert!(text.answer.contains(&format!("rows: {}", out.totals.rows)));
    assert!(text.answer.contains(&format!("columns: {}", out.fields.len())));
    for f in &out.fields {
        assert!(text.answer.contains(&f.name),
            "field '{}' missing from text: {}", f.name, text.answer);
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn json_mode_error_path_emits_structured_failure() {
    olorin::kernels::ffi::init().unwrap();
    let result = run_rune("eaparquet", "--json /tmp/does_not_exist_xyz_abc.parquet").unwrap();
    assert!(!result.success);
    let out = parse_answer(&result.answer);
    assert!(!out.success);
    let err = out.error.expect("error populated");
    assert!(err.contains("not found"), "unexpected error: {err}");
    assert!(out.fields.is_empty());
}

#[test]
fn json_mode_flag_position_does_not_matter() {
    olorin::kernels::ffi::init().unwrap();
    let path = stage_fixture();
    let prefix = run_rune("eaparquet", &format!("--json {}", path.display())).unwrap();
    let suffix = run_rune("eaparquet", &format!("{} --json", path.display())).unwrap();
    let a = parse_answer(&prefix.answer);
    let b = parse_answer(&suffix.answer);
    assert_eq!(a.fields, b.fields);
    assert_eq!(a.totals.rows, b.totals.rows);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn json_mode_non_parquet_input_emits_structured_failure() {
    olorin::kernels::ffi::init().unwrap();
    let dst = std::env::temp_dir().join(format!(
        "olorin_eaparquet_bad_json_{}.parquet", std::process::id()
    ));
    std::fs::write(&dst, b"definitely not a parquet file").unwrap();
    let result = run_rune("eaparquet", &format!("--json {}", dst.display())).unwrap();
    assert!(!result.success);
    let out = parse_answer(&result.answer);
    assert!(!out.success);
    assert!(out.error.unwrap_or_default().contains("decode failed"));
    let _ = std::fs::remove_file(&dst);
}

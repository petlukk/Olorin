//! eajson `--json` mode: structured RuneOutput exercising ALL FIVE
//! `FieldKind` variants — Number, Text, Bool, Timestamp, Mixed.
//!
//! eajson is the only rune that touches every variant; if any of them
//! doesn't round-trip, the schema's discriminator design fails here.

use olorin::runes::output::{FieldKind, FieldStats, RuneOutput};
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

fn find_field<'a>(out: &'a RuneOutput, name: &str) -> &'a FieldStats {
    out.fields.iter().find(|f| f.name == name)
        .unwrap_or_else(|| panic!("missing field '{name}' in {:?}",
            out.fields.iter().map(|f| f.name.as_str()).collect::<Vec<_>>()))
}

// Fixture hits all five FieldKind variants:
// - status: Number
// - level:  Text (low-cardinality)
// - cached: Bool
// - ts:     Timestamp (ISO 8601)
// - id:     Mixed (number then text)
// - cursor: Text (every value unique — suppressed in text view, kept in JSON)
const FIXTURE: &[u8] = b"\
{\"status\":200,\"level\":\"info\",\"cached\":true,\"ts\":\"2026-05-11T10:00:00Z\",\"id\":1,\"cursor\":\"abc-001\"}
{\"status\":404,\"level\":\"warn\",\"cached\":false,\"ts\":\"2026-05-11T10:00:05Z\",\"id\":2,\"cursor\":\"abc-002\"}
{\"status\":500,\"level\":\"error\",\"cached\":true,\"ts\":\"2026-05-11T10:00:10Z\",\"id\":\"x\",\"cursor\":\"abc-003\"}
{\"status\":200,\"level\":\"info\",\"cached\":true,\"ts\":\"2026-05-11T10:00:15Z\",\"id\":4,\"cursor\":\"abc-004\"}
";

#[test]
fn json_mode_exercises_all_field_kinds() {
    olorin::kernels::ffi::init().unwrap();
    let path = write_tmp("olorin_eajson_all_kinds.jsonl", FIXTURE);
    let result = run_rune("eajson", &format!("--json {path}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);

    let out = parse_answer(&result.answer);
    assert_eq!(out.rune, "eajson");
    let src = out.source.as_ref().expect("source populated");
    assert_eq!(src.format, "jsonl");
    assert_eq!(out.totals.rows, 4);

    let status = find_field(&out, "status");
    assert_eq!(status.kind, FieldKind::Number);
    let n = status.numeric.as_ref().expect("number stats");
    assert!((n.min - 200.0).abs() < 0.01);
    assert!((n.max - 500.0).abs() < 0.01);
    assert_eq!(status.count, 4);

    let level = find_field(&out, "level");
    assert_eq!(level.kind, FieldKind::Text);
    let t = level.text.as_ref().expect("text stats");
    assert_eq!(t.unique, 3);
    assert_eq!(t.top[0].value, "info");
    assert_eq!(t.top[0].count, 2);

    let cached = find_field(&out, "cached");
    assert_eq!(cached.kind, FieldKind::Bool);
    let b = cached.bool.as_ref().expect("bool stats");
    assert_eq!(b.true_count, 3);
    assert_eq!(b.false_count, 1);

    let ts = find_field(&out, "ts");
    assert_eq!(ts.kind, FieldKind::Timestamp);
    let tsv = ts.timestamp.as_ref().expect("timestamp stats");
    assert_eq!(tsv.min, "2026-05-11T10:00:00Z");
    assert_eq!(tsv.max, "2026-05-11T10:00:15Z");
    assert_eq!(tsv.unique, 4);

    let id = find_field(&out, "id");
    assert_eq!(id.kind, FieldKind::Mixed);
    assert!(id.numeric.is_none() && id.text.is_none() && id.bool.is_none());

    // High-cardinality text key stays in the structured form even though
    // it's suppressed from the legacy text view.
    let cursor = find_field(&out, "cursor");
    assert_eq!(cursor.kind, FieldKind::Text);
    let ct = cursor.text.as_ref().expect("text stats");
    assert_eq!(ct.unique, 4);
    assert_eq!(cursor.count, 4);
}

#[test]
fn text_mode_still_suppresses_high_cardinality_keys() {
    olorin::kernels::ffi::init().unwrap();
    let path = write_tmp("olorin_eajson_suppress.jsonl", FIXTURE);
    let text = run_rune("eajson", &path).unwrap();
    assert!(text.success);
    let a = &text.answer;

    // cursor (text where every value is unique) is suppressed.
    assert!(!a.contains("cursor (text):"),
        "cursor should not appear in text view: {a}");
    assert!(a.contains("(+1 high-cardinality keys suppressed)"),
        "suppressed-count annotation missing: {a}");
    assert!(a.contains("level (text):"),
        "low-cardinality text should remain: {a}");
}

#[test]
fn json_and_text_modes_agree_on_visible_data() {
    olorin::kernels::ffi::init().unwrap();
    let path = write_tmp("olorin_eajson_agree.jsonl", FIXTURE);

    let text = run_rune("eajson", &path).unwrap();
    let json = run_rune("eajson", &format!("--json {path}")).unwrap();
    let out = parse_answer(&json.answer);

    assert!(text.answer.contains(&format!("rows: {}", out.totals.rows)));

    // Every non-suppressed text/number/bool/timestamp/mixed field appears
    // in the text view by name. Suppressed text keys (unique == count)
    // do NOT.
    for f in &out.fields {
        let suppressed = f.kind == FieldKind::Text
            && f.count > 0
            && f.text.as_ref().map_or(false, |t| t.unique == f.count);
        if suppressed {
            assert!(!text.answer.contains(&format!("{} (text):", f.name)),
                "suppressed field '{}' leaked into text: {}", f.name, text.answer);
        } else {
            assert!(text.answer.contains(&f.name),
                "field '{}' missing from text: {}", f.name, text.answer);
        }
    }
}

#[test]
fn json_mode_error_path_emits_structured_failure() {
    olorin::kernels::ffi::init().unwrap();
    let result = run_rune("eajson", "--json /tmp/does_not_exist_xyz_77777.jsonl").unwrap();
    assert!(!result.success);
    let out = parse_answer(&result.answer);
    assert!(!out.success);
    let err = out.error.expect("error populated");
    assert!(err.contains("not found"), "unexpected error: {err}");
    assert!(out.fields.is_empty());
}

#[test]
fn null_values_counted_as_presence_with_null_count() {
    // Robustness wave one, finding #3: a key that is JSON `null` in some
    // records was silently dropped — `count` undercounted the non-null
    // values and `null_count` was omitted entirely (and a column null in
    // EVERY record vanished). Now `count` = total presence (non-null +
    // null), `null_count` is always populated, and stats cover the non-null
    // values — matching eaparquet's contract. A key that is null in every
    // record stays omitted: eajson is value-typed, with no schema to
    // declare an all-null column.
    olorin::kernels::ffi::init().unwrap();
    const NULLS: &[u8] = b"\
{\"a\":1,\"b\":\"x\",\"c\":10,\"d\":null,\"e\":true}
{\"a\":null,\"b\":null,\"c\":null,\"d\":null,\"e\":false}
{\"a\":3,\"b\":\"y\",\"c\":30,\"d\":null,\"e\":true}
";
    let path = write_tmp("olorin_eajson_nulls.jsonl", NULLS);
    let out = parse_answer(&run_rune("eajson", &format!("--json {path}")).unwrap().answer);
    assert_eq!(out.totals.rows, 3);

    // a: number, 2 non-null + 1 null. count = presence, stats over non-null.
    let a = find_field(&out, "a");
    assert_eq!(a.kind, FieldKind::Number);
    assert_eq!(a.count, 3, "presence = 2 non-null + 1 null");
    assert_eq!(a.null_count, Some(1));
    let n = a.numeric.as_ref().expect("number stats");
    assert!((n.min - 1.0).abs() < 1e-9, "min over non-null only");
    assert!((n.max - 3.0).abs() < 1e-9, "max over non-null only");

    // b: text, 2 non-null + 1 null.
    let b = find_field(&out, "b");
    assert_eq!(b.kind, FieldKind::Text);
    assert_eq!(b.count, 3);
    assert_eq!(b.null_count, Some(1));

    // c: number, 2 non-null + 1 null.
    let c = find_field(&out, "c");
    assert_eq!(c.count, 3);
    assert_eq!(c.null_count, Some(1));

    // e: bool present in every record, never null → null_count Some(0),
    // proving null_count is always populated (not just when > 0).
    let e = find_field(&out, "e");
    assert_eq!(e.kind, FieldKind::Bool);
    assert_eq!(e.count, 3);
    assert_eq!(e.null_count, Some(0));

    // d: null in all 3 records → all-null → omitted entirely.
    assert!(out.fields.iter().all(|f| f.name != "d"),
        "all-null key must be omitted, fields: {:?}",
        out.fields.iter().map(|f| f.name.as_str()).collect::<Vec<_>>());

    // No field may report more nulls than its total presence.
    for f in &out.fields {
        assert!(f.null_count.unwrap_or(0) <= f.count,
            "null_count > count for '{}'", f.name);
    }
}

#[test]
fn json_mode_flag_position_does_not_matter() {
    olorin::kernels::ffi::init().unwrap();
    let path = write_tmp("olorin_eajson_order.jsonl", FIXTURE);

    let prefix = run_rune("eajson", &format!("--json {path}")).unwrap();
    let suffix = run_rune("eajson", &format!("{path} --json")).unwrap();
    let a = parse_answer(&prefix.answer);
    let b = parse_answer(&suffix.answer);
    assert_eq!(a.fields, b.fields);
    assert_eq!(a.totals.rows, b.totals.rows);
}

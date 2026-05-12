//! eadiff per-FieldKind diff modes: covers Text, Bool, Timestamp
//! handling including v0.9.3 additions (Timestamp range shift seconds,
//! Text top-N value comparison). Split out of `tests/runes_eadiff.rs`
//! so each test file stays under the 500-LOC cap.

mod common;
use common::eadiff_helpers::{numeric_field, parse_answer, write_runeoutput};

use olorin::runes::output::{
    BoolStats, FieldKind, FieldStats, RuneOutput, TextEntry, TextStats, TimestampStats,
};
use olorin::runes::run_rune;

#[test]
fn eadiff_text_field_emits_unique_delta() {
    olorin::kernels::ffi::init().unwrap();
    let mut a = RuneOutput::new("eajson", 1);
    let mut b = RuneOutput::new("eajson", 1);
    a.fields.push(numeric_field("status", 200.0, 200.0, 200.0, 200000.0, 1000));
    b.fields.push(numeric_field("status", 404.0, 404.0, 404.0, 404000.0, 1000));
    a.fields.push(FieldStats {
        name: "level".into(), kind: FieldKind::Text, count: 1000,
        null_count: None, numeric: None,
        text: Some(TextStats { unique: 3, top: vec![] }),
        bool: None, timestamp: None,
    });
    b.fields.push(FieldStats {
        name: "level".into(), kind: FieldKind::Text, count: 1000,
        null_count: None, numeric: None,
        text: Some(TextStats { unique: 4, top: vec![] }),
        bool: None, timestamp: None,
    });
    let pa = write_runeoutput("olorin_eadiff_textfld_a.json", &a);
    let pb = write_runeoutput("olorin_eadiff_textfld_b.json", &b);

    let result = run_rune("eadiff", &format!("--json {pa} {pb}")).unwrap();
    let out = parse_answer(&result.answer);
    let names: Vec<&str> = out.fields.iter().map(|f| f.name.as_str()).collect();
    assert!(names.contains(&"status"), "status numeric delta missing: {names:?}");
    assert!(names.contains(&"level.unique_delta"),
        "text unique_delta missing: {names:?}");
    let level = out.fields.iter().find(|f| f.name == "level.unique_delta").unwrap();
    assert_eq!(level.numeric.as_ref().unwrap().mean, 1.0, "unique delta should be +1");
    let _ = std::fs::remove_file(&pa);
    let _ = std::fs::remove_file(&pb);
}

#[test]
fn eadiff_bool_field_emits_paired_deltas() {
    olorin::kernels::ffi::init().unwrap();
    let mut a = RuneOutput::new("eajson", 1);
    let mut b = RuneOutput::new("eajson", 1);
    a.fields.push(FieldStats {
        name: "cached".into(), kind: FieldKind::Bool, count: 100,
        null_count: None, numeric: None, text: None,
        bool: Some(BoolStats { true_count: 90, false_count: 10 }),
        timestamp: None,
    });
    b.fields.push(FieldStats {
        name: "cached".into(), kind: FieldKind::Bool, count: 100,
        null_count: None, numeric: None, text: None,
        bool: Some(BoolStats { true_count: 85, false_count: 15 }),
        timestamp: None,
    });
    let pa = write_runeoutput("olorin_eadiff_bool_a.json", &a);
    let pb = write_runeoutput("olorin_eadiff_bool_b.json", &b);

    let result = run_rune("eadiff", &format!("--json {pa} {pb}")).unwrap();
    let out = parse_answer(&result.answer);
    let by_name: std::collections::HashMap<&str, f64> = out.fields.iter()
        .filter_map(|f| f.numeric.as_ref().map(|n| (f.name.as_str(), n.mean)))
        .collect();
    assert_eq!(by_name.get("cached.true_delta"),  Some(&-5.0),
        "true count dropped by 5: {by_name:?}");
    assert_eq!(by_name.get("cached.false_delta"), Some(&5.0),
        "false count grew by 5: {by_name:?}");
    let _ = std::fs::remove_file(&pa);
    let _ = std::fs::remove_file(&pb);
}

#[test]
fn eadiff_timestamp_field_emits_unique_delta() {
    olorin::kernels::ffi::init().unwrap();
    let mut a = RuneOutput::new("eajson", 1);
    let mut b = RuneOutput::new("eajson", 1);
    a.fields.push(FieldStats {
        name: "ts".into(), kind: FieldKind::Timestamp, count: 1000,
        null_count: None, numeric: None, text: None, bool: None,
        timestamp: Some(TimestampStats {
            min: "2026-05-10T00:00:00Z".into(),
            max: "2026-05-10T23:59:59Z".into(),
            unique: 800,
        }),
    });
    b.fields.push(FieldStats {
        name: "ts".into(), kind: FieldKind::Timestamp, count: 1000,
        null_count: None, numeric: None, text: None, bool: None,
        timestamp: Some(TimestampStats {
            min: "2026-05-11T00:00:00Z".into(),
            max: "2026-05-11T23:59:59Z".into(),
            unique: 950,
        }),
    });
    let pa = write_runeoutput("olorin_eadiff_ts_a.json", &a);
    let pb = write_runeoutput("olorin_eadiff_ts_b.json", &b);

    let result = run_rune("eadiff", &format!("--json {pa} {pb}")).unwrap();
    let out = parse_answer(&result.answer);
    let ts = out.fields.iter().find(|f| f.name == "ts.unique_delta")
        .expect("timestamp unique_delta emitted");
    assert_eq!(ts.numeric.as_ref().unwrap().mean, 150.0, "unique_delta = +150");
    let _ = std::fs::remove_file(&pa);
    let _ = std::fs::remove_file(&pb);
}

#[test]
fn eadiff_timestamp_field_emits_range_shift_seconds() {
    // v0.9.3: Timestamp diff emits min_shift_s / max_shift_s alongside
    // unique_delta. Forward by one day = +86400 seconds is the cleanest
    // verification.
    olorin::kernels::ffi::init().unwrap();
    let mut a = RuneOutput::new("eajson", 1);
    let mut b = RuneOutput::new("eajson", 1);
    a.fields.push(FieldStats {
        name: "ts".into(), kind: FieldKind::Timestamp, count: 100,
        null_count: None, numeric: None, text: None, bool: None,
        timestamp: Some(TimestampStats {
            min: "2026-05-10T00:00:00Z".into(),
            max: "2026-05-10T12:00:00Z".into(),
            unique: 100,
        }),
    });
    b.fields.push(FieldStats {
        name: "ts".into(), kind: FieldKind::Timestamp, count: 100,
        null_count: None, numeric: None, text: None, bool: None,
        timestamp: Some(TimestampStats {
            min: "2026-05-11T00:00:00Z".into(),
            max: "2026-05-11T12:00:00Z".into(),
            unique: 100,
        }),
    });
    let pa = write_runeoutput("olorin_eadiff_ts_shift_a.json", &a);
    let pb = write_runeoutput("olorin_eadiff_ts_shift_b.json", &b);

    let result = run_rune("eadiff", &format!("--json {pa} {pb}")).unwrap();
    let out = parse_answer(&result.answer);
    let by_name: std::collections::HashMap<&str, f64> = out.fields.iter()
        .filter_map(|f| f.numeric.as_ref().map(|n| (f.name.as_str(), n.mean)))
        .collect();
    assert_eq!(by_name.get("ts.min_shift_s"), Some(&86400.0),
        "min shifted forward 1 day: {by_name:?}");
    assert_eq!(by_name.get("ts.max_shift_s"), Some(&86400.0),
        "max shifted forward 1 day: {by_name:?}");
    assert!(!by_name.contains_key("ts.unique_delta"),
        "unique unchanged, no unique_delta expected");
    let _ = std::fs::remove_file(&pa);
    let _ = std::fs::remove_file(&pb);
}

#[test]
fn eadiff_timestamp_invalid_iso_skips_shift_gracefully() {
    olorin::kernels::ffi::init().unwrap();
    let mut a = RuneOutput::new("eajson", 1);
    let mut b = RuneOutput::new("eajson", 1);
    a.fields.push(FieldStats {
        name: "ts".into(), kind: FieldKind::Timestamp, count: 10,
        null_count: None, numeric: None, text: None, bool: None,
        timestamp: Some(TimestampStats {
            min: "garbage".into(), max: "not-iso".into(), unique: 5,
        }),
    });
    b.fields.push(FieldStats {
        name: "ts".into(), kind: FieldKind::Timestamp, count: 10,
        null_count: None, numeric: None, text: None, bool: None,
        timestamp: Some(TimestampStats {
            min: "more-junk".into(), max: "still-bad".into(), unique: 7,
        }),
    });
    let pa = write_runeoutput("olorin_eadiff_ts_bad_a.json", &a);
    let pb = write_runeoutput("olorin_eadiff_ts_bad_b.json", &b);

    let result = run_rune("eadiff", &format!("--json {pa} {pb}")).unwrap();
    let out = parse_answer(&result.answer);
    let names: Vec<&str> = out.fields.iter().map(|f| f.name.as_str()).collect();
    assert!(names.contains(&"ts.unique_delta"),
        "unique_delta survives garbage min/max: {names:?}");
    assert!(!names.iter().any(|n| n.contains("shift_s")),
        "no shift fields when ISO parse fails: {names:?}");
    let _ = std::fs::remove_file(&pa);
    let _ = std::fs::remove_file(&pb);
}

#[test]
fn eadiff_text_top_n_emits_count_delta_and_markers() {
    olorin::kernels::ffi::init().unwrap();
    let mut a = RuneOutput::new("eajson", 1);
    let mut b = RuneOutput::new("eajson", 1);
    // "shared" in both top-N with count change. "dropped" in a only.
    // "new" in b only.
    a.fields.push(FieldStats {
        name: "level".into(), kind: FieldKind::Text, count: 100,
        null_count: None, numeric: None, bool: None, timestamp: None,
        text: Some(TextStats {
            unique: 10,
            top: vec![
                TextEntry { value: "shared".into(),  count: 50 },
                TextEntry { value: "dropped".into(), count: 30 },
            ],
        }),
    });
    b.fields.push(FieldStats {
        name: "level".into(), kind: FieldKind::Text, count: 100,
        null_count: None, numeric: None, bool: None, timestamp: None,
        text: Some(TextStats {
            unique: 10,
            top: vec![
                TextEntry { value: "shared".into(), count: 65 },
                TextEntry { value: "new".into(),    count: 20 },
            ],
        }),
    });
    let pa = write_runeoutput("olorin_eadiff_text_top_a.json", &a);
    let pb = write_runeoutput("olorin_eadiff_text_top_b.json", &b);

    let result = run_rune("eadiff", &format!("--json {pa} {pb}")).unwrap();
    let out = parse_answer(&result.answer);
    let names: Vec<&str> = out.fields.iter().map(|f| f.name.as_str()).collect();

    let shared = out.fields.iter().find(|f| f.name == "level:shared.count_delta")
        .expect("shared count_delta missing");
    assert_eq!(shared.numeric.as_ref().unwrap().mean, 15.0);

    assert!(names.contains(&"[appeared in top] level:new"),
        "appeared-in-top marker missing: {names:?}");
    assert!(names.contains(&"[disappeared from top] level:dropped"),
        "disappeared-from-top marker missing: {names:?}");

    let _ = std::fs::remove_file(&pa);
    let _ = std::fs::remove_file(&pb);
}

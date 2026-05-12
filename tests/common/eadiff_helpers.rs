//! Shared helpers for eadiff integration test files.
//!
//! Used by tests/runes_eadiff.rs (core + v1 behaviors) and
//! tests/runes_eadiff_kinds.rs (per-FieldKind diff modes — v2 + v3).
//! Split out so each test file stays under the 500-LOC cap.

use olorin::runes::output::{
    Category, FieldKind, FieldStats, NumericStats, RuneOutput, Totals,
};
use std::io::Write;

pub fn write_tmp(name: &str, bytes: &[u8]) -> String {
    let path = format!("/tmp/{name}");
    let mut f = std::fs::File::create(&path).expect("tmp create");
    f.write_all(bytes).expect("tmp write");
    path
}

pub fn parse_answer(answer: &str) -> RuneOutput {
    RuneOutput::from_json(answer.as_bytes())
        .unwrap_or_else(|e| panic!("not parseable JSON: {e}\nanswer={answer}"))
}

pub fn write_runeoutput(name: &str, out: &RuneOutput) -> String {
    let path = format!("/tmp/{name}");
    let mut f = std::fs::File::create(&path).expect("tmp create");
    f.write_all(out.to_json().as_bytes()).expect("tmp write");
    path
}

pub fn numeric_field(
    name: &str, min: f64, max: f64, mean: f64, sum: f64, count: u64,
) -> FieldStats {
    FieldStats {
        name: name.to_string(),
        kind: FieldKind::Number,
        count,
        null_count: None,
        numeric: Some(NumericStats { min, max, mean, sum }),
        text: None, bool: None, timestamp: None,
    }
}

pub fn fixture_with_numerics(rune: &str, bearing: f64, motor: f64) -> RuneOutput {
    let mut r = RuneOutput::new(rune, 1);
    r.totals = Totals { rows: 1000, scan_us: 0 };
    r.fields.push(numeric_field(
        "bearing_stock", bearing, bearing, bearing, bearing * 1000.0, 1000,
    ));
    r.fields.push(numeric_field(
        "motor_temp", motor, motor, motor, motor * 1000.0, 1000,
    ));
    r
}

pub fn fixture_with_hour_categories(
    rune: &str, hour_counts: &[(usize, u64)],
) -> RuneOutput {
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

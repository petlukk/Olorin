//! Round-trip + composition tests for `runes::output::RuneOutput` v1.
//!
//! Three checks:
//! 1. Encode → decode is lossless for a fully-populated record.
//! 2. NaN/Inf in `NumericStats` survive as `null` in JSON and decode to 0.0.
//! 3. A minimal `eadiff` prototype operates generically over two RuneOutputs,
//!    proving the schema actually enables composition (the entire point).

use olorin::runes::correlation::Correlation;
use olorin::runes::grouping::{AggResult, Group};
use olorin::runes::output::{
    Anomaly, BoolStats, Category, FieldKind, FieldStats, NumericStats, RuneOutput, Sample, Source,
    TextEntry, TextStats, TimestampStats, Totals, SCHEMA_VERSION,
};

fn rich_fixture() -> RuneOutput {
    RuneOutput {
        schema_version: SCHEMA_VERSION,
        rune:           "eacrunch".to_string(),
        rune_version:   1,
        success:        true,
        source: Some(Source {
            path:   "/tmp/stocks.csv".to_string(),
            bytes:  4_096,
            format: "csv".to_string(),
        }),
        totals: Totals { rows: 1_000, scan_us: 5_200 },
        fields: vec![
            FieldStats {
                name:       "bearing_stock".to_string(),
                kind:       FieldKind::Number,
                count:      1_000,
                null_count: Some(2),
                numeric:    Some(NumericStats {
                    min: 0.0, max: 120.0, mean: 47.5, sum: 47_500.0,
                }),
                text:       None,
                bool:       None,
                timestamp:  None,
            },
            FieldStats {
                name:       "supplier".to_string(),
                kind:       FieldKind::Text,
                count:      1_000,
                null_count: None,
                numeric:    None,
                text:       Some(TextStats {
                    unique: 3,
                    top: vec![
                        TextEntry { value: "SKF".to_string(),    count: 600 },
                        TextEntry { value: "FAG".to_string(),    count: 300 },
                        TextEntry { value: "Timken".to_string(), count: 100 },
                    ],
                }),
                bool:       None,
                timestamp:  None,
            },
            FieldStats {
                name:       "active".to_string(),
                kind:       FieldKind::Bool,
                count:      1_000,
                null_count: None,
                numeric:    None,
                text:       None,
                bool:       Some(BoolStats { true_count: 998, false_count: 2 }),
                timestamp:  None,
            },
            FieldStats {
                name:       "delivered_at".to_string(),
                kind:       FieldKind::Timestamp,
                count:      1_000,
                null_count: None,
                numeric:    None,
                text:       None,
                bool:       None,
                timestamp:  Some(TimestampStats {
                    min:    "2026-05-01T08:00:00".to_string(),
                    max:    "2026-05-11T17:30:00".to_string(),
                    unique: 873,
                }),
            },
        ],
        categories: vec![
            Category { name: "DEBUG".to_string(), count: 0 },
            Category { name: "INFO".to_string(),  count: 980 },
            Category { name: "WARN".to_string(),  count: 15 },
            Category { name: "ERROR".to_string(), count: 5 },
            Category { name: "FATAL".to_string(), count: 0 },
        ],
        samples: vec![
            Sample {
                byte_offset: Some(2_048),
                line:        Some(42),
                timestamp:   Some("2026-05-11T10:54:00".to_string()),
                text:        "ERROR: bearing temp exceeds 85C".to_string(),
            },
        ],
        anomalies: vec![
            Anomaly {
                bucket:   "2026-05-11T02:11:00".to_string(),
                count:    4_200,
                baseline: 50.0,
                ratio:    84.0,
                score:    37.5,
            },
        ],
        correlations: vec![
            Correlation {
                stream_a:      "syslog (errors)".to_string(),
                stream_b:      "deploys.csv".to_string(),
                lag_seconds:   240,
                // 4dp-exact so the lossless round-trip assertion holds
                // (the wire rounds score to 4 decimals).
                score:         0.9375,
                peak_bucket:   "2026-05-11T03:02:00".to_string(),
                events_a:      60,
                events_b:      3,
                width_seconds: 60,
            },
        ],
        groups: vec![
            Group {
                key:   "SKF".to_string(),
                count: 600,
                aggs:  vec![
                    AggResult { op: "mean".to_string(), col: "price".to_string(), value: 47.5 },
                    AggResult { op: "count".to_string(), col: String::new(), value: 600.0 },
                ],
            },
            Group {
                key:   "FAG".to_string(),
                count: 300,
                aggs:  vec![
                    AggResult { op: "mean".to_string(), col: "price".to_string(), value: 31.0 },
                    AggResult { op: "count".to_string(), col: String::new(), value: 300.0 },
                ],
            },
        ],
        group_by: Some("supplier".to_string()),
        incident: None,
        error: None,
    }
}

#[test]
fn round_trip_lossless() {
    let original = rich_fixture();
    let json = original.to_json();

    assert!(!json.contains('\n'), "JSONL output must be single-line");
    assert!(json.starts_with('{') && json.ends_with('}'));

    let decoded = RuneOutput::from_json(json.as_bytes()).expect("decode");
    assert_eq!(original, decoded);
}

#[test]
fn rejects_unknown_schema_version() {
    let mut fx = rich_fixture();
    fx.schema_version = 999;
    let json = fx.to_json();
    let err = RuneOutput::from_json(json.as_bytes()).unwrap_err();
    assert!(err.contains("schema_version"), "got: {err}");
}

#[test]
fn nonfinite_numeric_stats_survive_as_null() {
    // f32_stats returns NaN min/max on empty inputs. The schema emits
    // those as JSON null, and on the way back they decode to 0.0.
    let mut fx = RuneOutput::new("eacrunch", 1);
    fx.totals = Totals { rows: 0, scan_us: 0 };
    fx.fields.push(FieldStats {
        name:       "empty_col".to_string(),
        kind:       FieldKind::Number,
        count:      0,
        null_count: None,
        numeric:    Some(NumericStats {
            min:  f64::NAN,
            max:  f64::INFINITY,
            mean: f64::NEG_INFINITY,
            sum:  0.0,
        }),
        text: None, bool: None, timestamp: None,
    });

    let json = fx.to_json();
    assert!(json.contains("\"min\":null"));
    assert!(json.contains("\"max\":null"));
    assert!(json.contains("\"mean\":null"));
    assert!(json.contains("\"sum\":0.0"));

    let decoded = RuneOutput::from_json(json.as_bytes()).expect("decode");
    let n = decoded.fields[0].numeric.as_ref().unwrap();
    assert_eq!(n.min, 0.0);
    assert_eq!(n.max, 0.0);
    assert_eq!(n.mean, 0.0);
    assert_eq!(n.sum, 0.0);
}

#[test]
fn empty_arrays_omit_or_decode_to_empty() {
    let fx = RuneOutput::new("eatime", 1);
    let json = fx.to_json();
    let decoded = RuneOutput::from_json(json.as_bytes()).expect("decode");
    assert!(decoded.fields.is_empty());
    assert!(decoded.categories.is_empty());
    assert!(decoded.samples.is_empty());
    assert!(decoded.source.is_none());
    assert!(decoded.error.is_none());
}

#[test]
fn eadiff_prototype_over_two_outputs() {
    // The whole point of the schema: a 2-input rune like `eadiff` can
    // operate generically over any two prior RuneOutputs. This test is
    // the contract proof — the diff function does not branch on which
    // rune produced the inputs.
    let yesterday = {
        let mut r = RuneOutput::new("eacrunch", 1);
        r.totals = Totals { rows: 1_000, scan_us: 0 };
        r.fields.push(numeric_field("bearing_stock", 100.0, 100.0, 100.0, 100_000.0, 1_000));
        r.fields.push(numeric_field("motor_temp",     70.0,  72.0,  71.0,  71_000.0, 1_000));
        r.categories.push(Category { name: "ERROR".to_string(), count: 3 });
        r
    };
    let today = {
        let mut r = RuneOutput::new("eacrunch", 1);
        r.totals = Totals { rows: 1_000, scan_us: 0 };
        r.fields.push(numeric_field("bearing_stock",  85.0,  85.0,  85.0,  85_000.0, 1_000));
        r.fields.push(numeric_field("motor_temp",     72.0,  74.0,  73.0,  73_000.0, 1_000));
        r.categories.push(Category { name: "ERROR".to_string(), count: 12 });
        r
    };

    let delta = eadiff_prototype(&yesterday, &today);

    assert_eq!(delta.rune, "eadiff");
    assert_eq!(delta.schema_version, SCHEMA_VERSION);

    let bearing = delta.fields.iter().find(|f| f.name == "bearing_stock").unwrap();
    let n = bearing.numeric.as_ref().unwrap();
    assert_eq!(n.mean, -15.0, "stock dropped 15");
    assert_eq!(n.sum,  -15_000.0);

    let motor = delta.fields.iter().find(|f| f.name == "motor_temp").unwrap();
    let n = motor.numeric.as_ref().unwrap();
    assert_eq!(n.mean, 2.0, "temp rose 2");

    let err = delta.categories.iter().find(|c| c.name == "ERROR").unwrap();
    assert_eq!(err.count, 9, "ERROR count delta as u64-abs; sign in narration");
}

fn numeric_field(name: &str, min: f64, max: f64, mean: f64, sum: f64, count: u64) -> FieldStats {
    FieldStats {
        name:       name.to_string(),
        kind:       FieldKind::Number,
        count,
        null_count: None,
        numeric:    Some(NumericStats { min, max, mean, sum }),
        text: None, bool: None, timestamp: None,
    }
}

/// Generic numeric-and-category delta between two RuneOutputs. Not the
/// real `eadiff` rune — this is the schema's composition proof.
/// Match-by-name; non-numeric fields and unmatched names are skipped.
fn eadiff_prototype(a: &RuneOutput, b: &RuneOutput) -> RuneOutput {
    let mut out = RuneOutput::new("eadiff", 1);
    out.totals = Totals { rows: 0, scan_us: 0 };

    for fb in &b.fields {
        let Some(fa) = a.fields.iter().find(|f| f.name == fb.name) else { continue };
        let (Some(na), Some(nb)) = (fa.numeric.as_ref(), fb.numeric.as_ref()) else { continue };
        out.fields.push(FieldStats {
            name:       fb.name.clone(),
            kind:       FieldKind::Number,
            count:      fb.count,
            null_count: None,
            numeric:    Some(NumericStats {
                min:  nb.min  - na.min,
                max:  nb.max  - na.max,
                mean: nb.mean - na.mean,
                sum:  nb.sum  - na.sum,
            }),
            text: None, bool: None, timestamp: None,
        });
    }

    for cb in &b.categories {
        let Some(ca) = a.categories.iter().find(|c| c.name == cb.name) else { continue };
        let delta = (cb.count as i64 - ca.count as i64).unsigned_abs();
        out.categories.push(Category { name: cb.name.clone(), count: delta });
    }

    out
}

//! Regression: eacrunch must preserve precision for CSV numeric columns
//! holding integers larger than f32 can represent losslessly (>~16.7M).
//! The sibling bug to eajson — same `f32_stats` call, same data path.
//!
//! Same oracle pattern: 25 push-id-shaped integers; f64 arithmetic is
//! exact for u64 < 2^53, so a direct comparison works.

use olorin::runes::output::{FieldKind, RuneOutput};
use olorin::runes::run_rune;
use std::io::Write;

fn write_tmp(name: &str, bytes: &[u8]) -> String {
    let path = format!("/tmp/{name}");
    let mut f = std::fs::File::create(&path).expect("tmp create");
    f.write_all(bytes).expect("tmp write");
    path
}

const PUSH_IDS: &[u64] = &[
    34_404_460_741, 34_404_461_007, 34_404_461_017, 34_404_461_094,
    34_404_461_122, 34_404_461_201, 34_404_461_208, 34_404_461_276,
    34_404_461_279, 34_404_461_308, 34_404_461_400, 34_404_461_430,
    34_404_461_457, 34_404_461_465, 34_404_461_517, 34_404_461_549,
    34_404_461_594, 34_404_461_604, 34_404_461_605, 34_404_461_609,
    34_404_461_621, 34_404_461_633, 34_404_461_690, 34_404_461_766,
    34_404_461_885,
];

#[test]
fn csv_large_integers_round_trip_through_f64() {
    let mut buf = String::from("push_id\n");
    for id in PUSH_IDS {
        buf.push_str(&format!("{id}\n"));
    }
    let path = write_tmp("eacrunch_large_ints.csv", buf.as_bytes());

    let result = run_rune("eacrunch", &format!("--json {path}")).unwrap();
    assert!(result.success, "rune failed: {}", result.answer);
    let out = RuneOutput::from_json(result.answer.as_bytes())
        .unwrap_or_else(|e| panic!("not parseable JSON: {e}\nanswer={}", result.answer));

    let push = out.fields.iter().find(|f| f.name == "push_id")
        .expect("push_id column missing");
    assert_eq!(push.kind, FieldKind::Number, "push_id should be Number");

    let n = push.numeric.as_ref().expect("numeric stats present");

    let real_min: f64 = *PUSH_IDS.iter().min().unwrap() as f64;
    let real_max: f64 = *PUSH_IDS.iter().max().unwrap() as f64;
    let real_sum: f64 = PUSH_IDS.iter().map(|x| *x as f64).sum();
    let real_mean: f64 = real_sum / PUSH_IDS.len() as f64;

    assert_eq!(push.count, PUSH_IDS.len() as u64, "count");
    assert_eq!(n.min,  real_min,  "min: got {} want {real_min}",  n.min);
    assert_eq!(n.max,  real_max,  "max: got {} want {real_max}",  n.max);
    assert_eq!(n.sum,  real_sum,  "sum: got {} want {real_sum}",  n.sum);
    assert_eq!(n.mean, real_mean, "mean: got {} want {real_mean}", n.mean);

    assert_ne!(n.min, n.max, "min == max means precision collapsed");
}

#[test]
fn csv_small_floats_still_work() {
    let buf = "value\n1.5\n2.5\n3.5\n";
    let path = write_tmp("eacrunch_small_floats.csv", buf.as_bytes());
    let result = run_rune("eacrunch", &format!("--json {path}")).unwrap();
    let out = RuneOutput::from_json(result.answer.as_bytes()).expect("parse");
    let f = out.fields.iter().find(|f| f.name == "value").expect("value");
    let n = f.numeric.as_ref().unwrap();
    assert_eq!(n.min,  1.5);
    assert_eq!(n.max,  3.5);
    assert_eq!(n.sum,  7.5);
    assert_eq!(n.mean, 2.5);
}

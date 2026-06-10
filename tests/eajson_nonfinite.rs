//! Regression (runes robustness wave): a JSON number that overflows f64 to
//! ±inf (e.g. `1e400`) must not poison a numeric field's additive stats.
//!
//! Same bug class as the eacrunch nan/inf fix: the non-finite value propagated
//! through `sum`/`mean` (serialized as null by `finite_f64`) while `min`/`max`
//! survived, leaving an internally inconsistent, silently-wrong field summary.
//! eajson now drops non-finite parses, so the field summarizes its finite
//! values consistently. Found by differential testing.

use olorin::runes::output::RuneOutput;
use olorin::runes::run_rune;

fn ensure_kernels() {
    olorin::kernels::ffi::init().expect("kernel init");
}

fn field(jsonl: &str, tag: &str, name: &str) -> (u64, Option<(f64, f64, f64, f64)>) {
    ensure_kernels();
    let path = std::env::temp_dir().join(format!("olorin_ejnf_{tag}_{}.jsonl", std::process::id()));
    std::fs::write(&path, jsonl).unwrap();
    let res = run_rune("eajson", &format!("--json {}", path.display())).expect("eajson runs");
    let _ = std::fs::remove_file(&path);
    let out = RuneOutput::from_json(res.answer.as_bytes()).expect("parse RuneOutput");
    assert!(out.success, "eajson should succeed: {:?}", out.error);
    let f = out.fields.iter().find(|f| f.name == name).expect("field present");
    let stats = f.numeric.as_ref().map(|n| (n.min, n.max, n.mean, n.sum));
    (f.count, stats)
}

#[test]
fn overflow_inf_value_is_excluded() {
    let (count, stats) = field("{\"x\":1}\n{\"x\":2}\n{\"x\":1e400}\n", "inf", "x");
    let (min, max, mean, sum) = stats.expect("numeric stats present");
    assert_eq!(count, 2, "the inf value must be excluded from the count");
    assert_eq!(sum, 3.0, "sum must be over the finite values, not poisoned to null");
    assert_eq!(mean, 1.5, "mean must be over the finite values");
    assert_eq!(min, 1.0);
    assert_eq!(max, 2.0);
}

#[test]
fn negative_overflow_excluded_too() {
    let (count, stats) = field("{\"x\":10}\n{\"x\":-1e400}\n{\"x\":20}\n", "neginf", "x");
    let (_, _, _, sum) = stats.expect("numeric stats present");
    assert_eq!(count, 2);
    assert_eq!(sum, 30.0);
}

/// A field whose only values overflow degrades cleanly: still numeric, zero
/// finite values, no panic.
#[test]
fn all_overflow_field_degrades_cleanly() {
    let (count, _) = field("{\"x\":1e400}\n{\"x\":1e400}\n", "allinf", "x");
    assert_eq!(count, 0, "no finite values to count");
}

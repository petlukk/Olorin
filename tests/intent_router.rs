//! Intent router classifier tests.
//!
//! Covers CALC detection: real math expressions match, dates / hyphenated
//! phrases / date ranges do not. The state machine must not be fooled by a
//! bare `-` that is actually a hyphen.

use olorin::core::dispatch::{classify_intent, INTENT_CALC, INTENT_NONE, INTENT_TIME, INTENT_CPU, INTENT_WEATHER};
use olorin::kernels::ffi;

fn intent_of(text: &str) -> i32 {
    ffi::init().unwrap();
    let (intent, _start, _len) = classify_intent(text.as_bytes());
    intent
}

// ── CALC: true positives (real math) ─────────────────────────────────────────

#[test]
fn calc_plus_tight() { assert_eq!(intent_of("2+3"), INTENT_CALC); }

#[test]
fn calc_plus_spaced() { assert_eq!(intent_of("2 + 3"), INTENT_CALC); }

#[test]
fn calc_multiply() { assert_eq!(intent_of("6*7"), INTENT_CALC); }

#[test]
fn calc_divide() { assert_eq!(intent_of("10/2"), INTENT_CALC); }

#[test]
fn calc_power() { assert_eq!(intent_of("2^8"), INTENT_CALC); }

#[test]
fn calc_mod() { assert_eq!(intent_of("10%3"), INTENT_CALC); }

#[test]
fn calc_embedded_in_question() {
    assert_eq!(intent_of("what is 2+3"), INTENT_CALC);
}

#[test]
fn calc_minus_with_spaces() {
    // `-` is allowed as subtract when it has whitespace on at least one side.
    assert_eq!(intent_of("5 - 3"), INTENT_CALC);
    assert_eq!(intent_of("5 -3"), INTENT_CALC);
    assert_eq!(intent_of("5- 3"), INTENT_CALC);
}

// ── CALC: true negatives (hyphen/dash/date patterns) ────────────────────────

#[test]
fn not_calc_iso_date() {
    // ISO date 2024-01-15 must not hijack to calc.
    assert_eq!(intent_of("2024-01-15"), INTENT_NONE);
}

#[test]
fn not_calc_date_range() {
    // "Nov 3-7" has digit-hyphen-digit but is a date range.
    assert_eq!(intent_of("Nov 3-7"), INTENT_NONE);
}

#[test]
fn not_calc_hyphen_tight() {
    // A tight `2-3` is ambiguous with date/range; we conservatively treat it
    // as not-calc. Users can type `2 - 3` or use /calc explicitly.
    assert_eq!(intent_of("2-3"), INTENT_NONE);
}

#[test]
fn not_calc_top_ranking() {
    assert_eq!(intent_of("top-5 items"), INTENT_NONE);
}

#[test]
fn not_calc_four_fold() {
    assert_eq!(intent_of("4-fold improvement"), INTENT_NONE);
}

#[test]
fn not_calc_phone_number() {
    assert_eq!(intent_of("555-123-4567"), INTENT_NONE);
}

#[test]
fn not_calc_natural_prose_with_numbers() {
    // A stat-heavy summary like the ones Runes output should not hijack.
    assert_eq!(
        intent_of("Found 1 phone number, 1 email, 1 credit card"),
        INTENT_NONE
    );
}

// ── Other intents still work ─────────────────────────────────────────────────

#[test]
fn time_intent() { assert_eq!(intent_of("what time is it"), INTENT_TIME); }

#[test]
fn cpu_intent() { assert_eq!(intent_of("cpu usage please"), INTENT_CPU); }

#[test]
fn weather_intent() { assert_eq!(intent_of("weather in Malmö"), INTENT_WEATHER); }

#[test]
fn empty_is_none() { assert_eq!(intent_of(""), INTENT_NONE); }

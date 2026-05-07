//! Tests for the eaparquet rune — Parquet metadata summarizer.

use olorin::runes::{run_rune, RUNES, OutputSafety};

fn ensure_kernels() {
    olorin::kernels::ffi::init().expect("kernel init");
}

#[test]
fn eaparquet_is_registered() {
    let found = RUNES.iter().any(|r| r.name() == "eaparquet");
    assert!(found, "eaparquet rune missing from registry");
}

#[test]
fn eaparquet_output_safety_is_untrusted() {
    let r = RUNES.iter().find(|r| r.name() == "eaparquet")
        .expect("eaparquet registered");
    assert_eq!(r.output_safety(), OutputSafety::UntrustedQuoted,
        "Parquet column names + stats are file-derived; must be wrapped before reaching LLM");
}

#[test]
fn eaparquet_summarizes_fixture() {
    ensure_kernels();
    // Stage the repo fixture in /tmp (allowlist requires ~ or /tmp).
    let src = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/runes/tiny.parquet");
    let dst = std::env::temp_dir().join(format!(
        "olorin_eaparquet_{}.parquet", std::process::id()
    ));
    std::fs::copy(&src, &dst).expect("copy fixture to /tmp");

    let result = run_rune("eaparquet", dst.to_string_lossy().as_ref())
        .expect("eaparquet should be registered and runnable");
    assert!(result.success, "rune failed: {}", result.answer);
    let a = &result.answer;

    // 10 rows in the fixture (deterministic from gen_synthetic.py shape).
    assert!(a.contains("rows: 10"), "wrong row count: {a}");
    // 4 columns (id, category, amount, is_recurring).
    assert!(a.contains("columns: 4"), "wrong column count: {a}");
    // id is INT64 → numeric with min/max from precomputed stats.
    assert!(a.contains("id (number):"), "id key missing or wrong type: {a}");
    assert!(a.contains("min=1"), "id min missing: {a}");
    assert!(a.contains("max=10"), "id max missing: {a}");
    // amount is DOUBLE → numeric with float min/max.
    assert!(a.contains("amount (number):"), "amount key missing: {a}");
    assert!(a.contains("min=12.00"), "amount min missing: {a}");
    assert!(a.contains("max=1800.00"), "amount max missing: {a}");
    // is_recurring is BOOLEAN.
    assert!(a.contains("is_recurring (bool):"), "bool column missing: {a}");
    // category is BYTE_ARRAY → text with documented note.
    assert!(a.contains("category (text):"), "text column missing: {a}");
    assert!(a.contains("byte-array column"),
        "byte-array note missing: {a}");

    let _ = std::fs::remove_file(&dst);
}

#[test]
fn eaparquet_rejects_non_parquet() {
    ensure_kernels();
    let dst = std::env::temp_dir().join(format!(
        "olorin_eaparquet_bad_{}.parquet", std::process::id()
    ));
    std::fs::write(&dst, b"not a parquet file at all").unwrap();
    let result = run_rune("eaparquet", dst.to_string_lossy().as_ref())
        .expect("eaparquet runnable");
    assert!(!result.success, "should fail on non-parquet input");
    assert!(result.answer.contains("PAR1"),
        "error should mention missing magic: {}", result.answer);
    let _ = std::fs::remove_file(&dst);
}

#[test]
fn eaparquet_rejects_outside_allowlist() {
    let result = run_rune("eaparquet", "/etc/passwd")
        .expect("eaparquet runnable");
    assert!(!result.success);
    assert!(result.answer.contains("outside allowlist"),
        "unexpected error: {}", result.answer);
}

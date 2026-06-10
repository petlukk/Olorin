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

#[test]
fn eaparquet_decodes_unsigned_columns() {
    // UINT32/UINT64 columns are physically INT32/INT64; without reading the
    // ConvertedType they decode as signed and wrap to negative above the
    // signed max. Verify unsigned values come back positive.
    use olorin::storage::parquet::read_summary;
    let bytes = std::fs::read(
        std::env::current_dir().unwrap().join("tests/fixtures/runes/uint.parquet")
    ).expect("uint fixture exists");
    let s = read_summary(&bytes).expect("parse uint parquet");
    let col = |name: &str| s.columns.iter().find(|c| c.name == name)
        .unwrap_or_else(|| panic!("column {name} missing"));

    // u32: values above 2^31 decode exactly (u32 fits in i64/f64).
    let u32 = col("u32");
    assert_eq!(u32.min.unwrap().as_f64(), 3_000_000_000.0, "u32 min wrong/negative");
    assert_eq!(u32.max.unwrap().as_f64(), 4_000_000_000.0, "u32 max wrong/negative");

    // u64: values above 2^63 must be POSITIVE (previously wrapped negative)
    // AND carry their true magnitude. The fixture's u64 max is ~1.8e19. A weak
    // `> 9.0e18` check passed even when the stat-reduction round-tripped the
    // f64 back through `as i64`, whose saturating cast pinned anything above
    // i64::MAX to 2^63 (~9.22e18) — so assert the true magnitude. f64 tolerance
    // because >2^53 isn't exactly representable.
    let u64c = col("u64");
    assert!(u64c.min.unwrap().as_f64() > 0.0, "u64 min wrapped negative");
    let u64_max = u64c.max.unwrap().as_f64();
    assert!(
        (u64_max - 1.8e19).abs() < 1.0e16,
        "u64 max wrong: {u64_max} (expected ~1.8e19, not saturated to i64::MAX ~9.22e18)"
    );

    // Regression: a genuinely signed int64 column stays signed.
    let sig = col("sig");
    assert_eq!(sig.min.unwrap().as_f64(), -5.0);
    assert_eq!(sig.max.unwrap().as_f64(), 50.0);
}

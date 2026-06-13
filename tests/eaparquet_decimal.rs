//! eaparquet DECIMAL — the footer stores the *unscaled* integer (INT32/
//! INT64 little-endian, or FIXED_LEN_BYTE_ARRAY big-endian two's
//! complement); the reader decodes it to the real value `unscaled / 10^scale`
//! and renders the column as a Number.
//!
//! Fixture `decimal.parquet` (pyarrow, decimal128(10,2), FIXED_LEN_BYTE_ARRAY,
//! statistics present): price = [19.99, 1234.50, -7.25, 0.01], qty int32.
//! Oracle (pyarrow): min -7.25, max 1234.50.

use olorin::runes::output::{FieldKind, RuneOutput};
use olorin::runes::run_rune;

#[test]
fn decimal_column_decodes_to_scaled_number() {
    olorin::kernels::ffi::init().unwrap();
    let src = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/runes/decimal.parquet");
    let dst = std::env::temp_dir()
        .join(format!("olorin_decimal_{}.parquet", std::process::id()));
    std::fs::copy(&src, &dst).expect("copy fixture");

    let r = run_rune("eaparquet", &format!("--json {}", dst.display())).unwrap();
    let _ = std::fs::remove_file(&dst);
    assert!(r.success, "rune failed: {}", r.answer);
    let out = RuneOutput::from_json(r.answer.as_bytes()).expect("valid JSON");

    let price = out.fields.iter().find(|f| f.name == "price")
        .expect("price column present");
    assert_eq!(price.kind, FieldKind::Number,
        "DECIMAL must render as a number, got {:?}", price.kind);
    let n = price.numeric.as_ref().expect("decoded min/max");
    // Scaled values, not the raw unscaled integers (-725 / 123450).
    assert!((n.min - (-7.25)).abs() < 1e-9, "min should be -7.25, got {}", n.min);
    assert!((n.max - 1234.50).abs() < 1e-9, "max should be 1234.50, got {}", n.max);

    let qty = out.fields.iter().find(|f| f.name == "qty").expect("qty column");
    assert_eq!(qty.kind, FieldKind::Number, "plain int column unaffected");
}

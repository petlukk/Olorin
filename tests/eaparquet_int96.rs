//! eaparquet INT96 timestamps — the deprecated Spark/Hive/Impala encoding
//! (12 bytes: i64 nanos-of-day + i32 Julian day). The footer reader now
//! labels INT96 columns as timestamps and decodes their min/max stats to
//! ISO instants instead of skipping them.
//!
//! Two layers:
//!  1. Decode math — hand-crafted INT96 bytes for known instants. This is
//!     the rigorous proof: it doesn't depend on a writer emitting stats.
//!  2. End-to-end labeling — a pyarrow INT96 fixture. NOTE: pyarrow omits
//!     statistics for INT96 (undefined sort order), so min/max read back as
//!     "?"; the e2e check confirms the column is *labeled* a timestamp (no
//!     longer a bare number). Decoding real min/max is exercised by layer 1.

use olorin::runes::output::{FieldKind, RuneOutput};
use olorin::runes::run_rune;
use olorin::runes::timekey::unix_seconds_to_iso;
use olorin::storage::parquet::int96_to_epoch_seconds;

/// Build a 12-byte INT96 value: i64 nanos-of-day (LE) + i32 Julian day (LE).
fn int96(nanos_of_day: i64, julian_day: i32) -> [u8; 12] {
    let mut b = [0u8; 12];
    b[0..8].copy_from_slice(&nanos_of_day.to_le_bytes());
    b[8..12].copy_from_slice(&julian_day.to_le_bytes());
    b
}

#[test]
fn int96_decode_known_instants() {
    // 2022-12-31T23:59:59 → epoch 1672531199. nanos-of-day = 86399s,
    // Julian day = 2459945. 2023-06-01T12:00:00 → epoch 1685620800,
    // nanos-of-day = 43200s, Julian day = 2460097.
    let min_b = int96(86_399 * 1_000_000_000, 2_459_945);
    let max_b = int96(43_200 * 1_000_000_000, 2_460_097);

    assert_eq!(int96_to_epoch_seconds(&min_b), Some(1_672_531_199));
    assert_eq!(int96_to_epoch_seconds(&max_b), Some(1_685_620_800));

    // Round-trips through the same ISO renderer the rune uses.
    assert_eq!(unix_seconds_to_iso(1_672_531_199, false), "2022-12-31T23:59:59");
    assert_eq!(unix_seconds_to_iso(1_685_620_800, false), "2023-06-01T12:00:00");

    // Wrong length is rejected, never panics.
    assert_eq!(int96_to_epoch_seconds(&[0u8; 8]), None);
}

#[test]
fn int96_column_labeled_timestamp_e2e() {
    olorin::kernels::ffi::init().unwrap();
    let src = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/runes/int96.parquet");
    let dst = std::env::temp_dir()
        .join(format!("olorin_int96_{}.parquet", std::process::id()));
    std::fs::copy(&src, &dst).expect("copy fixture");

    let r = run_rune("eaparquet", &format!("--json {}", dst.display())).unwrap();
    let _ = std::fs::remove_file(&dst);
    assert!(r.success, "rune failed: {}", r.answer);
    let out = RuneOutput::from_json(r.answer.as_bytes()).expect("valid JSON");

    let ev = out.fields.iter().find(|f| f.name == "event_time")
        .expect("event_time column present");
    assert_eq!(ev.kind, FieldKind::Timestamp,
        "INT96 must be labeled a timestamp, got {:?}", ev.kind);
    // pyarrow omits INT96 stats → min/max are unknown here (decode proven
    // in int96_decode_known_instants instead).
    assert!(ev.timestamp.is_some(), "timestamp stats struct present");

    let id = out.fields.iter().find(|f| f.name == "id").expect("id column");
    assert_eq!(id.kind, FieldKind::Number, "plain int column unaffected");
}

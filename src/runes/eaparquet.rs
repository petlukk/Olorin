//! eaparquet — Parquet footer summarizer. Reads metadata only (never
//! decodes column data) and aggregates per-column stats via f64_stats.
//!
//! Limits: primitive types only (BOOLEAN/INT32/INT64/FLOAT/DOUBLE get
//! min/max; BYTE_ARRAY/INT96 are reported but stats are left absent).
//! Flat schemas only. File-metadata statistics must be present.

use super::{Rune, RuneResult, OutputSafety};
use super::common::{resolve_path, open_capped, truncate_answer, PathError};
use crate::storage::parquet::{read_summary, ColumnSummary, PhysicalType};
use std::path::PathBuf;
use std::time::Instant;

pub struct Eaparquet;
pub const RUNE: Eaparquet = Eaparquet;

impl Rune for Eaparquet {
    fn name(&self) -> &'static str { "eaparquet" }
    fn description(&self) -> &'static str {
        "Summarize a Parquet file via its metadata footer: row count, \
         per-column type (number/text/bool), and per-column statistics \
         (min/max/null_count) aggregated across row groups. Never \
         decodes column data — milliseconds even on multi-GB files. \
         Args: <path.parquet>."
    }
    fn usage(&self) -> &'static str { "eaparquet <path.parquet>" }
    fn output_safety(&self) -> OutputSafety { OutputSafety::UntrustedQuoted }

    fn run(&self, args: &str) -> RuneResult {
        let t0 = Instant::now();
        let path = args.trim();
        if path.is_empty() {
            return refusal(t0, "usage: eaparquet <path.parquet>");
        }
        let home = crate::home_dir().unwrap_or_else(|| PathBuf::from("/tmp"));
        let resolved = match resolve_path(path, &home) {
            Ok(p) => p,
            Err(PathError::OutsideAllowlist) =>
                return refusal(t0, "path rejected: outside allowlist (~ or /tmp only)"),
            Err(PathError::NotFound) =>
                return refusal(t0, "file not found"),
            Err(PathError::TooLarge(n)) =>
                return refusal(t0, &format!("file too large: {} bytes", n)),
            Err(PathError::Io(e)) =>
                return refusal(t0, &format!("io error: {e}")),
        };
        let bytes = match open_capped(&resolved, &home) {
            Ok(b) => b,
            Err(e) => return refusal(t0, &format!("open failed: {e:?}")),
        };
        let summary = match read_summary(&bytes) {
            Ok(s) => s,
            Err(e) => return refusal(t0, &format!("parquet decode failed: {e}")),
        };
        let answer = format_summary(&summary);
        RuneResult {
            answer: truncate_answer(&answer),
            details: None,
            success: true,
            timing_us: t0.elapsed().as_micros() as u64,
        }
    }
}

fn refusal(t0: Instant, msg: &str) -> RuneResult {
    RuneResult {
        answer: msg.to_string(),
        details: None,
        success: false,
        timing_us: t0.elapsed().as_micros() as u64,
    }
}

fn format_summary(s: &crate::storage::parquet::ParquetSummary) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "rows: {}\ncolumns: {} (across {} row group{})\n",
        s.num_rows,
        s.columns.len(),
        s.num_row_groups,
        if s.num_row_groups == 1 { "" } else { "s" },
    ));
    for col in &s.columns {
        out.push_str(&format_column(col));
        out.push('\n');
    }
    out
}

fn format_column(c: &ColumnSummary) -> String {
    let label = c.physical_type.label();
    let nulls = match c.null_count {
        Some(n) => format!(", nulls={n}"),
        None => String::new(),
    };
    match c.physical_type {
        PhysicalType::Boolean => {
            format!("{} (bool): values={}{}", c.name, c.total_values, nulls)
        }
        PhysicalType::Int32 | PhysicalType::Int64 | PhysicalType::Float | PhysicalType::Double => {
            let stats = format_minmax(c);
            format!("{} (number): values={}{}{}", c.name, c.total_values, stats, nulls)
        }
        PhysicalType::Int96 | PhysicalType::ByteArray | PhysicalType::FixedLenByteArray => {
            let note = if matches!(c.physical_type, PhysicalType::Int96) {
                " [INT96 timestamp; min/max not decoded]"
            } else {
                " [byte-array column; min/max not decoded]"
            };
            format!("{} ({label}): values={}{}{}", c.name, c.total_values, nulls, note)
        }
    }
}

fn format_minmax(c: &ColumnSummary) -> String {
    match (c.min, c.max) {
        (Some(min), Some(max)) => {
            let is_int = matches!(c.physical_type, PhysicalType::Int32 | PhysicalType::Int64);
            if is_int {
                format!(", min={}, max={}", min.as_f64() as i64, max.as_f64() as i64)
            } else {
                format!(", min={:.2}, max={:.2}", min.as_f64(), max.as_f64())
            }
        }
        _ => String::new(),
    }
}

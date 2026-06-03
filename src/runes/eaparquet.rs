//! eaparquet — Parquet footer summarizer. Reads metadata only (never
//! decodes column data) and aggregates per-column stats. Output: a
//! `RuneOutput` is built first; either serialized via `to_json()` when
//! `--json` is set, or rendered to legacy human-readable text.
//!
//! Schema encoding (no new schema fields needed):
//! - Boolean              → kind=Bool, count + null_count (no true/false breakdown)
//! - Int32/64/Float/Double → kind=Number, NumericStats { min, max, mean=0, sum=0 }
//! - Int96                → kind=Number, numeric=None  (signal: undecoded numeric)
//! - ByteArray / FixedLen → kind=Text,   text=None     (signal: undecoded text)

use super::{Rune, RuneResult, OutputSafety};
use super::common::{resolve_path, open_capped, truncate_answer, PathError};
use super::output::{
    BoolStats, FieldKind, FieldStats, NumericStats, RuneOutput, Source, Totals,
};
use crate::storage::parquet::{
    read_summary, ColumnSummary, ParquetSummary, PhysicalType,
};
use std::path::PathBuf;
use std::time::Instant;

const RUNE_VERSION: i64 = 1;

pub struct Eaparquet;
pub const RUNE: Eaparquet = Eaparquet;

impl Rune for Eaparquet {
    fn name(&self) -> &'static str { "eaparquet" }
    fn description(&self) -> &'static str {
        "Summarize a Parquet file via its metadata footer: row count, \
         per-column type (number/text/bool), and per-column statistics \
         (min/max/null_count) aggregated across row groups. Never \
         decodes column data — milliseconds even on multi-GB files. \
         Args: [--json] <path.parquet>."
    }
    fn usage(&self) -> &'static str { "eaparquet [--json] <path.parquet>" }
    fn output_safety(&self) -> OutputSafety { OutputSafety::UntrustedQuoted }

    fn run(&self, args: &str) -> RuneResult {
        let t0 = Instant::now();
        let (path, json_mode) = parse_args(args);
        let output = execute(&path);
        let answer = if json_mode {
            output.to_json()
        } else if let Some(err) = &output.error {
            err.clone()
        } else {
            format_text(&output)
        };
        RuneResult {
            answer:     truncate_answer(&answer),
            details:    None,
            success:    output.success,
            timing_us:  t0.elapsed().as_micros() as u64,
            structured: json_mode,
        }
    }
}

fn parse_args(args: &str) -> (String, bool) {
    let trimmed = args.trim();
    if let Some(rest) = trimmed.strip_prefix("--json ") {
        (rest.trim().to_string(), true)
    } else if let Some(rest) = trimmed.strip_suffix(" --json") {
        (rest.trim().to_string(), true)
    } else if trimmed == "--json" {
        (String::new(), true)
    } else {
        (trimmed.to_string(), false)
    }
}

fn execute(path: &str) -> RuneOutput {
    if path.is_empty() {
        return error_output("usage: eaparquet [--json] <path.parquet>");
    }
    let home = crate::home_dir().unwrap_or_else(|| PathBuf::from("/tmp"));
    let resolved = match resolve_path(path, &home) {
        Ok(p) => p,
        Err(PathError::OutsideAllowlist) =>
            return error_output("path rejected: outside allowlist (~ or /tmp only)"),
        Err(PathError::NotFound) =>
            return error_output("file not found"),
        Err(PathError::TooLarge(n)) =>
            return error_output(&format!("file too large: {n} bytes")),
        Err(PathError::Io(e)) =>
            return error_output(&format!("io error: {e}")),
    };
    let bytes = match open_capped(&resolved, &home) {
        Ok(b) => b,
        Err(PathError::NotFound) =>
            return error_output("file not found"),
        Err(PathError::TooLarge(n)) =>
            return error_output(&format!("file too large: {n} bytes")),
        Err(PathError::OutsideAllowlist) =>
            return error_output("path rejected: outside allowlist (~ or /tmp only)"),
        Err(PathError::Io(e)) =>
            return error_output(&format!("io error: {e}")),
    };
    let resolved_str = resolved.to_string_lossy().into_owned();

    let t_scan = Instant::now();
    let summary = match read_summary(&bytes) {
        Ok(s) => s,
        Err(e) => return error_output(&format!("parquet decode failed: {e}")),
    };
    let scan_us = t_scan.elapsed().as_micros() as u64;
    build_output(&summary, &bytes, resolved_str, scan_us)
}

fn error_output(msg: &str) -> RuneOutput {
    let mut out = RuneOutput::new("eaparquet", RUNE_VERSION);
    out.success = false;
    out.error = Some(msg.to_string());
    out
}

fn build_output(s: &ParquetSummary, bytes: &[u8], path: String, scan_us: u64) -> RuneOutput {
    let mut out = RuneOutput::new("eaparquet", RUNE_VERSION);
    out.source = Some(Source {
        path,
        bytes:  bytes.len() as u64,
        format: "parquet".to_string(),
    });
    out.totals = Totals { rows: s.num_rows.max(0) as u64, scan_us };
    out.fields = s.columns.iter().map(column_to_field).collect();
    out
}

fn column_to_field(c: &ColumnSummary) -> FieldStats {
    let count      = c.total_values.max(0) as u64;
    let null_count = c.null_count.map(|n| n.max(0) as u64);
    let blank = FieldStats {
        name: c.name.clone(), kind: FieldKind::Mixed, count, null_count,
        numeric: None, text: None, bool: None, timestamp: None,
    };
    match c.physical_type {
        PhysicalType::Boolean => FieldStats {
            kind: FieldKind::Bool,
            bool: Some(BoolStats { true_count: 0, false_count: 0 }),
            ..blank
        },
        PhysicalType::Int32 | PhysicalType::Int64
            | PhysicalType::Float | PhysicalType::Double =>
        {
            let numeric = match (c.min, c.max) {
                (Some(min), Some(max)) => Some(NumericStats {
                    min: min.as_f64(), max: max.as_f64(),
                    mean: 0.0, sum: 0.0,
                }),
                _ => None,
            };
            FieldStats { kind: FieldKind::Number, numeric, ..blank }
        }
        // INT96 is a deprecated timestamp encoding the reader doesn't
        // decode; the kind stays Number to match the legacy label, with
        // numeric=None as the "min/max not decoded" signal that
        // format_text picks up.
        PhysicalType::Int96 => FieldStats {
            kind: FieldKind::Number, numeric: None, ..blank
        },
        // Byte arrays: Text with text=None signals undecoded content.
        PhysicalType::ByteArray | PhysicalType::FixedLenByteArray => FieldStats {
            kind: FieldKind::Text, text: None, ..blank
        },
    }
}

fn format_text(out: &RuneOutput) -> String {
    let mut buf = String::new();
    buf.push_str(&format!("rows: {}\ncolumns: {}\n",
        out.totals.rows, out.fields.len()));
    for f in &out.fields {
        buf.push_str(&format_field(f));
    }
    buf
}

fn format_field(f: &FieldStats) -> String {
    let nulls = f.null_count.map(|n| format!(", nulls={n}")).unwrap_or_default();
    match f.kind {
        FieldKind::Bool => {
            format!("{} (bool): values={}{nulls}\n", f.name, f.count)
        }
        FieldKind::Number => match &f.numeric {
            Some(n) => format!(
                "{} (number): values={}, min={:.2}, max={:.2}{nulls}\n",
                f.name, f.count, n.min, n.max
            ),
            None => format!(
                "{} (number): values={}{nulls} [min/max not available]\n",
                f.name, f.count
            ),
        },
        FieldKind::Text => match &f.text {
            Some(t) => {
                let top: Vec<&str> = t.top.iter().map(|e| e.value.as_str()).collect();
                format!("{} (text): {} unique; top values: {}\n",
                    f.name, t.unique, top.join(", "))
            }
            None => format!(
                "{} (text): values={}{nulls} [byte-array column; min/max not decoded]\n",
                f.name, f.count
            ),
        },
        FieldKind::Mixed => format!("{} (mixed): values={}{nulls}\n", f.name, f.count),
        FieldKind::Timestamp => format!("{} (timestamp): values={}{nulls}\n", f.name, f.count),
    }
}

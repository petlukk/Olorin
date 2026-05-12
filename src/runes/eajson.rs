//! eajson — JSON Lines summarizer via SIMD `jsonl_struct_scan` + `f32_stats`.
//!
//! Features:
//! - Escape-aware quote walking — `\"` inside strings is correctly skipped.
//! - Nested-object flattening — `{"http": {"status":200}}` becomes
//!   `http.status` keys (one level deep).
//! - Byte-array decoding — `[27,91,...]` (systemd-style binary MESSAGE
//!   encoding) is decoded as UTF-8 with `�` replacement.
//! - ISO-8601 timestamp detection — text keys whose values look like
//!   timestamps report `range: min..max` instead of top-3 unique.
//! - Cardinality noise filter — text keys where every value is unique
//!   (cursors, sequence IDs) are suppressed from the *text* output.
//!   They remain in the structured `RuneOutput.fields[]` so downstream
//!   runes can chain on them.
//!
//! Output: a `RuneOutput` is built first from the scan + classify pass;
//! the rune's `answer` is either the JSONL serialization (when `--json`
//! is set) or the legacy human-readable text rendered from the same
//! structured form.
//!
//! See `eajson_aggregate.rs` for the per-line walker + per-key stats
//! materialization that this file orchestrates.

use super::{Rune, RuneResult, OutputSafety};
use super::common::{resolve_path, open_capped, truncate_answer, PathError};
use super::output::{FieldKind, FieldStats, RuneOutput, Source, Totals};
use super::eajson_aggregate::{
    build_field_stats, process_line, Aggregator,
};
use crate::storage::jsonl_parse::build_escaped_quote_set;
use std::path::PathBuf;
use std::time::Instant;

const RUNE_VERSION: i64 = 1;

pub struct Eajson;
pub const RUNE: Eajson = Eajson;

impl Rune for Eajson {
    fn name(&self) -> &'static str { "eajson" }
    fn description(&self) -> &'static str {
        "Summarize a JSON Lines file (one object per line) via SIMD: row \
         count, per-key type (number/text/bool/timestamp), per-numeric-key \
         stats (min/max/mean/sum), top-3 most frequent values for text keys. \
         Handles nested objects (flattened to parent.child), byte-array \
         strings (systemd MESSAGE format), and ISO-8601 timestamps. \
         Args: [--json] <path.jsonl>."
    }
    fn usage(&self) -> &'static str { "eajson [--json] <path.jsonl>" }
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
        return error_output("usage: eajson [--json] <path.jsonl>");
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
    match build_output(&bytes, resolved_str) {
        Ok(out) => out,
        Err(e)  => error_output(&format!("parse failed: {e}")),
    }
}

fn error_output(msg: &str) -> RuneOutput {
    let mut out = RuneOutput::new("eajson", RUNE_VERSION);
    out.success = false;
    out.error = Some(msg.to_string());
    out
}

fn build_output(bytes: &[u8], path: String) -> Result<RuneOutput, String> {
    use crate::kernels::ffi;

    if bytes.is_empty() {
        return Err("empty file".into());
    }
    if bytes.len() > i32::MAX as usize {
        return Err(format!(
            "file too large for jsonl_struct_scan: {} bytes (2 GB limit)",
            bytes.len()
        ));
    }

    let t_scan = Instant::now();
    let len = bytes.len() as i32;
    let mut newlines    = vec![0i32; bytes.len()];
    let mut quotes      = vec![0i32; bytes.len()];
    let mut colons      = vec![0i32; bytes.len()];
    let mut commas      = vec![0i32; bytes.len()];
    let mut backslashes = vec![0i32; bytes.len()];
    let mut n_nl = 0i32; let mut n_q = 0i32; let mut n_co = 0i32;
    let mut n_cm = 0i32; let mut n_bs = 0i32;
    let mut scratch = [0u8; 16];
    unsafe {
        ffi::jsonl_struct_scan(
            bytes.as_ptr(), len,
            newlines.as_mut_ptr(), quotes.as_mut_ptr(),
            colons.as_mut_ptr(),   commas.as_mut_ptr(), backslashes.as_mut_ptr(),
            &mut n_nl, &mut n_q, &mut n_co, &mut n_cm, &mut n_bs,
            scratch.as_mut_ptr(),
        );
    }
    let newlines    = &newlines[..n_nl as usize];
    let quotes_raw  = &quotes[..n_q as usize];
    let colons      = &colons[..n_co as usize];
    let backslashes = &backslashes[..n_bs as usize];

    let escaped = build_escaped_quote_set(quotes_raw, backslashes);
    let quotes: Vec<i32> = quotes_raw.iter().copied()
        .filter(|q| !escaped.contains(q)).collect();

    let mut line_ranges: Vec<(usize, usize)> = Vec::with_capacity(newlines.len() + 1);
    let mut start = 0usize;
    for &nl in newlines {
        line_ranges.push((start, nl as usize));
        start = nl as usize + 1;
    }
    if start < bytes.len() {
        line_ranges.push((start, bytes.len()));
    }

    let mut agg = Aggregator::new();
    let mut row_count: usize = 0;
    let mut q_cur = 0usize;
    let mut co_cur = 0usize;

    for &(line_start, line_end) in &line_ranges {
        while q_cur  < quotes.len() && (quotes[q_cur] as usize) < line_start { q_cur  += 1; }
        while co_cur < colons.len() && (colons[co_cur] as usize) < line_start { co_cur += 1; }

        let q_start = q_cur;
        let mut q_end = q_cur;
        while q_end < quotes.len() && (quotes[q_end] as usize) < line_end { q_end += 1; }
        let line_quotes = &quotes[q_start..q_end];
        if line_quotes.is_empty() { continue; }

        row_count += 1;
        process_line(bytes, line_quotes, colons, line_end, &mut co_cur, "", &mut agg, 0);
        q_cur = q_end;
    }

    if row_count == 0 {
        return Err("no JSON lines parsed".into());
    }

    let fields = build_field_stats(&agg);
    let scan_us = t_scan.elapsed().as_micros() as u64;

    let mut out = RuneOutput::new("eajson", RUNE_VERSION);
    out.source = Some(Source {
        path,
        bytes:  bytes.len() as u64,
        format: "jsonl".to_string(),
    });
    out.totals = Totals { rows: row_count as u64, scan_us };
    out.fields = fields;
    Ok(out)
}

fn format_text(out: &RuneOutput) -> String {
    // Re-apply the cardinality-noise filter for the text view only —
    // suppressed keys (text where every value is unique) stay in the
    // structured form so downstream runes can chain on them.
    let mut suppressed_count = 0usize;
    let mut visible: Vec<&FieldStats> = Vec::with_capacity(out.fields.len());
    for f in &out.fields {
        if f.kind == FieldKind::Text {
            let t = f.text.as_ref().expect("text kind has text stats");
            if f.count > 0 && t.unique == f.count {
                suppressed_count += 1;
                continue;
            }
        }
        visible.push(f);
    }

    let mut buf = String::new();
    buf.push_str(&format!("rows: {}\nkeys: {}", out.totals.rows, visible.len()));
    if suppressed_count > 0 {
        buf.push_str(&format!(" (+{suppressed_count} high-cardinality keys suppressed)"));
    }
    buf.push('\n');

    for f in visible {
        match f.kind {
            FieldKind::Number => {
                let n = f.numeric.as_ref().expect("number has numeric stats");
                buf.push_str(&format!(
                    "{} (number): count={}, mean={:.2}, min={:.2}, max={:.2}, sum={:.2}\n",
                    f.name, f.count, n.mean, n.min, n.max, n.sum
                ));
            }
            FieldKind::Timestamp => {
                let ts = f.timestamp.as_ref().expect("timestamp has timestamp stats");
                buf.push_str(&format!(
                    "{} (timestamp): {} unique of {}; range: {} .. {}\n",
                    f.name, ts.unique, f.count, ts.min, ts.max
                ));
            }
            FieldKind::Text => {
                let t = f.text.as_ref().expect("text has text stats");
                let top: Vec<&str> = t.top.iter().map(|e| e.value.as_str()).collect();
                buf.push_str(&format!(
                    "{} (text): {} unique; top values: {}\n",
                    f.name, t.unique, top.join(", ")
                ));
            }
            FieldKind::Bool => {
                let b = f.bool.as_ref().expect("bool has bool stats");
                buf.push_str(&format!(
                    "{} (bool): true={}, false={}\n",
                    f.name, b.true_count, b.false_count
                ));
            }
            FieldKind::Mixed => {
                buf.push_str(&format!(
                    "{} (mixed): inconsistent types across rows\n", f.name
                ));
            }
        }
    }
    buf
}

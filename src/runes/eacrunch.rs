//! eacrunch — CSV summarizer via SIMD csv_scan + f32_stats.
//!
//! Classifies columns as numeric vs text from the first 32 non-empty
//! rows, then streams stats per column. Returns a one-paragraph summary
//! suitable for a 2B model to narrate.

use super::{Rune, RuneResult, OutputSafety};
use super::common::{resolve_path, open_capped, truncate_answer, PathError};
use std::path::PathBuf;
use std::time::Instant;

pub struct Eacrunch;
pub const RUNE: Eacrunch = Eacrunch;

impl Rune for Eacrunch {
    fn name(&self) -> &'static str { "eacrunch" }
    fn description(&self) -> &'static str {
        "Summarize a CSV file via SIMD: row count, per-column type \
         (numeric/text), and per-numeric-column stats (min/max/mean/sum). \
         Top-3 most frequent values for text columns. Args: <path.csv>."
    }
    fn usage(&self) -> &'static str { "eacrunch <path.csv>" }
    fn output_safety(&self) -> OutputSafety { OutputSafety::UntrustedQuoted }

    fn run(&self, args: &str) -> RuneResult {
        let t0 = Instant::now();
        let path = args.trim();
        if path.is_empty() {
            return refusal(t0, "usage: eacrunch <path.csv>");
        }
        let home = std::env::var("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/tmp"));
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
        // `home` is passed so open_capped can re-check the canonical path
        // against the allowlist — catches symlinks that resolve outside it.
        let bytes = match open_capped(&resolved, &home) {
            Ok(b) => b,
            Err(e) => return refusal(t0, &format!("open failed: {e:?}")),
        };
        let summary = match summarize_csv(&bytes) {
            Ok(s) => s,
            Err(e) => return refusal(t0, &format!("parse failed: {e}")),
        };
        RuneResult {
            answer: truncate_answer(&summary),
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

fn summarize_csv(bytes: &[u8]) -> Result<String, String> {
    use crate::kernels::ffi;

    if bytes.is_empty() {
        return Err("empty file".into());
    }
    // csv_scan takes an i32 length. The 4 GB allowlist in open_capped is
    // wider than i32::MAX, so guard the narrowing cast before calling.
    // Lifting the kernel to i64 is deferred to the streaming MVP+1 pass.
    if bytes.len() > i32::MAX as usize {
        return Err(format!(
            "file too large for csv_scan: {} bytes (2 GB limit)",
            bytes.len()
        ));
    }

    let len = bytes.len() as i32;
    let mut commas   = vec![0i32; bytes.len()];
    let mut newlines = vec![0i32; bytes.len()];
    let mut n_comma  = 0i32;
    let mut n_newln  = 0i32;
    unsafe {
        ffi::csv_scan(
            bytes.as_ptr(), len,
            commas.as_mut_ptr(), newlines.as_mut_ptr(),
            &mut n_comma, &mut n_newln,
        );
    }
    let commas   = &commas[..n_comma as usize];
    let newlines = &newlines[..n_newln as usize];

    // Build row boundaries (start, end-exclusive). Each row ends at a
    // newline. If the final row lacks a trailing newline, include it too.
    let mut row_starts: Vec<usize> = Vec::with_capacity(newlines.len() + 1);
    let mut row_ends:   Vec<usize> = Vec::with_capacity(newlines.len() + 1);
    row_starts.push(0);
    for (i, &nl) in newlines.iter().enumerate() {
        row_ends.push(nl as usize);
        if i + 1 < newlines.len() {
            row_starts.push(nl as usize + 1);
        } else if (nl as usize) + 1 < bytes.len() {
            // Trailing bytes past the last newline — treat as a final row.
            row_starts.push(nl as usize + 1);
            row_ends.push(bytes.len());
        }
    }
    if row_ends.is_empty() {
        // File had no newlines — treat the whole thing as one row.
        row_ends.push(bytes.len());
    }

    let n_rows = row_ends.len();

    // Split each row into field byte ranges using commas within the row.
    let mut rows_fields: Vec<Vec<(usize, usize)>> = Vec::with_capacity(n_rows);
    let mut comma_cursor = 0usize;
    for r in 0..n_rows {
        let start = row_starts[r];
        let end = row_ends[r];
        let mut fields: Vec<(usize, usize)> = Vec::with_capacity(8);
        let mut last = start;
        while comma_cursor < commas.len() && (commas[comma_cursor] as usize) < end {
            let c = commas[comma_cursor] as usize;
            if c >= start {
                fields.push((last, c));
                last = c + 1;
            }
            comma_cursor += 1;
        }
        fields.push((last, end));
        rows_fields.push(fields);
    }

    let header_fields = &rows_fields[0];
    let n_cols = header_fields.len();
    let headers: Vec<String> = header_fields.iter()
        .map(|&(s, e)| String::from_utf8_lossy(&bytes[s..e]).trim().to_string())
        .collect();

    if n_rows < 2 {
        return Err("no data rows".into());
    }

    // Sniff first up-to-32 data rows to classify columns as numeric vs text
    // (numeric if >= 75% of sniffed cells parse as f32).
    let sniff_rows = 32.min(n_rows - 1);
    let mut is_numeric = vec![false; n_cols];
    for c in 0..n_cols {
        let mut ok = 0u32;
        for r in 1..=sniff_rows {
            if c >= rows_fields[r].len() { continue; }
            let (s, e) = rows_fields[r][c];
            let txt = std::str::from_utf8(&bytes[s..e]).unwrap_or("").trim();
            if txt.parse::<f32>().is_ok() { ok += 1; }
        }
        is_numeric[c] = ok * 100 >= (sniff_rows as u32) * 75;
    }

    // Collect per-column f32 values (numeric cols) and unique-value counts
    // (text cols, simple HashMap).
    let mut numeric_vals: Vec<Vec<f32>> = vec![Vec::new(); n_cols];
    let mut text_tops: Vec<std::collections::HashMap<String, u32>> =
        vec![std::collections::HashMap::new(); n_cols];
    let n_data = n_rows - 1;
    for r in 1..n_rows {
        for c in 0..n_cols {
            if c >= rows_fields[r].len() { continue; }
            let (s, e) = rows_fields[r][c];
            let txt = std::str::from_utf8(&bytes[s..e]).unwrap_or("").trim();
            if is_numeric[c] {
                if let Ok(v) = txt.parse::<f32>() {
                    numeric_vals[c].push(v);
                }
            } else {
                *text_tops[c].entry(txt.to_string()).or_insert(0) += 1;
            }
        }
    }

    // Stats per numeric column via the SIMD kernel; top-3 per text column.
    let mut out = String::new();
    out.push_str(&format!("rows: {}\ncolumns: {}\n", n_data, n_cols));
    for c in 0..n_cols {
        if is_numeric[c] {
            let vals = &numeric_vals[c];
            let (count, sum, min_v, max_v) = unsafe {
                let mut count = 0i32;
                let mut sum   = 0f32;
                let mut mn    = 0f32;
                let mut mx    = 0f32;
                ffi::f32_stats(
                    vals.as_ptr(), vals.len() as i32,
                    &mut count, &mut sum, &mut mn, &mut mx,
                );
                (count, sum, mn, mx)
            };
            let mean = if count > 0 { sum / count as f32 } else { 0.0 };
            out.push_str(&format!(
                "{} (number): count={count}, mean={mean:.2}, min={min_v:.2}, max={max_v:.2}, sum={sum:.2}\n",
                headers[c]
            ));
        } else {
            // Top-3 most frequent values.
            let mut pairs: Vec<(&String, &u32)> = text_tops[c].iter().collect();
            pairs.sort_by(|a, b| b.1.cmp(a.1));
            let top: Vec<String> = pairs.iter().take(3)
                .map(|(k, _)| (*k).clone()).collect();
            out.push_str(&format!(
                "{} (text): {} unique; top values: {}\n",
                headers[c], pairs.len(), top.join(", ")
            ));
        }
    }
    Ok(out)
}

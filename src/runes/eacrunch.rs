//! eacrunch — CSV summarizer via SIMD csv_scan + f64_stats.
//!
//! Classifies columns as numeric vs text from the first 32 non-empty
//! rows, then streams stats per column. Output: the structured
//! `RuneOutput` is built first; either serialized to JSON (when
//! `--json` is set) or rendered to the legacy human-readable text.

use super::{Rune, RuneResult, OutputSafety};
use super::common::{resolve_path, open_capped, truncate_answer, unquote, PathError};
use super::grouping::{self, AggSpec};
use super::output::{
    FieldKind, FieldStats, NumericStats, RuneOutput, Source, TextEntry, TextStats, Totals,
};
use std::path::PathBuf;
use std::time::Instant;

const RUNE_VERSION: i64 = 1;
const SNIFF_ROWS: usize = 32;
const TEXT_CARDINALITY_CAP: usize = 10_000;
const TOP_N: usize = 3;

pub struct Eacrunch;
pub const RUNE: Eacrunch = Eacrunch;

impl Rune for Eacrunch {
    fn name(&self) -> &'static str { "eacrunch" }
    fn description(&self) -> &'static str {
        "Summarize a CSV file via SIMD: row count, per-column type \
         (numeric/text), and per-numeric-column stats (min/max/mean/sum). \
         Top-3 most frequent values for text columns. GROUP BY a column with \
         `--by <col>` and aggregate others with `--agg <op:col,...>` (ops: \
         count/sum/mean/min/max). Args: [--json] [--by <col> [--agg <op:col,...>]] <path.csv>."
    }
    fn usage(&self) -> &'static str {
        "eacrunch [--json] [--by <col> [--agg <op:col,...>]] <path.csv>"
    }
    fn output_safety(&self) -> OutputSafety { OutputSafety::UntrustedQuoted }

    fn run(&self, args: &str) -> RuneResult {
        let t0 = Instant::now();
        let a = parse_args(args);
        let output = execute(&a);
        let answer = if a.json {
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
            structured: a.json,
        }
    }
}

/// Parsed eacrunch invocation. `by`/`agg` select GROUP BY mode; absent,
/// eacrunch does its classic whole-file column summary.
struct Args {
    path: String,
    json: bool,
    by:   Option<String>,
    agg:  Option<String>,
}

/// Whitespace-tokenized flag parse. `--by`/`--agg` each consume the next
/// token as their value; the lone non-flag token is the path. (Paths with
/// embedded spaces aren't supported in flag mode — staged rune paths under
/// ~/ or /tmp never contain them.)
fn parse_args(args: &str) -> Args {
    let mut a = Args { path: String::new(), json: false, by: None, agg: None };
    let mut it = args.split_whitespace();
    while let Some(tok) = it.next() {
        match tok {
            "--json" => a.json = true,
            "--by"   => a.by  = it.next().map(str::to_string),
            "--agg"  => a.agg = it.next().map(str::to_string),
            _ => if a.path.is_empty() { a.path = tok.to_string(); },
        }
    }
    a
}

fn execute(a: &Args) -> RuneOutput {
    if a.path.is_empty() {
        return error_output(
            "usage: eacrunch [--json] [--by <col> [--agg <op:col,...>]] <path.csv>",
        );
    }
    if a.agg.is_some() && a.by.is_none() {
        return error_output("--agg requires --by (it aggregates rows grouped by a column)");
    }
    let specs = match &a.agg {
        Some(s) => match grouping::parse_agg_specs(s) {
            Ok(v)  => v,
            Err(e) => return error_output(&e),
        },
        None => Vec::new(),
    };
    let home = crate::home_dir().unwrap_or_else(|| PathBuf::from("/tmp"));
    let resolved = match resolve_path(&a.path, &home) {
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
    match build_output(&bytes, resolved_str, a.by.as_deref(), &specs) {
        Ok(out) => out,
        Err(e)  => error_output(&format!("parse failed: {e}")),
    }
}

fn error_output(msg: &str) -> RuneOutput {
    let mut out = RuneOutput::new("eacrunch", RUNE_VERSION);
    out.success = false;
    out.error = Some(msg.to_string());
    out
}

fn build_output(
    bytes: &[u8],
    path:  String,
    by:    Option<&str>,
    specs: &[AggSpec],
) -> Result<RuneOutput, String> {
    use crate::kernels::ffi;

    if bytes.is_empty() {
        return Err("empty file".into());
    }
    // csv_scan takes an i32 length. Guard the narrowing cast before calling;
    // lifting the kernel to i64 is deferred to the streaming MVP+1 pass.
    if bytes.len() > i32::MAX as usize {
        return Err(format!(
            "file too large for csv_scan: {} bytes (2 GB limit)", bytes.len()
        ));
    }

    // GROUP BY mode takes the fused path: the `csv_groupby_scan` kernel
    // projects only the key + aggregation columns in one pass, so neither
    // the two `len`-sized delimiter arrays below nor the full field grid
    // are ever allocated (scratch is O(rows·cols_needed), not O(bytes)).
    if let Some(by) = by {
        let t_scan = Instant::now();
        let outcome = grouping::build_groups(bytes, by, specs)?;
        let scan_us = t_scan.elapsed().as_micros() as u64;
        let mut out = RuneOutput::new("eacrunch", RUNE_VERSION);
        out.source = Some(Source {
            path,
            bytes:  bytes.len() as u64,
            format: "csv".to_string(),
        });
        out.totals = Totals { rows: outcome.data_rows, scan_us };
        out.groups = outcome.groups;
        out.group_by = Some(by.to_string());
        return Ok(out);
    }

    let t_scan = Instant::now();
    let len = bytes.len() as i32;
    let mut commas   = vec![0i32; bytes.len()];
    let mut newlines = vec![0i32; bytes.len()];
    let mut n_comma  = 0i32;
    let mut n_newln  = 0i32;
    let mut scratch  = [0u8; 16];
    unsafe {
        ffi::csv_scan(
            bytes.as_ptr(), len,
            commas.as_mut_ptr(), newlines.as_mut_ptr(),
            &mut n_comma, &mut n_newln,
            scratch.as_mut_ptr(),
        );
    }
    let commas   = &commas[..n_comma as usize];
    let newlines = &newlines[..n_newln as usize];

    // Build row boundaries (start, end-exclusive). If the final row lacks
    // a trailing newline, include it as a row too.
    let mut row_starts: Vec<usize> = Vec::with_capacity(newlines.len() + 1);
    let mut row_ends:   Vec<usize> = Vec::with_capacity(newlines.len() + 1);
    row_starts.push(0);
    for (i, &nl) in newlines.iter().enumerate() {
        row_ends.push(nl as usize);
        if i + 1 < newlines.len() {
            row_starts.push(nl as usize + 1);
        } else if (nl as usize) + 1 < bytes.len() {
            row_starts.push(nl as usize + 1);
            row_ends.push(bytes.len());
        }
    }
    if row_ends.is_empty() {
        row_ends.push(bytes.len());
    }

    let n_rows = row_ends.len();

    // Split rows into field byte ranges using commas within each row.
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
        .map(|&(s, e)| {
            let raw = String::from_utf8_lossy(&bytes[s..e]);
            unquote(raw.trim()).into_owned()
        })
        .collect();

    if n_rows < 2 {
        return Err("no data rows".into());
    }

    // Sniff first up-to-32 data rows to classify columns as numeric vs
    // text (numeric if >= 75% of sniffed cells parse as f64).
    let sniff_rows = SNIFF_ROWS.min(n_rows - 1);
    let mut is_numeric = vec![false; n_cols];
    for c in 0..n_cols {
        let mut ok = 0u32;
        for r in 1..=sniff_rows {
            if c >= rows_fields[r].len() { continue; }
            let (s, e) = rows_fields[r][c];
            let raw = std::str::from_utf8(&bytes[s..e]).unwrap_or("").trim();
            let txt = unquote(raw);
            if txt.parse::<f64>().is_ok() { ok += 1; }
        }
        is_numeric[c] = ok * 100 >= (sniff_rows as u32) * 75;
    }

    // Per-column numeric values + text-frequency maps.
    let mut numeric_vals: Vec<Vec<f64>> = vec![Vec::new(); n_cols];
    let mut text_tops: Vec<std::collections::HashMap<String, u32>> =
        vec![std::collections::HashMap::new(); n_cols];
    let n_data = n_rows - 1;
    for r in 1..n_rows {
        for c in 0..n_cols {
            if c >= rows_fields[r].len() { continue; }
            let (s, e) = rows_fields[r][c];
            let raw = std::str::from_utf8(&bytes[s..e]).unwrap_or("").trim();
            let txt = unquote(raw);
            if is_numeric[c] {
                // Exclude non-finite parses. Rust's `f64::parse` accepts the
                // literals "nan"/"inf"/"infinity" (case-insensitive), and a
                // single such cell otherwise poisons the additive stats — NaN
                // propagates through `sum`/`mean` (→ serialized as null) while
                // `min`/`max` survive, leaving an internally inconsistent
                // column summary. Treat them as junk: drop from the numeric
                // vector so count/min/max/sum/mean stay consistent over the
                // finite values (matching pandas' skipna behavior).
                if let Ok(v) = txt.parse::<f64>() {
                    if v.is_finite() {
                        numeric_vals[c].push(v);
                    }
                }
            } else if let Some(ct) = text_tops[c].get_mut(txt.as_ref()) {
                *ct += 1;
            } else if text_tops[c].len() < TEXT_CARDINALITY_CAP {
                text_tops[c].insert(txt.into_owned(), 1);
            }
        }
    }

    // Per-column stats → FieldStats.
    let mut fields: Vec<FieldStats> = Vec::with_capacity(n_cols);
    for c in 0..n_cols {
        if is_numeric[c] {
            let vals = &numeric_vals[c];
            let (count, sum, min_v, max_v) = unsafe {
                let mut count = 0i32;
                let mut sum   = 0f64;
                let mut mn    = 0f64;
                let mut mx    = 0f64;
                ffi::f64_stats(
                    vals.as_ptr(), vals.len() as i32,
                    &mut count, &mut sum, &mut mn, &mut mx,
                );
                (count, sum, mn, mx)
            };
            let mean = if count > 0 { sum / count as f64 } else { 0.0 };
            fields.push(FieldStats {
                name:       headers[c].clone(),
                kind:       FieldKind::Number,
                count:      count as u64,
                null_count: None,
                numeric:    Some(NumericStats {
                    min:  min_v,
                    max:  max_v,
                    mean,
                    sum,
                }),
                text: None, bool: None, timestamp: None,
            });
        } else {
            let mut pairs: Vec<(&String, &u32)> = text_tops[c].iter().collect();
            // Sort by count desc, then value asc — without the tiebreaker the
            // top-N order is HashMap-seed-dependent and varies across calls.
            pairs.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
            let top: Vec<TextEntry> = pairs.iter().take(TOP_N)
                .map(|(k, v)| TextEntry { value: (*k).clone(), count: **v as u64 })
                .collect();
            fields.push(FieldStats {
                name:       headers[c].clone(),
                kind:       FieldKind::Text,
                count:      pairs.iter().map(|(_, v)| **v as u64).sum(),
                null_count: None,
                numeric:    None,
                text:       Some(TextStats { unique: pairs.len() as u64, top }),
                bool:       None,
                timestamp:  None,
            });
        }
    }

    let scan_us = t_scan.elapsed().as_micros() as u64;

    let mut out = RuneOutput::new("eacrunch", RUNE_VERSION);
    out.source = Some(Source {
        path,
        bytes:  bytes.len() as u64,
        format: "csv".to_string(),
    });
    out.totals = Totals { rows: n_data as u64, scan_us };
    out.fields = fields;
    Ok(out)
}

/// Cap on group rows rendered in the human-readable answer (the `--json`
/// path emits all groups). Mirrors easql's TOP_TABLES — keeps the LLM /
/// REPL summary bounded when a column has many distinct values.
const TOP_GROUPS: usize = 40;

fn format_text(out: &RuneOutput) -> String {
    if !out.groups.is_empty() {
        return format_groups(out);
    }
    let mut buf = String::new();
    buf.push_str(&format!("rows: {}\ncolumns: {}\n",
        out.totals.rows, out.fields.len()));
    for f in &out.fields {
        match f.kind {
            FieldKind::Number => {
                let n = f.numeric.as_ref().expect("numeric kind has numeric stats");
                buf.push_str(&format!(
                    "{} (number): count={}, mean={:.2}, min={:.2}, max={:.2}, sum={:.2}\n",
                    f.name, f.count, n.mean, n.min, n.max, n.sum
                ));
            }
            FieldKind::Text => {
                let t = f.text.as_ref().expect("text kind has text stats");
                let top_str = t.top.iter()
                    .map(|e| e.value.as_str()).collect::<Vec<_>>().join(", ");
                buf.push_str(&format!(
                    "{} (text): {} unique; top values: {}\n",
                    f.name, t.unique, top_str
                ));
            }
            // CSV sniffer only emits Number or Text. Other kinds would only
            // appear if a future caller constructed a RuneOutput by hand and
            // passed it to this formatter — keep the contract explicit.
            FieldKind::Bool | FieldKind::Timestamp | FieldKind::Mixed => {
                buf.push_str(&format!("{} ({}): {} values\n",
                    f.name, f.kind.as_str(), f.count));
            }
        }
    }
    buf
}

/// Human-readable GROUP BY table. One line per group: the key, the row
/// count, then each requested aggregation as `op(col)=value`. Capped at
/// TOP_GROUPS rows (groups are already sorted biggest-first).
fn format_groups(out: &RuneOutput) -> String {
    let by = out.group_by.as_deref().unwrap_or("?");
    let mut buf = format!(
        "group by {by}: {} group(s) over {} rows\n",
        out.groups.len(), out.totals.rows,
    );
    for g in out.groups.iter().take(TOP_GROUPS) {
        buf.push_str(&format!("  {} — count={}", g.key, g.count));
        for a in &g.aggs {
            if a.col.is_empty() {
                buf.push_str(&format!("  {}={:.2}", a.op, a.value));
            } else {
                buf.push_str(&format!("  {}({})={:.2}", a.op, a.col, a.value));
            }
        }
        buf.push('\n');
    }
    if out.groups.len() > TOP_GROUPS {
        buf.push_str(&format!("  … {} more group(s) (use --json for all)\n",
            out.groups.len() - TOP_GROUPS));
    }
    buf
}

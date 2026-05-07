//! eajson — JSON Lines summarizer via SIMD `jsonl_struct_scan` + `f32_stats`.
//!
//! v2 features (vs the original sketch):
//! - Escape-aware quote walking — `\"` inside strings is correctly skipped.
//! - Nested-object flattening — `{"http": {"status":200}}` becomes
//!   `http.status` keys (one level deep).
//! - Byte-array decoding — `[27,91,...]` (systemd-style binary MESSAGE
//!   encoding) is decoded as UTF-8 with `�` replacement.
//! - ISO-8601 timestamp detection — text keys whose values look like
//!   timestamps report `range: min..max` instead of top-3 unique.
//! - Cardinality noise filter — text keys where every value is unique
//!   (cursors, sequence IDs) are suppressed from the output.

use super::{Rune, RuneResult, OutputSafety};
use super::common::{resolve_path, open_capped, truncate_answer, PathError};
use crate::storage::jsonl_parse::{
    build_escaped_quote_set, classify_scalar, decode_byte_array, find_matching,
    looks_iso8601, scalar_end, skip_ws, trim_ws, unescape_json_string, ScalarKind,
};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

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
         Args: <path.jsonl>."
    }
    fn usage(&self) -> &'static str { "eajson <path.jsonl>" }
    fn output_safety(&self) -> OutputSafety { OutputSafety::UntrustedQuoted }

    fn run(&self, args: &str) -> RuneResult {
        let t0 = Instant::now();
        let path = args.trim();
        if path.is_empty() {
            return refusal(t0, "usage: eajson <path.jsonl>");
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
        let summary = match summarize_jsonl(&bytes) {
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

#[derive(Debug, Clone, Copy, PartialEq)]
enum KeyType { Number, Text, Bool, Timestamp, Mixed }

const TEXT_CARDINALITY_CAP: usize = 10_000;
const NESTED_FLATTEN_MAX_DEPTH: usize = 1;

/// Aggregator for per-key stats — scope-named so the helper signatures
/// don't drown in HashMap parameters.
#[derive(Default)]
struct Aggregator {
    key_order:    Vec<String>,
    key_types:    HashMap<String, KeyType>,
    numeric_vals: HashMap<String, Vec<f32>>,
    text_tops:    HashMap<String, HashMap<String, u32>>,
    text_counts:  HashMap<String, u32>,
    bool_counts:  HashMap<String, (u32, u32)>,
}

fn summarize_jsonl(bytes: &[u8]) -> Result<String, String> {
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

    // SIMD structural scan: one pass over `bytes`, five output arrays
    // sized at the worst-case `len`.
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

    // Filter out escape-hidden quotes so the orchestrator's quote-pair
    // walk operates only on real string boundaries.
    let escaped = build_escaped_quote_set(quotes_raw, backslashes);
    let quotes: Vec<i32> = quotes_raw.iter().copied()
        .filter(|q| !escaped.contains(q)).collect();

    // Build line ranges from newline positions.
    let mut line_ranges: Vec<(usize, usize)> = Vec::with_capacity(newlines.len() + 1);
    let mut start = 0usize;
    for &nl in newlines {
        line_ranges.push((start, nl as usize));
        start = nl as usize + 1;
    }
    if start < bytes.len() {
        line_ranges.push((start, bytes.len()));
    }

    let mut agg = Aggregator::default();
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

    format_summary(row_count, &agg)
}

fn process_line(
    bytes: &[u8],
    line_quotes: &[i32],
    colons: &[i32],
    line_end: usize,
    co_cur: &mut usize,
    key_prefix: &str,
    agg: &mut Aggregator,
    depth: usize,
) {
    let mut pair_i = 0usize;
    while pair_i + 1 < line_quotes.len() {
        let key_open  = line_quotes[pair_i] as usize;
        let key_close = line_quotes[pair_i + 1] as usize;
        let key_bytes = &bytes[key_open + 1..key_close];

        let colon_pos = colons[*co_cur..].iter()
            .find(|&&p| (p as usize) > key_close && (p as usize) < line_end)
            .copied();
        let Some(colon_pos) = colon_pos else { break; };
        let colon_pos = colon_pos as usize;

        let val_start = skip_ws(bytes, colon_pos + 1, line_end);
        if val_start >= line_end { break; }

        let key_str = std::str::from_utf8(key_bytes).unwrap_or("");
        let full_key = if key_prefix.is_empty() {
            key_str.to_string()
        } else {
            format!("{key_prefix}.{key_str}")
        };

        let first = bytes[val_start];

        if first == b'"' {
            if pair_i + 3 >= line_quotes.len() { break; }
            let v_open  = line_quotes[pair_i + 2] as usize;
            let v_close = line_quotes[pair_i + 3] as usize;
            let val_bytes = &bytes[v_open + 1..v_close];
            pair_i += 4;
            advance_cursors(colons, co_cur, v_close + 1);
            ingest_scalar(&full_key, val_bytes, ScalarKind::Text, agg);
        } else if first == b'{' {
            let obj_end = match find_matching(bytes, val_start, line_end, b'{', b'}') {
                Some(e) => e,
                None => break,
            };
            if depth < NESTED_FLATTEN_MAX_DEPTH {
                let inner_quotes: Vec<i32> = line_quotes.iter().copied()
                    .filter(|&q| (q as usize) > val_start && (q as usize) < obj_end)
                    .collect();
                let mut inner_co = *co_cur;
                process_line(bytes, &inner_quotes, colons, obj_end,
                    &mut inner_co, &full_key, agg, depth + 1);
            }
            while pair_i < line_quotes.len() && (line_quotes[pair_i] as usize) < obj_end {
                pair_i += 1;
            }
            advance_cursors(colons, co_cur, obj_end + 1);
        } else if first == b'[' {
            let arr_end = match find_matching(bytes, val_start, line_end, b'[', b']') {
                Some(e) => e,
                None => break,
            };
            if let Some(text) = decode_byte_array(&bytes[val_start..=arr_end]) {
                ingest_scalar(&full_key, text.as_bytes(), ScalarKind::Text, agg);
            }
            while pair_i < line_quotes.len() && (line_quotes[pair_i] as usize) < arr_end {
                pair_i += 1;
            }
            advance_cursors(colons, co_cur, arr_end + 1);
        } else {
            let val_end = scalar_end(bytes, val_start, line_end);
            let val_bytes = trim_ws(&bytes[val_start..val_end]);
            let kind = classify_scalar(val_bytes);
            pair_i += 2;
            advance_cursors(colons, co_cur, val_end);
            if kind != ScalarKind::Skip {
                ingest_scalar(&full_key, val_bytes, kind, agg);
            }
        }
    }
}

fn ingest_scalar(full_key: &str, val_bytes: &[u8], kind: ScalarKind, agg: &mut Aggregator) {
    let observed = match kind {
        ScalarKind::Number => KeyType::Number,
        ScalarKind::Text   => KeyType::Text,
        ScalarKind::Bool   => KeyType::Bool,
        ScalarKind::Skip   => return,
    };

    if !agg.key_types.contains_key(full_key) {
        agg.key_order.push(full_key.to_string());
        agg.key_types.insert(full_key.to_string(), observed);
    } else if agg.key_types[full_key] != observed && agg.key_types[full_key] != KeyType::Mixed {
        agg.key_types.insert(full_key.to_string(), KeyType::Mixed);
    }
    if agg.key_types[full_key] == KeyType::Mixed { return; }

    match kind {
        ScalarKind::Number => {
            let txt = std::str::from_utf8(val_bytes).unwrap_or("");
            if let Ok(v) = txt.parse::<f32>() {
                agg.numeric_vals.entry(full_key.to_string()).or_default().push(v);
            }
        }
        ScalarKind::Text => {
            // Unescape JSON string sequences (`\"` → `"`, `\\` → `\`, etc.)
            // so the user sees the decoded form. Cow avoids alloc when the
            // value has no backslashes (the common case for log data).
            let s = unescape_json_string(val_bytes).into_owned();
            *agg.text_counts.entry(full_key.to_string()).or_insert(0) += 1;
            let counts = agg.text_tops.entry(full_key.to_string()).or_default();
            if let Some(ct) = counts.get_mut(&s) {
                *ct += 1;
            } else if counts.len() < TEXT_CARDINALITY_CAP {
                counts.insert(s, 1);
            }
        }
        ScalarKind::Bool => {
            let entry = agg.bool_counts.entry(full_key.to_string()).or_insert((0, 0));
            if val_bytes == b"true" { entry.0 += 1; } else { entry.1 += 1; }
        }
        ScalarKind::Skip => {}
    }
}

fn format_summary(row_count: usize, agg: &Aggregator) -> Result<String, String> {
    use crate::kernels::ffi;

    // Pre-pass: classify text fields.
    // Order matters: timestamp detection runs FIRST so ISO-8601 fields
    // (which are typically one-per-row, hence all-unique) get a range
    // report instead of being suppressed as noise. After that, text
    // fields where every value is unique (cursors, sequence IDs) are
    // suppressed from the output.
    let mut effective_types: HashMap<&str, KeyType> = HashMap::new();
    let mut suppressed: HashSet<&str> = HashSet::new();
    for k in &agg.key_order {
        let kt = agg.key_types[k];
        if kt != KeyType::Text {
            effective_types.insert(k, kt);
            continue;
        }
        if let Some(sample) = agg.text_tops.get(k).and_then(|h| h.keys().next()) {
            if looks_iso8601(sample) {
                effective_types.insert(k, KeyType::Timestamp);
                continue;
            }
        }
        let count = *agg.text_counts.get(k).unwrap_or(&0);
        let unique = agg.text_tops.get(k).map(|h| h.len() as u32).unwrap_or(0);
        if count > 0 && unique == count {
            suppressed.insert(k);
            continue;
        }
        effective_types.insert(k, KeyType::Text);
    }

    let shown = agg.key_order.iter().filter(|k| !suppressed.contains(k.as_str())).count();
    let suppressed_n = suppressed.len();
    let mut out = String::new();
    out.push_str(&format!("rows: {row_count}\nkeys: {shown}"));
    if suppressed_n > 0 {
        out.push_str(&format!(" (+{suppressed_n} high-cardinality keys suppressed)"));
    }
    out.push('\n');

    for k in &agg.key_order {
        if suppressed.contains(k.as_str()) { continue; }
        match effective_types[k.as_str()] {
            KeyType::Number => {
                let vals = agg.numeric_vals.get(k).map(|v| v.as_slice()).unwrap_or(&[]);
                let (count, sum, min_v, max_v) = unsafe {
                    let mut count = 0i32;
                    let mut sum = 0f32; let mut mn = 0f32; let mut mx = 0f32;
                    ffi::f32_stats(
                        vals.as_ptr(), vals.len() as i32,
                        &mut count, &mut sum, &mut mn, &mut mx,
                    );
                    (count, sum, mn, mx)
                };
                let mean = if count > 0 { sum / count as f32 } else { 0.0 };
                out.push_str(&format!(
                    "{k} (number): count={count}, mean={mean:.2}, min={min_v:.2}, max={max_v:.2}, sum={sum:.2}\n"
                ));
            }
            KeyType::Timestamp => {
                let counts = agg.text_tops.get(k).cloned().unwrap_or_default();
                let mut min_s: Option<&String> = None;
                let mut max_s: Option<&String> = None;
                for s in counts.keys() {
                    if min_s.map_or(true, |m| s < m) { min_s = Some(s); }
                    if max_s.map_or(true, |m| s > m) { max_s = Some(s); }
                }
                let total = agg.text_counts.get(k).copied().unwrap_or(0);
                out.push_str(&format!(
                    "{k} (timestamp): {} unique of {total}; range: {} .. {}\n",
                    counts.len(),
                    min_s.map(|s| s.as_str()).unwrap_or("?"),
                    max_s.map(|s| s.as_str()).unwrap_or("?")
                ));
            }
            KeyType::Text => {
                let counts = agg.text_tops.get(k).cloned().unwrap_or_default();
                let mut pairs: Vec<(&String, &u32)> = counts.iter().collect();
                pairs.sort_by(|a, b| b.1.cmp(a.1));
                let top: Vec<String> = pairs.iter().take(3)
                    .map(|(s, _)| (*s).clone()).collect();
                out.push_str(&format!(
                    "{k} (text): {} unique; top values: {}\n",
                    pairs.len(), top.join(", ")
                ));
            }
            KeyType::Bool => {
                let (t, f) = agg.bool_counts.get(k).copied().unwrap_or((0, 0));
                out.push_str(&format!("{k} (bool): true={t}, false={f}\n"));
            }
            KeyType::Mixed => {
                out.push_str(&format!("{k} (mixed): inconsistent types across rows\n"));
            }
        }
    }

    Ok(out)
}

fn advance_cursors(colons: &[i32], co_cur: &mut usize, past: usize) {
    while *co_cur < colons.len() && (colons[*co_cur] as usize) < past { *co_cur += 1; }
}

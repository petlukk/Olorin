//! eajson scan-pass aggregator: walks line-by-line quote/colon position
//! arrays from `jsonl_struct_scan` to extract per-key values, then
//! materializes a `Vec<FieldStats>` for the structured RuneOutput.
//!
//! Split out of `eajson.rs` so each file stays under the 500-LOC cap;
//! the rune entry point in `eajson.rs` still owns the scan kernel call,
//! the line walk, and the legacy text rendering.

use super::output::{
    FieldKind, FieldStats, NumericStats, BoolStats, TextEntry, TextStats, TimestampStats,
};
use crate::storage::jsonl_parse::{
    decode_byte_array, find_matching, looks_iso8601, scalar_end,
    skip_ws, trim_ws, classify_scalar, unescape_json_string, ScalarKind,
};
use std::collections::HashMap;

pub const TEXT_CARDINALITY_CAP: usize = 10_000;
pub const NESTED_FLATTEN_MAX_DEPTH: usize = 1;
const TOP_N: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq)]
enum KeyType { Number, Text, Bool, Timestamp, Mixed }

#[derive(Default)]
pub struct Aggregator {
    key_order:    Vec<String>,
    key_types:    HashMap<String, KeyType>,
    numeric_vals: HashMap<String, Vec<f32>>,
    text_tops:    HashMap<String, HashMap<String, u32>>,
    text_counts:  HashMap<String, u32>,
    bool_counts:  HashMap<String, (u32, u32)>,
}

impl Aggregator {
    pub fn new() -> Self { Self::default() }
}

pub fn process_line(
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

pub fn advance_cursors(colons: &[i32], co_cur: &mut usize, past: usize) {
    while *co_cur < colons.len() && (colons[*co_cur] as usize) < past { *co_cur += 1; }
}

pub fn build_field_stats(agg: &Aggregator) -> Vec<FieldStats> {
    let mut fields: Vec<FieldStats> = Vec::with_capacity(agg.key_order.len());
    for k in &agg.key_order {
        let kt = agg.key_types[k];
        let effective = match kt {
            KeyType::Text => {
                // Promote to Timestamp if a sample value looks ISO-8601.
                // HashMap iteration order doesn't affect detection because
                // looks_iso8601 is a per-value predicate.
                let sample = agg.text_tops.get(k).and_then(|h| h.keys().next());
                if sample.map_or(false, |s| looks_iso8601(s)) {
                    KeyType::Timestamp
                } else {
                    KeyType::Text
                }
            }
            other => other,
        };
        fields.push(make_field(k, effective, agg));
    }
    fields
}

fn make_field(k: &str, kind: KeyType, agg: &Aggregator) -> FieldStats {
    use crate::kernels::ffi;
    let blank = FieldStats {
        name: k.to_string(), kind: FieldKind::Mixed, count: 0, null_count: None,
        numeric: None, text: None, bool: None, timestamp: None,
    };
    match kind {
        KeyType::Number => {
            let vals = agg.numeric_vals.get(k).map(|v| v.as_slice()).unwrap_or(&[]);
            let (count, sum, min_v, max_v) = unsafe {
                let mut count = 0i32; let mut sum = 0f32;
                let mut mn = 0f32; let mut mx = 0f32;
                ffi::f32_stats(vals.as_ptr(), vals.len() as i32,
                    &mut count, &mut sum, &mut mn, &mut mx);
                (count, sum, mn, mx)
            };
            let mean = if count > 0 { sum / count as f32 } else { 0.0 };
            FieldStats {
                kind: FieldKind::Number, count: count as u64,
                numeric: Some(NumericStats {
                    min: min_v as f64, max: max_v as f64,
                    mean: mean as f64, sum: sum as f64,
                }),
                ..blank
            }
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
            FieldStats {
                kind: FieldKind::Timestamp, count: total as u64,
                timestamp: Some(TimestampStats {
                    min: min_s.cloned().unwrap_or_else(|| "?".to_string()),
                    max: max_s.cloned().unwrap_or_else(|| "?".to_string()),
                    unique: counts.len() as u64,
                }),
                ..blank
            }
        }
        KeyType::Text => {
            let counts = agg.text_tops.get(k).cloned().unwrap_or_default();
            let mut pairs: Vec<(&String, &u32)> = counts.iter().collect();
            // count desc, value asc — same deterministic tie-break as eacrunch.
            pairs.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
            let top: Vec<TextEntry> = pairs.iter().take(TOP_N)
                .map(|(s, c)| TextEntry { value: (*s).clone(), count: **c as u64 })
                .collect();
            let total = agg.text_counts.get(k).copied().unwrap_or(0);
            FieldStats {
                kind: FieldKind::Text, count: total as u64,
                text: Some(TextStats { unique: pairs.len() as u64, top }),
                ..blank
            }
        }
        KeyType::Bool => {
            let (t, f) = agg.bool_counts.get(k).copied().unwrap_or((0, 0));
            FieldStats {
                kind: FieldKind::Bool, count: (t + f) as u64,
                bool: Some(BoolStats {
                    true_count: t as u64, false_count: f as u64,
                }),
                ..blank
            }
        }
        KeyType::Mixed => blank,
    }
}

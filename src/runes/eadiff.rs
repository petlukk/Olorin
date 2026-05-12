//! eadiff — structural delta between two prior `--json` rune outputs.
//!
//! Reads two JSONL files produced by any rune's `--json` mode, then
//! emits a `RuneOutput { rune: "eadiff" }` whose `fields[]` and
//! `categories[]` carry signed deltas of matched entries.
//!
//! v1 scope (deliberate):
//! - `fields[]` with `kind: Number` are diffed: numeric (min/max/mean/sum)
//!   becomes b.numeric − a.numeric. Number fields without `numeric`
//!   (the "undecoded values" eaparquet pattern) are skipped.
//! - `categories[]` matched by name emit directional entries: a bucket
//!   that grew emits as `+<name>` with count=|delta|; one that shrank
//!   emits as `-<name>`; unchanged buckets are omitted. The structured
//!   form preserves direction; downstream code parses the leading sign.
//! - Text, Bool, Timestamp, Mixed fields and samples[] are skipped.
//!   Adding them is straightforward but each requires a sign-encoding
//!   decision; v1 picks the smallest set that powers the
//!   `eatime × eatime` and `eacrunch × eacrunch` use cases.
//! - Asymmetric keys (in one input but not the other) are skipped.

use super::{Rune, RuneResult, OutputSafety};
use super::common::{resolve_path, open_capped, truncate_answer, PathError};
use super::output::{
    Category, FieldKind, FieldStats, NumericStats, RuneOutput, Totals,
};
use std::path::PathBuf;
use std::time::Instant;

const RUNE_VERSION: i64 = 1;

pub struct Eadiff;
pub const RUNE: Eadiff = Eadiff;

impl Rune for Eadiff {
    fn name(&self) -> &'static str { "eadiff" }
    fn description(&self) -> &'static str {
        "Compute a structural delta between two prior --json rune outputs. \
         Each argument is a path to a JSONL file produced by `--json` on \
         any rune. Numeric fields and categories are diffed by name; \
         deltas are signed (b - a). Args: [--json] <a.json> <b.json>."
    }
    fn usage(&self) -> &'static str { "eadiff [--json] <a.json> <b.json>" }
    fn output_safety(&self) -> OutputSafety { OutputSafety::UntrustedQuoted }

    fn run(&self, args: &str) -> RuneResult {
        let t0 = Instant::now();
        // --json detection runs before parse_args so usage errors still
        // emit structured JSON when the caller asked for it.
        let json_mode = args.split_whitespace().any(|t| t == "--json");
        let output = match parse_args(args) {
            Ok((paths, _)) => execute(&paths.0, &paths.1),
            Err(msg) => error_output(&msg),
        };
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

fn parse_args(args: &str) -> Result<((String, String), bool), String> {
    let mut tokens: Vec<&str> = args.split_whitespace().collect();
    let json_mode = tokens.iter().any(|t| *t == "--json");
    tokens.retain(|t| *t != "--json");
    if tokens.len() != 2 {
        return Err(format!(
            "usage: eadiff [--json] <a.json> <b.json> (got {} non-flag arg(s))",
            tokens.len()
        ));
    }
    Ok(((tokens[0].to_string(), tokens[1].to_string()), json_mode))
}

fn execute(path_a: &str, path_b: &str) -> RuneOutput {
    let a = match load_rune_output(path_a, "a") {
        Ok(out) => out,
        Err(msg) => return error_output(&msg),
    };
    let b = match load_rune_output(path_b, "b") {
        Ok(out) => out,
        Err(msg) => return error_output(&msg),
    };
    build_output(&a, &b)
}

fn load_rune_output(path_arg: &str, label: &str) -> Result<RuneOutput, String> {
    let home = crate::home_dir().unwrap_or_else(|| PathBuf::from("/tmp"));
    let resolved = resolve_path(path_arg, &home).map_err(|e| match e {
        PathError::OutsideAllowlist =>
            format!("{label}: path rejected (outside allowlist)"),
        PathError::NotFound => format!("{label}: file not found"),
        PathError::TooLarge(n) => format!("{label}: file too large: {n} bytes"),
        PathError::Io(s) => format!("{label}: io error: {s}"),
    })?;
    let bytes = open_capped(&resolved, &home).map_err(|e| match e {
        PathError::NotFound => format!("{label}: file not found"),
        PathError::TooLarge(n) => format!("{label}: file too large: {n} bytes"),
        PathError::OutsideAllowlist =>
            format!("{label}: path rejected (outside allowlist)"),
        PathError::Io(s) => format!("{label}: io error: {s}"),
    })?;
    RuneOutput::from_json(&bytes)
        .map_err(|e| format!("{label}: not a valid RuneOutput JSON: {e}"))
}

fn error_output(msg: &str) -> RuneOutput {
    let mut out = RuneOutput::new("eadiff", RUNE_VERSION);
    out.success = false;
    out.error = Some(msg.to_string());
    out
}

fn build_output(a: &RuneOutput, b: &RuneOutput) -> RuneOutput {
    let t0 = Instant::now();

    let mut fields: Vec<FieldStats> = Vec::new();
    // Symmetric: name in both inputs. Dispatch by matching kind.
    for fb in &b.fields {
        let Some(fa) = a.fields.iter().find(|f| f.name == fb.name) else { continue };
        match (fa.kind.clone(), fb.kind.clone()) {
            (FieldKind::Number, FieldKind::Number) => {
                if let Some(diff) = diff_number(&fb.name, fa, fb) { fields.push(diff); }
            }
            (FieldKind::Bool, FieldKind::Bool) => {
                fields.extend(diff_bool(&fb.name, fa, fb));
            }
            (FieldKind::Timestamp, FieldKind::Timestamp) => {
                fields.extend(diff_timestamp(&fb.name, fa, fb));
            }
            (FieldKind::Text, FieldKind::Text) => {
                fields.extend(diff_text(&fb.name, fa, fb));
            }
            // Mismatched-kind same name → flag as Mixed; the schema
            // shouldn't produce two runs with different kinds for the
            // same name from a stable input, so this signals a real
            // schema-evolution event the caller probably wants to see.
            _ => fields.push(mixed_marker(
                &format!("[kind-changed] {}", fb.name),
                fb.count,
            )),
        }
    }
    // Asymmetric fields: appeared in b but not a, or vice versa.
    for fb in &b.fields {
        if a.fields.iter().any(|f| f.name == fb.name) { continue; }
        fields.push(mixed_marker(
            &format!("[appeared] {}", fb.name),
            fb.count,
        ));
    }
    for fa in &a.fields {
        if b.fields.iter().any(|f| f.name == fa.name) { continue; }
        fields.push(mixed_marker(
            &format!("[disappeared] {}", fa.name),
            fa.count,
        ));
    }

    // Categories: directional delta on symmetric names; appeared /
    // disappeared markers on asymmetric ones.
    let mut categories: Vec<Category> = Vec::new();
    for cb in &b.categories {
        if let Some(ca) = a.categories.iter().find(|c| c.name == cb.name) {
            if cb.count == ca.count { continue; }
            let (sign, magnitude) = if cb.count > ca.count {
                ('+', cb.count - ca.count)
            } else {
                ('-', ca.count - cb.count)
            };
            categories.push(Category {
                name:  format!("{sign}{}", cb.name),
                count: magnitude,
            });
        } else {
            categories.push(Category {
                name:  format!("[appeared] {}", cb.name),
                count: cb.count,
            });
        }
    }
    for ca in &a.categories {
        if !b.categories.iter().any(|c| c.name == ca.name) {
            categories.push(Category {
                name:  format!("[disappeared] {}", ca.name),
                count: ca.count,
            });
        }
    }

    let scan_us = t0.elapsed().as_micros() as u64;
    let mut out = RuneOutput::new("eadiff", RUNE_VERSION);
    out.totals = Totals { rows: 0, scan_us };
    out.fields = fields;
    out.categories = categories;
    out
}

fn diff_number(name: &str, fa: &FieldStats, fb: &FieldStats) -> Option<FieldStats> {
    let (na, nb) = (fa.numeric.as_ref()?, fb.numeric.as_ref()?);
    Some(FieldStats {
        name:       name.to_string(),
        kind:       FieldKind::Number,
        count:      fb.count,
        null_count: None,
        numeric:    Some(NumericStats {
            min:  nb.min  - na.min,
            max:  nb.max  - na.max,
            mean: nb.mean - na.mean,
            sum:  nb.sum  - na.sum,
        }),
        text: None, bool: None, timestamp: None,
    })
}

// Bool: split into two Number fields named `<col>.true_delta` and
// `<col>.false_delta`, each carrying a signed delta in min=max=mean=sum.
// Two counts that move independently → two single-purpose fields.
fn diff_bool(name: &str, fa: &FieldStats, fb: &FieldStats) -> Vec<FieldStats> {
    let (Some(ba), Some(bb)) = (fa.bool.as_ref(), fb.bool.as_ref()) else { return Vec::new() };
    let t_delta = bb.true_count  as i64 - ba.true_count  as i64;
    let f_delta = bb.false_count as i64 - ba.false_count as i64;
    vec![
        scalar_delta_field(&format!("{name}.true_delta"),  t_delta as f64, fb.count),
        scalar_delta_field(&format!("{name}.false_delta"), f_delta as f64, fb.count),
    ]
}

// Timestamp diff: emits unique_delta plus signed second-deltas for the
// range endpoints (min_shift_s, max_shift_s). All three are Number
// fields with a single value in numeric.mean. Suffix names disambiguate.
fn diff_timestamp(name: &str, fa: &FieldStats, fb: &FieldStats) -> Vec<FieldStats> {
    let (Some(ta), Some(tb)) = (fa.timestamp.as_ref(), fb.timestamp.as_ref()) else {
        return Vec::new();
    };
    let mut out: Vec<FieldStats> = Vec::new();
    if ta.unique != tb.unique {
        let delta = tb.unique as i64 - ta.unique as i64;
        out.push(scalar_delta_field(
            &format!("{name}.unique_delta"), delta as f64, fb.count,
        ));
    }
    if let (Some(min_a), Some(min_b)) = (iso_to_seconds(&ta.min), iso_to_seconds(&tb.min)) {
        if min_a != min_b {
            out.push(scalar_delta_field(
                &format!("{name}.min_shift_s"), (min_b - min_a) as f64, fb.count,
            ));
        }
    }
    if let (Some(max_a), Some(max_b)) = (iso_to_seconds(&ta.max), iso_to_seconds(&tb.max)) {
        if max_a != max_b {
            out.push(scalar_delta_field(
                &format!("{name}.max_shift_s"), (max_b - max_a) as f64, fb.count,
            ));
        }
    }
    out
}

// Text diff: emits unique_delta plus per-value top-N comparison entries.
//   - Value in both a.top and b.top with different count → Number field
//     `<col>:<value>.count_delta` with signed delta in numeric.mean.
//   - Value in b.top only → `[appeared in top] <col>:<value>` Mixed marker.
//   - Value in a.top only → `[disappeared from top] <col>:<value>` Mixed marker.
fn diff_text(name: &str, fa: &FieldStats, fb: &FieldStats) -> Vec<FieldStats> {
    let (Some(ta), Some(tb)) = (fa.text.as_ref(), fb.text.as_ref()) else {
        return Vec::new();
    };
    let mut out: Vec<FieldStats> = Vec::new();
    if ta.unique != tb.unique {
        let delta = tb.unique as i64 - ta.unique as i64;
        out.push(scalar_delta_field(
            &format!("{name}.unique_delta"), delta as f64, fb.count,
        ));
    }
    let a_top: std::collections::HashMap<&str, u64> = ta.top.iter()
        .map(|e| (e.value.as_str(), e.count)).collect();
    let b_top: std::collections::HashMap<&str, u64> = tb.top.iter()
        .map(|e| (e.value.as_str(), e.count)).collect();
    // Symmetric: same value in both top-N, count may differ.
    for (val, &cb) in &b_top {
        if let Some(&ca) = a_top.get(val) {
            if ca != cb {
                let delta = cb as i64 - ca as i64;
                out.push(scalar_delta_field(
                    &format!("{name}:{val}.count_delta"), delta as f64, fb.count,
                ));
            }
        }
    }
    // Asymmetric: appeared in b.top, disappeared from a.top.
    for (val, &cb) in &b_top {
        if !a_top.contains_key(val) {
            out.push(mixed_marker(
                &format!("[appeared in top] {name}:{val}"), cb,
            ));
        }
    }
    for (val, &ca) in &a_top {
        if !b_top.contains_key(val) {
            out.push(mixed_marker(
                &format!("[disappeared from top] {name}:{val}"), ca,
            ));
        }
    }
    out
}

// Parse `YYYY-MM-DDTHH:MM:SS` (any trailing timezone / fraction is
// ignored) to seconds since 2000-01-01T00:00:00. Returns None on any
// digit failure or out-of-range component. The fixed epoch only needs
// to be monotonic across calls — eadiff only uses the difference of
// two values, never the absolute number.
fn iso_to_seconds(s: &str) -> Option<i64> {
    let bytes = s.as_bytes();
    if bytes.len() < 19 { return None; }
    let year:   i64 = parse_uint(&bytes[0..4])?  as i64;
    if bytes[4] != b'-' { return None; }
    let month:  i64 = parse_uint(&bytes[5..7])?  as i64;
    if bytes[7] != b'-' { return None; }
    let day:    i64 = parse_uint(&bytes[8..10])? as i64;
    if bytes[10] != b'T' { return None; }
    let hour:   i64 = parse_uint(&bytes[11..13])? as i64;
    if bytes[13] != b':' { return None; }
    let minute: i64 = parse_uint(&bytes[14..16])? as i64;
    if bytes[16] != b':' { return None; }
    let second: i64 = parse_uint(&bytes[17..19])? as i64;
    if month < 1 || month > 12 || day < 1 || day > 31 { return None; }
    if hour > 23 || minute > 59 || second > 60 { return None; }
    let days = days_since_2000(year, month, day);
    Some(days * 86400 + hour * 3600 + minute * 60 + second)
}

fn parse_uint(s: &[u8]) -> Option<u32> {
    let mut acc: u32 = 0;
    for &b in s {
        if !(b'0'..=b'9').contains(&b) { return None; }
        acc = acc * 10 + (b - b'0') as u32;
    }
    Some(acc)
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_since_2000(year: i64, month: i64, day: i64) -> i64 {
    const MONTH_DAYS: [i64; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut total: i64 = 0;
    // Year accumulation. Supports years on either side of 2000 — pre-
    // 2000 years count negatively. eadiff only uses the difference of
    // two timestamps, so the absolute origin is arbitrary.
    if year >= 2000 {
        for y in 2000..year {
            total += if is_leap(y) { 366 } else { 365 };
        }
    } else {
        for y in year..2000 {
            total -= if is_leap(y) { 366 } else { 365 };
        }
    }
    for m in 0..(month - 1) as usize {
        total += MONTH_DAYS[m];
    }
    if month > 2 && is_leap(year) { total += 1; }
    total + (day - 1)
}

// Helper: emit a Number FieldStats carrying a single signed value in
// all four numeric slots. Used for synthesized scalar deltas
// (bool counts, unique counts) that don't have a meaningful min/max
// distinction — consumers read `numeric.mean` and ignore the rest.
fn scalar_delta_field(name: &str, value: f64, count: u64) -> FieldStats {
    FieldStats {
        name:       name.to_string(),
        kind:       FieldKind::Number,
        count,
        null_count: None,
        numeric:    Some(NumericStats {
            min: value, max: value, mean: value, sum: value,
        }),
        text: None, bool: None, timestamp: None,
    }
}

fn mixed_marker(name: &str, count: u64) -> FieldStats {
    FieldStats {
        name:       name.to_string(),
        kind:       FieldKind::Mixed,
        count,
        null_count: None,
        numeric:    None, text: None, bool: None, timestamp: None,
    }
}

fn format_text(out: &RuneOutput) -> String {
    let mut buf = String::with_capacity(256);
    buf.push_str(&format!(
        "fields-diffed:     {}\ncategories-diffed: {}\n",
        out.fields.len(), out.categories.len()
    ));
    if !out.fields.is_empty() {
        buf.push('\n');
        buf.push_str("field deltas (b - a):\n");
        for f in &out.fields {
            match f.kind {
                FieldKind::Mixed => {
                    // Asymmetric markers: name carries the [appeared]/
                    // [disappeared] prefix already.
                    buf.push_str(&format!("  {}\n", f.name));
                }
                FieldKind::Number => {
                    let n = f.numeric.as_ref().expect("Number field has numeric");
                    if f.name.ends_with("_delta") || f.name.ends_with("_shift_s") {
                        // Synthesized scalar delta (bool / unique-count /
                        // timestamp-shift / text-top count delta): single
                        // meaningful value, print compactly.
                        buf.push_str(&format!("  {}: {:+.2}\n", f.name, n.mean));
                    } else {
                        buf.push_str(&format!(
                            "  {}: mean={:+.2}, min={:+.2}, max={:+.2}, sum={:+.2}\n",
                            f.name, n.mean, n.min, n.max, n.sum
                        ));
                    }
                }
                // Bool / Text / Timestamp shouldn't appear in eadiff
                // output today (they're always rewritten into Number
                // fields or skipped). Future-proofed with a line.
                _ => buf.push_str(&format!("  {} ({})\n", f.name, f.kind.as_str())),
            }
        }
    }
    if !out.categories.is_empty() {
        buf.push('\n');
        buf.push_str("category deltas:\n");
        for c in &out.categories {
            buf.push_str(&format!("  {} {:>12}\n", c.name, c.count));
        }
    }
    if out.fields.is_empty() && out.categories.is_empty() {
        buf.push('\n');
        buf.push_str("(no diffable structure overlap between the two inputs)\n");
    }
    buf
}

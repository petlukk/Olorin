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
            answer:    truncate_answer(&answer),
            details:   None,
            success:   output.success,
            timing_us: t0.elapsed().as_micros() as u64,
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
    for fb in &b.fields {
        if fb.kind != FieldKind::Number { continue; }
        let Some(fa) = a.fields.iter().find(|f| f.name == fb.name) else { continue };
        if fa.kind != FieldKind::Number { continue; }
        let (Some(na), Some(nb)) = (fa.numeric.as_ref(), fb.numeric.as_ref()) else { continue };
        fields.push(FieldStats {
            name:       fb.name.clone(),
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
        });
    }

    // Categories: directional delta. + prefix = grew, - prefix = shrank.
    // Buckets with zero delta are omitted to keep output compact for
    // sparse changes (typical case: a handful of hour buckets differ).
    let mut categories: Vec<Category> = Vec::new();
    for cb in &b.categories {
        let Some(ca) = a.categories.iter().find(|c| c.name == cb.name) else { continue };
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
    }

    let scan_us = t0.elapsed().as_micros() as u64;
    let mut out = RuneOutput::new("eadiff", RUNE_VERSION);
    out.totals = Totals { rows: 0, scan_us };
    out.fields = fields;
    out.categories = categories;
    out
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
            let n = f.numeric.as_ref().expect("eadiff Number field has numeric");
            buf.push_str(&format!(
                "  {}: mean={:+.2}, min={:+.2}, max={:+.2}, sum={:+.2}\n",
                f.name, n.mean, n.min, n.max, n.sum
            ));
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

//! eatime — ISO-8601 timestamp histogram. Bucketizes every
//! `YYYY-MM-DDTHH:MM:SS` occurrence in the file by hour-of-day,
//! emitting a 24-bucket histogram as `categories[]`.
//!
//! SIMD strategy: `timestamp_scan` kernel does a one-pass position
//! find for the 11-byte prefix `YYYY-MM-DDT`; Rust extracts the two
//! HH digits at offset+11..+13 and increments a 24-slot counter.
//! Pattern matches the rune family: SIMD for the bandwidth-bound byte
//! scan, scalar for the per-position dispatch.
//!
//! Output: a `RuneOutput` is built first; either serialized via
//! `to_json()` when `--json` is set, or rendered to human-readable
//! text. Both views share the same structured form.

use super::{Rune, RuneResult, OutputSafety};
use super::common::{resolve_path, open_capped, truncate_answer, PathError};
use super::output::{Anomaly, Category, RuneOutput, Source, Totals};
use super::stream::{self, Format, MAX_POSITIONS};
use super::timekey::seconds_to_iso;
use super::anomaly;
use std::path::PathBuf;
use std::time::Instant;

const RUNE_VERSION: i64 = 1;

pub struct Eatime;
pub const RUNE: Eatime = Eatime;

impl Rune for Eatime {
    fn name(&self) -> &'static str { "eatime" }
    fn description(&self) -> &'static str {
        "Bucketize ISO-8601 timestamps in a file via SIMD. Default \
         --bucket hour emits a 24-slot hour-of-day histogram; \
         --bucket weekday emits a 7-slot Mon..Sun histogram via \
         Zeller's congruence on the kernel positions; --bucket series \
         emits a chronological histogram (auto bucket width) with \
         robust spike detection — buckets where the event rate broke \
         from baseline are reported as anomalies. Auto-detects ISO-8601 \
         (YYYY-MM-DDTHH:MM:SS) and Common Log Format ([dd/MMM/yyyy:hh:mm:ss], \
         the Apache/nginx access-log default); force with --format. Args: \
         [--json] [--bucket hour|weekday|series] [--format iso|clf|auto] <path>."
    }
    fn usage(&self) -> &'static str { "eatime [--json] [--bucket hour|weekday|series] [--format iso|clf|auto] <path>" }
    fn output_safety(&self) -> OutputSafety { OutputSafety::UntrustedQuoted }

    fn run(&self, args: &str) -> RuneResult {
        let t0 = Instant::now();
        let parsed = parse_args(args);
        let output = match parsed {
            Ok((path, json_mode, bucket, format)) => {
                let r = execute(&path, bucket, format);
                (r, json_mode)
            }
            Err((msg, json_mode)) => (error_output(&msg), json_mode),
        };
        let (out, json_mode) = output;
        let answer = if json_mode {
            out.to_json()
        } else if let Some(err) = &out.error {
            err.clone()
        } else {
            format_text(&out)
        };
        RuneResult {
            answer:     truncate_answer(&answer),
            details:    None,
            success:    out.success,
            timing_us:  t0.elapsed().as_micros() as u64,
            structured: json_mode,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Bucket { Hour, Weekday, Series }

fn parse_args(args: &str) -> Result<(String, bool, Bucket, Option<Format>), (String, bool)> {
    let mut json_mode = false;
    let mut bucket = Bucket::Hour;
    let mut format: Option<Format> = None;
    let mut path_tokens: Vec<&str> = Vec::new();
    let mut tokens = args.split_whitespace();
    while let Some(tok) = tokens.next() {
        match tok {
            "--json" => json_mode = true,
            "--bucket" => match tokens.next() {
                Some("hour")    => bucket = Bucket::Hour,
                Some("weekday") => bucket = Bucket::Weekday,
                Some("series")  => bucket = Bucket::Series,
                Some(other) => return Err((
                    format!("unknown --bucket: {other} (expected hour|weekday|series)"),
                    json_mode,
                )),
                None => return Err((
                    "missing value after --bucket".to_string(),
                    json_mode,
                )),
            },
            "--format" => match tokens.next() {
                Some("auto")               => format = None,
                Some("iso") | Some("iso8601") => format = Some(Format::Iso),
                Some("clf")                => format = Some(Format::Clf),
                Some(other) => return Err((
                    format!("unknown --format: {other} (expected iso|clf|auto)"),
                    json_mode,
                )),
                None => return Err((
                    "missing value after --format".to_string(),
                    json_mode,
                )),
            },
            other => path_tokens.push(other),
        }
    }
    if path_tokens.is_empty() {
        return Err((
            "usage: eatime [--json] [--bucket hour|weekday|series] [--format iso|clf|auto] <path>".to_string(),
            json_mode,
        ));
    }
    Ok((path_tokens.join(" "), json_mode, bucket, format))
}

fn execute(path: &str, bucket: Bucket, format: Option<Format>) -> RuneOutput {
    if path.is_empty() {
        return error_output("usage: eatime [--json] [--bucket hour|weekday|series] [--format iso|clf|auto] <path>");
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

    if bytes.len() > i32::MAX as usize {
        return error_output(&format!(
            "file too large for timestamp_scan: {} bytes (2 GB limit)", bytes.len()
        ));
    }

    build_output(&bytes, resolved_str, bucket, format)
}

fn error_output(msg: &str) -> RuneOutput {
    let mut out = RuneOutput::new("eatime", RUNE_VERSION);
    out.success = false;
    out.error = Some(msg.to_string());
    out
}

fn build_output(bytes: &[u8], path: String, bucket: Bucket, fmt: Option<Format>) -> RuneOutput {
    // Auto-detect the grammar when not forced, then scan with that kernel
    // and decode every hit to epoch-seconds. All three bucket modes work
    // off the epoch list, so CLF and ISO share one code path.
    let format = fmt.unwrap_or_else(|| stream::detect_format(bytes));
    let scan = stream::scan_for(bytes, format, MAX_POSITIONS);
    let scan_us = scan.scan_us;
    let epochs = stream::positions_to_epochs(bytes, &scan.positions, format);

    let mut out = RuneOutput::new("eatime", RUNE_VERSION);
    let (categories, total) = match bucket {
        Bucket::Hour    => hour_buckets(&epochs),
        Bucket::Weekday => weekday_buckets(&epochs),
        Bucket::Series  => {
            let (cats, total, anomalies) = series_buckets(&epochs);
            out.anomalies = anomalies;
            (cats, total)
        }
    };

    out.source = Some(Source {
        path,
        bytes:  bytes.len() as u64,
        format: format.tag().to_string(),
    });
    out.totals = Totals { rows: total, scan_us };
    out.categories = categories;
    out
}

fn hour_buckets(epochs: &[i64]) -> (Vec<Category>, u64) {
    let mut counts = [0u64; 24];
    for &e in epochs {
        let hour = (e.rem_euclid(86400) / 3600) as usize;
        counts[hour] += 1;
    }
    let categories = (0..24).map(|h| Category {
        name:  format!("{h:02}:00"),
        count: counts[h],
    }).collect();
    (categories, epochs.len() as u64)
}

const WEEKDAY_NAMES: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

fn weekday_buckets(epochs: &[i64]) -> (Vec<Category>, u64) {
    let mut counts = [0u64; 7];
    for &e in epochs {
        counts[weekday_index(e)] += 1;
    }
    let categories = (0..7).map(|i| Category {
        name:  WEEKDAY_NAMES[i].to_string(),
        count: counts[i],
    }).collect();
    (categories, epochs.len() as u64)
}

/// Epoch day 0 (2000-01-01) was a Saturday = index 5 in WEEKDAY_NAMES
/// (Mon=0). Works for negative epochs via floored div/mod.
fn weekday_index(epoch: i64) -> usize {
    let days = epoch.div_euclid(86400);
    (days.rem_euclid(7) as usize + 5) % 7
}

const TARGET_BUCKETS: i64 = 120;

/// Absolute ceiling on series buckets. `auto_width` caps the bucket width
/// at the largest "nice" value (1 week), so a pathological span still
/// grows the bucket count without bound — eatime scans *every* ISO instant
/// in the text, and one outlier (a junk nested timestamp spanning years or
/// millennia, as in real GH Archive data) would blow the series up to
/// 100 k+ buckets. Past this many buckets we widen further than any nice
/// width to keep the output sane and the labels meaningful.
const MAX_SERIES_BUCKETS: usize = 1000;

/// Chronological histogram with spike detection. Decodes each kernel
/// position to epoch-seconds, bins the span into auto-width buckets, and
/// hands the count series to `anomaly::detect`. Buckets are labelled with
/// the ISO timestamp of their start instant.
fn series_buckets(epochs: &[i64]) -> (Vec<Category>, u64, Vec<Anomaly>) {
    if epochs.is_empty() {
        return (Vec::new(), 0, Vec::new());
    }
    let min = *epochs.iter().min().unwrap();
    let max = *epochs.iter().max().unwrap();
    let span = max - min;
    let mut width = stream::auto_width(span, TARGET_BUCKETS);
    // Hard-bound the bucket count against pathological spans. `+1` guarantees
    // width > span / MAX_SERIES_BUCKETS, so n lands at or below the ceiling.
    if (span / width) as usize + 1 > MAX_SERIES_BUCKETS {
        width = span / MAX_SERIES_BUCKETS as i64 + 1;
    }
    let n = (span / width) as usize + 1;
    let mut counts = vec![0u64; n];
    for &e in epochs {
        counts[((e - min) / width) as usize] += 1;
    }
    let categories = (0..n).map(|i| Category {
        name:  seconds_to_iso(min + (i as i64) * width),
        count: counts[i],
    }).collect();
    let anomalies = anomaly::detect(&counts, min, width);
    (categories, epochs.len() as u64, anomalies)
}

fn format_text(out: &RuneOutput) -> String {
    // Series labels are ISO instants (contain 'T'); they also end in
    // ":00" at minute/hour widths, so this branch must precede the
    // hour-of-day heuristic below or a timeline would mislabel itself.
    if out.categories.first().map_or(false, |c| c.name.contains('T')) {
        return format_series_text(out);
    }
    let src = out.source.as_ref().expect("build_output populates source on success");
    let total = out.totals.rows;
    let mut buf = String::with_capacity(512);
    buf.push_str(&format!("bytes:       {}\n", format_bytes(src.bytes as usize)));
    buf.push_str(&format!("timestamps:  {total}\n"));
    buf.push_str(&format!("scan:        {}\n", super::common::format_scan_time(out.totals.scan_us)));
    buf.push('\n');
    if total == 0 {
        buf.push_str("(no ISO-8601 timestamps found)\n");
        return buf;
    }
    // Header label is data-driven so weekday vs hour-of-day output
    // reads correctly without the formatter knowing the bucket mode.
    let label = match out.categories.first().map(|c| c.name.as_str()) {
        Some(n) if n.ends_with(":00") => "hour-of-day:",
        Some("Mon")                   => "weekday:",
        _                             => "buckets:",
    };
    buf.push_str(label);
    buf.push('\n');

    let mut peak = 0u64;
    let mut peak_name = String::new();
    for c in &out.categories {
        if c.count > peak { peak = c.count; peak_name = c.name.clone(); }
        let pct = if total > 0 {
            (c.count as f64) * 100.0 / (total as f64)
        } else {
            0.0
        };
        buf.push_str(&format!(
            "  {} {:>12}  ({:>5.2}%)\n", c.name, c.count, pct
        ));
    }
    if peak > 0 {
        buf.push_str(&format!("\npeak: {peak_name} ({peak} timestamps)\n"));
    }
    buf
}

// Series mode prints a compact incident summary — span, peak bucket,
// and the flagged spikes — rather than every bucket (the full series
// lives in `categories[]` for `--json` consumers).
fn format_series_text(out: &RuneOutput) -> String {
    let src = out.source.as_ref().expect("build_output populates source on success");
    let total = out.totals.rows;
    let mut buf = String::with_capacity(512);
    buf.push_str(&format!("bytes:       {}\n", format_bytes(src.bytes as usize)));
    buf.push_str(&format!("timestamps:  {total}\n"));
    buf.push_str(&format!("buckets:     {}\n", out.categories.len()));
    buf.push_str(&format!("scan:        {}\n", super::common::format_scan_time(out.totals.scan_us)));
    buf.push('\n');
    if total == 0 {
        buf.push_str("(no ISO-8601 timestamps found)\n");
        return buf;
    }
    if let (Some(first), Some(last)) = (out.categories.first(), out.categories.last()) {
        buf.push_str(&format!("span:        {} .. {}\n", first.name, last.name));
    }
    let peak = out.categories.iter().max_by_key(|c| c.count);
    if let Some(p) = peak {
        buf.push_str(&format!("peak bucket: {} ({} timestamps)\n", p.name, p.count));
    }
    buf.push('\n');
    if out.anomalies.is_empty() {
        buf.push_str("anomalies:   none (rate within baseline)\n");
    } else {
        buf.push_str(&format!("anomalies:   {} spike(s) detected\n", out.anomalies.len()));
        for a in &out.anomalies {
            let ratio = if a.ratio.is_finite() {
                format!("{:.1}×", a.ratio)
            } else {
                "∞".to_string()
            };
            buf.push_str(&format!(
                "  {} count={} ({ratio} baseline {:.0})\n",
                a.bucket, a.count, a.baseline
            ));
        }
    }
    buf
}

fn format_bytes(n: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = 1024 * KB;
    const GB: usize = 1024 * MB;
    if n >= GB {
        format!("{:.2} GB", n as f64 / GB as f64)
    } else if n >= MB {
        format!("{:.2} MB", n as f64 / MB as f64)
    } else if n >= KB {
        format!("{:.1} KB", n as f64 / KB as f64)
    } else {
        format!("{n} B")
    }
}

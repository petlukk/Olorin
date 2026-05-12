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
use super::output::{Category, RuneOutput, Source, Totals};
use crate::kernels::ffi;
use std::path::PathBuf;
use std::time::Instant;

const RUNE_VERSION: i64 = 1;
/// Caller-side cap on positions stored from the kernel. Each emitted
/// position is 4 bytes; ~16M timestamps fits in 64 MB which is well
/// inside the 4 GB input limit. The kernel saturates at this cap;
/// past that, the rune reports the cap was hit.
const MAX_POSITIONS: usize = 16_000_000;

pub struct Eatime;
pub const RUNE: Eatime = Eatime;

impl Rune for Eatime {
    fn name(&self) -> &'static str { "eatime" }
    fn description(&self) -> &'static str {
        "Bucketize ISO-8601 timestamps in a file via SIMD. Default \
         --bucket hour emits a 24-slot hour-of-day histogram; \
         --bucket weekday emits a 7-slot Mon..Sun histogram via \
         Zeller's congruence on the kernel positions. Recognizes \
         YYYY-MM-DDTHH:MM:SS anywhere in the file. Args: \
         [--json] [--bucket hour|weekday] <path>."
    }
    fn usage(&self) -> &'static str { "eatime [--json] [--bucket hour|weekday] <path>" }
    fn output_safety(&self) -> OutputSafety { OutputSafety::UntrustedQuoted }

    fn run(&self, args: &str) -> RuneResult {
        let t0 = Instant::now();
        let parsed = parse_args(args);
        let output = match parsed {
            Ok((path, json_mode, bucket)) => {
                let r = execute(&path, bucket);
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
enum Bucket { Hour, Weekday }

fn parse_args(args: &str) -> Result<(String, bool, Bucket), (String, bool)> {
    let mut json_mode = false;
    let mut bucket = Bucket::Hour;
    let mut path_tokens: Vec<&str> = Vec::new();
    let mut tokens = args.split_whitespace();
    while let Some(tok) = tokens.next() {
        match tok {
            "--json" => json_mode = true,
            "--bucket" => match tokens.next() {
                Some("hour")    => bucket = Bucket::Hour,
                Some("weekday") => bucket = Bucket::Weekday,
                Some(other) => return Err((
                    format!("unknown --bucket: {other} (expected hour|weekday)"),
                    json_mode,
                )),
                None => return Err((
                    "missing value after --bucket".to_string(),
                    json_mode,
                )),
            },
            other => path_tokens.push(other),
        }
    }
    if path_tokens.is_empty() {
        return Err((
            "usage: eatime [--json] [--bucket hour|weekday] <path>".to_string(),
            json_mode,
        ));
    }
    Ok((path_tokens.join(" "), json_mode, bucket))
}

fn execute(path: &str, bucket: Bucket) -> RuneOutput {
    if path.is_empty() {
        return error_output("usage: eatime [--json] [--bucket hour|weekday] <path>");
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

    build_output(&bytes, resolved_str, bucket)
}

fn error_output(msg: &str) -> RuneOutput {
    let mut out = RuneOutput::new("eatime", RUNE_VERSION);
    out.success = false;
    out.error = Some(msg.to_string());
    out
}

fn build_output(bytes: &[u8], path: String, bucket: Bucket) -> RuneOutput {
    let positions = scan_timestamps(bytes);
    let scan_us = positions.scan_us;

    let (categories, total_timestamps) = match bucket {
        Bucket::Hour    => hour_buckets(bytes, &positions.positions),
        Bucket::Weekday => weekday_buckets(bytes, &positions.positions),
    };

    let mut out = RuneOutput::new("eatime", RUNE_VERSION);
    out.source = Some(Source {
        path,
        bytes:  bytes.len() as u64,
        format: "iso8601".to_string(),
    });
    out.totals = Totals { rows: total_timestamps, scan_us };
    out.categories = categories;
    out
}

struct ScanResult {
    positions: Vec<i32>,
    scan_us:   u64,
}

fn scan_timestamps(bytes: &[u8]) -> ScanResult {
    if bytes.is_empty() {
        return ScanResult { positions: Vec::new(), scan_us: 0 };
    }
    let t_scan = Instant::now();
    let mut positions = vec![0i32; MAX_POSITIONS];
    let mut n_positions = 0i32;
    let mut scratch = [0u8; 16];
    unsafe {
        ffi::timestamp_scan(
            bytes.as_ptr(), bytes.len() as i32,
            positions.as_mut_ptr(), MAX_POSITIONS as i32, &mut n_positions,
            scratch.as_mut_ptr(),
        );
    }
    positions.truncate(n_positions as usize);
    ScanResult { positions, scan_us: t_scan.elapsed().as_micros() as u64 }
}

fn hour_buckets(bytes: &[u8], positions: &[i32]) -> (Vec<Category>, u64) {
    let mut counts = [0u64; 24];
    let mut total = 0u64;
    for &pos in positions {
        let p = pos as usize;
        if p + 13 > bytes.len() { continue; }
        let h_tens = bytes[p + 11].wrapping_sub(b'0');
        let h_ones = bytes[p + 12].wrapping_sub(b'0');
        if h_tens > 9 || h_ones > 9 { continue; }
        let hour = (h_tens as usize) * 10 + (h_ones as usize);
        if hour >= 24 { continue; }
        counts[hour] += 1;
        total += 1;
    }
    let categories = (0..24).map(|h| Category {
        name:  format!("{h:02}:00"),
        count: counts[h],
    }).collect();
    (categories, total)
}

const WEEKDAY_NAMES: [&str; 7] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

fn weekday_buckets(bytes: &[u8], positions: &[i32]) -> (Vec<Category>, u64) {
    let mut counts = [0u64; 7];
    let mut total  = 0u64;
    for &pos in positions {
        let p = pos as usize;
        if p + 10 > bytes.len() { continue; }
        // YYYY at p+0..p+4, MM at p+5..p+7, DD at p+8..p+10.
        let (Some(year), Some(month), Some(day)) = (
            parse_uint(&bytes[p..p + 4]),
            parse_uint(&bytes[p + 5..p + 7]),
            parse_uint(&bytes[p + 8..p + 10]),
        ) else { continue };
        if month < 1 || month > 12 || day < 1 || day > 31 { continue; }
        let wd = zeller_weekday(year as i64, month as i64, day as i64);
        counts[wd] += 1;
        total += 1;
    }
    let categories = (0..7).map(|i| Category {
        name:  WEEKDAY_NAMES[i].to_string(),
        count: counts[i],
    }).collect();
    (categories, total)
}

fn parse_uint(s: &[u8]) -> Option<u32> {
    let mut acc: u32 = 0;
    for &b in s {
        if !(b'0'..=b'9').contains(&b) { return None; }
        acc = acc * 10 + (b - b'0') as u32;
    }
    Some(acc)
}

// Zeller's congruence (Gregorian). Returns 0=Mon..6=Sun so the
// emitted category order matches the WEEKDAY_NAMES array.
fn zeller_weekday(mut year: i64, mut month: i64, day: i64) -> usize {
    if month < 3 { month += 12; year -= 1; }
    let k = year % 100;
    let j = year / 100;
    let h = (day + 13 * (month + 1) / 5 + k + k / 4 + j / 4 + 5 * j) % 7;
    // Zeller h: 0=Sat, 1=Sun, 2=Mon, ..., 6=Fri.
    ((h + 5) % 7) as usize
}

fn format_text(out: &RuneOutput) -> String {
    let src = out.source.as_ref().expect("build_output populates source on success");
    let total = out.totals.rows;
    let mut buf = String::with_capacity(512);
    buf.push_str(&format!("bytes:       {}\n", format_bytes(src.bytes as usize)));
    buf.push_str(&format!("timestamps:  {total}\n"));
    buf.push_str(&format!("scan:        {} ms\n", out.totals.scan_us / 1000));
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

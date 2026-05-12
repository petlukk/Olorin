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
        "Bucketize ISO-8601 timestamps in a file by hour-of-day via SIMD: \
         24-bucket histogram showing when events cluster. Recognizes \
         the YYYY-MM-DDTHH:MM:SS prefix wherever it appears (log lines, \
         CSV cells, JSONL values). Args: [--json] <path>."
    }
    fn usage(&self) -> &'static str { "eatime [--json] <path>" }
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
            answer:    truncate_answer(&answer),
            details:   None,
            success:   output.success,
            timing_us: t0.elapsed().as_micros() as u64,
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
        return error_output("usage: eatime [--json] <path>");
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

    build_output(&bytes, resolved_str)
}

fn error_output(msg: &str) -> RuneOutput {
    let mut out = RuneOutput::new("eatime", RUNE_VERSION);
    out.success = false;
    out.error = Some(msg.to_string());
    out
}

fn build_output(bytes: &[u8], path: String) -> RuneOutput {
    let mut hour_counts = [0u64; 24];
    let mut total_timestamps: u64 = 0;
    let scan_us;

    if bytes.is_empty() {
        scan_us = 0;
    } else {
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
        scan_us = t_scan.elapsed().as_micros() as u64;

        for &pos in &positions[..n_positions as usize] {
            let p = pos as usize;
            // Defense in depth: the kernel only validates the 11-byte
            // prefix, so we re-check that the two HH digits past 'T' are
            // present and parse before incrementing.
            if p + 13 > bytes.len() { continue; }
            let h_tens = bytes[p + 11].wrapping_sub(b'0');
            let h_ones = bytes[p + 12].wrapping_sub(b'0');
            if h_tens > 9 || h_ones > 9 { continue; }
            let hour = (h_tens as usize) * 10 + (h_ones as usize);
            if hour >= 24 { continue; }
            hour_counts[hour] += 1;
            total_timestamps += 1;
        }
    }

    let mut out = RuneOutput::new("eatime", RUNE_VERSION);
    out.source = Some(Source {
        path,
        bytes:  bytes.len() as u64,
        format: "iso8601".to_string(),
    });
    out.totals = Totals { rows: total_timestamps, scan_us };
    out.categories = (0..24).map(|h| Category {
        name:  format!("{h:02}:00"),
        count: hour_counts[h],
    }).collect();
    out
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
    buf.push_str("hour-of-day:\n");
    let mut peak = 0u64;
    let mut peak_hour = 0usize;
    for (i, c) in out.categories.iter().enumerate() {
        if c.count > peak { peak = c.count; peak_hour = i; }
    }
    for c in &out.categories {
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
        buf.push_str(&format!("\npeak: {:02}:00 ({peak} timestamps)\n", peak_hour));
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

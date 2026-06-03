//! ealog — Multi-keyword severity scanner for log files. Counts
//! word-bounded DEBUG / INFO / WARN / ERROR / FATAL occurrences and
//! the line count via a single SIMD pass, plus records byte offsets
//! of the first N ERROR / FATAL matches so the rune can surface
//! sample lines for the LLM to narrate.
//!
//! Output: the structured `RuneOutput` is built first from the kernel
//! results, then either serialized to JSON (when `--json` is set) or
//! rendered to the legacy human-readable text. Both paths share the
//! same source of truth — the text and the JSON can never disagree.

use super::{Rune, RuneResult, OutputSafety};
use super::common::{resolve_path, open_capped, truncate_answer, PathError};
use super::output::{Category, RuneOutput, Sample, Source, Totals};
use crate::kernels::ffi;
use std::path::PathBuf;
use std::time::Instant;

const MAX_HIGH_SEVERITY_SAMPLES: usize = 5;
const SAMPLE_LINE_TRUNCATE: usize = 160;
const RUNE_VERSION: i64 = 1;

pub struct Ealog;
pub const RUNE: Ealog = Ealog;

impl Rune for Ealog {
    fn name(&self) -> &'static str { "ealog" }
    fn description(&self) -> &'static str {
        "Summarize a log file via SIMD: per-severity counts \
         (DEBUG/INFO/WARN/ERROR/FATAL, case-insensitive), total line \
         count, and bytes scanned. Word-bounded matching ignores compound \
         identifiers like ERROR_HANDLER. Args: [--json] <path.log>."
    }
    fn usage(&self) -> &'static str { "ealog [--json] <path.log>" }
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
        return error_output("usage: ealog [--json] <path.log>");
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
    build_output(&bytes, resolved_str)
}

fn error_output(msg: &str) -> RuneOutput {
    let mut out = RuneOutput::new("ealog", RUNE_VERSION);
    out.success = false;
    out.error = Some(msg.to_string());
    out
}

fn build_output(bytes: &[u8], path: String) -> RuneOutput {
    let t_scan = Instant::now();
    let mut counts = [0i32; 6];
    let mut positions = [0i32; MAX_HIGH_SEVERITY_SAMPLES];
    let mut n_pos = 0i32;
    let mut scratch = [0u8; 16];
    unsafe {
        ffi::log_level_scan(
            bytes.as_ptr(), bytes.len() as i32,
            counts.as_mut_ptr(),
            positions.as_mut_ptr(), MAX_HIGH_SEVERITY_SAMPLES as i32, &mut n_pos,
            scratch.as_mut_ptr(),
        );
    }
    let scan_us = t_scan.elapsed().as_micros() as u64;

    let [c_debug, c_info, c_warn, c_error, c_fatal, c_nl] = [
        counts[0] as u64, counts[1] as u64, counts[2] as u64,
        counts[3] as u64, counts[4] as u64, counts[5] as u64,
    ];
    // Trailing partial line (no terminating newline) bumps the count.
    let lines = if !bytes.is_empty() && *bytes.last().unwrap() != b'\n' {
        c_nl + 1
    } else {
        c_nl
    };

    let mut out = RuneOutput::new("ealog", RUNE_VERSION);
    out.source = Some(Source {
        path,
        bytes:  bytes.len() as u64,
        format: detect_format(bytes).to_string(),
    });
    out.totals = Totals { rows: lines, scan_us };
    out.categories = vec![
        Category { name: "DEBUG".to_string(), count: c_debug },
        Category { name: "INFO".to_string(),  count: c_info },
        Category { name: "WARN".to_string(),  count: c_warn },
        Category { name: "ERROR".to_string(), count: c_error },
        Category { name: "FATAL".to_string(), count: c_fatal },
    ];
    for &offset in &positions[..n_pos as usize] {
        let (line_num, line) = extract_line_at(bytes, offset as usize);
        out.samples.push(Sample {
            byte_offset: Some(offset as u64),
            line:        Some(line_num as u64),
            timestamp:   None,
            text:        truncate_line(line),
        });
    }
    out
}

fn format_text(out: &RuneOutput) -> String {
    let src = out.source.as_ref().expect("build_output populates source on success");
    let total: u64 = out.categories.iter().map(|c| c.count).sum();

    let mut buf = String::with_capacity(512);
    buf.push_str(&format!("bytes:   {}\n", format_bytes(src.bytes as usize)));
    buf.push_str(&format!("lines:   {}\n", out.totals.rows));
    buf.push_str(&format!("format:  {}\n", src.format));
    buf.push_str(&format!("scan:    {} ms\n", out.totals.scan_us / 1000));
    buf.push('\n');
    buf.push_str("severity:\n");
    for c in &out.categories {
        buf.push_str(&fmt_level(&c.name, c.count, total));
    }
    if total == 0 {
        buf.push_str("  (no severity keywords found — file may not be a log)\n");
    }
    if !out.samples.is_empty() {
        buf.push('\n');
        buf.push_str("high-severity sample:\n");
        for s in &out.samples {
            let line = s.line.unwrap_or(0);
            buf.push_str(&format!("  L{line}: {}\n", s.text));
        }
    }
    buf
}

fn extract_line_at(bytes: &[u8], offset: usize) -> (usize, &[u8]) {
    let start = bytes[..offset].iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
    let end = bytes[offset..].iter().position(|&b| b == b'\n').map_or(bytes.len(), |i| offset + i);
    let line_num = bytes[..start].iter().filter(|&&b| b == b'\n').count() + 1;
    (line_num, &bytes[start..end])
}

fn truncate_line(line: &[u8]) -> String {
    let s = String::from_utf8_lossy(line);
    let trimmed = s.trim_end_matches('\r');
    if trimmed.len() > SAMPLE_LINE_TRUNCATE {
        let cut = (0..=SAMPLE_LINE_TRUNCATE)
            .rev()
            .find(|&i| trimmed.is_char_boundary(i))
            .unwrap_or(0);
        format!("{}…", &trimmed[..cut])
    } else {
        trimmed.to_string()
    }
}

fn fmt_level(name: &str, count: u64, total: u64) -> String {
    let pct = if total > 0 {
        (count as f64) * 100.0 / (total as f64)
    } else {
        0.0
    };
    format!("  {name:<6} {count:>12}  ({pct:>5.2}%)\n")
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

fn detect_format(bytes: &[u8]) -> &'static str {
    let head = &bytes[..bytes.len().min(4096)];
    let mut brace_lines = 0usize;
    let mut total_lines = 0usize;
    let mut line_start = 0;
    for (i, &b) in head.iter().enumerate() {
        if b == b'\n' {
            let line = &head[line_start..i];
            let trimmed = line.iter().position(|&c| c != b' ' && c != b'\t');
            if let Some(off) = trimmed {
                if line[off] == b'{' {
                    brace_lines += 1;
                }
            }
            total_lines += 1;
            line_start = i + 1;
        }
    }
    if total_lines == 0 {
        return "plaintext";
    }
    if brace_lines * 2 > total_lines &&
        (head.windows(7).any(|w| w == b"\"level\"") ||
         head.windows(10).any(|w| w == b"\"severity\""))
    {
        "jsonl"
    } else {
        "plaintext"
    }
}

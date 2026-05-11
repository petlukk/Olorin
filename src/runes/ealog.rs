//! ealog — Multi-keyword severity scanner for log files. Counts
//! word-bounded DEBUG / INFO / WARN / ERROR / FATAL occurrences and
//! the line count via a single SIMD pass, plus records byte offsets
//! of the first N ERROR / FATAL matches so the rune can surface
//! sample lines for the LLM to narrate.

use super::{Rune, RuneResult, OutputSafety};
use super::common::{resolve_path, open_capped, truncate_answer, PathError};
use crate::kernels::ffi;
use std::path::PathBuf;
use std::time::Instant;

const MAX_HIGH_SEVERITY_SAMPLES: usize = 5;
const SAMPLE_LINE_TRUNCATE: usize = 160;

pub struct Ealog;
pub const RUNE: Ealog = Ealog;

impl Rune for Ealog {
    fn name(&self) -> &'static str { "ealog" }
    fn description(&self) -> &'static str {
        "Summarize a log file via SIMD: per-severity counts \
         (DEBUG/INFO/WARN/ERROR/FATAL), total line count, and bytes \
         scanned. Word-bounded matching ignores compound identifiers \
         like ERROR_HANDLER. Args: <path.log>."
    }
    fn usage(&self) -> &'static str { "ealog <path.log>" }
    fn output_safety(&self) -> OutputSafety { OutputSafety::UntrustedQuoted }

    fn run(&self, args: &str) -> RuneResult {
        let t0 = Instant::now();
        let path = args.trim();
        if path.is_empty() {
            return refusal(t0, "usage: ealog <path.log>");
        }
        let home = crate::home_dir().unwrap_or_else(|| PathBuf::from("/tmp"));
        let resolved = match resolve_path(path, &home) {
            Ok(p) => p,
            Err(PathError::OutsideAllowlist) =>
                return refusal(t0, "path rejected: outside allowlist (~ or /tmp only)"),
            Err(PathError::NotFound) =>
                return refusal(t0, "file not found"),
            Err(PathError::TooLarge(n)) =>
                return refusal(t0, &format!("file too large: {n} bytes")),
            Err(PathError::Io(e)) =>
                return refusal(t0, &format!("io error: {e}")),
        };
        let bytes = match open_capped(&resolved, &home) {
            Ok(b) => b,
            Err(e) => return refusal(t0, &format!("open failed: {e:?}")),
        };

        let answer = scan_and_format(&bytes);
        RuneResult {
            answer: truncate_answer(&answer),
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

fn scan_and_format(bytes: &[u8]) -> String {
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
    let scan_us = t_scan.elapsed().as_micros();

    let [c_debug, c_info, c_warn, c_error, c_fatal, c_nl] =
        [counts[0] as u64, counts[1] as u64, counts[2] as u64,
         counts[3] as u64, counts[4] as u64, counts[5] as u64];

    // Newlines in the buffer = complete-line count. If the file does not
    // end in a newline, the trailing partial line bumps the count by one.
    let lines = if !bytes.is_empty() && *bytes.last().unwrap() != b'\n' {
        c_nl + 1
    } else {
        c_nl
    };

    let total = c_debug + c_info + c_warn + c_error + c_fatal;
    let format = detect_format(bytes);

    let mut out = String::with_capacity(512);
    out.push_str(&format!("bytes:   {}\n", format_bytes(bytes.len())));
    out.push_str(&format!("lines:   {lines}\n"));
    out.push_str(&format!("format:  {format}\n"));
    out.push_str(&format!("scan:    {} ms\n", scan_us / 1000));
    out.push('\n');
    out.push_str("severity:\n");
    out.push_str(&fmt_level("DEBUG", c_debug, total));
    out.push_str(&fmt_level("INFO",  c_info,  total));
    out.push_str(&fmt_level("WARN",  c_warn,  total));
    out.push_str(&fmt_level("ERROR", c_error, total));
    out.push_str(&fmt_level("FATAL", c_fatal, total));
    if total == 0 {
        out.push_str("  (no severity keywords found — file may not be a log)\n");
    }

    let sample_count = n_pos as usize;
    if sample_count > 0 {
        out.push('\n');
        out.push_str("high-severity sample:\n");
        for &offset in &positions[..sample_count] {
            let (line_num, line) = extract_line_at(bytes, offset as usize);
            let display = truncate_line(line);
            out.push_str(&format!("  L{line_num}: {display}\n"));
        }
    }
    out
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

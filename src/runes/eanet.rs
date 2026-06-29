//! eanet — pcap network-triage rune. Streams a packet capture through the
//! `pcap_scan` kernel + `netflow` aggregator and reports the hosts that stand
//! out: top talkers by bytes, source fan-out (horizontal-scan signal), and
//! destination fan-in (DDoS/brute signal). The model narrates the standout
//! host and what it's doing; deep per-packet inspection stays Wireshark's job.
//!
//! Kernel/Rust split mirrors eatime: the kernel does the per-byte parse, Rust
//! does the hash bookkeeping. v1 reads classic-pcap Ethernet captures only.

use super::common::{open_capped, resolve_path, truncate_answer, format_scan_time, PathError};
use super::netflow::{self, Triage};
use super::output::{Anomaly, Category, RuneOutput, Source, Totals};
use super::{OutputSafety, Rune, RuneResult};
use std::io::Cursor;
use std::path::PathBuf;
use std::time::Instant;

const RUNE_VERSION: i64 = 1;
// A host is flagged only when it clears an absolute floor AND dwarfs the median
// — so quiet captures never manufacture an "anomaly".
const SCAN_RATIO: f64 = 5.0;
const SCAN_FLOOR: u64 = 20; // distinct destinations
const TALKER_RATIO: f64 = 10.0;

pub struct Eanet;
pub const RUNE: Eanet = Eanet;

impl Rune for Eanet {
    fn name(&self) -> &'static str { "eanet" }
    fn description(&self) -> &'static str {
        "Triage a packet capture (.pcap) via SIMD. Walks Ethernet/IPv4 TCP-UDP \
         packets and ranks hosts by bytes (top talkers), distinct destinations \
         contacted (source fan-out — a horizontal-scan signal), and distinct \
         sources connecting in (destination fan-in — a DDoS/brute signal), so \
         the model can name suspicious activity. For bulk triage, not deep \
         packet inspection. Reads classic-pcap Ethernet captures (not pcapng or \
         live capture). Args: [--json] <path>."
    }
    fn usage(&self) -> &'static str { "eanet [--json] <path>" }
    fn output_safety(&self) -> OutputSafety { OutputSafety::UntrustedQuoted }

    fn run(&self, args: &str) -> RuneResult {
        let t0 = Instant::now();
        let (path, json_mode) = match parse_args(args) {
            Ok(v) => v,
            Err((msg, json_mode)) => return result(error_output(&msg), None, json_mode, t0),
        };
        match execute(&path) {
            Ok((out, triage)) => result(out, Some(triage), json_mode, t0),
            Err(out) => result(out, None, json_mode, t0),
        }
    }
}

fn result(out: RuneOutput, triage: Option<Triage>, json_mode: bool, t0: Instant) -> RuneResult {
    // The narration path (build_narration_prompt) feeds the LLM `answer` ONLY,
    // while the REPL/web body shows `answer` + `details`. So `answer` is the
    // compact summary (stats + findings) the small model narrates cleanly —
    // exactly the shape eatime narrates well — and the verbose ranking tables
    // go in `details`, seen by the user but never drowning the model in grid.
    let (answer, details) = if json_mode {
        (out.to_json(), None)
    } else if let Some(err) = &out.error {
        (err.clone(), None)
    } else {
        let t = triage.as_ref().expect("success path carries the triage");
        (format_answer(&out, t), Some(format_details(t)))
    };
    RuneResult {
        answer: truncate_answer(&answer),
        details: details.map(|d| truncate_answer(&d)),
        success: out.success,
        timing_us: t0.elapsed().as_micros() as u64,
        structured: json_mode,
    }
}

fn parse_args(args: &str) -> Result<(String, bool), (String, bool)> {
    let mut json_mode = false;
    let mut path_tokens: Vec<&str> = Vec::new();
    for tok in args.split_whitespace() {
        match tok {
            "--json" => json_mode = true,
            other => path_tokens.push(other),
        }
    }
    if path_tokens.is_empty() {
        return Err(("usage: eanet [--json] <path>".to_string(), json_mode));
    }
    Ok((path_tokens.join(" "), json_mode))
}

fn error_output(msg: &str) -> RuneOutput {
    let mut out = RuneOutput::new("eanet", RUNE_VERSION);
    out.success = false;
    out.error = Some(msg.to_string());
    out
}

fn execute(path: &str) -> Result<(RuneOutput, Triage), RuneOutput> {
    let home = crate::home_dir().unwrap_or_else(|| PathBuf::from("/tmp"));
    let resolved = match resolve_path(path, &home) {
        Ok(p) => p,
        Err(PathError::OutsideAllowlist) => return Err(error_output("path rejected: outside allowlist (~ or /tmp only)")),
        Err(PathError::NotFound) => return Err(error_output("file not found")),
        Err(PathError::TooLarge(n)) => return Err(error_output(&format!("file too large: {n} bytes"))),
        Err(PathError::Io(e)) => return Err(error_output(&format!("io error: {e}"))),
    };
    // v1 reuses open_capped (audited allowlist + symlink-safety + 4 GB cap);
    // triage still streams the bytes in chunks. Disk-streaming for >RAM captures
    // is a later optimisation — triage already takes a Read.
    let bytes = match open_capped(&resolved, &home) {
        Ok(b) => b,
        Err(PathError::NotFound) => return Err(error_output("file not found")),
        Err(PathError::TooLarge(n)) => return Err(error_output(&format!("file too large: {n} bytes (4 GB limit)"))),
        Err(PathError::OutsideAllowlist) => return Err(error_output("path rejected: outside allowlist (~ or /tmp only)")),
        Err(PathError::Io(e)) => return Err(error_output(&format!("io error: {e}"))),
    };
    let resolved_str = resolved.to_string_lossy().into_owned();
    let n_bytes = bytes.len() as u64;

    let t_scan = Instant::now();
    let triage = match netflow::triage(&mut Cursor::new(bytes)) {
        Ok(t) => t,
        Err(e) => return Err(error_output(&e)),
    };
    let scan_us = t_scan.elapsed().as_micros() as u64;

    Ok((build_output(&triage, resolved_str, n_bytes, scan_us), triage))
}

fn ip(v: u32) -> String {
    let b = v.to_be_bytes();
    format!("{}.{}.{}.{}", b[0], b[1], b[2], b[3])
}

fn build_output(t: &Triage, path: String, n_bytes: u64, scan_us: u64) -> RuneOutput {
    let mut out = RuneOutput::new("eanet", RUNE_VERSION);
    out.source = Some(Source { path, bytes: n_bytes, format: "pcap".to_string() });
    out.totals = Totals { rows: t.packets, scan_us };
    out.categories = vec![
        Category { name: "tcp".to_string(), count: t.tcp },
        Category { name: "udp".to_string(), count: t.udp },
    ];

    // Likely horizontal scanner: top fan-out clears the floor and the median.
    if let Some(&(src, n_dst)) = t.top_fanout.first() {
        let base = t.fanout_median.max(1);
        let ratio = n_dst as f64 / base as f64;
        if n_dst >= SCAN_FLOOR && ratio >= SCAN_RATIO {
            out.anomalies.push(Anomaly {
                bucket: format!("{} fan-out", ip(src)),
                count: n_dst,
                baseline: base as f64,
                ratio,
                score: ratio,
            });
        }
    }
    // Heavy talker (exfil candidate): top conversation dwarfs the median.
    if let Some(&(src, dst, bytes)) = t.top_talkers.first() {
        let base = t.talker_median.max(1);
        let ratio = bytes as f64 / base as f64;
        if ratio >= TALKER_RATIO {
            out.anomalies.push(Anomaly {
                bucket: format!("{} -> {}", ip(src), ip(dst)),
                count: bytes,
                baseline: base as f64,
                ratio,
                score: ratio,
            });
        }
    }
    out
}

/// The compact, LLM-facing summary: stats + the prose findings headline. This
/// is what `build_narration_prompt` feeds the model — no ranking tables, so the
/// small model summarises the finding instead of drowning in grid (the same
/// compact shape eatime's text answer narrates well). The full tables live in
/// `format_details`, shown to the user but withheld from the model.
fn format_answer(out: &RuneOutput, t: &Triage) -> String {
    let src = out.source.as_ref().expect("build_output populates source on success");
    let mut buf = String::with_capacity(512);
    buf.push_str("NETWORK FLOW TRIAGE\n");
    buf.push_str(&format!("packets:       {}\n", t.packets));
    buf.push_str(&format!("bytes:         {}\n", format_bytes(src.bytes)));
    buf.push_str(&format!("conversations: {}\n", t.conversations));
    buf.push_str(&format!("protocols:     tcp {} / udp {}\n", t.tcp, t.udp));
    buf.push_str(&format!("scan:          {}\n\n", format_scan_time(out.totals.scan_us)));

    if t.packets == 0 {
        buf.push_str("(no IPv4 TCP/UDP packets found)\n");
        return buf;
    }
    if out.anomalies.is_empty() {
        // No flagged scan/exfil — still hand the model one concrete headline.
        if let Some(&(s, d, b)) = t.top_talkers.first() {
            buf.push_str(&format!(
                "findings:\n  • no scan or exfil signal stood out; the busiest conversation was {} → {} ({})\n",
                ip(s), ip(d), format_bytes(b)
            ));
        } else {
            buf.push_str("findings:\n  • no notable activity\n");
        }
    } else {
        buf.push_str(&format_findings(&out.anomalies));
    }
    buf
}

/// The verbose ranking tables — shown to the user (REPL/web append `details`
/// after the answer) but NOT fed to the narration model.
fn format_details(t: &Triage) -> String {
    let mut buf = String::with_capacity(512);
    buf.push_str("top source fan-out (distinct destinations — scan signal):\n");
    for &(s, c) in &t.top_fanout {
        buf.push_str(&format!("  {}   {} destinations\n", ip(s), c));
    }
    buf.push('\n');
    buf.push_str("top talkers (src -> dst by bytes):\n");
    for &(s, d, b) in &t.top_talkers {
        buf.push_str(&format!("  {} -> {}   {}\n", ip(s), ip(d), format_bytes(b)));
    }
    buf.push('\n');
    buf.push_str("top destination fan-in (distinct sources):\n");
    for &(d, c) in &t.top_fanin {
        buf.push_str(&format!("  {}   {} sources\n", ip(d), c));
    }
    buf
}

/// Headline findings as plain-English prose derived from the flagged anomalies
/// — the host and the concrete magnitude, so narration names the figure.
fn format_findings(anomalies: &[Anomaly]) -> String {
    if anomalies.is_empty() {
        return String::new();
    }
    let mut s = String::from("findings:\n");
    for a in anomalies {
        if let Some(host) = a.bucket.strip_suffix(" fan-out") {
            // The absolute count is the concrete, narratable signal — the ratio
            // (vs a tiny median) reads as an absurd number and confuses the
            // small model, so we lead with the magnitude instead.
            s.push_str(&format!(
                "  • {host} contacted {} distinct destinations — likely a horizontal scan\n",
                a.count
            ));
        } else {
            s.push_str(&format!(
                "  • {} moved {} to a single destination — heavy talker, possible exfiltration\n",
                a.bucket,
                format_bytes(a.count)
            ));
        }
    }
    s.push('\n');
    s
}

fn format_bytes(n: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
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

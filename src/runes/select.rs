//! Deterministic rune selection for the file-drop analyst.
//!
//! When a user drops a file into the web UI, the gesture itself is the
//! "analyze this" intent — there is no model decision to make. So the choice
//! of WHICH rune to run must be deterministic and content-driven, never a
//! model call: that keeps it arch-independent and bulletproof on the Pi
//! (where the 2B's autonomous tool-call decision is unreliable).
//!
//! Selection is extension-first, with a timestamp sniff to split ambiguous
//! log/text files between `eatime` (timing-spike forensics) and `ealog`
//! (severity counts). `eadiff` is intentionally excluded — it consumes two
//! rune outputs, not a dropped file.

use crate::runes::{Rune, RUNES};

/// Pick the rune to analyze a dropped file. Returns `None` when no rune fits;
/// the caller surfaces a friendly "not sure how to analyze this" message.
pub fn pick_rune(filename: &str, bytes: &[u8]) -> Option<&'static (dyn Rune + Sync)> {
    let name = pick_rune_name(filename, bytes)?;
    RUNES.iter().find(|r| r.name() == name).copied()
}

/// The selection decision as a pure (filename, bytes) → rune-name function,
/// testable without the generated `RUNES` registry.
pub fn pick_rune_name(filename: &str, bytes: &[u8]) -> Option<&'static str> {
    match extension(filename).as_deref() {
        Some("csv") => Some("eacrunch"),
        Some("jsonl") | Some("ndjson") | Some("json") => Some("eajson"),
        Some("parquet") => Some("eaparquet"),
        Some("sql") => Some("easql"),
        // Logs and untyped text: forensics-first — prefer timing-spike
        // detection when the file carries timestamps, else severity counts.
        Some("log") | Some("txt") | None => {
            Some(if has_timestamp(bytes) { "eatime" } else { "ealog" })
        }
        _ => None,
    }
}

/// Human-readable list of the file types the file-drop analyst accepts, kept
/// right next to `pick_rune_name` so the two can't drift — a drift-guard test
/// asserts every extension routed above appears in this string. Surfaced by the
/// "unsupported file" message, the REPL `/help`, and the CLI.
pub fn supported_formats() -> &'static str {
    "  • CSV (.csv) — row counts, group-by, aggregates\n  \
     • JSON / JSON Lines (.json, .jsonl, .ndjson) — field stats\n  \
     • Parquet (.parquet) — columnar summaries\n  \
     • SQL (.sql) — schema / query inspection\n  \
     • Logs & text (.log, .txt, or no extension) — timing spikes or severity counts"
}

/// Default rune arguments for a file-drop analysis (flags only; the file path
/// is appended by the caller). For `eatime` we lead with spike detection —
/// the forensic "when did the rate break" story — rather than the plain
/// hour-of-day histogram. Other runes need no flags by default.
pub fn default_args(rune_name: &str) -> &'static str {
    match rune_name {
        "eatime" => "--bucket series",
        _ => "",
    }
}

/// Lowercased extension (no dot) of a filename's basename, or `None` if it has
/// no extension.
fn extension(filename: &str) -> Option<String> {
    let base = filename.rsplit(['/', '\\']).next().unwrap_or(filename);
    let (stem, ext) = base.rsplit_once('.')?;
    if stem.is_empty() {
        return None; // dotfile like ".log" — treat as no extension
    }
    Some(ext.to_ascii_lowercase())
}

/// True if the first chunk of the file contains an ISO-8601 date
/// (`YYYY-MM-DD`) or a Common Log Format date (`DD/Mon/YYYY`). Scans only a
/// bounded prefix — one timestamped line near the top is enough to classify.
fn has_timestamp(bytes: &[u8]) -> bool {
    let prefix = &bytes[..bytes.len().min(8192)];
    let d = |b: u8| b.is_ascii_digit();
    let a = |b: u8| b.is_ascii_alphabetic();

    // ISO-8601 date: dddd-dd-dd
    if prefix.len() >= 10 {
        for w in prefix.windows(10) {
            if d(w[0]) && d(w[1]) && d(w[2]) && d(w[3]) && w[4] == b'-'
                && d(w[5]) && d(w[6]) && w[7] == b'-' && d(w[8]) && d(w[9])
            {
                return true;
            }
        }
    }
    // CLF date: dd/Mon/dddd  (e.g. 10/Oct/2000)
    if prefix.len() >= 11 {
        for w in prefix.windows(11) {
            if d(w[0]) && d(w[1]) && w[2] == b'/'
                && a(w[3]) && a(w[4]) && a(w[5]) && w[6] == b'/'
                && d(w[7]) && d(w[8]) && d(w[9]) && d(w[10])
            {
                return true;
            }
        }
    }
    false
}

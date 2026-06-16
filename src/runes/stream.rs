//! Event-stream extraction — the shared front half of every
//! timestamp-driven rune: kernel position scan → epoch-seconds decode →
//! chronological grid sizing.
//!
//! Extracted from `eatime` so multi-file runes (`eacorrelate`) can build
//! epoch streams from several files without re-implementing the scan
//! orchestration. `eatime` remains the single-file consumer; bucketing
//! and formatting stay rune-local.

use super::timekey::{clf_bytes_to_seconds, iso_bytes_to_seconds, json_epoch_bytes_to_seconds, syslog_bytes_to_seconds};
use crate::kernels::ffi;
use std::time::Instant;

/// Caller-side cap on positions stored from the kernel. Each emitted
/// position is 4 bytes; ~16M timestamps fits in 64 MB which is well
/// inside the 4 GB input limit. The kernel saturates at this cap;
/// past that, the rune reports the cap was hit.
pub const MAX_POSITIONS: usize = 16_000_000;

/// Timestamp grammar. `Iso` = `YYYY-MM-DD[T| ]HH:MM:SS` (timestamp_scan);
/// `Clf` = `[dd/MMM/yyyy:hh:mm:ss` Common Log Format (clf_scan); `Syslog` =
/// classic BSD `MMM DD HH:MM:SS` (syslog_scan, yearless — fixed reference year);
/// `JsonEpoch` = a numeric Unix epoch under a JSON timestamp key
/// (`"ts":1749600000`, json_epoch_scan) — ISO-string JSON timestamps are `Iso`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Format { Iso, Clf, Syslog, JsonEpoch }

impl Format {
    pub fn tag(self) -> &'static str {
        match self {
            Format::Iso       => "iso8601",
            Format::Clf       => "clf",
            Format::Syslog    => "syslog",
            Format::JsonEpoch => "json-epoch",
        }
    }
}

pub struct ScanResult {
    pub positions: Vec<i32>,
    pub scan_us:   u64,
}

/// FFI signature shared by every position-scan kernel.
type ScanFn = unsafe fn(*const u8, i32, *mut i32, i32, *mut i32, *mut u8);

/// Run the format's position kernel over `bytes`, capturing up to
/// `max_positions` hits.
pub fn scan_for(bytes: &[u8], format: Format, max_positions: usize) -> ScanResult {
    if bytes.is_empty() {
        return ScanResult { positions: Vec::new(), scan_us: 0 };
    }
    let kernel: ScanFn = match format {
        Format::Iso       => ffi::timestamp_scan,
        Format::Clf       => ffi::clf_scan,
        Format::Syslog    => ffi::syslog_scan,
        Format::JsonEpoch => ffi::json_epoch_scan,
    };
    let t_scan = Instant::now();
    let mut positions = vec![0i32; max_positions];
    let mut n_positions = 0i32;
    let mut scratch = [0u8; 16];
    unsafe {
        kernel(
            bytes.as_ptr(), bytes.len() as i32,
            positions.as_mut_ptr(), max_positions as i32, &mut n_positions,
            scratch.as_mut_ptr(),
        );
    }
    positions.truncate(n_positions as usize);
    ScanResult { positions, scan_us: t_scan.elapsed().as_micros() as u64 }
}

/// Sniff the timestamp grammar by running both kernels over a head sample
/// and picking whichever matches more. Using the kernels themselves means
/// the sniff can never disagree with the scan that follows.
pub fn detect_format(bytes: &[u8]) -> Format {
    const SNIFF_BYTES: usize = 64 * 1024;
    const SNIFF_CAP: usize = 4096;
    let head = &bytes[..bytes.len().min(SNIFF_BYTES)];
    let iso = scan_for(head, Format::Iso, SNIFF_CAP).positions.len();
    let clf = scan_for(head, Format::Clf, SNIFF_CAP).positions.len();
    let sys = scan_for(head, Format::Syslog, SNIFF_CAP).positions.len();
    let json = scan_for(head, Format::JsonEpoch, SNIFF_CAP).positions.len();
    // Pick the grammar that matches most. JSON-epoch wins outright when it leads
    // (its `"ts":<digit>` anchor never fires on ISO-string JSON, which has no
    // digit after the colon). Otherwise ISO wins ties (the most common), then
    // CLF over syslog — a real CLF/ISO line never matches the syslog grammar.
    if json > iso && json > clf && json > sys { Format::JsonEpoch }
    else if sys > iso && sys > clf { Format::Syslog }
    else if clf > iso { Format::Clf }
    else { Format::Iso }
}

/// Decode each kernel position to epoch-seconds with the format's decoder;
/// positions that don't decode (truncated / out-of-range) are dropped.
pub fn positions_to_epochs(bytes: &[u8], positions: &[i32], format: Format) -> Vec<i64> {
    let mut epochs: Vec<i64> = Vec::with_capacity(positions.len());
    for &pos in positions {
        let p = pos as usize;
        if p >= bytes.len() { continue; }
        let decoded = match format {
            Format::Iso       => iso_bytes_to_seconds(&bytes[p..]),
            Format::Clf       => clf_bytes_to_seconds(&bytes[p..]),
            Format::Syslog    => syslog_bytes_to_seconds(&bytes[p..]),
            Format::JsonEpoch => json_epoch_bytes_to_seconds(&bytes[p..]),
        };
        if let Some(secs) = decoded { epochs.push(secs); }
    }
    epochs
}

/// Candidate chronological bucket widths in seconds: 1s, 5s, 10s, 30s,
/// 1m, 5m, 10m, 30m, 1h, 6h, 12h, 1d, 1w. `auto_width` snaps up to the
/// first width that keeps the series near `target_buckets`.
pub const NICE_WIDTHS: [i64; 13] =
    [1, 5, 10, 30, 60, 300, 600, 1800, 3600, 21600, 43200, 86400, 604800];

pub fn auto_width(span_secs: i64, target_buckets: i64) -> i64 {
    if span_secs <= 0 { return 1; }
    let raw = span_secs / target_buckets;
    for w in NICE_WIDTHS {
        if w >= raw { return w; }
    }
    NICE_WIDTHS[NICE_WIDTHS.len() - 1]
}

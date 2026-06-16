//! Error sub-streams: per-log, the subset of events that are "errors", surfaced
//! as a second correlatable stream alongside the log's all-events stream.
//!
//! ISO/syslog logs use ERROR/FATAL keyword matches (`log_level_scan`); CLF
//! access logs use HTTP 5xx status matches (`clf_status_scan`). Each kernel
//! emits byte offsets of its matches; this module maps every offset back to its
//! line's timestamp — the greatest timestamp position <= the offset, so a
//! stack-trace line without its own timestamp attributes to the last stamped
//! line above it — and decodes that to an epoch.

use super::timekey::{
    clf_bytes_to_seconds, iso_bytes_to_seconds, json_epoch_bytes_to_seconds,
    syslog_bytes_to_seconds,
};
use crate::kernels::ffi;

/// Cap on recorded error positions per file (4 MB of i32).
const MAX_ERROR_POSITIONS: usize = 1_000_000;

/// ERROR/FATAL keyword events for an ISO/syslog log, mapped to line epochs.
pub fn iso_errors(bytes: &[u8], ts_positions: &[i32]) -> Vec<i64> {
    keyword_errors(bytes, ts_positions, iso_bytes_to_seconds)
}

/// ERROR/FATAL keyword events for a classic BSD syslog log (`MMM DD HH:MM:SS`),
/// mapped to each line's yearless-syslog epoch.
pub fn syslog_errors(bytes: &[u8], ts_positions: &[i32]) -> Vec<i64> {
    keyword_errors(bytes, ts_positions, syslog_bytes_to_seconds)
}

/// Error events for a JSON/ndjson log, mapped to each line's numeric-epoch
/// timestamp. Covers both severity conventions: string levels
/// (`"level":"error"`, zap/zerolog) via the ERROR/FATAL keyword scan, and
/// numeric levels (`"level":50`, pino/bunyan) via `json_level_scan`. A line
/// uses one convention or the other, so the two position sets are disjoint;
/// their epochs are unioned (bucketing is a histogram — order-independent).
pub fn json_errors(bytes: &[u8], ts_positions: &[i32]) -> Vec<i64> {
    let mut epochs = keyword_errors(bytes, ts_positions, json_epoch_bytes_to_seconds);
    epochs.extend(numeric_level_errors(bytes, ts_positions));
    epochs
}

/// Numeric error-or-worse level events (`"level":50`/`60`, pino/bunyan) via
/// `json_level_scan`, mapped to line epochs. The kernel emits only level-digit
/// offsets whose value is >= 50, so this is the numeric twin of the keyword
/// ERROR/FATAL sub-stream.
fn numeric_level_errors(bytes: &[u8], ts_positions: &[i32]) -> Vec<i64> {
    if ts_positions.is_empty() {
        return Vec::new();
    }
    let mut positions = vec![0i32; MAX_ERROR_POSITIONS];
    let mut n_positions = 0i32;
    let mut scratch = [0u8; 16];
    unsafe {
        ffi::json_level_scan(
            bytes.as_ptr(), bytes.len() as i32,
            positions.as_mut_ptr(), MAX_ERROR_POSITIONS as i32, &mut n_positions,
            scratch.as_mut_ptr(),
        );
    }
    positions.truncate(n_positions as usize);
    map_to_epochs(bytes, ts_positions, &positions, json_epoch_bytes_to_seconds)
}

/// ERROR/FATAL keyword sub-stream via `log_level_scan`, mapped to line epochs
/// with `decode` (ISO bytes, or JSON numeric epoch). Shared by the text-log and
/// JSON paths — they differ only in how the line's timestamp is decoded.
fn keyword_errors(bytes: &[u8], ts_positions: &[i32], decode: fn(&[u8]) -> Option<i64>) -> Vec<i64> {
    if ts_positions.is_empty() {
        return Vec::new();
    }
    let mut counts = [0i32; 6];
    let mut positions = vec![0i32; MAX_ERROR_POSITIONS];
    let mut n_positions = 0i32;
    let mut scratch = [0u8; 16];
    unsafe {
        ffi::log_level_scan(
            bytes.as_ptr(), bytes.len() as i32,
            counts.as_mut_ptr(),
            positions.as_mut_ptr(), MAX_ERROR_POSITIONS as i32, &mut n_positions,
            scratch.as_mut_ptr(),
        );
    }
    positions.truncate(n_positions as usize);
    map_to_epochs(bytes, ts_positions, &positions, decode)
}

/// HTTP 5xx events for a CLF access log, mapped to line epochs. The kernel emits
/// the status-digit offset of each `" 5DD "` field; that offset always falls
/// after its line's `[` timestamp and before the next line's, so the same
/// greatest-timestamp-<=-offset attribution applies.
pub fn clf_errors(bytes: &[u8], ts_positions: &[i32]) -> Vec<i64> {
    if ts_positions.is_empty() {
        return Vec::new();
    }
    let mut positions = vec![0i32; MAX_ERROR_POSITIONS];
    let mut n_positions = 0i32;
    let mut scratch = [0u8; 16];
    unsafe {
        ffi::clf_status_scan(
            bytes.as_ptr(), bytes.len() as i32,
            positions.as_mut_ptr(), MAX_ERROR_POSITIONS as i32, &mut n_positions,
            scratch.as_mut_ptr(),
        );
    }
    positions.truncate(n_positions as usize);
    map_to_epochs(bytes, ts_positions, &positions, clf_bytes_to_seconds)
}

/// Attribute each error offset to its line's timestamp and decode to epoch
/// seconds. `decode` reads a timestamp from the start of the given slice.
fn map_to_epochs(
    bytes: &[u8], ts_positions: &[i32], err_positions: &[i32],
    decode: fn(&[u8]) -> Option<i64>,
) -> Vec<i64> {
    let mut epochs: Vec<i64> = Vec::with_capacity(err_positions.len());
    for &err_pos in err_positions {
        let idx = ts_positions.partition_point(|&t| t <= err_pos);
        if idx == 0 { continue; } // error before the first timestamp
        let t = ts_positions[idx - 1] as usize;
        if let Some(secs) = decode(&bytes[t..]) {
            epochs.push(secs);
        }
    }
    epochs
}

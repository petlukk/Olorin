//! Data-plane SIMD kernel FFI wrappers — the kernels that data-analysis
//! runes call (csv_scan, jsonl_struct_scan, log_level_scan,
//! timestamp_scan, f32_stats, f64_stats). Split out of `ffi.rs` to keep
//! that file under the 500-LOC cap.
//!
//! Re-exported from `ffi` so existing `kernels::ffi::<name>` call sites
//! continue to compile.

use super::ffi::k;

/// Scan CSV bytes, writing comma and newline byte-indices into the two
/// output arrays and the counts into `out_n_comma` / `out_n_newline`.
///
/// # Safety
/// - `text` must point to `len` readable bytes.
/// - `out_commas` and `out_newlines` must each be valid for at least `len`
///   writable `i32` elements (worst case is every byte being a delimiter).
/// - `out_n_comma` and `out_n_newline` must each be valid for one writable
///   `i32`. Passing non-null dangling pointers is UB even when `len == 0`.
/// - `scratch` must be valid for 16 writable bytes. The kernel overwrites
///   it once per 16-byte chunk; the contents after the call are unspecified.
pub unsafe fn csv_scan(
    text: *const u8, len: i32,
    out_commas: *mut i32, out_newlines: *mut i32,
    out_n_comma: *mut i32, out_n_newline: *mut i32,
    scratch: *mut u8,
) {
    (k().csv_scan)(text, len, out_commas, out_newlines, out_n_comma, out_n_newline, scratch);
}

/// Single-pass JSONL structural scan: writes positions of newlines, quotes,
/// colons, commas, and backslashes across `text` to five caller-allocated
/// arrays. Backslash positions allow callers to filter out escaped quotes
/// (`\"` inside a JSON string is *not* a structural quote).
///
/// # Safety
/// - `text` must point to `len` readable bytes.
/// - Each `out_*` array must be writable for `len` `i32` elements (worst
///   case: every byte is a structural character of that type).
/// - `out_n_*` must each be valid for one writable `i32`.
/// - `scratch` must be writable for 16 bytes; contents after the call are
///   unspecified (overwritten once per 16-byte chunk).
pub unsafe fn jsonl_struct_scan(
    text: *const u8, len: i32,
    out_newlines: *mut i32, out_quotes: *mut i32,
    out_colons: *mut i32,   out_commas: *mut i32, out_backslashes: *mut i32,
    out_n_newline: *mut i32, out_n_quote: *mut i32,
    out_n_colon: *mut i32,   out_n_comma: *mut i32, out_n_backslash: *mut i32,
    scratch: *mut u8,
) {
    (k().jsonl_struct_scan)(
        text, len,
        out_newlines, out_quotes, out_colons, out_commas, out_backslashes,
        out_n_newline, out_n_quote, out_n_colon, out_n_comma, out_n_backslash,
        scratch,
    );
}

/// Multi-keyword severity scanner. Counts word-bounded occurrences of
/// DEBUG, INFO, WARN, ERROR, FATAL plus newline bytes in `text`, and
/// optionally records the byte offsets of valid ERROR / FATAL matches
/// (in scan order, capped at `max_positions`). Word boundary = bounded
/// by one of: space, tab, newline, CR, '[', ']', '"', ':'. Start/end of
/// buffer count as implicit delimiters.
///
/// # Safety
/// - `text` must point to `len` readable bytes.
/// - `out_counts` must be valid for **six** writable `i32` elements; the
///   kernel writes [DEBUG, INFO, WARN, ERROR, FATAL, NEWLINES] in that
///   order and always zeroes all six before counting (safe to call with
///   `len == 0`).
/// - `out_positions` must be valid for `max_positions` writable `i32`
///   elements. May be dangling when `max_positions == 0`.
/// - `out_n_positions` must be valid for one writable `i32`; the kernel
///   zeroes it before counting and writes the final number of recorded
///   positions (`<= max_positions`).
/// - `scratch` must be writable for 16 bytes; contents after the call
///   are unspecified. Touched only when `max_positions > 0`, but must
///   still be a valid pointer in that branch.
pub unsafe fn log_level_scan(
    text: *const u8, len: i32,
    out_counts: *mut i32,
    out_positions: *mut i32, max_positions: i32, out_n_positions: *mut i32,
    scratch: *mut u8,
) {
    (k().log_level_scan)(
        text, len,
        out_counts,
        out_positions, max_positions, out_n_positions,
        scratch,
    );
}

/// Scan `text` for ISO-8601 timestamp prefixes (`YYYY-MM-DDT`) and
/// emit each match's start byte offset. Used by `eatime` — the caller
/// extracts HH:MM:SS from `text[offset+11..offset+19]` after each hit
/// (the kernel deliberately does not validate the trailing 8 bytes,
/// keeping the SIMD body to one tight pass and the tail safe).
///
/// # Safety
/// - `text` must point to `len` readable bytes (any value of `len >= 0`
///   is safe; the kernel handles tiny buffers via a scalar fallback).
/// - `out_positions` must be writable for at least `max_positions` `i32`s
///   when `max_positions > 0`. The kernel clamps writes at that capacity.
/// - `out_n_positions` must point to one writable `i32` (always set to 0
///   on entry, then incremented per emitted position).
/// - `scratch` must be writable for 16 bytes; contents are unspecified
///   after the call. Touched only when the SIMD body runs.
pub unsafe fn timestamp_scan(
    text: *const u8, len: i32,
    out_positions: *mut i32, max_positions: i32, out_n_positions: *mut i32,
    scratch: *mut u8,
) {
    (k().timestamp_scan)(
        text, len,
        out_positions, max_positions, out_n_positions,
        scratch,
    );
}

/// Scan `text` for Common Log Format timestamps (`[dd/MMM/yyyy:hh:mm:ss`,
/// the Apache/nginx access-log default) and emit each match's `[` byte
/// offset. Used by `eatime` — the caller decodes the fixed-width fields
/// (textual month, zone) from `text[offset..offset+21]`.
///
/// # Safety
/// Identical contract to [`timestamp_scan`]: `text` readable for `len`
/// bytes; `out_positions` writable for `max_positions` `i32`s when
/// positive; `out_n_positions` a writable `i32` (zeroed on entry);
/// `scratch` writable for 16 bytes (touched only when the SIMD body runs).
pub unsafe fn clf_scan(
    text: *const u8, len: i32,
    out_positions: *mut i32, max_positions: i32, out_n_positions: *mut i32,
    scratch: *mut u8,
) {
    (k().clf_scan)(
        text, len,
        out_positions, max_positions, out_n_positions,
        scratch,
    );
}

/// Streaming stats over `len` f32 elements. Writes count, sum, min, max.
///
/// # Safety
/// - `data` must point to `len` readable `f32` elements; may be dangling
///   when `len == 0` (the kernel does not dereference it in that case).
/// - `out_count`, `out_sum`, `out_min`, `out_max` must each be valid for
///   one writable element **even when `len == 0`** — the kernel always
///   writes zeros to all four before the early return.
pub unsafe fn f32_stats(
    data: *const f32, len: i32,
    out_count: *mut i32,
    out_sum: *mut f32, out_min: *mut f32, out_max: *mut f32,
) {
    (k().f32_stats)(data, len, out_count, out_sum, out_min, out_max);
}

/// f64 streaming stats — the double-precision counterpart to `f32_stats`.
/// Used by eaparquet for INT64 statistics aggregation where f32's
/// 24-bit mantissa would lose precision on large integer values.
///
/// # Safety
/// Same as `f32_stats`: data must be valid for `len` reads (or len==0,
/// in which case data is not dereferenced); the four out pointers must
/// each be valid for one write.
pub unsafe fn f64_stats(
    data: *const f64, len: i32,
    out_count: *mut i32,
    out_sum: *mut f64, out_min: *mut f64, out_max: *mut f64,
) {
    (k().f64_stats)(data, len, out_count, out_sum, out_min, out_max);
}

//! timekey — shared ISO-8601 ↔ epoch-seconds conversion for runes.
//!
//! Both `eadiff` (timestamp range shifts) and `eatime` (chronological
//! series bucketing) need to turn a `YYYY-MM-DDTHH:MM:SS` instant into a
//! single sortable integer and back. Keeping one implementation here
//! prevents two divergent copies of the civil-date / leap-year math —
//! exactly the silent-divergence class the rune differential audit hunts.
//!
//! Epoch origin is `2000-01-01T00:00:00`. The absolute value is
//! arbitrary; callers only ever use differences and bucket indices, and
//! `seconds_to_iso(iso_to_seconds(t)) == t` round-trips for any valid
//! in-range timestamp.

/// Parse `YYYY-MM-DDTHH:MM:SS` from the front of `s` to seconds since
/// 2000-01-01. Any trailing timezone / fractional part is ignored.
/// Returns `None` on any digit failure or out-of-range component.
pub fn iso_to_seconds(s: &str) -> Option<i64> {
    iso_bytes_to_seconds(s.as_bytes())
}

/// Byte-slice form of [`iso_to_seconds`]: parses the 19-byte prefix of
/// `b`. Lets `eatime` decode directly at a kernel-emitted position
/// (`&bytes[pos..]`) without allocating a `&str`.
pub fn iso_bytes_to_seconds(b: &[u8]) -> Option<i64> {
    if b.len() < 19 { return None; }
    let year:   i64 = parse_uint(&b[0..4])?  as i64;
    if b[4] != b'-' { return None; }
    let month:  i64 = parse_uint(&b[5..7])?  as i64;
    if b[7] != b'-' { return None; }
    let day:    i64 = parse_uint(&b[8..10])? as i64;
    // 'T' (RFC-3339) or ' ' (Postgres/MySQL/Python-logging/OpenStack style).
    if b[10] != b'T' && b[10] != b' ' { return None; }
    let hour:   i64 = parse_uint(&b[11..13])? as i64;
    if b[13] != b':' { return None; }
    let minute: i64 = parse_uint(&b[14..16])? as i64;
    if b[16] != b':' { return None; }
    let second: i64 = parse_uint(&b[17..19])? as i64;
    if month < 1 || month > 12 || day < 1 || day > 31 { return None; }
    if hour > 23 || minute > 59 || second > 60 { return None; }
    let days = days_since_2000(year, month, day);
    Some(days * 86400 + hour * 3600 + minute * 60 + second)
}

/// Inverse of [`iso_to_seconds`]: render seconds-since-2000 as
/// `YYYY-MM-DDTHH:MM:SS`. Used for chronological bucket labels.
pub fn seconds_to_iso(secs: i64) -> String {
    let days_2000 = secs.div_euclid(86400);
    let tod       = secs.rem_euclid(86400);
    // Hinnant `civil_from_days` expects days since 1970-01-01; our epoch
    // is 2000-01-01, which is 10957 days later.
    let (y, m, d) = civil_from_days(days_2000 + 10957);
    let h  = tod / 3600;
    let mi = (tod % 3600) / 60;
    let s  = tod % 60;
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{s:02}")
}

/// Render Unix epoch seconds (since 1970-01-01) as ISO-8601. Parquet/Arrow
/// timestamps are Unix-epoch, unlike this module's internal 2000-epoch
/// [`seconds_to_iso`]. A trailing `Z` is appended when the source declared
/// the instant UTC-adjusted (Parquet `isAdjustedToUTC`); a naive local
/// instant gets none.
pub fn unix_seconds_to_iso(secs: i64, utc: bool) -> String {
    let days = secs.div_euclid(86400);
    let tod  = secs.rem_euclid(86400);
    let (y, m, d) = civil_from_days(days);
    let h  = tod / 3600;
    let mi = (tod % 3600) / 60;
    let s  = tod % 60;
    let zone = if utc { "Z" } else { "" };
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{s:02}{zone}")
}

/// Parse a Common Log Format instant `[dd/MMM/yyyy:hh:mm:ss` from the
/// front of `b` to seconds since 2000-01-01. The trailing ` ±HHMM` zone
/// is ignored — within a single log it is constant and cancels out of
/// bucket indices, so bucketing happens on the log's own wall clock.
/// Returns `None` on any malformed field.
pub fn clf_bytes_to_seconds(b: &[u8]) -> Option<i64> {
    if b.len() < 21 { return None; }
    if b[0] != b'[' { return None; }
    let day:   i64 = parse_uint(&b[1..3])? as i64;
    if b[3] != b'/' { return None; }
    let month: i64 = month_from_name(&b[4..7])? as i64;
    if b[7] != b'/' { return None; }
    let year:  i64 = parse_uint(&b[8..12])? as i64;
    if b[12] != b':' { return None; }
    let hour:   i64 = parse_uint(&b[13..15])? as i64;
    if b[15] != b':' { return None; }
    let minute: i64 = parse_uint(&b[16..18])? as i64;
    if b[18] != b':' { return None; }
    let second: i64 = parse_uint(&b[19..21])? as i64;
    if day < 1 || day > 31 { return None; }
    if hour > 23 || minute > 59 || second > 60 { return None; }
    let days = days_since_2000(year, month, day);
    Some(days * 86400 + hour * 3600 + minute * 60 + second)
}

/// Three-letter English month abbreviation → 1..12, case-insensitive.
/// Folds each byte with `| 0x20` (the same letter-fold idiom as
/// `log_level_scan`) so `Oct`/`OCT`/`oct` all match. `None` if not a month.
fn month_from_name(b: &[u8]) -> Option<u32> {
    const MONTHS: [&[u8; 3]; 12] = [
        b"jan", b"feb", b"mar", b"apr", b"may", b"jun",
        b"jul", b"aug", b"sep", b"oct", b"nov", b"dec",
    ];
    let folded = [b[0] | 0x20, b[1] | 0x20, b[2] | 0x20];
    for (i, m) in MONTHS.iter().enumerate() {
        if folded == **m {
            return Some(i as u32 + 1);
        }
    }
    None
}

fn parse_uint(s: &[u8]) -> Option<u32> {
    let mut acc: u32 = 0;
    for &b in s {
        if !(b'0'..=b'9').contains(&b) { return None; }
        acc = acc * 10 + (b - b'0') as u32;
    }
    Some(acc)
}

fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

/// Days from 2000-01-01 to `year-month-day`. Negative for pre-2000
/// dates; callers only use the difference of two values.
fn days_since_2000(year: i64, month: i64, day: i64) -> i64 {
    const MONTH_DAYS: [i64; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut total: i64 = 0;
    if year >= 2000 {
        for y in 2000..year {
            total += if is_leap(y) { 366 } else { 365 };
        }
    } else {
        for y in year..2000 {
            total -= if is_leap(y) { 366 } else { 365 };
        }
    }
    for m in 0..(month - 1) as usize {
        total += MONTH_DAYS[m];
    }
    if month > 2 && is_leap(year) { total += 1; }
    total + (day - 1)
}

/// Howard Hinnant's `civil_from_days`: days since 1970-01-01 →
/// (year, month, day). Exact for the full proleptic Gregorian range.
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097;                       // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y   = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp  = (5 * doy + 2) / 153;                      // [0, 11]
    let d   = doy - (153 * mp + 2) / 5 + 1;            // [1, 31]
    let m   = if mp < 10 { mp + 3 } else { mp - 9 };   // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

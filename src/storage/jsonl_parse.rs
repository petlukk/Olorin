//! Pure helpers for parsing JSON Lines from the SIMD `jsonl_struct_scan`
//! output. Stateless — every function takes byte slices + integer indices
//! and returns positions or decoded values. No allocations except where
//! decoding inherently requires one (`decode_byte_array`, escape set).
//!
//! Lives under `storage/` next to `json.rs` because it operates on raw
//! bytes from the same problem domain. Kept out of `runes/` so build.rs's
//! "every file in runes/ must be a Rune" rule isn't tripped.

use std::collections::HashSet;

/// Whether a parsed scalar value (post-trim) is a number, text, bool, or
/// should be skipped (null / unknown).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarKind { Number, Text, Bool, Skip }

const BYTE_ARRAY_MAX_LEN: usize = 8192;

/// Build the set of quote positions that are escape-hidden (preceded by an
/// odd number of consecutive backslashes). Walk backslashes in order and
/// mark the immediately-following byte as escape-hidden iff that byte is a
/// known structural quote.
pub fn build_escaped_quote_set(quotes: &[i32], backslashes: &[i32]) -> HashSet<i32> {
    let mut escaped = HashSet::new();
    if backslashes.is_empty() { return escaped; }
    let quote_set: HashSet<i32> = quotes.iter().copied().collect();

    let mut i = 0usize;
    while i < backslashes.len() {
        let mut run_end = i;
        while run_end + 1 < backslashes.len()
            && backslashes[run_end + 1] == backslashes[run_end] + 1
        {
            run_end += 1;
        }
        let run_len = run_end - i + 1;
        let next_pos = backslashes[run_end] + 1;
        if (run_len % 2 == 1) && quote_set.contains(&next_pos) {
            escaped.insert(next_pos);
        }
        i = run_end + 1;
    }
    escaped
}

/// Find the position of the matching close bracket for `open` starting at
/// `start` (which must point at `open`). Tracks nested depth and skips
/// brackets inside string literals (with escape awareness). Returns the
/// position of the matching close, or None if the line ends first.
pub fn find_matching(bytes: &[u8], start: usize, end: usize, open: u8, close: u8) -> Option<usize> {
    let mut depth = 0i32;
    let mut i = start;
    let mut in_string = false;
    let mut prev_backslash = false;
    while i < end {
        let b = bytes[i];
        if in_string {
            if b == b'\\' && !prev_backslash {
                prev_backslash = true;
            } else {
                if b == b'"' && !prev_backslash { in_string = false; }
                prev_backslash = false;
            }
        } else if b == b'"' {
            in_string = true;
        } else if b == open {
            depth += 1;
        } else if b == close {
            depth -= 1;
            if depth == 0 { return Some(i); }
        }
        i += 1;
    }
    None
}

/// Decode a JSON array of integers (0..=255) as a UTF-8 string with `�`
/// replacement on invalid sequences. Used for systemd-style binary MESSAGE
/// fields that arrive as `[27,91,51,...]`. Returns None when the array
/// isn't well-formed bytes (any non-integer element, out-of-range value,
/// empty, or longer than BYTE_ARRAY_MAX_LEN).
pub fn decode_byte_array(arr: &[u8]) -> Option<String> {
    if arr.len() < 2 || arr[0] != b'[' || arr[arr.len()-1] != b']' {
        return None;
    }
    let inner = &arr[1..arr.len()-1];
    let mut buf = Vec::with_capacity(inner.len() / 3);
    let mut i = 0;
    while i < inner.len() && buf.len() <= BYTE_ARRAY_MAX_LEN {
        while i < inner.len() && (inner[i] == b' ' || inner[i] == b',') { i += 1; }
        if i >= inner.len() { break; }
        let mut j = i;
        while j < inner.len() && inner[j].is_ascii_digit() { j += 1; }
        if j == i { return None; }
        let n: u32 = std::str::from_utf8(&inner[i..j]).ok()?.parse().ok()?;
        if n > 255 { return None; }
        buf.push(n as u8);
        i = j;
    }
    if buf.is_empty() || buf.len() > BYTE_ARRAY_MAX_LEN { return None; }
    Some(String::from_utf8_lossy(&buf).into_owned())
}

/// Skip ASCII whitespace forward from `start`, capped at `end`.
pub fn skip_ws(bytes: &[u8], start: usize, end: usize) -> usize {
    let mut i = start;
    while i < end && bytes[i].is_ascii_whitespace() { i += 1; }
    i
}

/// Find the end of a non-string scalar value: stops at `,`, `}`, or end.
pub fn scalar_end(bytes: &[u8], start: usize, end: usize) -> usize {
    let mut i = start;
    while i < end {
        let b = bytes[i];
        if b == b',' || b == b'}' { return i; }
        i += 1;
    }
    end
}

/// Trim ASCII whitespace from both ends of a byte slice.
pub fn trim_ws(s: &[u8]) -> &[u8] {
    let mut start = 0;
    while start < s.len() && s[start].is_ascii_whitespace() { start += 1; }
    let mut end = s.len();
    while end > start && s[end - 1].is_ascii_whitespace() { end -= 1; }
    &s[start..end]
}

/// Classify a non-string scalar's bytes: number, bool, or skip (null).
pub fn classify_scalar(v: &[u8]) -> ScalarKind {
    if v == b"true" || v == b"false" { return ScalarKind::Bool; }
    if v == b"null" || v.is_empty()  { return ScalarKind::Skip; }
    let first = v[0];
    if first == b'-' || first == b'+' || first == b'.' || first.is_ascii_digit() {
        return ScalarKind::Number;
    }
    ScalarKind::Skip
}

/// Unescape a JSON string value (the bytes between unescaped quote pairs).
/// Handles the common sequences: `\"`, `\\`, `\/`, `\n`, `\t`, `\r`, `\b`,
/// `\f`. Unicode escape sequences (`\uXXXX`) and any other unknown escapes
/// are emitted as-is (the original two characters preserved). Returns the
/// input slice borrowed when no backslashes are present — avoids the
/// allocation in the common case.
pub fn unescape_json_string(bytes: &[u8]) -> std::borrow::Cow<'_, str> {
    if !bytes.contains(&b'\\') {
        return std::borrow::Cow::Borrowed(std::str::from_utf8(bytes).unwrap_or(""));
    }
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\' && i + 1 < bytes.len() {
            let esc = bytes[i + 1];
            let decoded: u8 = match esc {
                b'"'  => b'"',
                b'\\' => b'\\',
                b'/'  => b'/',
                b'n'  => b'\n',
                b't'  => b'\t',
                b'r'  => b'\r',
                b'b'  => 0x08,
                b'f'  => 0x0C,
                _ => {
                    // Unknown escape (including \uXXXX) — pass both bytes through.
                    out.push(b'\\');
                    out.push(esc);
                    i += 2;
                    continue;
                }
            };
            out.push(decoded);
            i += 2;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    std::borrow::Cow::Owned(String::from_utf8_lossy(&out).into_owned())
}

/// Cheap ISO-8601 sniff: at least 19 chars, "YYYY-MM-DDTHH:MM:SS" pattern.
/// Lexicographic min/max comparison works correctly on this format because
/// every component is fixed-width zero-padded.
pub fn looks_iso8601(s: &str) -> bool {
    let b = s.as_bytes();
    if b.len() < 19 { return false; }
    b[0..4].iter().all(|c| c.is_ascii_digit()) &&
        b[4] == b'-' &&
        b[5..7].iter().all(|c| c.is_ascii_digit()) &&
        b[7] == b'-' &&
        b[8..10].iter().all(|c| c.is_ascii_digit()) &&
        (b[10] == b'T' || b[10] == b' ') &&
        b[11..13].iter().all(|c| c.is_ascii_digit()) &&
        b[13] == b':' &&
        b[14..16].iter().all(|c| c.is_ascii_digit()) &&
        b[16] == b':' &&
        b[17..19].iter().all(|c| c.is_ascii_digit())
}


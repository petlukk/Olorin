//! Thread-count and cache-size detection via Linux sysfs.
//!
//! Re-exported by `threadpool` so callers keep the
//! `olorin::inference::threadpool::detect_thread_count` path.

use std::collections::HashSet;

/// Count physical (non-SMT) CPU cores via Linux sysfs.
/// Returns None if the sysfs interface isn't available or unreadable
/// (e.g. non-Linux, sandboxed containers).
fn physical_core_count_sysfs() -> Option<usize> {
    let entries = std::fs::read_dir("/sys/devices/system/cpu").ok()?;
    let mut siblings_first: HashSet<u32> = HashSet::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("cpu") || !name[3..].chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let path = entry.path().join("topology/thread_siblings_list");
        let Ok(txt) = std::fs::read_to_string(&path) else { continue };
        // Format: "0,8" or "0-1" or "0". First sibling ID = physical core representative.
        let first = txt
            .trim()
            .split(|c: char| c == ',' || c == '-')
            .next()?
            .parse::<u32>()
            .ok()?;
        siblings_first.insert(first);
    }
    if siblings_first.is_empty() { None } else { Some(siblings_first.len()) }
}

/// Decide worker thread count. Priority:
/// 1. `OLORIN_THREADS` env var (positive integer).
/// 2. Physical-core count from sysfs — ignores SMT siblings on x86,
///    equals logical count on ARM (no SMT on Cortex-A76 / Pi 5).
/// 3. `std::thread::available_parallelism()` fallback.
/// 4. `1` last-resort.
pub fn detect_thread_count() -> usize {
    if let Ok(s) = std::env::var("OLORIN_THREADS") {
        if let Ok(n) = s.trim().parse::<usize>() {
            if n >= 1 { return n; }
        }
    }
    if let Some(n) = physical_core_count_sysfs() {
        return n;
    }
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Read the largest shared-cache size from Linux sysfs, in bytes.
/// Walks `/sys/devices/system/cpu/cpu0/cache/index*/size`, picks the
/// entry with the highest `level`, returns its size in bytes.
/// Returns None on non-Linux or when the files are unreadable.
fn largest_cache_bytes_sysfs() -> Option<usize> {
    let mut best: Option<(u32, usize)> = None; // (level, bytes)
    for entry in std::fs::read_dir("/sys/devices/system/cpu/cpu0/cache").ok()?.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("index") { continue; }
        let level_str = std::fs::read_to_string(entry.path().join("level")).ok()?;
        let level: u32 = level_str.trim().parse().ok()?;
        let size_str = std::fs::read_to_string(entry.path().join("size")).ok()?;
        // Format: "512K", "8192K", "2M" — integer followed by one suffix char.
        let s = size_str.trim();
        let (num_part, mult) = match s.as_bytes().last()? {
            b'K' | b'k' => (&s[..s.len()-1], 1024usize),
            b'M' | b'm' => (&s[..s.len()-1], 1024 * 1024),
            b'G' | b'g' => (&s[..s.len()-1], 1024 * 1024 * 1024),
            _ => (s, 1),
        };
        let bytes: usize = num_part.parse::<usize>().ok()? * mult;
        match best {
            None => best = Some((level, bytes)),
            Some((lvl, _)) if level > lvl => best = Some((level, bytes)),
            _ => {}
        }
    }
    best.map(|(_, bytes)| bytes)
}

/// Decide the default prefill ubatch size. Priority:
/// 1. `OLORIN_PREFILL_UBATCH` env var (≥1 or 0 to disable).
/// 2. If the largest CPU cache is ≥ 8 MB → 64 (empirically the sweet
///    spot on Zen 1 Ryzen 7 1700 with a 16 MB L3 split across 2 CCXes;
///    keeps gemm_down's activation + weight working set well inside L3).
/// 3. Otherwise → `usize::MAX` (no chunking). Pi 5 (Cortex-A76, 2 MB
///    shared L3) falls here because the weight already can't fit in
///    cache, so the ubatch savings wouldn't outweigh the 4× weight
///    re-reads.
///
/// Caller treats `usize::MAX` as "process the whole prompt in one pass."
pub fn detect_prefill_ubatch() -> usize {
    if let Ok(s) = std::env::var("OLORIN_PREFILL_UBATCH") {
        if let Ok(k) = s.trim().parse::<usize>() {
            if k >= 1 { return k; }
            return usize::MAX; // "0" or negative → disabled
        }
    }
    match largest_cache_bytes_sysfs() {
        Some(bytes) if bytes >= 8 * 1024 * 1024 => 64,
        _ => usize::MAX,
    }
}

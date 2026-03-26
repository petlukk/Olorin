//! Histogram computation and xxHash64 for vault integrity.

/// Compute byte-frequency histogram of data.
/// Each byte value 0-255 gets a count, saturating at 255.
pub fn compute_histogram(data: &[u8]) -> [u8; 256] {
    let mut hist = [0u8; 256];
    for &b in data {
        hist[b as usize] = hist[b as usize].saturating_add(1);
    }
    hist
}

// xxHash64 constants
const PRIME64_1: u64 = 0x9E3779B185EBCA87;
const PRIME64_2: u64 = 0xC2B2AE3D27D4EB4F;
const PRIME64_3: u64 = 0x165667B19E3779F9;
const PRIME64_4: u64 = 0x85EBCA77C2B2AE63;
const PRIME64_5: u64 = 0x27D4EB2F165667C5;

fn round(acc: u64, input: u64) -> u64 {
    acc.wrapping_add(input.wrapping_mul(PRIME64_2))
        .rotate_left(31)
        .wrapping_mul(PRIME64_1)
}

fn merge_round(acc: u64, val: u64) -> u64 {
    let val = round(0, val);
    acc.bitxor(val)
        .wrapping_mul(PRIME64_1)
        .wrapping_add(PRIME64_4)
}

fn avalanche(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(PRIME64_2);
    h ^= h >> 29;
    h = h.wrapping_mul(PRIME64_3);
    h ^= h >> 32;
    h
}

fn read_u64_le(buf: &[u8]) -> u64 {
    u64::from_le_bytes(buf[..8].try_into().unwrap())
}

fn read_u32_le(buf: &[u8]) -> u64 {
    u32::from_le_bytes(buf[..4].try_into().unwrap()) as u64
}

use std::ops::BitXor;

/// xxHash64 — fast non-cryptographic hash for integrity checking.
pub fn xxhash64(data: &[u8], seed: u64) -> u64 {
    let len = data.len();
    let mut h: u64;

    if len >= 32 {
        let mut v1 = seed.wrapping_add(PRIME64_1).wrapping_add(PRIME64_2);
        let mut v2 = seed.wrapping_add(PRIME64_2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(PRIME64_1);

        let mut i = 0;
        while i + 32 <= len {
            v1 = round(v1, read_u64_le(&data[i..]));
            v2 = round(v2, read_u64_le(&data[i + 8..]));
            v3 = round(v3, read_u64_le(&data[i + 16..]));
            v4 = round(v4, read_u64_le(&data[i + 24..]));
            i += 32;
        }

        h = v1.rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18));

        h = merge_round(h, v1);
        h = merge_round(h, v2);
        h = merge_round(h, v3);
        h = merge_round(h, v4);
    } else {
        h = seed.wrapping_add(PRIME64_5);
    }

    h = h.wrapping_add(len as u64);

    // Process remaining bytes after the 32-byte stripe loop
    let remaining = &data[len & !31..];
    let mut i = 0;
    while i + 8 <= remaining.len() {
        let k = read_u64_le(&remaining[i..]);
        h ^= round(0, k);
        h = h.rotate_left(27).wrapping_mul(PRIME64_1).wrapping_add(PRIME64_4);
        i += 8;
    }
    if i + 4 <= remaining.len() {
        h ^= read_u32_le(&remaining[i..]).wrapping_mul(PRIME64_1);
        h = h.rotate_left(23).wrapping_mul(PRIME64_2).wrapping_add(PRIME64_3);
        i += 4;
    }
    while i < remaining.len() {
        h ^= (remaining[i] as u64).wrapping_mul(PRIME64_5);
        h = h.rotate_left(11).wrapping_mul(PRIME64_1);
        i += 1;
    }

    avalanche(h)
}

/// Normalize a byte histogram to unit vector (for cosine similarity).
pub fn normalize_histogram(hist: &[u8; 256]) -> [f32; 256] {
    let mut norm = [0.0f32; 256];
    let mut sum_sq: f32 = 0.0;
    for i in 0..256 {
        let v = hist[i] as f32;
        norm[i] = v;
        sum_sq += v * v;
    }
    let mag = sum_sq.sqrt();
    if mag > 0.0 {
        for i in 0..256 {
            norm[i] /= mag;
        }
    }
    norm
}

/// Cosine similarity between two normalized 256-dim vectors.
pub fn cosine_similarity(a: &[f32; 256], b: &[f32; 256]) -> f32 {
    let mut dot = 0.0f32;
    for i in 0..256 {
        dot += a[i] * b[i];
    }
    dot
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram() {
        let data = b"aaaaabbc";
        let hist = compute_histogram(data);
        assert_eq!(hist[b'a' as usize], 5);
        assert_eq!(hist[b'b' as usize], 2);
        assert_eq!(hist[b'c' as usize], 1);
        assert_eq!(hist[b'd' as usize], 0);
    }

    #[test]
    fn test_histogram_saturates() {
        let data = vec![0x42u8; 300];
        let hist = compute_histogram(&data);
        assert_eq!(hist[0x42], 255);
    }

    #[test]
    fn test_xxhash64_empty() {
        // Known test vector: xxhash64("", seed=0) = 0xEF46DB3751D8E999
        assert_eq!(xxhash64(b"", 0), 0xEF46DB3751D8E999);
    }

    #[test]
    fn test_xxhash64_short() {
        // Known test vector: xxhash64("abc", seed=0)
        let h = xxhash64(b"abc", 0);
        assert_eq!(h, 0x44BC2CF5AD770999);
    }

    #[test]
    fn test_xxhash64_longer() {
        // 32+ byte input exercises the stripe loop
        let data = b"0123456789abcdef0123456789abcdef";
        let h = xxhash64(data, 0);
        // Determinism: same input always gives same output
        assert_eq!(xxhash64(data, 0), h);
        // Different seed gives different result
        assert_ne!(xxhash64(data, 1), h);
    }

    #[test]
    fn test_xxhash64_deterministic() {
        let data = b"The quick brown fox jumps over the lazy dog";
        let h1 = xxhash64(data, 42);
        let h2 = xxhash64(data, 42);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_normalize_histogram() {
        let mut hist = [0u8; 256];
        hist[0] = 3;
        hist[1] = 4;
        let norm = normalize_histogram(&hist);
        // magnitude = sqrt(9 + 16) = 5
        let eps = 1e-6;
        assert!((norm[0] - 0.6).abs() < eps);
        assert!((norm[1] - 0.8).abs() < eps);
        assert!((norm[2] - 0.0).abs() < eps);
        // verify unit length
        let mag_sq: f32 = norm.iter().map(|x| x * x).sum();
        assert!((mag_sq - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_normalize_histogram_zero() {
        let hist = [0u8; 256];
        let norm = normalize_histogram(&hist);
        for v in &norm {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_cosine_similarity() {
        let mut a = [0.0f32; 256];
        let mut b = [0.0f32; 256];
        a[0] = 1.0;
        b[0] = 1.0;
        // identical unit vectors => similarity 1.0
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);

        // orthogonal vectors => similarity 0.0
        let mut c = [0.0f32; 256];
        c[1] = 1.0;
        assert!((cosine_similarity(&a, &c)).abs() < 1e-6);
    }
}

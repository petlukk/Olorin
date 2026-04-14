//! Unit tests for ngram_lookup kernel: longest-match draft extraction,
//! right-to-left priority, N=2 fallback, edge cases.

use olorin::kernels::ffi;
use olorin::kernels::ffi_inference::ngram_lookup;

fn init() {
    ffi::init().unwrap();
}

#[test]
fn match_at_end_returns_zero_tokens() {
    init();
    // Key matches at positions 0 AND 6. Right-to-left prefers position 6.
    // After position 6's "10 11 12", ctx has nothing — 0 tokens available.
    let ctx: Vec<u32> = vec![10, 11, 12, 13, 14, 15, 10, 11, 12];
    let key = [10u32, 11, 12];
    let mut out = vec![0u32; 4];
    let n = ngram_lookup(&ctx, &key, 4, &mut out);
    assert_eq!(n, 0);
}

#[test]
fn match_prefers_recent_not_earliest() {
    init();
    // Matches at positions 0 and 5. Recent (position 5) is preferred.
    let ctx: Vec<u32> = vec![10, 11, 12, 99, 98, 10, 11, 12, 77, 88];
    let key = [10u32, 11, 12];
    let mut out = vec![0u32; 4];
    let n = ngram_lookup(&ctx, &key, 4, &mut out);
    assert_eq!(n, 2);
    assert_eq!(&out[..2], &[77, 88]);
}

#[test]
fn no_match_returns_zero() {
    init();
    let ctx: Vec<u32> = vec![1, 2, 3, 4, 5, 6];
    let key = [99u32, 99, 99];
    let mut out = vec![0u32; 4];
    assert_eq!(ngram_lookup(&ctx, &key, 4, &mut out), 0);
}

#[test]
fn n3_miss_n2_hit() {
    init();
    // 3-gram [10,20,30] misses. 2-gram [20,30] matches right-to-left at position 3.
    let ctx: Vec<u32> = vec![5, 20, 30, 20, 30, 42, 43];
    let key = [10u32, 20, 30];
    let mut out = vec![0u32; 4];
    let n = ngram_lookup(&ctx, &key, 4, &mut out);
    assert_eq!(n, 2);
    assert_eq!(&out[..2], &[42, 43]);
}

#[test]
fn context_shorter_than_key_returns_zero() {
    init();
    let ctx: Vec<u32> = vec![1, 2]; // only 2 tokens
    let key = [1u32, 2, 3];
    let mut out = vec![0u32; 4];
    assert_eq!(ngram_lookup(&ctx, &key, 4, &mut out), 0);
}

#[test]
fn respects_k_limit() {
    init();
    // Pattern [1,2,3] at position 0; tail is [10,11,12,13,14,15].
    let ctx: Vec<u32> = vec![1, 2, 3, 10, 11, 12, 13, 14, 15];
    let key = [1u32, 2, 3];
    let mut out = vec![0u32; 3];
    let n = ngram_lookup(&ctx, &key, 3, &mut out);
    assert_eq!(n, 3);
    assert_eq!(&out[..3], &[10, 11, 12]);
}

#[test]
fn simd_boundary_long_context_recent_match() {
    init();
    // Force SIMD loop + tail sweep interaction. Create ~40-token context
    // with earlier decoys and a recent match that straddles SIMD boundaries.
    let mut ctx: Vec<u32> = Vec::new();
    for i in 0..32u32 { ctx.push(100 + i); }
    // Insert an early match at position 10..12:
    ctx[10] = 7; ctx[11] = 8; ctx[12] = 9;
    // And a later match at position 29..31:
    ctx[29] = 7; ctx[30] = 8; ctx[31] = 9;
    // No tokens after position 31 (ctx.len() == 32).
    let key = [7u32, 8, 9];
    let mut out = vec![0u32; 4];
    assert_eq!(ngram_lookup(&ctx, &key, 4, &mut out), 0);

    // Now append a tail so the recent match produces draft tokens:
    ctx.push(500); ctx.push(501); ctx.push(502);
    let n = ngram_lookup(&ctx, &key, 4, &mut out);
    assert_eq!(n, 3);
    assert_eq!(&out[..3], &[500, 501, 502]);
}

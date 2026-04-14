//! Unit tests for verify_draft: fused per-row argmax + find-first-mismatch.

use olorin::kernels::ffi;
use olorin::kernels::ffi_inference::verify_draft;

fn init() {
    ffi::init().unwrap();
}

/// Build K × vocab logits row-major, with row r's argmax at `peaks[r]`.
fn make_logits(peaks: &[usize], vocab: usize) -> Vec<f32> {
    let k = peaks.len();
    let mut logits = vec![-1.0f32; k * vocab];
    for (r, &p) in peaks.iter().enumerate() {
        logits[r * vocab + p] = 10.0;
    }
    logits
}

#[test]
fn full_accept_returns_k() {
    init();
    let vocab = 128;
    // K=4 rows, argmaxes: [8, 9, 99, 42]. Drafts (K-1=3): [8, 9, 99]. All match.
    let logits = make_logits(&[8, 9, 99, 42], vocab);
    let drafts: Vec<u32> = vec![8, 9, 99];
    let mut out = vec![0u32; 4];
    let j = verify_draft(&logits, vocab, &drafts, 4, &mut out);
    assert_eq!(j, 4);
    assert_eq!(out, vec![8, 9, 99, 42]);
}

#[test]
fn immediate_reject_returns_zero() {
    init();
    let vocab = 128;
    // Row 0 argmax = 7; draft expected 8. Mismatch at row 0.
    let logits = make_logits(&[7, 20, 30, 40], vocab);
    let drafts: Vec<u32> = vec![8, 20, 30];
    let mut out = vec![0u32; 4];
    let j = verify_draft(&logits, vocab, &drafts, 4, &mut out);
    assert_eq!(j, 0);
    assert_eq!(out[0], 7); // correction token written
}

#[test]
fn partial_accept_returns_middle_index() {
    init();
    let vocab = 128;
    // Argmaxes: [8, 9, 77, 99]. Drafts [8, 9, 30] -> matches rows 0,1; mismatch at row 2.
    let logits = make_logits(&[8, 9, 77, 99], vocab);
    let drafts: Vec<u32> = vec![8, 9, 30];
    let mut out = vec![0u32; 4];
    let j = verify_draft(&logits, vocab, &drafts, 4, &mut out);
    assert_eq!(j, 2);
    assert_eq!(&out[..3], &[8, 9, 77]);
}

#[test]
fn realistic_vocab_full_accept() {
    init();
    let vocab = 262144; // Gemma 4 vocab
    let peaks = [100_000, 200_000, 50_000, 150_000];
    let logits = make_logits(&peaks, vocab);
    let drafts: Vec<u32> = vec![100_000, 200_000, 50_000];
    let mut out = vec![0u32; 4];
    let j = verify_draft(&logits, vocab, &drafts, 4, &mut out);
    assert_eq!(j, 4);
    assert_eq!(out, vec![100_000, 200_000, 50_000, 150_000]);
}

#[test]
fn single_row_k_equals_one() {
    init();
    // K=1: one row, zero drafts, full-accept path.
    let vocab = 64;
    let logits = make_logits(&[17], vocab);
    let drafts: Vec<u32> = vec![];
    let mut out = vec![0u32; 1];
    let j = verify_draft(&logits, vocab, &drafts, 1, &mut out);
    assert_eq!(j, 1);
    assert_eq!(out[0], 17);
}

#[test]
fn tie_prefers_lowest_index() {
    init();
    // Two lanes with identical max values. Argmax must be the lower index.
    let vocab = 128;
    let mut logits = vec![-1.0f32; vocab];
    logits[10] = 5.0;
    logits[50] = 5.0; // same value at higher index
    // K=1, no drafts
    let mut out = vec![0u32; 1];
    let j = verify_draft(&logits, vocab, &[], 1, &mut out);
    assert_eq!(j, 1);
    assert_eq!(out[0], 10, "ties must prefer lowest index");
}

#[test]
fn tail_lanes_covered_when_vocab_not_multiple_of_4() {
    init();
    // vocab = 7 -> 1 SIMD window of 4, tail of 3. Max at index 6 (in tail).
    let vocab = 7;
    let mut logits = vec![-1.0f32; vocab];
    logits[6] = 20.0;
    let mut out = vec![0u32; 1];
    let j = verify_draft(&logits, vocab, &[], 1, &mut out);
    assert_eq!(j, 1);
    assert_eq!(out[0], 6);
}

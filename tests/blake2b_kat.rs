//! Blake2b known-answer tests against the RFC 7693 reference and the
//! Blake2 official test vectors.  Covers the three things that can go
//! wrong independently:
//!
//! 1. Compression-function correctness — exercised by every vector.
//! 2. Parameter block construction (digest length XOR into h[0]) —
//!    exercised by the 32-byte / 64-byte split.
//! 3. Streaming behaviour (deferred-final + counter accounting) —
//!    exercised by the chunked update + multi-block paths below.

use olorin::kernels::ffi;
use olorin::storage::blake2b::{hash, Hasher};

fn setup() {
    ffi::init().expect("kernel init");
}

fn hex_to_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn blake2b_512_abc_matches_rfc7693_appendix_a() {
    setup();
    // RFC 7693 Appendix A: Blake2b("abc") with 64-byte output.
    let expected = hex_to_bytes(
        "ba80a53f981c4d0d6a2797b69f12f6e94c212f14685ac4b74b12bb6fdbffa2d1\
         7d87c5392aab792dc252d5de4533cc9518d38aa8dbf1925ab92386edd4009923",
    );
    let mut out = [0u8; 64];
    hash(b"abc", &mut out);
    assert_eq!(out.as_slice(), expected.as_slice());
}

#[test]
fn blake2b_512_empty_matches_reference() {
    setup();
    let expected = hex_to_bytes(
        "786a02f742015903c6c6fd852552d272912f4740e15847618a86e217f71f5419\
         d25e1031afee585313896444934eb04b903a685b1448b755d56f701afe9be2ce",
    );
    let mut out = [0u8; 64];
    hash(b"", &mut out);
    assert_eq!(out.as_slice(), expected.as_slice());
}

#[test]
fn blake2b_256_empty_matches_reference() {
    setup();
    // Standard Blake2b-256 reference output for empty input.
    let expected =
        hex_to_bytes("0e5751c026e543b2e8ab2eb06099daa1d1e5df47778f7787faab45cdf12fe3a8");
    let mut out = [0u8; 32];
    hash(b"", &mut out);
    assert_eq!(out.as_slice(), expected.as_slice());
}

#[test]
fn streaming_update_matches_one_shot_for_short_input() {
    setup();
    let input = b"The quick brown fox jumps over the lazy dog";
    let mut one_shot = [0u8; 64];
    hash(input, &mut one_shot);

    let mut streamed = [0u8; 64];
    let mut h = Hasher::new(64);
    h.update(&input[..10]);
    h.update(&input[10..25]);
    h.update(&input[25..]);
    h.finalize(&mut streamed);

    assert_eq!(one_shot, streamed);
}

#[test]
fn streaming_handles_exact_block_boundary() {
    // Argon2id will feed Blake2b inputs that frequently land exactly on
    // a 128-byte boundary; the deferred-final logic in Hasher::update
    // must compress the staged block only once it knows another byte
    // is coming.
    setup();
    let input: Vec<u8> = (0..128).map(|i| (i * 31) as u8).collect();

    let mut one_shot = [0u8; 64];
    hash(&input, &mut one_shot);

    let mut split_at_128 = [0u8; 64];
    let mut h = Hasher::new(64);
    h.update(&input);
    h.finalize(&mut split_at_128);
    assert_eq!(one_shot, split_at_128);

    let mut chunked = [0u8; 64];
    let mut h = Hasher::new(64);
    h.update(&input[..64]);
    h.update(&input[64..128]);
    h.finalize(&mut chunked);
    assert_eq!(one_shot, chunked);
}

#[test]
fn streaming_handles_multi_block_input() {
    setup();
    let input: Vec<u8> = (0..512).map(|i| (i ^ (i >> 3)) as u8).collect();

    let mut one_shot = [0u8; 64];
    hash(&input, &mut one_shot);

    let mut chunked = [0u8; 64];
    let mut h = Hasher::new(64);
    for chunk in input.chunks(37) {
        h.update(chunk);
    }
    h.finalize(&mut chunked);

    assert_eq!(one_shot, chunked);
}

#[test]
fn output_lengths_are_distinct() {
    // Smaller digests are NOT a truncation of larger ones — the digest
    // length is mixed into h[0] before compression, so changing it
    // changes every output byte.
    setup();
    let mut out32 = [0u8; 32];
    let mut out64 = [0u8; 64];
    hash(b"abc", &mut out32);
    hash(b"abc", &mut out64);

    assert_ne!(
        out32.as_slice(),
        &out64[..32],
        "Blake2b-256 must not equal first 32 bytes of Blake2b-512 — \
         the digest-length parameter XOR distinguishes them"
    );
}

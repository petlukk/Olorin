//! ChaCha20-Poly1305 AEAD bit-parity sweep against the RustCrypto
//! `chacha20poly1305` crate (RFC 8439 reference).  100 random trials
//! across a range of msg/aad sizes — ciphertext and tag must match
//! byte-for-byte; if they don't, our kernel disagrees with the spec.

use chacha20poly1305::aead::{Aead, Payload};
use chacha20poly1305::{ChaCha20Poly1305, KeyInit, Nonce};
use olorin::kernels::ffi;
use olorin::storage::aead;

fn ref_seal(key: &[u8; 32], nonce: &[u8; 12], aad: &[u8], pt: &[u8]) -> (Vec<u8>, [u8; 16]) {
    let cipher = ChaCha20Poly1305::new(key.into());
    let mut ct_and_tag = cipher
        .encrypt(Nonce::from_slice(nonce), Payload { msg: pt, aad })
        .expect("ref encrypt");
    let tag_off = ct_and_tag.len() - 16;
    let mut tag = [0u8; 16];
    tag.copy_from_slice(&ct_and_tag[tag_off..]);
    ct_and_tag.truncate(tag_off);
    (ct_and_tag, tag)
}

fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

#[test]
fn parity_random_sweep() {
    ffi::init().expect("kernel init");

    let mut state: u64 = 0xfeed_face_cafe_beef;
    // Sizes spanning under-block, exact-block, over-block, and several blocks
    // worth of plaintext so we exercise all padding-and-length code paths.
    let sizes = [0usize, 1, 15, 16, 17, 63, 64, 65, 127, 128, 256, 1024, 16384];

    for trial in 0..100u32 {
        let len = sizes[(xorshift64(&mut state) % sizes.len() as u64) as usize];

        let mut key = [0u8; 32];
        for b in key.iter_mut() {
            *b = xorshift64(&mut state) as u8;
        }
        let mut nonce = [0u8; 12];
        for b in nonce.iter_mut() {
            *b = xorshift64(&mut state) as u8;
        }
        let aad_len = (xorshift64(&mut state) % 256) as usize;
        let mut aad = vec![0u8; aad_len];
        for b in aad.iter_mut() {
            *b = xorshift64(&mut state) as u8;
        }
        let mut pt = vec![0u8; len];
        for b in pt.iter_mut() {
            *b = xorshift64(&mut state) as u8;
        }

        let (ref_ct, ref_tag) = ref_seal(&key, &nonce, &aad, &pt);

        let mut our_buf = pt.clone();
        let mut our_tag = [0u8; 16];
        aead::seal(&key, &nonce, &aad, &mut our_buf, &mut our_tag);

        assert_eq!(
            our_buf, ref_ct,
            "trial {trial} len {len} aad_len {aad_len}: ciphertext mismatch"
        );
        assert_eq!(
            our_tag, ref_tag,
            "trial {trial} len {len} aad_len {aad_len}: tag mismatch"
        );
    }
}

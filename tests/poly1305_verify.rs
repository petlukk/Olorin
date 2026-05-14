//! Constant-time Poly1305 tag verification tests.
//!
//! Covers the correctness side: accept on a correct tag, reject on any
//! single-byte flip, reject zero tag with nonzero message.  The
//! constant-time property itself is statistically tested in Task 15.

use olorin::kernels::ffi;

fn poly1305_verify(key: &[u8; 32], msg: &[u8], tag: &[u8; 16]) -> bool {
    unsafe {
        ffi::poly1305_verify(key.as_ptr(), msg.as_ptr(), msg.len() as i32, tag.as_ptr()) != 0
    }
}

fn compute_correct_tag(key: &[u8; 32], msg: &[u8]) -> [u8; 16] {
    let mut tag = [0u8; 16];
    unsafe {
        ffi::poly1305_mac(key.as_ptr(), msg.as_ptr(), msg.len() as i32, tag.as_mut_ptr());
    }
    tag
}

#[test]
fn verify_accepts_correct_tag() {
    ffi::init().expect("kernel init");
    let key = [7u8; 32];
    let msg = b"hello world";
    let tag = compute_correct_tag(&key, msg);
    assert!(poly1305_verify(&key, msg, &tag));
}

#[test]
fn verify_rejects_byte_flip_any_position() {
    ffi::init().expect("kernel init");
    let key = [7u8; 32];
    let msg = b"hello world";
    let correct = compute_correct_tag(&key, msg);
    for i in 0..16 {
        let mut tag = correct;
        tag[i] ^= 0x01;
        assert!(!poly1305_verify(&key, msg, &tag), "byte {i} flip not detected");
    }
}

#[test]
fn verify_rejects_zero_tag_with_nonzero_message() {
    ffi::init().expect("kernel init");
    let key = [9u8; 32];
    let msg = b"x";
    let zero = [0u8; 16];
    assert!(!poly1305_verify(&key, msg, &zero));
}

#[test]
fn verify_rejects_every_single_bit_flip() {
    // Exhaustive 128-bit flip sweep — every bit position in every byte.
    ffi::init().expect("kernel init");
    let key = [0x42u8; 32];
    let msg = b"the quick brown fox jumps over the lazy dog";
    let correct = compute_correct_tag(&key, msg);
    for byte_i in 0..16 {
        for bit_i in 0..8u8 {
            let mut tag = correct;
            tag[byte_i] ^= 1 << bit_i;
            assert!(
                !poly1305_verify(&key, msg, &tag),
                "byte {byte_i} bit {bit_i} flip not detected",
            );
        }
    }
}

#[test]
fn verify_accepts_empty_message() {
    ffi::init().expect("kernel init");
    let key = [0xa5u8; 32];
    let msg: &[u8] = &[];
    let tag = compute_correct_tag(&key, msg);
    assert!(poly1305_verify(&key, msg, &tag));
}

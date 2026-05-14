//! ChaCha20-Poly1305 AEAD round-trip + tamper-rejection tests.

use olorin::kernels::ffi;
use olorin::storage::aead;

#[test]
fn seal_then_open_round_trip() {
    ffi::init().expect("kernel init");
    let key = [0x42u8; 32];
    let nonce = [0x07u8; 12];
    let aad = b"some-aad";
    let pt_original = b"the quick brown fox jumps over the lazy dog".to_vec();
    let mut buf = pt_original.clone();
    let mut tag = [0u8; 16];

    aead::seal(&key, &nonce, aad, &mut buf, &mut tag);
    assert_ne!(buf, pt_original, "ciphertext must not equal plaintext");

    aead::open(&key, &nonce, aad, &mut buf, &tag).expect("open should succeed");
    assert_eq!(buf, pt_original);
}

#[test]
fn open_rejects_tag_tampering() {
    ffi::init().expect("kernel init");
    let key = [0x42u8; 32];
    let nonce = [0x07u8; 12];
    let aad = b"";
    let mut buf = b"secret data".to_vec();
    let mut tag = [0u8; 16];
    aead::seal(&key, &nonce, aad, &mut buf, &mut tag);

    let ct_snapshot = buf.clone();
    tag[0] ^= 0x01;
    let result = aead::open(&key, &nonce, aad, &mut buf, &tag);
    assert!(result.is_err());
    // Critical: buf must be UNCHANGED if open fails — never decrypt on bad tag.
    assert_eq!(buf, ct_snapshot, "buf must not be decrypted when tag fails");
}

#[test]
fn open_rejects_aad_tampering() {
    ffi::init().expect("kernel init");
    let key = [0x42u8; 32];
    let nonce = [0x07u8; 12];
    let mut buf = b"data".to_vec();
    let mut tag = [0u8; 16];
    aead::seal(&key, &nonce, b"aad-A", &mut buf, &mut tag);

    let ct_snapshot = buf.clone();
    let result = aead::open(&key, &nonce, b"aad-B", &mut buf, &tag);
    assert!(result.is_err());
    assert_eq!(buf, ct_snapshot);
}

#[test]
fn open_rejects_ciphertext_tampering() {
    ffi::init().expect("kernel init");
    let key = [0x42u8; 32];
    let nonce = [0x07u8; 12];
    let mut buf = b"twelve bytes".to_vec();
    let mut tag = [0u8; 16];
    aead::seal(&key, &nonce, b"", &mut buf, &mut tag);

    buf[0] ^= 0x01;
    let ct_snapshot = buf.clone();
    let result = aead::open(&key, &nonce, b"", &mut buf, &tag);
    assert!(result.is_err());
    assert_eq!(buf, ct_snapshot);
}

#[test]
fn empty_plaintext_is_valid() {
    ffi::init().expect("kernel init");
    let key = [0u8; 32];
    let nonce = [0u8; 12];
    let mut buf: Vec<u8> = vec![];
    let mut tag = [0u8; 16];
    aead::seal(&key, &nonce, b"aad", &mut buf, &mut tag);
    assert_eq!(buf.len(), 0);
    aead::open(&key, &nonce, b"aad", &mut buf, &tag).expect("empty pt should open");
}

#[test]
fn nonce_change_breaks_round_trip() {
    ffi::init().expect("kernel init");
    let key = [0xabu8; 32];
    let mut buf = b"sensitive payload".to_vec();
    let mut tag = [0u8; 16];
    aead::seal(&key, &[0u8; 12], b"", &mut buf, &mut tag);

    let ct_snapshot = buf.clone();
    let result = aead::open(&key, &[1u8; 12], b"", &mut buf, &tag);
    assert!(result.is_err());
    assert_eq!(buf, ct_snapshot);
}

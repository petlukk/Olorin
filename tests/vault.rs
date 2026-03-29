use olorin::kernels::ffi;
use olorin::storage::crypto;
use olorin::storage::secure::SecureBuffer;

#[test]
fn test_chacha20_encrypt_decrypt_roundtrip() {
    ffi::init().unwrap();
    let key = [0x42u8; 32];
    let nonce = [0x01u8; 12];
    let plaintext = b"The Wakeful Mind in Ea";
    let mut ct = plaintext.to_vec();
    crypto::encrypt(&key, &nonce, 0, &mut ct);
    assert_ne!(&ct[..], plaintext);
    crypto::decrypt(&key, &nonce, 0, &mut ct);
    assert_eq!(&ct[..], plaintext);
}

#[test]
fn test_secure_buffer_basics() {
    ffi::init().unwrap();
    let mut buf = SecureBuffer::new(4096);
    assert_eq!(buf.len(), 4096);
    buf.as_mut_slice()[0] = 0xFF;
    assert_eq!(buf.as_slice()[0], 0xFF);
}

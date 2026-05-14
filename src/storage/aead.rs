//! ChaCha20-Poly1305 AEAD per RFC 8439, layered on the Eä SIMD kernels.
//!
//! `seal` encrypts in-place and writes the 16-byte authentication tag.
//! `open` constant-time-verifies the tag first and only decrypts on success
//! (the ciphertext buffer is left untouched on integrity failure, so the
//! caller cannot accidentally consume unauthenticated plaintext).

use crate::error::{Error, Result};
use crate::kernels::ffi;
use crate::storage::crypto;

/// Encrypt `pt` in-place using ChaCha20 with `key` + `nonce`, authenticate
/// `aad` together with the resulting ciphertext, and write the 16-byte
/// Poly1305 tag.  Caller chooses nonce uniqueness (single-use per key).
pub fn seal(
    key: &[u8; 32],
    nonce: &[u8; 12],
    aad: &[u8],
    pt: &mut [u8],
    tag: &mut [u8; 16],
) {
    let mut otk = [0u8; 32];
    crypto::keystream(key, nonce, 0, &mut otk);

    crypto::encrypt(key, nonce, 1, pt);

    let mac_msg = build_mac_message(aad, pt);
    unsafe {
        ffi::poly1305_mac(
            otk.as_ptr(),
            mac_msg.as_ptr(),
            mac_msg.len() as i32,
            tag.as_mut_ptr(),
        );
        ffi::zeroize(otk.as_mut_ptr(), 32);
    }
}

/// Verify `tag` against `ct`+`aad` in constant time using `key`+`nonce`,
/// then decrypt `ct` in-place.  On integrity failure, `ct` is left
/// unchanged and `Error::Vault` is returned.
pub fn open(
    key: &[u8; 32],
    nonce: &[u8; 12],
    aad: &[u8],
    ct: &mut [u8],
    tag: &[u8; 16],
) -> Result<()> {
    let mut otk = [0u8; 32];
    crypto::keystream(key, nonce, 0, &mut otk);

    let mac_msg = build_mac_message(aad, ct);
    let ok = unsafe {
        ffi::poly1305_verify(
            otk.as_ptr(),
            mac_msg.as_ptr(),
            mac_msg.len() as i32,
            tag.as_ptr(),
        )
    };
    unsafe {
        ffi::zeroize(otk.as_mut_ptr(), 32);
    }
    if ok == 0 {
        return Err(Error::Vault("integrity check failed"));
    }

    crypto::decrypt(key, nonce, 1, ct);
    Ok(())
}

/// Build the Poly1305 input per RFC 8439 §2.8.1:
///   aad || pad16(aad) || ct || pad16(ct) || len64_le(aad) || len64_le(ct)
fn build_mac_message(aad: &[u8], ct: &[u8]) -> Vec<u8> {
    let pad_aad = (16 - (aad.len() % 16)) % 16;
    let pad_ct = (16 - (ct.len() % 16)) % 16;
    let total = aad.len() + pad_aad + ct.len() + pad_ct + 16;
    let mut buf = Vec::with_capacity(total);
    buf.extend_from_slice(aad);
    buf.resize(aad.len() + pad_aad, 0);
    buf.extend_from_slice(ct);
    buf.resize(aad.len() + pad_aad + ct.len() + pad_ct, 0);
    buf.extend_from_slice(&(aad.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(ct.len() as u64).to_le_bytes());
    buf
}

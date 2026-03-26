//! Rust FFI wrapper for ChaCha20 Eä SIMD kernels.
//!
//! Loads `libchacha20.so` at runtime via libloading and calls the
//! `chacha20_encrypt` symbol. Key must be 32 bytes, nonce 12 bytes.

use libloading::{Library, Symbol};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ChachaError {
    #[error("kernel load failed: {0}")]
    KernelLoad(String),
    #[error("invalid key length: expected 32, got {0}")]
    InvalidKeyLen(usize),
    #[error("invalid nonce length: expected 12, got {0}")]
    InvalidNonceLen(usize),
}

/// Encrypt plaintext using ChaCha20.
/// Key must be 32 bytes, nonce must be 12 bytes.
/// Returns ciphertext (same length as plaintext).
pub fn encrypt(
    plaintext: &[u8],
    key: &[u8; 32],
    nonce: &[u8; 12],
    lib_path: &std::path::Path,
) -> Result<Vec<u8>, ChachaError> {
    chacha20_xor(plaintext, key, nonce, 1, lib_path)
}

/// Decrypt ciphertext using ChaCha20.
/// Same as encrypt (ChaCha20 is symmetric XOR).
pub fn decrypt(
    ciphertext: &[u8],
    key: &[u8; 32],
    nonce: &[u8; 12],
    lib_path: &std::path::Path,
) -> Result<Vec<u8>, ChachaError> {
    chacha20_xor(ciphertext, key, nonce, 1, lib_path)
}

// chacha20_encrypt kernel signature (from Eä SIMD compiler output):
//   key_i32:   *const i32  — 8 i32s (little-endian words from 32-byte key)
//   nonce_i32: *const i32  — 3 i32s (little-endian words from 12-byte nonce)
//   counter:   i32         — initial block counter
//   plaintext: *const u8   — input bytes
//   ciphertext: *mut u8    — output bytes (same length)
//   len:       i32         — byte length
//   ks_i32:    *mut i32    — scratch: 64 i32s (keystream words)
//   ks_u8:     *mut u8     — scratch: 256 u8s (same memory, byte view)
//   pt_i32:    *mut i32    — plaintext cast as i32 pointer
//   ct_i32:    *mut i32    — ciphertext cast as i32 pointer
type KernelFn = unsafe extern "C" fn(
    *const i32, *const i32, i32,
    *const u8, *mut u8, i32,
    *mut i32, *mut u8,
    *mut i32, *mut i32,
);

fn chacha20_xor(
    input: &[u8],
    key: &[u8; 32],
    nonce: &[u8; 12],
    counter: i32,
    lib_path: &std::path::Path,
) -> Result<Vec<u8>, ChachaError> {
    if input.is_empty() {
        return Ok(Vec::new());
    }

    // Convert 32-byte key to 8 little-endian i32 words.
    let key_i32: [i32; 8] = {
        let mut arr = [0i32; 8];
        for (i, chunk) in key.chunks_exact(4).enumerate() {
            arr[i] = i32::from_le_bytes(chunk.try_into().unwrap());
        }
        arr
    };

    // Convert 12-byte nonce to 3 little-endian i32 words.
    let nonce_i32: [i32; 3] = {
        let mut arr = [0i32; 3];
        for (i, chunk) in nonce.chunks_exact(4).enumerate() {
            arr[i] = i32::from_le_bytes(chunk.try_into().unwrap());
        }
        arr
    };

    // Allocate output buffer and scratch (256 bytes = 64 i32s).
    let mut output: Vec<u8> = vec![0u8; input.len()];
    let mut scratch: Vec<i32> = vec![0i32; 64];

    // Load the .so and call the kernel.
    let lib = unsafe {
        Library::new(lib_path)
            .map_err(|e| ChachaError::KernelLoad(e.to_string()))?
    };

    unsafe {
        let func: Symbol<KernelFn> = lib
            .get(b"chacha20_encrypt\0")
            .map_err(|e| ChachaError::KernelLoad(e.to_string()))?;

        let ks_i32_ptr = scratch.as_mut_ptr();
        let ks_u8_ptr = ks_i32_ptr as *mut u8;
        // pt_i32 and ct_i32 are the same buffers viewed as i32 pointers;
        // the kernel uses them for 4-byte aligned XOR operations.
        let pt_i32_ptr = input.as_ptr() as *mut i32;
        let ct_i32_ptr = output.as_mut_ptr() as *mut i32;

        func(
            key_i32.as_ptr(),
            nonce_i32.as_ptr(),
            counter,
            input.as_ptr(),
            output.as_mut_ptr(),
            input.len() as i32,
            ks_i32_ptr,
            ks_u8_ptr,
            pt_i32_ptr,
            ct_i32_ptr,
        );
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn so_path() -> PathBuf {
        // Locate libchacha20.so relative to the workspace root (dev machine).
        let manifest = env!("CARGO_MANIFEST_DIR");
        let root = PathBuf::from(manifest).join("..").join("..");
        root.join("kernels/prebuilt/x86/libchacha20.so")
    }

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let path = so_path();
        if !path.exists() {
            eprintln!("skipping: {} not found", path.display());
            return;
        }
        let key = [0x42u8; 32];
        let nonce = [0x24u8; 12];
        let plaintext = b"hello, ChaCha20 from Rust!";

        let ct = encrypt(plaintext, &key, &nonce, &path).unwrap();
        assert_ne!(ct, plaintext.as_slice());

        let recovered = decrypt(&ct, &key, &nonce, &path).unwrap();
        assert_eq!(recovered, plaintext.as_slice());
    }

    #[test]
    fn empty_input_returns_empty() {
        let path = so_path();
        let key = [0u8; 32];
        let nonce = [0u8; 12];
        // encrypt/decrypt of empty slice doesn't load the .so at all.
        let ct = encrypt(&[], &key, &nonce, &path).unwrap();
        assert!(ct.is_empty());
    }

    #[test]
    fn encrypt_decrypt_longer_message() {
        let path = so_path();
        if !path.exists() {
            eprintln!("skipping: {} not found", path.display());
            return;
        }
        let key = [0xABu8; 32];
        let nonce = [0xCDu8; 12];
        let plaintext: Vec<u8> = (0u8..=255).cycle().take(1024).collect();

        let ct = encrypt(&plaintext, &key, &nonce, &path).unwrap();
        assert_eq!(ct.len(), plaintext.len());

        let recovered = decrypt(&ct, &key, &nonce, &path).unwrap();
        assert_eq!(recovered, plaintext);
    }
}

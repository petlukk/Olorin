//! ChaCha20 encrypt/decrypt via Eä SIMD kernel.
//!
//! Key: 32 bytes. Nonce: 12 bytes. Counter: i32 (typically 0 or 1).
//! Operates in-place: `encrypt` and `decrypt` are the same XOR operation.

use crate::kernels::ffi;

/// Encrypt `buf` in-place using ChaCha20.
/// `counter` is the initial block counter (use 0 for a fresh message).
pub fn encrypt(key: &[u8; 32], nonce: &[u8; 12], counter: i32, buf: &mut [u8]) {
    chacha20_xor(key, nonce, counter, buf);
}

/// Decrypt `buf` in-place using ChaCha20 (identical to encrypt).
pub fn decrypt(key: &[u8; 32], nonce: &[u8; 12], counter: i32, buf: &mut [u8]) {
    chacha20_xor(key, nonce, counter, buf);
}

/// Write `out.len()` bytes of ChaCha20 keystream into `out`.
/// Used to derive the Poly1305 one-time key from counter=0.
pub fn keystream(key: &[u8; 32], nonce: &[u8; 12], counter: i32, out: &mut [u8]) {
    out.iter_mut().for_each(|b| *b = 0);
    chacha20_xor(key, nonce, counter, out);
}

fn chacha20_xor(key: &[u8; 32], nonce: &[u8; 12], counter: i32, buf: &mut [u8]) {
    if buf.is_empty() {
        return;
    }

    // Convert 32-byte key to 8 little-endian i32 words.
    let mut key_i32: [i32; 8] = {
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

    // Scratch: 64 i32s (256 bytes) for the keystream.
    let mut scratch = vec![0i32; 64];

    // i32-aligned staging buffers for SIMD kernel (CLAUDE.md: "Use Vec<i32> not Vec<u8>")
    let i32_len = (buf.len() + 3) / 4;
    let mut input_i32: Vec<i32> = vec![0i32; i32_len];
    let mut output_i32: Vec<i32> = vec![0i32; i32_len];

    unsafe {
        std::ptr::copy_nonoverlapping(buf.as_ptr(), input_i32.as_mut_ptr() as *mut u8, buf.len());

        let ks_i32_ptr = scratch.as_mut_ptr();
        let ks_u8_ptr  = ks_i32_ptr as *mut u8;

        ffi::chacha20_encrypt(
            key_i32.as_ptr(),
            nonce_i32.as_ptr(),
            counter,
            input_i32.as_ptr() as *const u8,
            output_i32.as_mut_ptr() as *mut u8,
            buf.len() as i32,
            ks_i32_ptr,
            ks_u8_ptr,
            input_i32.as_mut_ptr(),
            output_i32.as_mut_ptr(),
        );

        std::ptr::copy_nonoverlapping(output_i32.as_ptr() as *const u8, buf.as_mut_ptr(), buf.len());

        // Plaintext, keystream, and key fragments leave the scope below
        // — but the underlying heap/stack memory is NOT zeroed by Rust
        // on drop. SIMD-zeroize them now via the FFI kernel (opaque to
        // the optimizer, so the writes can't be elided) before they
        // become reusable allocations holding residual secrets.
        ffi::zeroize(scratch.as_mut_ptr() as *mut u8, (scratch.len() * 4) as i32);
        ffi::zeroize(input_i32.as_mut_ptr() as *mut u8, (input_i32.len() * 4) as i32);
        ffi::zeroize(output_i32.as_mut_ptr() as *mut u8, (output_i32.len() * 4) as i32);
        ffi::zeroize(key_i32.as_mut_ptr() as *mut u8, 32);
    }
}

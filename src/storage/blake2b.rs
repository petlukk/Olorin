//! Blake2b (RFC 7693) variable-output hash, layered on the Ea
//! `blake2b_compress` kernel.
//!
//! Surface needed by Argon2id (RFC 9106):
//! - 64-byte fixed-output `hash` for the H function in §3.3
//! - Streaming-style `Hasher` for incremental input (Argon2id absorbs
//!   passphrase + salt + params + auxiliary fields one chunk at a time)
//!
//! Blake2b state is 8 × u64.  The compression kernel processes one
//! 128-byte block at a time and updates the chaining state in place;
//! we walk the input in those blocks, count total bytes seen in the
//! 64-bit counter, and finalise by passing `is_final = 1` on the last
//! block.  Trailing partial blocks are zero-padded to 128 bytes — the
//! counter records the actual byte count, not the padded length.

use crate::kernels::ffi;

const BLOCK_BYTES: usize = 128;
const STATE_WORDS: usize = 8;
const BLOCK_WORDS: usize = 16;

/// Blake2b IV constants (RFC 7693 §2.6) plus the all-ones finalization
/// mask at index 8.  Lives in Rust because the kernel's source language
/// caps hex literals at i64::MAX; passing them through as a `*const u64`
/// also matches the data-vs-algorithm split used by every other kernel
/// in Olorin.
const CONSTANTS: [u64; 9] = [
    0x6A09E667F3BCC908,
    0xBB67AE8584CAA73B,
    0x3C6EF372FE94F82B,
    0xA54FF53A5F1D36F1,
    0x510E527FADE682D1,
    0x9B05688C2B3E6C1F,
    0x1F83D9ABFB41BD6B,
    0x5BE0CD19137E2179,
    0xFFFFFFFFFFFFFFFF,
];

/// Streaming Blake2b hasher, parameterised by the desired output length
/// (1..=64 bytes).  No-key, no-salt, no-personalisation — the parameter
/// block reduces to `0x01010000 ^ nn` mixed into `h[0]`.
pub struct Hasher {
    h: [u64; STATE_WORDS],
    buf: [u8; BLOCK_BYTES],
    buf_len: usize,
    bytes_seen: u64,
    out_len: u8,
}

impl Hasher {
    /// Initialise a Blake2b hasher producing `out_len` bytes.
    /// Panics if `out_len` is 0 or > 64.
    pub fn new(out_len: usize) -> Self {
        assert!((1..=64).contains(&out_len), "Blake2b out_len must be 1..=64");
        let mut h = [
            CONSTANTS[0], CONSTANTS[1], CONSTANTS[2], CONSTANTS[3],
            CONSTANTS[4], CONSTANTS[5], CONSTANTS[6], CONSTANTS[7],
        ];
        // Parameter block XOR (RFC 7693 §2.5): digest_length=out_len,
        // key_length=0, fanout=1, depth=1, leaf_length=0, node_offset=0,
        // node_depth=0, inner_length=0, reserved/salt/personal=0.
        // The non-zero bytes occupy h[0]'s low 32 bits: 0x0101_kk_nn.
        h[0] ^= 0x0101_0000 | (out_len as u64);
        Self {
            h,
            buf: [0u8; BLOCK_BYTES],
            buf_len: 0,
            bytes_seen: 0,
            out_len: out_len as u8,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        // Fill the staging buffer first; only compress when we have
        // proof another byte is coming (RFC 7693 §3.3 — the final
        // block must run through F with f=1, so we never compress the
        // current tail until we know the next byte exists).
        if self.buf_len > 0 {
            let take = (BLOCK_BYTES - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == BLOCK_BYTES && !data.is_empty() {
                self.bytes_seen = self.bytes_seen.wrapping_add(BLOCK_BYTES as u64);
                let block = bytes_to_words(&self.buf);
                unsafe { compress(&mut self.h, &block, self.bytes_seen, false); }
                self.buf_len = 0;
            }
        }

        while data.len() > BLOCK_BYTES {
            self.bytes_seen = self.bytes_seen.wrapping_add(BLOCK_BYTES as u64);
            let block = bytes_to_words(&data[..BLOCK_BYTES]);
            unsafe { compress(&mut self.h, &block, self.bytes_seen, false); }
            data = &data[BLOCK_BYTES..];
        }

        if !data.is_empty() {
            self.buf[self.buf_len..self.buf_len + data.len()].copy_from_slice(data);
            self.buf_len += data.len();
        }
    }

    pub fn finalize(mut self, out: &mut [u8]) {
        assert_eq!(out.len(), self.out_len as usize, "output length mismatch");

        // Zero-pad the tail to 128 bytes; the counter still reflects
        // only the real byte count.
        for b in &mut self.buf[self.buf_len..] {
            *b = 0;
        }
        self.bytes_seen = self.bytes_seen.wrapping_add(self.buf_len as u64);
        let block = bytes_to_words(&self.buf);
        unsafe { compress(&mut self.h, &block, self.bytes_seen, true); }

        for (i, b) in out.iter_mut().enumerate() {
            *b = (self.h[i / 8] >> ((i % 8) * 8)) as u8;
        }
    }
}

/// One-shot Blake2b hash.  `out.len()` must be 1..=64.
pub fn hash(input: &[u8], out: &mut [u8]) {
    let mut h = Hasher::new(out.len());
    h.update(input);
    h.finalize(out);
}

fn bytes_to_words(block: &[u8]) -> [u64; BLOCK_WORDS] {
    debug_assert_eq!(block.len(), BLOCK_BYTES);
    let mut words = [0u64; BLOCK_WORDS];
    for (i, w) in words.iter_mut().enumerate() {
        let off = i * 8;
        *w = u64::from_le_bytes(block[off..off + 8].try_into().unwrap());
    }
    words
}

/// # Safety
/// `h` and `block` must hold STATE_WORDS / BLOCK_WORDS u64s respectively.
unsafe fn compress(
    h: &mut [u64; STATE_WORDS],
    block: &[u64; BLOCK_WORDS],
    counter: u64,
    is_final: bool,
) {
    let mut v_scratch = [0u64; 16];
    ffi::blake2b_compress(
        h.as_mut_ptr(),
        block.as_ptr(),
        CONSTANTS.as_ptr(),
        counter,
        0,
        if is_final { 1 } else { 0 },
        v_scratch.as_mut_ptr(),
    );
}

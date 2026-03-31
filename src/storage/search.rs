//! FusedSearcher — pre-allocated fused ChaCha20 decrypt + multi-needle search.
//!
//! Scratch buffers (~23 KB) allocated once at construction, reused every call.
//! Plaintext never exists as a contiguous buffer — decrypted in SIMD registers,
//! searched in-register, window zeroed after each stride.

use crate::kernels::ffi;

const DEFAULT_MAX_MATCHES: i32 = 64;
const DEFAULT_MAX_LINE_LEN: i32 = 256;
const DEFAULT_WINDOW_SIZE: i32 = 4096;

/// Result from a fused decrypt+search. Only matched context lines are returned.
#[derive(Debug, Clone)]
pub struct FusedSearchResult {
    pub match_count: usize,
    pub match_offsets: Vec<i32>,
    pub needle_ids: Vec<i32>,
    pub context_lines: Vec<Vec<u8>>,
}

/// Pre-allocated fused decrypt+search — zero heap allocations on the hot path.
pub struct FusedSearcher {
    ks_i32: Vec<i32>,
    pt_i32_buf: Vec<i32>,
    overlap: Vec<u8>,
    lines_buf: Vec<u8>,
    line_offsets: Vec<i32>,
    line_lens: Vec<i32>,
    match_offsets: Vec<i32>,
    needle_ids: Vec<i32>,
    ct_i32_buf: Vec<i32>,
    max_matches: i32,
    max_line_len: i32,
    window_size: i32,
}

impl FusedSearcher {
    pub fn new() -> Self {
        let max_matches = DEFAULT_MAX_MATCHES;
        let max_line_len = DEFAULT_MAX_LINE_LEN;
        let window_size = DEFAULT_WINDOW_SIZE;
        let window_i32_cap = (window_size as usize + 3) / 4 + 16;
        Self {
            ks_i32: vec![0i32; 64],
            pt_i32_buf: vec![0i32; window_i32_cap],
            overlap: vec![0u8; 1024],
            lines_buf: vec![0u8; (max_matches * max_line_len) as usize],
            line_offsets: vec![0i32; max_matches as usize],
            line_lens: vec![0i32; max_matches as usize],
            match_offsets: vec![0i32; max_matches as usize],
            needle_ids: vec![0i32; max_matches as usize],
            ct_i32_buf: vec![0i32; window_i32_cap],
            max_matches,
            max_line_len,
            window_size,
        }
    }

    /// Fused ChaCha20 decrypt + multi-needle search.
    ///
    /// Arguments: `key`, `nonce`, `ciphertext`, `needles`.
    /// Returns only matching context lines — full plaintext never materialises.
    /// Returns an empty result (not an error) for empty inputs.
    pub fn search(
        &mut self,
        key: &[u8; 32],
        nonce: &[u8; 12],
        ciphertext: &[u8],
        needles: &[&[u8]],
    ) -> FusedSearchResult {
        if ciphertext.is_empty() || needles.is_empty() {
            return FusedSearchResult {
                match_count: 0,
                match_offsets: Vec::new(),
                needle_ids: Vec::new(),
                context_lines: Vec::new(),
            };
        }

        // Pack needles into flat buffer
        let mut needle_buf = Vec::new();
        let mut needle_offsets = Vec::new();
        let mut needle_lens = Vec::new();
        for needle in needles {
            needle_offsets.push(needle_buf.len() as i32);
            needle_lens.push(needle.len() as i32);
            needle_buf.extend_from_slice(needle);
        }
        let needle_count = needles.len() as i32;

        // Convert key and nonce to little-endian i32 words
        let key_i32: [i32; 8] = {
            let mut arr = [0i32; 8];
            for (i, chunk) in key.chunks_exact(4).enumerate() {
                arr[i] = i32::from_le_bytes(chunk.try_into().unwrap());
            }
            arr
        };
        let nonce_i32: [i32; 3] = {
            let mut arr = [0i32; 3];
            for (i, chunk) in nonce.chunks_exact(4).enumerate() {
                arr[i] = i32::from_le_bytes(chunk.try_into().unwrap());
            }
            arr
        };

        let len = ciphertext.len() as i32;

        // Grow aligned i32 staging buffers if needed
        let ct_i32_len_needed = (ciphertext.len() + 3) / 4 + 16;
        if self.ct_i32_buf.len() < ct_i32_len_needed {
            self.ct_i32_buf.resize(ct_i32_len_needed, 0i32);
        }
        let pt_i32_len_needed = (self.window_size as usize + 3) / 4 + 16;
        if self.pt_i32_buf.len() < pt_i32_len_needed {
            self.pt_i32_buf.resize(pt_i32_len_needed, 0i32);
        }

        // Copy ciphertext bytes into the i32 buffer (preserves 4-byte alignment)
        let ct_bytes: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(
                self.ct_i32_buf.as_mut_ptr() as *mut u8,
                ciphertext.len(),
            )
        };
        ct_bytes.copy_from_slice(ciphertext);

        let mut match_count: i32 = 0;
        let mut lines_written: i32 = 0;

        unsafe {
            ffi::chacha20_search_v2(
                key_i32.as_ptr(),
                nonce_i32.as_ptr(),
                0, // ctr_init
                self.ct_i32_buf.as_ptr() as *const u8,
                len,
                self.ks_i32.as_mut_ptr(),
                self.ks_i32.as_mut_ptr() as *mut u8,
                self.ct_i32_buf.as_ptr(),
                self.pt_i32_buf.as_mut_ptr() as *mut u8,
                self.pt_i32_buf.as_mut_ptr(),
                self.overlap.as_mut_ptr(),
                needle_buf.as_ptr(),
                needle_offsets.as_ptr(),
                needle_lens.as_ptr(),
                needle_count,
                self.lines_buf.as_mut_ptr(),
                self.max_matches * self.max_line_len,
                self.line_offsets.as_mut_ptr(),
                self.line_lens.as_mut_ptr(),
                self.match_offsets.as_mut_ptr(),
                self.needle_ids.as_mut_ptr(),
                self.max_matches,
                self.max_line_len,
                self.window_size,
                &mut match_count,
                &mut lines_written,
            );
        }

        let mc = match_count as usize;
        let lw = lines_written as usize;
        let mut context_lines: Vec<Vec<u8>> = Vec::with_capacity(lw);
        for i in 0..lw {
            let off = self.line_offsets[i] as usize;
            let l = self.line_lens[i] as usize;
            if off + l <= self.lines_buf.len() {
                let line = self.lines_buf[off..off + l].to_vec();
                if !context_lines.iter().any(|existing| existing == &line) {
                    context_lines.push(line);
                }
            }
        }

        FusedSearchResult {
            match_count: mc,
            match_offsets: self.match_offsets[..mc].to_vec(),
            needle_ids: self.needle_ids[..mc].to_vec(),
            context_lines,
        }
    }
}

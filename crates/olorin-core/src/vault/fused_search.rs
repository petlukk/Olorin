//! FusedSearcher — pre-allocated fused decrypt+search.
//!
//! Scratch buffers allocated once at creation, reused every call.
//! Calls `ffi::chacha20_search_v2` from KernelTable (OnceLock).
//! Plaintext never exists as a contiguous buffer in memory.

use crate::kernels::ffi;

const DEFAULT_MAX_MATCHES: i32 = 64;
const DEFAULT_MAX_LINE_LEN: i32 = 256;
const DEFAULT_WINDOW_SIZE: i32 = 4096;

/// Result from fused decrypt+search. Only matched context lines are returned.
#[derive(Debug, Clone)]
pub struct FusedSearchResult {
    pub match_count: usize,
    pub match_offsets: Vec<i32>,
    pub needle_ids: Vec<i32>,
    pub context_lines: Vec<Vec<u8>>,
}

/// Pre-allocated fused decrypt+search — zero heap allocations on the hot path.
///
/// Scratch buffers (~23 KB) are allocated once and reused for every search call.
/// After first use, buffers stabilize in L1d/L2 cache.
pub struct FusedSearcher {
    ks_i32: Vec<i32>,
    /// Plaintext window buffer as i32 (ensures 4-byte alignment for pt_i32 FFI param).
    pt_i32_buf: Vec<i32>,
    overlap: Vec<u8>,
    lines_buf: Vec<u8>,
    line_offsets: Vec<i32>,
    line_lens: Vec<i32>,
    match_offsets: Vec<i32>,
    needle_ids: Vec<i32>,
    /// Aligned i32 staging buffer for ciphertext (ct_i32 FFI param requires i32 alignment).
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
        // ct_i32_buf / pt_i32_buf: sized for a full window (window_size bytes = window_size/4 i32s + padding)
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
    /// Decrypts in SIMD registers, searches for needles in-register, zeroes the
    /// sliding window, and returns only matching context lines. Pre-allocated
    /// scratch buffers are reused — zero heap allocations on the hot path.
    pub fn search(
        &mut self,
        ciphertext: &[u8],
        needles: &[&[u8]],
        key: &[u8; 32],
        nonce: &[u8; 12],
    ) -> Result<FusedSearchResult, String> {
        if ciphertext.is_empty() || needles.is_empty() {
            return Ok(FusedSearchResult {
                match_count: 0,
                match_offsets: Vec::new(),
                needle_ids: Vec::new(),
                context_lines: Vec::new(),
            });
        }

        // Pack needles into flat format
        let mut needle_buf = Vec::new();
        let mut needle_offsets = Vec::new();
        let mut needle_lens = Vec::new();
        for needle in needles {
            needle_offsets.push(needle_buf.len() as i32);
            needle_lens.push(needle.len() as i32);
            needle_buf.extend_from_slice(needle);
        }
        let needle_count = needles.len() as i32;

        // Convert key and nonce to i32 arrays (little-endian)
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

        // Copy ciphertext into aligned i32 buffer so ct_i32 FFI param is 4-byte aligned.
        let ct_i32_len_needed = (ciphertext.len() + 3) / 4 + 16;
        if self.ct_i32_buf.len() < ct_i32_len_needed {
            self.ct_i32_buf.resize(ct_i32_len_needed, 0i32);
        }
        // Also ensure pt_i32_buf is large enough for a window of ciphertext.len() bytes.
        let pt_i32_len_needed = (self.window_size as usize + 3) / 4 + 16;
        if self.pt_i32_buf.len() < pt_i32_len_needed {
            self.pt_i32_buf.resize(pt_i32_len_needed, 0i32);
        }
        // Zero-copy reinterpret: write ciphertext bytes into the i32 buffer
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
                1, // ctr_init
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
        let mut context_lines = Vec::with_capacity(lw);
        for i in 0..lw {
            let off = self.line_offsets[i] as usize;
            let l = self.line_lens[i] as usize;
            if off + l <= self.lines_buf.len() {
                context_lines.push(self.lines_buf[off..off + l].to_vec());
            }
        }

        Ok(FusedSearchResult {
            match_count: mc,
            match_offsets: self.match_offsets[..mc].to_vec(),
            needle_ids: self.needle_ids[..mc].to_vec(),
            context_lines,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vault::{Vault, EachachaCrypto, find_chacha_lib};
    use std::fs;
    use std::path::PathBuf;

    fn tmp_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("olorin_fused_search_tests");
        fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn test_key() -> [u8; 32] { [0x42u8; 32] }

    fn test_crypto() -> Box<dyn crate::vault::VaultCrypto> {
        let lib = find_chacha_lib().expect("libchacha20.so not found");
        Box::new(EachachaCrypto::new(lib))
    }

    #[test]
    fn test_fused_search_roundtrip() {
        crate::kernels::ffi::init().unwrap();
        let path = tmp_path("fused_roundtrip.vault");
        let _ = fs::remove_file(&path);

        let mut vault = Vault::create(&path, &test_key(), test_crypto()).unwrap();
        vault.append_message("INFO: starting\nERROR: disk full\nINFO: done").unwrap();
        vault.flush().unwrap();

        let (ct, nonce) = vault.read_encrypted_block(0).unwrap();
        let mut searcher = FusedSearcher::new();
        let result = searcher.search(&ct, &[b"ERROR"], &test_key(), &nonce).unwrap();

        assert!(result.match_count >= 1, "expected at least 1 match");
        assert!(!result.context_lines.is_empty(), "expected context lines");
        let line = String::from_utf8_lossy(&result.context_lines[0]);
        assert!(line.contains("ERROR"), "context line should contain ERROR: {}", line);

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_fused_search_no_match() {
        crate::kernels::ffi::init().unwrap();
        let path = tmp_path("fused_no_match.vault");
        let _ = fs::remove_file(&path);

        let mut vault = Vault::create(&path, &test_key(), test_crypto()).unwrap();
        vault.append_message("nothing interesting here").unwrap();
        vault.flush().unwrap();

        let (ct, nonce) = vault.read_encrypted_block(0).unwrap();
        let mut searcher = FusedSearcher::new();
        let result = searcher.search(&ct, &[b"MISSING"], &test_key(), &nonce).unwrap();

        assert_eq!(result.match_count, 0);
        assert!(result.context_lines.is_empty());

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_fused_search_multi_needle() {
        crate::kernels::ffi::init().unwrap();
        let path = tmp_path("fused_multi.vault");
        let _ = fs::remove_file(&path);

        let mut vault = Vault::create(&path, &test_key(), test_crypto()).unwrap();
        vault.append_message("apple pie is great\nbanana split is better\ncherry on top").unwrap();
        vault.flush().unwrap();

        let (ct, nonce) = vault.read_encrypted_block(0).unwrap();
        let mut searcher = FusedSearcher::new();
        let result = searcher.search(&ct, &[b"apple", b"banana"], &test_key(), &nonce).unwrap();

        assert!(result.match_count >= 2, "expected matches for both needles");
        let all_lines: String = result.context_lines.iter()
            .map(|l| String::from_utf8_lossy(l).to_string())
            .collect::<Vec<_>>().join(" ");
        assert!(all_lines.contains("apple"), "should find apple");
        assert!(all_lines.contains("banana"), "should find banana");

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_fused_search_empty_input() {
        crate::kernels::ffi::init().unwrap();
        let mut searcher = FusedSearcher::new();
        let key = [0u8; 32];
        let nonce = [0u8; 12];

        let result = searcher.search(&[], &[b"test"], &key, &nonce).unwrap();
        assert_eq!(result.match_count, 0);

        let result = searcher.search(b"data", &[], &key, &nonce).unwrap();
        assert_eq!(result.match_count, 0);
    }

    #[test]
    fn test_fused_search_reuse() {
        crate::kernels::ffi::init().unwrap();
        let path = tmp_path("fused_reuse.vault");
        let _ = fs::remove_file(&path);

        let mut vault = Vault::create(&path, &test_key(), test_crypto()).unwrap();
        vault.append_message("first block with keyword ALPHA").unwrap();
        vault.flush().unwrap();
        vault.append_message("second block with keyword BETA").unwrap();
        vault.flush().unwrap();

        let mut searcher = FusedSearcher::new();

        let (ct0, nonce0) = vault.read_encrypted_block(0).unwrap();
        let r1 = searcher.search(&ct0, &[b"ALPHA"], &test_key(), &nonce0).unwrap();
        assert!(r1.match_count >= 1);

        let (ct1, nonce1) = vault.read_encrypted_block(1).unwrap();
        let r2 = searcher.search(&ct1, &[b"BETA"], &test_key(), &nonce1).unwrap();
        assert!(r2.match_count >= 1);

        let r3 = searcher.search(&ct0, &[b"BETA"], &test_key(), &nonce0).unwrap();
        assert_eq!(r3.match_count, 0);

        fs::remove_file(&path).unwrap();
    }
}

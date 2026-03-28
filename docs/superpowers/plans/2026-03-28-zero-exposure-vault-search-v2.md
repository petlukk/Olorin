# Zero-Exposure Vault Search v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace vault's decrypt-then-search with a cache-resident fused search pipeline using pre-allocated buffers and the `chacha20_search_v2` kernel loaded once via `KernelTable`.

**Architecture:** The `chacha20_search_v2` Ea kernel joins `KernelTable` (OnceLock, loaded at startup). A `FusedSearcher` struct owns ~23 KB of pre-allocated scratch buffers that stay warm in L1d/L2. `Vault` owns a `FusedSearcher`. Search calls go through `ffi::chacha20_search_v2` — no per-call library loading, no per-call heap allocations on the hot path.

**Tech Stack:** Rust, Ea compiler, libloading (via existing KernelTable), `chacha20_search_v2.ea`

---

### File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `crates/olorin-core/build.rs` | Compile and embed `chacha20_search_v2.ea` |
| Modify | `crates/olorin-core/src/kernels/ffi.rs` | Add `SearchV2Fn` to KernelTable, public wrapper |
| Create | `crates/olorin-core/src/vault/fused_search.rs` | `FusedSearcher` struct with pre-allocated buffers |
| Modify | `crates/olorin-core/src/vault/mod.rs` | Vault owns FusedSearcher, add `read_encrypted_block()` |
| Modify | `crates/olorin-core/src/vault/search.rs` | `SearchResult.lines`, use FusedSearcher |
| Modify | `olorin-cli/src/repl.rs` | Update `recall_context()` to use `.lines` |
| Modify | `olorin-cli/src/repl_commands.rs` | Update `/recall` display to use `.lines` |
| Modify | `crates/olorin-core/tests/integration_vault_and_tools.rs` | Update vault search assertions |

---

### Task 1: Compile and embed chacha20_search_v2 kernel

**Files:**
- Modify: `crates/olorin-core/build.rs`

- [ ] **Step 1: Add chacha20_search_v2 to the kernel compilation list**

In `crates/olorin-core/build.rs`, add `"chacha20_search_v2"` to the `kernels` array (line 67-78). The kernel source lives in `kernels/eachacha/` not `kernels/olorin/`, so it needs its own compilation block before the olorin kernels loop.

After the existing chacha20 compilation block (line 59-64), add:

```rust
    // Compile chacha20_search_v2 for fused vault search.
    let search_v2_ea = eachacha_dir.join("chacha20_search_v2.ea");
    let search_v2_so = out_dir.join("libchacha20_search_v2.so");
    compile_kernel(&ea, &search_v2_ea, &search_v2_so, is_arm);
```

Then add `"chacha20_search_v2"` to the `kernels` array so it gets embedded via `include_bytes!`. But wait — the kernels array compiles from `kernels_dir` (kernels/olorin/), and chacha20_search_v2 is in `eachacha_dir`. The `.so` is already compiled to `out_dir` above, so we just need to include it in the embedding loop. The simplest approach: add it to the embedding code after the loop.

After the `for name in &kernels` loop (after line 98), add:

```rust
    // Embed chacha20_search_v2 (compiled from eachacha dir above)
    {
        let abs = fs::canonicalize(&search_v2_so)
            .unwrap_or_else(|e| panic!("cannot resolve chacha20_search_v2: {e}"));
        let bytes = fs::read(&abs)
            .unwrap_or_else(|e| panic!("cannot read chacha20_search_v2: {e}"));
        bytes.hash(&mut hasher);
        code.push_str(&format!(
            "pub const CHACHA20_SEARCH_V2: &[u8] = include_bytes!(\"{}\");\n",
            abs.display(),
        ));
    }
```

- [ ] **Step 2: Verify it compiles**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build -p olorin-core 2>&1 | tail -5`
Expected: build succeeds, no errors

- [ ] **Step 3: Commit**

```bash
git add crates/olorin-core/build.rs
git commit -m "build: compile and embed chacha20_search_v2 kernel"
```

---

### Task 2: Add chacha20_search_v2 to KernelTable

**Files:**
- Modify: `crates/olorin-core/src/kernels/ffi.rs`

- [ ] **Step 1: Add type alias for the kernel signature**

After line 32 (the `JlProjectBatchFn` type), add:

```rust
type SearchV2Fn = unsafe extern "C" fn(
    *const i32, *const i32, i32,        // key, nonce, ctr_init
    *const u8, i32,                      // ct_u8, len
    *mut i32, *mut u8,                   // ks_i32, ks_u8
    *const i32,                          // ct_i32
    *mut u8, *mut i32,                   // pt_buf, pt_i32
    *mut u8,                             // overlap
    *const u8, *const i32,               // needles, needle_offsets
    *const i32, i32,                     // needle_lens, needle_count
    *mut u8, i32,                        // lines_buf, lines_buf_cap
    *mut i32, *mut i32,                  // line_offsets, line_lens
    *mut i32, *mut i32,                  // match_offsets, needle_ids
    i32, i32, i32,                       // max_matches, max_line_len, window_size
    *mut i32, *mut i32,                  // match_count, lines_written
);
```

- [ ] **Step 2: Add field to KernelTable**

In the `KernelTable` struct (line 34-54), add after `jl_project_batch`:

```rust
    chacha20_search_v2: SearchV2Fn,
```

- [ ] **Step 3: Add to extract_kernels**

In `extract_kernels()` (line 103-114), add to the kernels array:

```rust
        ("libchacha20_search_v2.so", embedded::CHACHA20_SEARCH_V2),
```

- [ ] **Step 4: Add to load_kernels**

In `load_kernels()` (line 128-213), after `let jl_project_lib = load("jl_project")?;` (line 157), add:

```rust
    let chacha20_search_v2_lib = load("chacha20_search_v2")?;
```

In the `KernelTable` construction (inside the `unsafe` block), before `_libs`, add:

```rust
            chacha20_search_v2: std::mem::transmute(
                sym(&chacha20_search_v2_lib, b"chacha20_search_v2\0")?),
```

Add `chacha20_search_v2_lib` to the `_libs` vec:

```rust
            _libs: vec![
                byte_classifier, json_scanner, command_router,
                leak_scanner, sanitizer, fused_safety, search,
                turbo_rotate_lib, jl_project_lib, chacha20_search_v2_lib,
            ],
```

- [ ] **Step 5: Add public wrapper**

After the `jl_project_batch` wrapper (after line 311), add:

```rust
#[allow(clippy::too_many_arguments)]
pub unsafe fn chacha20_search_v2(
    key: *const i32, nonce: *const i32, ctr_init: i32,
    ct_u8: *const u8, len: i32,
    ks_i32: *mut i32, ks_u8: *mut u8,
    ct_i32: *const i32,
    pt_buf: *mut u8, pt_i32: *mut i32,
    overlap: *mut u8,
    needles: *const u8, needle_offsets: *const i32,
    needle_lens: *const i32, needle_count: i32,
    lines_buf: *mut u8, lines_buf_cap: i32,
    line_offsets: *mut i32, line_lens: *mut i32,
    match_offsets: *mut i32, needle_ids: *mut i32,
    max_matches: i32, max_line_len: i32, window_size: i32,
    match_count: *mut i32, lines_written: *mut i32,
) {
    (k().chacha20_search_v2)(
        key, nonce, ctr_init,
        ct_u8, len,
        ks_i32, ks_u8,
        ct_i32,
        pt_buf, pt_i32,
        overlap,
        needles, needle_offsets,
        needle_lens, needle_count,
        lines_buf, lines_buf_cap,
        line_offsets, line_lens,
        match_offsets, needle_ids,
        max_matches, max_line_len, window_size,
        match_count, lines_written,
    );
}
```

- [ ] **Step 6: Verify it compiles**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build -p olorin-core 2>&1 | tail -5`
Expected: build succeeds

- [ ] **Step 7: Commit**

```bash
git add crates/olorin-core/src/kernels/ffi.rs
git commit -m "feat(ffi): add chacha20_search_v2 to KernelTable"
```

---

### Task 3: Create FusedSearcher struct

**Files:**
- Create: `crates/olorin-core/src/vault/fused_search.rs`
- Modify: `crates/olorin-core/src/vault/mod.rs` (add `pub mod fused_search;`)

- [ ] **Step 1: Write failing test**

Create `crates/olorin-core/src/vault/fused_search.rs` with the test first:

```rust
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
    pt_buf: Vec<u8>,
    overlap: Vec<u8>,
    lines_buf: Vec<u8>,
    line_offsets: Vec<i32>,
    line_lens: Vec<i32>,
    match_offsets: Vec<i32>,
    needle_ids: Vec<i32>,
    max_matches: i32,
    max_line_len: i32,
    window_size: i32,
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
```

- [ ] **Step 2: Add module declaration**

In `crates/olorin-core/src/vault/mod.rs`, after `pub mod search;` (line 6), add:

```rust
pub mod fused_search;
```

- [ ] **Step 3: Add read_encrypted_block to Vault**

In `crates/olorin-core/src/vault/mod.rs`, after `decrypt_block()` (after line 262), add:

```rust
    /// Read raw encrypted block bytes and derive its nonce.
    /// Used by fused search — no decryption happens here.
    pub(crate) fn read_encrypted_block(&mut self, block_index: usize)
        -> Result<(Vec<u8>, [u8; 12]), VaultError>
    {
        if block_index >= self.index.len() {
            return Err(VaultError::InvalidFormat(
                format!("block index {} out of range (have {})", block_index, self.index.len()),
            ));
        }
        let entry = &self.index[block_index];
        let offset = entry.offset;
        let length = entry.length as usize;
        let nonce_counter = entry.nonce_counter;

        let mut ciphertext = vec![0u8; length];
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read_exact(&mut ciphertext)?;

        let nonce = derive_nonce(&self.nonce_seed, nonce_counter);
        Ok((ciphertext, nonce))
    }

    /// Get the vault encryption key (for fused search).
    pub(crate) fn key(&self) -> &[u8; 32] {
        &self.key
    }
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test -p olorin-core fused_search 2>&1 | tail -20`
Expected: FAIL — `FusedSearcher::new()` and `search()` not implemented yet

- [ ] **Step 5: Implement FusedSearcher**

In `crates/olorin-core/src/vault/fused_search.rs`, add the implementation before the `#[cfg(test)]` block:

```rust
impl FusedSearcher {
    pub fn new() -> Self {
        let max_matches = DEFAULT_MAX_MATCHES;
        let max_line_len = DEFAULT_MAX_LINE_LEN;
        let window_size = DEFAULT_WINDOW_SIZE;
        Self {
            ks_i32: vec![0i32; 64],
            pt_buf: vec![0u8; window_size as usize],
            overlap: vec![0u8; 1024],
            lines_buf: vec![0u8; (max_matches * max_line_len) as usize],
            line_offsets: vec![0i32; max_matches as usize],
            line_lens: vec![0i32; max_matches as usize],
            match_offsets: vec![0i32; max_matches as usize],
            needle_ids: vec![0i32; max_matches as usize],
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
        let mut match_count: i32 = 0;
        let mut lines_written: i32 = 0;

        unsafe {
            ffi::chacha20_search_v2(
                key_i32.as_ptr(),
                nonce_i32.as_ptr(),
                1, // ctr_init
                ciphertext.as_ptr(),
                len,
                self.ks_i32.as_mut_ptr(),
                self.ks_i32.as_mut_ptr() as *mut u8,
                ciphertext.as_ptr() as *const i32,
                self.pt_buf.as_mut_ptr(),
                self.pt_buf.as_mut_ptr() as *mut i32,
                self.overlap.as_mut_ptr(),
                needle_buf.as_ptr(),
                needle_offsets.as_ptr(),
                needle_lens.as_ptr(),
                needle_count,
                self.lines_buf.as_mut_ptr(),
                (self.max_matches * self.max_line_len),
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
```

- [ ] **Step 6: Run tests**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test -p olorin-core fused_search 2>&1`
Expected: all 5 tests pass

- [ ] **Step 7: Commit**

```bash
git add crates/olorin-core/src/vault/fused_search.rs crates/olorin-core/src/vault/mod.rs
git commit -m "feat(vault): FusedSearcher with pre-allocated scratch buffers"
```

---

### Task 4: Wire fused search into Vault::search()

**Files:**
- Modify: `crates/olorin-core/src/vault/search.rs`
- Modify: `crates/olorin-core/src/vault/mod.rs`

- [ ] **Step 1: Add FusedSearcher to Vault struct**

In `crates/olorin-core/src/vault/mod.rs`, add the import at the top (after line 5):

```rust
use fused_search::FusedSearcher;
```

Add `searcher` field to the `Vault` struct (after `crypto` on line 109):

```rust
    searcher: FusedSearcher,
```

In `Vault::create()` (line 142-152), add `searcher: FusedSearcher::new(),` to the `Self` construction.

In `Vault::open()` (line 173-183), add `searcher: FusedSearcher::new(),` to the `Self` construction.

- [ ] **Step 2: Rewrite search.rs**

Replace `crates/olorin-core/src/vault/search.rs` entirely:

```rust
//! Vault search — SIMD cosine similarity over byte histograms with recency boost.
//! Uses fused ChaCha20 decrypt+search: plaintext never exists in memory.

use super::{Vault, VaultError};
use super::index::{compute_histogram, normalize_histogram};
use crate::kernels::search;

const DIM: usize = 256;

/// A single search result with score and matched context lines.
/// Only lines matching the query are returned — the full block
/// is never decrypted to memory.
pub struct SearchResult {
    pub block_index: usize,
    pub score: f32,
    pub lines: Vec<String>,
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

impl Vault {
    /// Search vault for blocks most similar to the query.
    /// Returns top-k results sorted by score (descending).
    /// Uses fused decrypt+search: only matched context lines are returned.
    pub fn search(&mut self, query: &str, top_k: usize) -> Result<Vec<SearchResult>, VaultError> {
        if self.index.is_empty() {
            return Ok(vec![]);
        }

        let n = self.index.len();

        // Compute and normalize query histogram
        let query_hist = compute_histogram(query.as_bytes());
        let mut query_norm = normalize_histogram(&query_hist);
        search::normalize_vectors(&mut query_norm, DIM, 1);
        let qnorm = l2_norm(&query_norm);

        if qnorm < 1e-9 {
            return Ok(vec![]);
        }

        // Build flat buffer of normalized block histograms for SIMD batch search
        let mut vecs = vec![0.0f32; n * DIM];
        for (i, entry) in self.index.iter().enumerate() {
            let norm = normalize_histogram(&entry.histogram);
            vecs[i * DIM..(i + 1) * DIM].copy_from_slice(&norm);
        }
        search::normalize_vectors(&mut vecs, DIM, n);

        // SIMD batch cosine similarity
        let mut scores = search::batch_cosine(&query_norm, qnorm, &vecs, DIM, n);

        // Apply recency boost
        for (i, score) in scores.iter_mut().enumerate() {
            let recency = if n <= 1 {
                1.0
            } else {
                i as f32 / (n - 1) as f32
            };
            *score *= 0.85 + 0.15 * recency;
        }

        // SIMD top-k
        let (indices, top_scores) = search::top_k(&scores, top_k);

        // Collect and sort candidates
        let mut scored: Vec<(usize, f32)> = indices
            .into_iter()
            .zip(top_scores)
            .filter(|(_, s)| *s > 0.01)
            .map(|(idx, s)| (idx as usize, s))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Tokenize query into needles
        let needle_strs: Vec<&[u8]> = query.split_whitespace()
            .map(|w| w.as_bytes())
            .collect();

        // Fused decrypt+search per block
        let mut results = Vec::with_capacity(scored.len());
        for (block_idx, score) in scored {
            let (ciphertext, nonce) = self.read_encrypted_block(block_idx)?;

            let fused = self.searcher.search(
                &ciphertext,
                &needle_strs,
                self.key(),
                &nonce,
            ).map_err(|e| VaultError::Crypto(e))?;

            let lines: Vec<String> = fused.context_lines
                .into_iter()
                .map(|l| String::from_utf8_lossy(&l).to_string())
                .collect();

            results.push(SearchResult { block_index: block_idx, score, lines });
        }

        Ok(results)
    }

    /// Decrypt the last N blocks (for /teleport greeting generation).
    /// This is an explicit user action — full decrypt is intentional.
    pub fn decrypt_last_n(&mut self, n: usize) -> Result<Vec<Vec<u8>>, VaultError> {
        let start = self.index.len().saturating_sub(n);
        let mut blocks = Vec::with_capacity(n);
        for i in start..self.index.len() {
            blocks.push(self.decrypt_block(i)?);
        }
        Ok(blocks)
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use std::fs;
    use std::path::PathBuf;

    fn tmp_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("olorin_vault_search_tests");
        fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn test_key() -> [u8; 32] { [0x42u8; 32] }

    fn test_crypto() -> Box<dyn VaultCrypto> {
        let lib = find_chacha_lib().expect("libchacha20.so not found");
        Box::new(EachachaCrypto::new(lib))
    }

    #[test]
    fn test_vault_search_empty() {
        crate::kernels::ffi::init().unwrap();
        let path = tmp_path("search_empty.vault");
        let _ = fs::remove_file(&path);

        let mut vault = Vault::create(&path, &test_key(), test_crypto()).unwrap();
        let results = vault.search("anything", 5).unwrap();
        assert!(results.is_empty());

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_vault_search_finds_relevant() {
        crate::kernels::ffi::init().unwrap();
        let path = tmp_path("search_relevant.vault");
        let _ = fs::remove_file(&path);

        let mut vault = Vault::create(&path, &test_key(), test_crypto()).unwrap();

        vault.append_message("stars planets galaxies nebula cosmos astronomy telescope").unwrap();
        vault.flush().unwrap();

        vault.append_message("recipe flour sugar butter eggs bake oven kitchen cooking").unwrap();
        vault.flush().unwrap();

        vault.append_message("star constellation orbit planet astronomy celestial moon").unwrap();
        vault.flush().unwrap();

        let results = vault.search("stars planets astronomy cosmos", 3).unwrap();
        assert!(!results.is_empty());
        assert!(!results[0].lines.is_empty(), "should have context lines");

        let top = &results[0];
        assert!(top.block_index == 0 || top.block_index == 2,
            "expected astronomy block, got block {}", top.block_index);

        if results.len() >= 2 {
            assert!(results[0].score >= results[1].score);
        }

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_vault_search_recency_boost() {
        crate::kernels::ffi::init().unwrap();
        let path = tmp_path("search_recency.vault");
        let _ = fs::remove_file(&path);

        let mut vault = Vault::create(&path, &test_key(), test_crypto()).unwrap();

        let content = "identical content for recency test abcdefg";
        vault.append_message(content).unwrap();
        vault.flush().unwrap();
        vault.append_message(content).unwrap();
        vault.flush().unwrap();

        let results = vault.search(content, 2).unwrap();
        assert_eq!(results.len(), 2);

        assert_eq!(results[0].block_index, 1);
        assert_eq!(results[1].block_index, 0);
        assert!(results[0].score > results[1].score);

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_vault_decrypt_last_n() {
        let path = tmp_path("decrypt_last_n.vault");
        let _ = fs::remove_file(&path);

        let mut vault = Vault::create(&path, &test_key(), test_crypto()).unwrap();
        for i in 0..5 {
            vault.append_message(&format!("block number {}", i)).unwrap();
            vault.flush().unwrap();
        }

        let last2 = vault.decrypt_last_n(2).unwrap();
        assert_eq!(last2.len(), 2);
        assert_eq!(last2[0], b"block number 3");
        assert_eq!(last2[1], b"block number 4");

        let all = vault.decrypt_last_n(100).unwrap();
        assert_eq!(all.len(), 5);

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_vault_decrypt_last_n_empty() {
        let path = tmp_path("decrypt_last_n_empty.vault");
        let _ = fs::remove_file(&path);

        let mut vault = Vault::create(&path, &test_key(), test_crypto()).unwrap();
        let blocks = vault.decrypt_last_n(5).unwrap();
        assert!(blocks.is_empty());

        fs::remove_file(&path).unwrap();
    }
}
```

- [ ] **Step 3: Verify it compiles**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo check -p olorin-core 2>&1 | tail -10`
Expected: may have errors from consumers still using `.text` — those are fixed in Task 5

- [ ] **Step 4: Commit**

```bash
git add crates/olorin-core/src/vault/search.rs crates/olorin-core/src/vault/mod.rs
git commit -m "feat(vault): zero-exposure search via FusedSearcher"
```

---

### Task 5: Update consumers of SearchResult

**Files:**
- Modify: `olorin-cli/src/repl.rs:188-192`
- Modify: `olorin-cli/src/repl_commands.rs:160-168`
- Modify: `crates/olorin-core/tests/integration_vault_and_tools.rs:42-73`

- [ ] **Step 1: Update repl.rs recall_context()**

In `olorin-cli/src/repl.rs`, replace lines 188-192:

```rust
        // Before:
        let ctx: Vec<String> = filtered
            .iter()
            .map(|r| String::from_utf8_lossy(&r.text).to_string())
            .collect();
        format!("\n[Recall context]\n{}\n", ctx.join("\n---\n"))

        // After:
        let ctx: Vec<String> = filtered
            .iter()
            .flat_map(|r| r.lines.iter().cloned())
            .collect();
        format!("\n[Recall context]\n{}\n", ctx.join("\n"))
```

- [ ] **Step 2: Update repl_commands.rs /recall display**

In `olorin-cli/src/repl_commands.rs`, replace lines 160-168:

```rust
                // Before:
                for (i, r) in results.iter().enumerate() {
                    let text = String::from_utf8_lossy(&r.text);
                    let preview: String = text.chars().take(120).collect();
                    out.push_str(&format!(
                        "  [{}] (score {:.2}) {}\n",
                        i + 1,
                        r.score,
                        preview
                    ));
                }

                // After:
                for (i, r) in results.iter().enumerate() {
                    let preview = if r.lines.is_empty() {
                        "(no matching lines)".to_string()
                    } else {
                        r.lines.iter().take(3).cloned().collect::<Vec<_>>().join(" | ")
                    };
                    let truncated: String = preview.chars().take(120).collect();
                    out.push_str(&format!(
                        "  [{}] (score {:.2}) {}\n",
                        i + 1,
                        r.score,
                        truncated
                    ));
                }
```

- [ ] **Step 3: Update integration tests**

In `crates/olorin-core/tests/integration_vault_and_tools.rs`:

Add `crate::kernels::ffi::init().unwrap();` at the start of `test_vault_full_lifecycle` (after line 10, before vault creation). Actually, since this is an integration test, use the full path:

At the very start of `test_vault_full_lifecycle()` (line 11), add:

```rust
    olorin_core::kernels::ffi::init().unwrap();
```

Replace line 44-48:

```rust
        // Before:
        let top_text = String::from_utf8_lossy(&results[0].text);
        assert!(
            top_text.contains("AVX-512") || top_text.contains("x86") || top_text.contains("zmm"),
            "Top result should be about x86: {:.80}", top_text
        );

        // After:
        let top_lines = results[0].lines.join(" ");
        assert!(
            top_lines.contains("AVX-512") || top_lines.contains("x86") || top_lines.contains("zmm"),
            "Top result should be about x86: {:.80}", top_lines
        );
```

Replace line 53-54:

```rust
        // Before:
        let arm_text = String::from_utf8_lossy(&arm_results[0].text);
        assert!(arm_text.contains("NEON") || arm_text.contains("ARM") || arm_text.contains("128"));

        // After:
        let arm_lines = arm_results[0].lines.join(" ");
        assert!(arm_lines.contains("NEON") || arm_lines.contains("ARM") || arm_lines.contains("128"));
```

Replace line 71-72:

```rust
        // Before:
        let top_text = String::from_utf8_lossy(&results[0].text);
        assert!(top_text.contains("cache") || top_text.contains("64"));

        // After:
        let top_lines = results[0].lines.join(" ");
        assert!(top_lines.contains("cache") || top_lines.contains("64"));
```

- [ ] **Step 4: Verify full workspace compiles**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo check --workspace 2>&1 | tail -5`
Expected: no errors

- [ ] **Step 5: Run full test suite**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test --workspace 2>&1 | grep "test result"`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add olorin-cli/src/repl.rs olorin-cli/src/repl_commands.rs crates/olorin-core/tests/integration_vault_and_tools.rs
git commit -m "refactor: update SearchResult consumers to use .lines"
```

---

### Task 6: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Full release build**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | tail -5`
Expected: build succeeds, no errors, no warnings

- [ ] **Step 2: Full test suite**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test --workspace 2>&1 | grep "test result"`
Expected: all pass

- [ ] **Step 3: Verify binary size hasn't bloated**

Run: `ls -lh target/release/olorin 2>/dev/null || ls -lh target/release/olorin-cli 2>/dev/null`
Expected: still in the ~400 KB range (the search_v2 kernel adds ~20-30 KB embedded)

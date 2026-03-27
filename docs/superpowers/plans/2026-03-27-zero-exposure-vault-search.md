# Zero-Exposure Vault Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace vault's decrypt-then-search with fused in-register ChaCha20 decrypt+search so plaintext never exists in memory during recall.

**Architecture:** Wire the existing `chacha20_search_v2` Ea kernel into the vault search pipeline. The eachacha crate gets a new `search_fused()` FFI wrapper, vault search calls it instead of `decrypt_block()`, and consumers get context lines instead of full plaintext. The kernel is compiled by olorin-core's build.rs and found at runtime via `find_chacha_lib` pattern.

**Tech Stack:** Rust, libloading, `chacha20_search_v2.ea` Ea kernel, eachacha crate, olorin-core vault

---

### File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `crates/olorin-core/build.rs` | Also compile `chacha20_search_v2.ea`, expose path via env var |
| Rewrite | `crates/eachacha/src/search.rs` | Replace decrypt+windows with FFI call to `chacha20_search_v2` kernel |
| Modify | `crates/olorin-core/src/vault/search.rs` | Use fused search, change `SearchResult.text` → `.lines` |
| Modify | `crates/olorin-core/src/vault/mod.rs` | Add `read_encrypted_block()` pub(crate) method |
| Modify | `olorin-cli/src/repl.rs` | Update `recall_context()` to use `.lines` |
| Modify | `olorin-cli/src/repl_commands.rs` | Update `/recall` display to use `.lines` |
| Modify | `crates/olorin-core/tests/integration_vault_and_tools.rs` | Update vault search assertions |

---

### Task 1: Compile chacha20_search_v2 in olorin-core build.rs

**Files:**
- Modify: `crates/olorin-core/build.rs`

- [ ] **Step 1: Add chacha20_search_v2 compilation alongside chacha20**

In `crates/olorin-core/build.rs`, after the chacha20 compilation block (around line 52-78), add a second block for chacha20_search_v2:

```rust
    // Compile chacha20_search_v2 for fused vault search.
    let search_ea = eachacha_dir.join("chacha20_search_v2.ea");
    let search_so = out_dir.join("libchacha20_search_v2.so");
    compile_kernel(&ea, &search_ea, &search_so, is_arm);
    let search_abs = fs::canonicalize(&search_so).unwrap();
    println!("cargo:rustc-env=CHACHA_SEARCH_LIB_PATH={}", search_abs.display());
```

- [ ] **Step 2: Verify it compiles**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build -p olorin-core 2>&1 | grep -E "error|warning:|CHACHA_SEARCH"`
Expected: no errors, the env var is set

- [ ] **Step 3: Commit**

```bash
git add crates/olorin-core/build.rs
git commit -m "build: compile chacha20_search_v2.ea in olorin-core"
```

---

### Task 2: Rewrite eachacha search.rs with fused kernel FFI

**Files:**
- Rewrite: `crates/eachacha/src/search.rs`

- [ ] **Step 1: Write failing test for fused search**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::encrypt;

    fn find_search_lib() -> PathBuf {
        // Build-time path from olorin-core build.rs
        if let Some(p) = option_env!("CHACHA_SEARCH_LIB_PATH") {
            let path = PathBuf::from(p);
            if path.is_file() { return path; }
        }
        // Runtime: search ~/.olorin/lib/
        let home = std::env::var("HOME").unwrap();
        let lib_base = PathBuf::from(&home).join(".olorin/lib");
        for entry in std::fs::read_dir(&lib_base).unwrap() {
            let entry = entry.unwrap();
            let so = entry.path().join("libchacha20_search_v2.so");
            if so.is_file() { return so; }
        }
        panic!("libchacha20_search_v2.so not found");
    }

    fn find_chacha_lib() -> PathBuf {
        if let Some(p) = option_env!("CHACHA_LIB_PATH") {
            let path = PathBuf::from(p);
            if path.is_file() { return path; }
        }
        let home = std::env::var("HOME").unwrap();
        let lib_base = PathBuf::from(&home).join(".olorin/lib");
        for entry in std::fs::read_dir(&lib_base).unwrap() {
            let entry = entry.unwrap();
            let so = entry.path().join("libchacha20.so");
            if so.is_file() { return so; }
        }
        panic!("libchacha20.so not found");
    }

    #[test]
    fn test_fused_search_single_needle() {
        let chacha_path = find_chacha_lib();
        let search_path = find_search_lib();
        let key = [0x42u8; 32];
        let nonce = [0x24u8; 12];
        let plaintext = b"INFO: starting\nERROR: disk full\nINFO: done";

        let ct = encrypt(plaintext.as_slice(), &key, &nonce, &chacha_path).unwrap();
        let result = search_fused(
            &ct, &[b"ERROR"], &key, &nonce, &search_path,
            64, 256, 4096,
        ).unwrap();

        assert!(result.match_count >= 1, "expected at least 1 match");
        assert!(!result.context_lines.is_empty(), "expected context lines");
        let line = String::from_utf8_lossy(&result.context_lines[0]);
        assert!(line.contains("ERROR"), "context line should contain ERROR: {}", line);
    }

    #[test]
    fn test_fused_search_no_match() {
        let chacha_path = find_chacha_lib();
        let search_path = find_search_lib();
        let key = [0xCCu8; 32];
        let nonce = [0xDDu8; 12];
        let plaintext = b"nothing interesting here";

        let ct = encrypt(plaintext.as_slice(), &key, &nonce, &chacha_path).unwrap();
        let result = search_fused(
            &ct, &[b"MISSING"], &key, &nonce, &search_path,
            64, 256, 4096,
        ).unwrap();

        assert_eq!(result.match_count, 0);
        assert!(result.context_lines.is_empty());
    }

    #[test]
    fn test_fused_search_multi_needle() {
        let chacha_path = find_chacha_lib();
        let search_path = find_search_lib();
        let key = [0xAAu8; 32];
        let nonce = [0xBBu8; 12];
        let plaintext = b"apple pie is great\nbanana split is better\ncherry on top";

        let ct = encrypt(plaintext.as_slice(), &key, &nonce, &chacha_path).unwrap();
        let result = search_fused(
            &ct, &[b"apple", b"banana"], &key, &nonce, &search_path,
            64, 256, 4096,
        ).unwrap();

        assert!(result.match_count >= 2, "expected matches for both needles");
        let all_lines: String = result.context_lines.iter()
            .map(|l| String::from_utf8_lossy(l).to_string())
            .collect::<Vec<_>>().join(" ");
        assert!(all_lines.contains("apple"), "should find apple");
        assert!(all_lines.contains("banana"), "should find banana");
    }

    #[test]
    fn test_fused_search_empty_input() {
        let search_path = find_search_lib();
        let key = [0u8; 32];
        let nonce = [0u8; 12];

        let result = search_fused(&[], &[b"test"], &key, &nonce, &search_path, 64, 256, 4096).unwrap();
        assert_eq!(result.match_count, 0);

        let result = search_fused(b"data", &[], &key, &nonce, &search_path, 64, 256, 4096).unwrap();
        assert_eq!(result.match_count, 0);
    }
}
```

- [ ] **Step 2: Implement search_fused()**

Replace the entire `search.rs` with the fused implementation. The key challenge is packing needles into the flat format the kernel expects.

The kernel expects:
- `needles`: all needle bytes concatenated into one buffer
- `needle_offsets`: i32 array, byte offset of each needle in the concatenated buffer
- `needle_lens`: i32 array, byte length of each needle
- `needle_count`: i32

```rust
//! Fused ChaCha20 decrypt + search — plaintext never exists in memory.
//!
//! Calls the `chacha20_search_v2` Ea SIMD kernel which decrypts in registers,
//! searches for needles, zeroes the sliding window, and copies out only
//! matching context lines.

use crate::ChachaError;
use libloading::{Library, Symbol};
use std::path::Path;

/// Result from fused decrypt+search. Only matched context lines are returned.
#[derive(Debug, Clone)]
pub struct FusedSearchResult {
    pub match_count: usize,
    pub match_offsets: Vec<i32>,
    pub needle_ids: Vec<i32>,
    pub context_lines: Vec<Vec<u8>>,
}

// Kernel signature from chacha20_search_v2.ea
type SearchV2Fn = unsafe extern "C" fn(
    /* key */ *const i32, /* nonce */ *const i32, /* ctr_init */ i32,
    /* ct_u8 */ *const u8, /* len */ i32,
    /* ks_i32 */ *mut i32, /* ks_u8 */ *mut u8,
    /* ct_i32 */ *const i32,
    /* pt_buf */ *mut u8, /* pt_i32 */ *mut i32,
    /* overlap */ *mut u8,
    /* needles */ *const u8, /* needle_offsets */ *const i32,
    /* needle_lens */ *const i32, /* needle_count */ i32,
    /* lines_buf */ *mut u8, /* lines_buf_cap */ i32,
    /* line_offsets */ *mut i32, /* line_lens */ *mut i32,
    /* match_offsets */ *mut i32, /* needle_ids */ *mut i32,
    /* max_matches */ i32, /* max_line_len */ i32, /* window_size */ i32,
    /* match_count */ *mut i32, /* lines_written */ *mut i32,
);

/// Find `libchacha20_search_v2.so` — build-time path first, then ~/.olorin/lib/.
pub fn find_search_lib() -> Option<std::path::PathBuf> {
    if let Some(p) = option_env!("CHACHA_SEARCH_LIB_PATH") {
        let path = std::path::PathBuf::from(p);
        if path.is_file() { return Some(path); }
    }
    let home = std::env::var("HOME").ok()?;
    let lib_base = std::path::PathBuf::from(&home).join(".olorin/lib");
    if !lib_base.is_dir() { return None; }
    let mut best: Option<(std::path::PathBuf, std::time::SystemTime)> = None;
    for entry in std::fs::read_dir(&lib_base).ok()? {
        let entry = entry.ok()?;
        let so = entry.path().join("libchacha20_search_v2.so");
        if so.is_file() {
            let mtime = so.metadata().ok()?.modified().ok()?;
            if best.as_ref().map_or(true, |(_, t)| mtime > *t) {
                best = Some((so, mtime));
            }
        }
    }
    best.map(|(p, _)| p)
}

/// Fused ChaCha20 decrypt + multi-needle search.
///
/// Decrypts in SIMD registers, searches for needles in-register, zeroes the
/// sliding window, and returns only matching context lines. Plaintext never
/// exists as a contiguous buffer in memory.
pub fn search_fused(
    ciphertext: &[u8],
    needles: &[&[u8]],
    key: &[u8; 32],
    nonce: &[u8; 12],
    lib_path: &Path,
    max_matches: i32,
    max_line_len: i32,
    window_size: i32,
) -> Result<FusedSearchResult, ChachaError> {
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

    // Scratch buffers
    let mut ks_i32 = vec![0i32; 64];       // keystream as i32 (256 bytes)
    let mut pt_buf = vec![0u8; window_size as usize]; // plaintext sliding window
    let mut overlap = vec![0u8; 1024];       // overlap buffer for boundary matches

    // Output buffers
    let lines_buf_cap = max_matches * max_line_len;
    let mut lines_buf = vec![0u8; lines_buf_cap as usize];
    let mut line_offsets = vec![0i32; max_matches as usize];
    let mut line_lens = vec![0i32; max_matches as usize];
    let mut match_offsets_buf = vec![0i32; max_matches as usize];
    let mut needle_ids_buf = vec![0i32; max_matches as usize];
    let mut match_count: i32 = 0;
    let mut lines_written: i32 = 0;

    // Load kernel
    let lib = unsafe {
        Library::new(lib_path)
            .map_err(|e| ChachaError::KernelLoad(e.to_string()))?
    };
    let func: Symbol<SearchV2Fn> = unsafe {
        lib.get(b"chacha20_search_v2\0")
            .map_err(|e| ChachaError::KernelLoad(e.to_string()))?
    };

    unsafe {
        func(
            key_i32.as_ptr(),
            nonce_i32.as_ptr(),
            1, // ctr_init
            ciphertext.as_ptr(),
            len,
            ks_i32.as_mut_ptr(),
            ks_i32.as_mut_ptr() as *mut u8,
            ciphertext.as_ptr() as *const i32,
            pt_buf.as_mut_ptr(),
            pt_buf.as_mut_ptr() as *mut i32,
            overlap.as_mut_ptr(),
            needle_buf.as_ptr(),
            needle_offsets.as_ptr(),
            needle_lens.as_ptr(),
            needle_count,
            lines_buf.as_mut_ptr(),
            lines_buf_cap,
            line_offsets.as_mut_ptr(),
            line_lens.as_mut_ptr(),
            match_offsets_buf.as_mut_ptr(),
            needle_ids_buf.as_mut_ptr(),
            max_matches,
            max_line_len,
            window_size,
            &mut match_count,
            &mut lines_written,
        );
    }

    // Unpack context lines
    let mc = match_count as usize;
    let lw = lines_written as usize;
    let mut context_lines = Vec::with_capacity(lw);
    for i in 0..lw {
        let off = line_offsets[i] as usize;
        let len = line_lens[i] as usize;
        if off + len <= lines_buf.len() {
            context_lines.push(lines_buf[off..off + len].to_vec());
        }
    }

    Ok(FusedSearchResult {
        match_count: mc,
        match_offsets: match_offsets_buf[..mc].to_vec(),
        needle_ids: needle_ids_buf[..mc].to_vec(),
        context_lines,
    })
}
```

- [ ] **Step 3: Run tests**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test -p eachacha 2>&1`
Expected: all tests pass (including new fused search tests)

- [ ] **Step 4: Commit**

```bash
git add crates/eachacha/src/search.rs
git commit -m "feat(eachacha): fused decrypt+search via chacha20_search_v2 kernel"
```

---

### Task 3: Add read_encrypted_block to Vault

**Files:**
- Modify: `crates/olorin-core/src/vault/mod.rs`

- [ ] **Step 1: Add pub(crate) method to read raw encrypted block + nonce**

Add after `decrypt_block()` (around line 262):

```rust
    /// Read raw encrypted block bytes and its derived nonce.
    /// Used by fused search to avoid decrypting the whole block.
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

- [ ] **Step 2: Verify it compiles**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo check -p olorin-core 2>&1`
Expected: compiles (methods may warn as unused until Task 4)

- [ ] **Step 3: Commit**

```bash
git add crates/olorin-core/src/vault/mod.rs
git commit -m "feat(vault): add read_encrypted_block for fused search"
```

---

### Task 4: Wire fused search into Vault::search()

**Files:**
- Modify: `crates/olorin-core/src/vault/search.rs`

- [ ] **Step 1: Change SearchResult and rewrite search loop**

```rust
//! Vault search — SIMD cosine similarity over byte histograms with recency boost.
//! Uses fused ChaCha20 decrypt+search: plaintext never exists in memory.

use super::{Vault, VaultError};
use super::index::{compute_histogram, normalize_histogram};
use crate::kernels::search;
use crate::vault::find_chacha_lib;

const DIM: usize = 256;
const MAX_MATCHES: i32 = 64;
const MAX_LINE_LEN: i32 = 256;
const WINDOW_SIZE: i32 = 4096;

/// A single search result with score and matched context lines.
/// Only the lines matching the query are returned — the full block
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

        // Find the search kernel .so
        let search_lib = eachacha::search::find_search_lib()
            .ok_or_else(|| VaultError::Crypto(
                "libchacha20_search_v2.so not found".to_string()
            ))?;

        // Tokenize query into needles
        let needle_strs: Vec<&[u8]> = query.split_whitespace()
            .map(|w| w.as_bytes())
            .collect();

        // Fused decrypt+search per block
        let mut results = Vec::with_capacity(scored.len());
        for (block_idx, score) in scored {
            let (ciphertext, nonce) = self.read_encrypted_block(block_idx)?;

            let fused = eachacha::search::search_fused(
                &ciphertext,
                &needle_strs,
                self.key(),
                &nonce,
                &search_lib,
                MAX_MATCHES,
                MAX_LINE_LEN,
                WINDOW_SIZE,
            ).map_err(|e| VaultError::Crypto(e.to_string()))?;

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
```

Note: the `xxhash` integrity check is removed from the search path because fused search doesn't produce full plaintext to hash. Integrity is still verified on `decrypt_block()` (used by `/teleport`).

- [ ] **Step 2: Verify it compiles**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo check -p olorin-core 2>&1`
Expected: may have errors from consumers still using `.text` — fix in next task

- [ ] **Step 3: Commit**

```bash
git add crates/olorin-core/src/vault/search.rs
git commit -m "feat(vault): zero-exposure search via fused ChaCha20 kernel"
```

---

### Task 5: Update consumers of SearchResult

**Files:**
- Modify: `olorin-cli/src/repl.rs`
- Modify: `olorin-cli/src/repl_commands.rs`
- Modify: `crates/olorin-core/tests/integration_vault_and_tools.rs`

- [ ] **Step 1: Update repl.rs recall_context()**

In `olorin-cli/src/repl.rs`, around line 180-192, change:

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

In `olorin-cli/src/repl_commands.rs`, around line 159-167, change:

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

In `crates/olorin-core/tests/integration_vault_and_tools.rs`, the vault search assertions reference `results[0].text`. Update them to use `results[0].lines`:

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

Apply the same pattern for all `.text` → `.lines.join(" ")` replacements in the integration test.

- [ ] **Step 4: Update vault/search.rs tests**

The existing tests in `vault/search.rs` need updating. `test_vault_search_finds_relevant` currently checks `top.block_index` which still works (no change needed). `test_vault_search_recency_boost` checks scores (no change needed). The test `test_vault_decrypt_last_n` doesn't use SearchResult at all.

But `test_vault_search_finds_relevant` should also verify that lines are returned:

```rust
let results = vault.search("stars planets astronomy cosmos", 3).unwrap();
assert!(!results.is_empty());
assert!(!results[0].lines.is_empty(), "should have context lines");
```

- [ ] **Step 5: Run full test suite**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test --workspace 2>&1 | grep "test result"`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add olorin-cli/src/repl.rs olorin-cli/src/repl_commands.rs crates/olorin-core/tests/integration_vault_and_tools.rs crates/olorin-core/src/vault/search.rs
git commit -m "refactor: update all SearchResult consumers to use .lines"
```

---

### Task 6: Final verification and push

**Files:** none (verification only)

- [ ] **Step 1: Full workspace build**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build --release 2>&1 | grep -E "error|warning:|Finished"`
Expected: no errors, no warnings

- [ ] **Step 2: Full test suite**

Run: `PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test --workspace 2>&1 | grep "test result"`
Expected: all pass

- [ ] **Step 3: Push**

```bash
git push origin web-ui
```

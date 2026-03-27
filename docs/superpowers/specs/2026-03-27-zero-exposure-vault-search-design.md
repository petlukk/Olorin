# Zero-Exposure Vault Search

Replace decrypt-then-search with fused in-register ChaCha20 decrypt+search so plaintext never exists in memory during recall. Only matched context lines are returned.

## Problem

`Vault::search()` currently calls `decrypt_block()` which decrypts entire blocks to `Vec<u8>` in RAM. This defeats the purpose of an encrypted vault — 100% of block content is exposed as plaintext during every recall query.

## Solution

Wire the existing `chacha20_search_v2` Ea kernel into the vault search pipeline. The kernel decrypts in SIMD registers, searches for needles in-register, zeroes the sliding window after each pass, and copies out only matching context lines. ~95% of block content never exists as plaintext.

## What Changes

### `eachacha/src/search.rs` — new `search_fused()`

Replace the current decrypt+windows implementation with a proper FFI call to the `chacha20_search_v2` kernel.

```rust
pub struct FusedSearchResult {
    pub match_count: usize,
    pub match_offsets: Vec<i32>,
    pub needle_ids: Vec<i32>,
    pub context_lines: Vec<Vec<u8>>,
}

pub fn search_fused(
    ciphertext: &[u8],
    needles: &[&[u8]],
    key: &[u8; 32],
    nonce: &[u8; 12],
    lib_path: &Path,
    max_matches: i32,
    max_line_len: i32,
    window_size: i32,
) -> Result<FusedSearchResult, ChachaError>
```

The kernel signature from `chacha20_search_v2.ea`:

```
chacha20_search_v2(
    key, nonce, ctr_init,
    ct_u8, len,
    ks_i32, ks_u8,           // scratch: keystream
    ct_i32,                   // ciphertext as i32 view
    pt_buf, pt_i32,           // scratch: sliding window plaintext
    overlap,                  // scratch: overlap buffer
    needles, needle_offsets,  // packed needle data
    needle_lens, needle_count,
    lines_buf, lines_buf_cap, // output: context lines buffer
    line_offsets, line_lens,   // output: per-line offset+length
    match_offsets, needle_ids, // output: per-match offset+needle
    max_matches, max_line_len, window_size,
    match_count, lines_written // output: counts
)
```

The Rust wrapper allocates scratch buffers, packs needles into the flat format the kernel expects, calls the kernel, and unpacks results into `FusedSearchResult`.

The old `search()` function is removed — it was never called from production code and its decrypt-then-search approach is the anti-pattern we're eliminating.

### `vault/search.rs` — `Vault::search()` uses fused search

Current flow:
1. Histogram match → top-k block indices
2. `decrypt_block(idx)` → full plaintext `Vec<u8>`
3. Return plaintext in `SearchResult`

New flow:
1. Histogram match → top-k block indices (unchanged)
2. Tokenize query into needles (split on whitespace)
3. Per block: `search_fused(encrypted_block, needles, key, nonce, ...)` → context lines only
4. Return context lines in `SearchResult`

### `SearchResult` changes

```rust
// Before
pub struct SearchResult {
    pub block_index: usize,
    pub score: f32,
    pub text: Vec<u8>,        // full decrypted block
}

// After
pub struct SearchResult {
    pub block_index: usize,
    pub score: f32,
    pub lines: Vec<String>,   // only matched context lines
}
```

### Consumers of `SearchResult`

Check all code that reads `SearchResult.text` and update to use `SearchResult.lines`. The main consumer is `recall.rs` / `synthesize_context()` which injects recall results into the LLM prompt.

## What Does NOT Change

- **Index/histogram scanning** — already uses unencrypted metadata (histograms stored in index entries)
- **JL projection, batch_cosine, top_k** — unchanged SIMD operations
- **`decrypt_block()`** — kept for explicit user actions (e.g. `/teleport` greeting via `decrypt_last_n`)
- **`chacha20_search_v2.ea`** kernel source — unchanged, already compiled and embedded
- **`.eakv` / vault binary format** — unchanged
- **`chacha20.ea`** encrypt/decrypt kernel — still used by `EachachaCrypto` for vault block write/read

## Security Model

| Operation | Exposure | Method |
|-----------|----------|--------|
| Recall (background, LLM context) | Zero-exposure: only matched lines | `chacha20_search_v2` fused kernel |
| `/teleport` (user requests history) | Full decrypt | `decrypt_block()` — user explicitly asked to see data |
| Vault write | Encrypted immediately | `EachachaCrypto` via `chacha20.ea` |

## Data Flow

```
query "AVX-512 SIMD"
    |
    v
tokenize -> needles: ["AVX-512", "SIMD"]
    |
    v
histogram match -> block_idx: [0, 2, 5]
    |
    v
per block:
    encrypted bytes --> chacha20_search_v2 kernel
                         |-- decrypt in SIMD registers (XOR)
                         |-- needle-match in register
                         |-- zero sliding window
                         |-- copy only matched lines out
    |
    v
context_lines: ["Use 512-bit registers with zmm0-zmm31"]
    |
    v
inject in LLM prompt
```

## Defaults

- `window_size`: 4096 bytes (one page, matches kernel's sliding window design)
- `max_matches`: 64 per block
- `max_line_len`: 256 bytes (truncate long lines)

## Testing

- Roundtrip: encrypt known text, search with fused kernel, verify matched lines contain expected content
- No-match: search for nonexistent needle, verify empty result
- Multi-needle: verify correct needle_ids mapping
- Vault integration: create vault with blocks, run `vault.search()`, verify context lines without full decrypt
- Security: verify that after search, no full plaintext `Vec<u8>` exists (the old `text` field is gone)

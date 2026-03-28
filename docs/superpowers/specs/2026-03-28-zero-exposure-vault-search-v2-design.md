# Zero-Exposure Vault Search v2

Replace decrypt-then-search with a cache-resident fused search pipeline. The `chacha20_search_v2` kernel joins `KernelTable` (OnceLock, loaded once at startup). A `FusedSearcher` struct owns pre-allocated scratch buffers that stay warm in L1/L2 — zero heap allocations on the hot path. Plaintext never exists in memory during recall.

## Problem

`Vault::search()` currently calls `decrypt_block()` which decrypts entire blocks to `Vec<u8>` in RAM. Two problems:

1. **Security.** 100% of block content is exposed as plaintext during every recall query.
2. **Performance.** The previous plan (v1) fixed security but used per-call heap allocations — ~25 KB of scratch buffers allocated and freed every search call. That scatters plaintext fragments across the heap (never zeroed by `free()`) and causes cold cache misses on every call.

## Solution

Follow the established Olorin patterns:

- **Kernel in KernelTable** (like `fused_safety`, `batch_cosine`, `top_k`) — `chacha20_search_v2` embedded via `include_bytes!`, extracted at startup, symbol cached in OnceLock. Zero per-call overhead.
- **FusedSearcher struct** (like `FusedScanner`, `VectorStore`) — owns pre-allocated scratch buffers. After first use, buffers stabilize in L1d/L2 and every subsequent search is cache-hot. Same virtual addresses reused every call.
- **Vault owns FusedSearcher** (like `Agent` owns `SafetyLayer` and `VectorStore`) — lifetime is clear, no global state beyond KernelTable.

## What Changes

### `ffi.rs` — add `chacha20_search_v2` to KernelTable

New type alias for the kernel signature:

```rust
type SearchV2Fn = unsafe extern "C" fn(
    *const i32, *const i32, i32,           // key, nonce, ctr_init
    *const u8, i32,                         // ct_u8, len
    *mut i32, *mut u8,                      // ks_i32, ks_u8 (scratch: keystream)
    *const i32,                             // ct_i32
    *mut u8, *mut i32,                      // pt_buf, pt_i32 (scratch: sliding window)
    *mut u8,                                // overlap (scratch)
    *const u8, *const i32,                  // needles, needle_offsets
    *const i32, i32,                        // needle_lens, needle_count
    *mut u8, i32,                           // lines_buf, lines_buf_cap
    *mut i32, *mut i32,                     // line_offsets, line_lens
    *mut i32, *mut i32,                     // match_offsets, needle_ids
    i32, i32, i32,                          // max_matches, max_line_len, window_size
    *mut i32, *mut i32,                     // match_count, lines_written
);
```

Add to `KernelTable`:
- `chacha20_search_v2: SearchV2Fn`

Add to `extract_kernels()`:
- `("libchacha20_search_v2.so", embedded::CHACHA20_SEARCH_V2)`

Add to `load_kernels()`:
- Load `chacha20_search_v2` library, extract symbol

Add public wrapper:
- `pub unsafe fn chacha20_search_v2(...)` that calls `k().chacha20_search_v2`

### `build.rs` — compile and embed `chacha20_search_v2.ea`

Add `chacha20_search_v2.ea` to the kernel compilation list. The compiled `.so` gets embedded via `include_bytes!` in the generated `embedded_kernels.rs`, alongside the existing 10 kernels.

### `vault/fused_search.rs` — new file, FusedSearcher struct

```rust
/// Pre-allocated fused decrypt+search — zero heap allocations on the hot path.
///
/// Scratch buffers are allocated once at creation and reused for every search.
/// After first use, buffers stabilize in L1d/L2 cache (~25 KB total).
/// The kernel decrypts in SIMD registers, searches in-register, zeroes the
/// sliding window, and copies only matched context lines to the output buffer.
pub struct FusedSearcher {
    // Scratch — same virtual addresses every call, warm in cache
    ks_i32: Vec<i32>,           // 64 entries (256 bytes) — keystream
    pt_buf: Vec<u8>,            // window_size bytes — sliding window plaintext
    overlap: Vec<u8>,           // 1024 bytes — boundary match overlap

    // Output — pre-allocated, kernel writes into these
    lines_buf: Vec<u8>,         // max_matches * max_line_len
    line_offsets: Vec<i32>,     // max_matches
    line_lens: Vec<i32>,        // max_matches
    match_offsets: Vec<i32>,    // max_matches
    needle_ids: Vec<i32>,       // max_matches

    // Config
    max_matches: i32,           // 64
    max_line_len: i32,          // 256
    window_size: i32,           // 4096
}
```

**`FusedSearcher::new()`** — allocates with defaults. ~25 KB total:
- keystream: 256 B
- sliding window: 4096 B
- overlap: 1024 B
- output: 64 * 256 = 16384 B
- metadata: 64 * 4 * 4 = 1024 B

**`FusedSearcher::search(&mut self, ciphertext, needles, key, nonce) -> FusedSearchResult`**:
1. Early return if ciphertext or needles empty
2. Pack needles into flat format (concat bytes + offsets + lens) — these are small, temporary Vecs proportional to query size
3. Convert key/nonce to i32 arrays (stack, not heap)
4. Call `ffi::chacha20_search_v2` with pre-allocated scratch buffers
5. Unpack results: read match_count and lines_written from kernel output, copy matched lines into `FusedSearchResult`

```rust
pub struct FusedSearchResult {
    pub match_count: usize,
    pub match_offsets: Vec<i32>,
    pub needle_ids: Vec<i32>,
    pub context_lines: Vec<Vec<u8>>,
}
```

The result allocates — it should. It's output leaving the struct.

### `vault/mod.rs` — Vault owns FusedSearcher

```rust
pub struct Vault {
    path: PathBuf,
    file: File,
    header: VaultHeader,
    index: Vec<IndexEntry>,
    buffer: Vec<u8>,
    key: [u8; 32],
    nonce_seed: [u8; 12],
    crypto: Box<dyn VaultCrypto>,
    searcher: FusedSearcher,        // new
}
```

New `pub(crate)` method:

```rust
/// Read raw encrypted block bytes and derive its nonce.
/// Used by fused search — no decryption happens here.
pub(crate) fn read_encrypted_block(&mut self, block_index: usize)
    -> Result<(Vec<u8>, [u8; 12]), VaultError>
```

### `vault/search.rs` — use FusedSearcher

`SearchResult` changes:

```rust
pub struct SearchResult {
    pub block_index: usize,
    pub score: f32,
    pub lines: Vec<String>,    // only matched context lines (was: text: Vec<u8>)
}
```

`Vault::search()` new flow:
1. Histogram match → top-k block indices (unchanged — SIMD via batch_cosine, top_k)
2. Tokenize query into needles (split on whitespace)
3. Per block: `read_encrypted_block()` → `self.searcher.search(ct, needles, key, nonce)` → context lines
4. Return `SearchResult { block_index, score, lines }`

The xxhash integrity check is removed from the search path — fused search doesn't produce full plaintext to hash. Integrity is still verified on `decrypt_block()` (used by `/teleport`).

### Consumers of SearchResult

- `repl.rs` — `recall_context()`: `.text` → `.lines`
- `repl_commands.rs` — `/recall` display: `.text` → `.lines`
- `integration_vault_and_tools.rs` — assertions: `.text` → `.lines`

## What Does NOT Change

- **Index/histogram scanning** — already uses unencrypted metadata
- **JL projection, batch_cosine, top_k** — unchanged SIMD operations
- **`decrypt_block()`** — kept for explicit user actions (`/teleport` via `decrypt_last_n`)
- **`chacha20_search_v2.ea` kernel source** — unchanged, already compiled
- **Vault binary format** — unchanged
- **`chacha20.ea` encrypt/decrypt kernel** — still used by `EachachaCrypto` for vault write/read
- **eachacha crate** — its own `search.rs` is eachacha's API, not Olorin's concern

## Security Model

| Operation | Exposure | Method |
|-----------|----------|--------|
| Recall (background, auto) | Zero-exposure: only matched lines | `chacha20_search_v2` via FusedSearcher |
| `/teleport` (user requests) | Full decrypt | `decrypt_block()` — explicit user action |
| Vault write | Encrypted immediately | `EachachaCrypto` via `chacha20.ea` |

Cache-resident scratch buffers mean plaintext fragments stay at the same virtual addresses and get overwritten on every search call. No heap scatter. `free()` never sees plaintext.

## Data Flow

```
query "AVX-512 SIMD"
    |
    v
tokenize -> needles: ["AVX-512", "SIMD"]
    |
    v
histogram match -> block_idx: [0, 2, 5]   (SIMD: batch_cosine + top_k)
    |
    v
per block:
    read_encrypted_block(idx) -> (ciphertext, nonce)
    |
    v
    FusedSearcher.search(ct, needles, key, nonce)
        |-- ffi::chacha20_search_v2 (KernelTable, OnceLock)
        |-- decrypt in SIMD registers (XOR with keystream)
        |-- needle-match in register
        |-- zero sliding window (same 4 KB buffer, warm in L1d)
        |-- copy only matched lines to pre-allocated output buffer
    |
    v
context_lines: ["Use 512-bit registers with zmm0-zmm31"]
    |
    v
SearchResult { lines: [...] }
    |
    v
inject in LLM prompt
```

## Memory Profile

FusedSearcher total: ~23 KB (fits in L1d on ARM Cortex-A76, 64 KB)

| Buffer | Size | Cache |
|--------|------|-------|
| ks_i32 (keystream) | 256 B | L1d |
| pt_buf (sliding window) | 4096 B | L1d |
| overlap | 1024 B | L1d |
| lines_buf (output) | 16384 B | L1d/L2 |
| metadata (offsets, lens, ids) | 1024 B | L1d |

After first search call, all buffers are warm. Subsequent calls: zero cache misses on scratch, zero heap allocations.

## Defaults

- `window_size`: 4096 bytes (one page, matches kernel's sliding window)
- `max_matches`: 64 per block
- `max_line_len`: 256 bytes (truncate long lines)

## Testing

- **Roundtrip:** encrypt known text, fused search with FusedSearcher, verify matched lines contain expected content
- **No-match:** search for nonexistent needle, verify empty result
- **Multi-needle:** verify correct needle_ids mapping
- **Empty input:** empty ciphertext and empty needles both return empty result
- **Vault integration:** create vault with blocks, run `vault.search()`, verify context lines without full decrypt
- **Reuse:** call search twice on same FusedSearcher, verify second call works correctly (buffers reused, not corrupted)
- **Security:** verify that `SearchResult` has no `.text` field — the old full-plaintext path is gone

# eakv C→Rust Port

Port the eakv C orchestration layer (~1000 LOC) to pure Rust, eliminating the C build dependency and aligning with the rest of Olorin's Rust + Ea architecture.

## What Gets Removed

- **Entire `csrc/` directory** — cache.c, attention.c, io.c, ggml_type.c, llama_bridge.c, all headers
- **C build machinery** in `build.rs` — cc::Build, .o file linking, C compiler dependency
- **FFI bindings** in lib.rs — extern "C" declarations, opaque eakv_cache_t pointer

### Why remove ggml_type.c and llama_bridge.c

- `ggml_type.c` implements llama.cpp ggml type callbacks (quantize_row, dequantize_row, vec_dot). These are never called from Rust or Cougar — dead code.
- `llama_bridge.c` requires llama.cpp headers and is already excluded from the build.
- Olorin uses Cougar for inference, not llama.cpp.

## New Rust Modules

### `src/cache.rs` (~120 LOC) — replaces cache.c

EakvCache struct with fields directly in Rust (no opaque pointer):

```rust
pub struct EakvCache {
    n_layers: i32,
    n_kv_heads: i32,
    head_dim: i32,
    max_seq_len: i32,
    seq_len: i32,
    groups_per_token: i32,
    max_groups: i32,
    data_buf: Vec<u8>,        // single flat allocation
    kv: Vec<KvSliceInfo>,     // offset/length views into data_buf
    jl_signs: [f32; 64],      // TurboQuant sign mask
    kernels: KernelTable,     // loaded .so handles
}

struct KvSliceInfo {
    weights_offset: usize,    // into data_buf, packed Q4 nibbles
    scales_offset: usize,     // per-group f32 scales
    biases_offset: usize,     // per-group f32 biases
}
```

Functions: `create()`, `load_raw()`, `append()`, `advance()`, `clear()`, `checkpoint()`, `restore()`, `seq_len()`, `n_layers()`, `n_heads()`, `head_dim()`, `max_seq_len()`, `compression_ratio()`.

`gen_jl_signs()` — deterministic xorshift PRNG, identical output to C version.

`rotate_groups()` and `inverse_rotate_groups()` — Rust loops calling turbo_rotate/fwht_inplace/sign_flip kernels via KernelTable.

### `src/attention.rs` (~60 LOC) — replaces attention.c

Two functions:
- `attention_scores(cache, queries, layer, n_q_heads, n_kv_heads) → scores`
- `attention_output(cache, weights, layer, n_q_heads, n_kv_heads) → output`

Routes to the correct kernel based on:
- head_dim == 64 vs other → `_64` variant
- n_q_heads == n_kv_heads (MHA) vs != (GQA) → `_gqa` variant

Same 4-way dispatch as C version. Queries get rotate_groups before scoring; output gets inverse_rotate_groups after weighted sum.

### `src/io.rs` (~100 LOC) — replaces io.c

Binary .eakv format, backwards compatible:
- Same header struct (EAKV magic, version 1, 512-byte header)
- Same index table (layer offsets as u64)
- Same 64-byte aligned data blocks

Uses `std::fs::File` + `std::io::{Read,Write,Seek}` instead of C stdio.

### `src/kernels.rs` (~50 LOC) — kernel loading

```rust
pub struct KernelTable {
    q4_quantize: Symbol<QuantizeFn>,
    turbo_rotate: Symbol<RotateFn>,
    fwht_inplace: Symbol<FwhtFn>,
    sign_flip: Symbol<SignFlipFn>,
    fused_k_score: Symbol<KScoreFn>,
    fused_k_score_64: Symbol<KScoreFn>,
    k_score_gqa: Symbol<KScoreGqaFn>,
    k_score_gqa_64: Symbol<KScoreGqaFn>,
    fused_v_sum: Symbol<VSumFn>,
    fused_v_sum_64: Symbol<VSumFn>,
    v_sum_gqa: Symbol<VSumGqaFn>,
    v_sum_gqa_64: Symbol<VSumGqaFn>,
    // ...
}
```

Loads .so files via `libloading` from `~/.olorin/lib/` — same pattern as eachacha and olorin-core. Uses `find_eakv_libs()` searching extracted kernel directories.

### `src/lib.rs` — public API

Re-exports `EakvCache` with the same safe interface:
- `EakvCache::new()`, `checkpoint()`, `restore()`, `seq_len()`
- `save()`, `load()` — using Rust io module
- `attention_scores()`, `attention_output()` — delegates to attention module
- `load_raw()`, `append()`, `advance()`, `clear()`
- `Drop` impl zeros and drops the Vec

## Kernel Loading

Dynamic loading via `libloading`, matching the rest of Olorin:
- Kernels compiled by olorin-cli unified build.rs from `kernels/eakv/*.ea`
- Embedded in binary, extracted to `~/.olorin/lib/<version>/`
- eakv loads them at EakvCache::new() time

The .o static linking path is removed entirely.

## build.rs

Minimal — no cc crate, no C compiler. Only:
```rust
fn main() {
    println!("cargo:rustc-link-lib=dl");
}
```

Or removed entirely if libloading handles dl linking itself.

## What Does NOT Change

- **Ea kernel sources** (`kernels/eakv/*.ea`) — untouched
- **Public API** signatures — same types, same semantics
- **Binary .eakv format** — backwards compatible, same header/layout
- **TurboQuant rotation** — same kernels (turbo_rotate, fwht_inplace, sign_flip)
- **JL projection** — same deterministic sign generation
- **Q4 split-pack format** — same nibble layout (lo=val[k], hi=val[k+32])

## Testing

- Port existing C tests to Rust integration tests
- Roundtrip test: create → load_raw → checkpoint → restore → verify seq_len
- Attention test: create cache, load known data, verify scores match expected
- IO test: save → load → verify data identical
- Kernel loading test: verify all required .so files load successfully
- Cross-validate: create .eakv with new code, verify old format compatibility

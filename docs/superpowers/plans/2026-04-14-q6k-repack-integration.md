# Q6K Repack Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the proven Q6K 4-row repacked kernel into Olorin's inference pipeline, achieving ~3× speedup on the 17 Q6K ffn_down layers (estimated ~77ms savings on prefill).

**Architecture:** Copy the autoresearch Ea kernel into Olorin's `kernels/` directory. Add a Rust `q6k_repack_4row` function (pure Rust memcpy, no Ea kernel needed for repack). Wire through FFI, LayerWeights, engine_helpers, and the two matmul dispatch paths (work-stealing graph + sequential). Follow the exact pattern established by Q4K repack.

**Tech Stack:** Ea SIMD kernels, Rust FFI, existing `build.rs` auto-discovery

---

## Key differences from Q4K repack

| Aspect | Q4K repack | Q6K repack |
|--------|-----------|-----------|
| Tile | 8 rows × 1152 bytes | 4 rows × 840 bytes |
| Repack impl | Ea kernel (`q4k_repack_8x8`) | Rust memcpy (trivial layout) |
| Kernel name | `q4k_8x8_q8k_matvec` | `q6k_dot_q8k_4row_repacked` |
| d_arr | Extracted inline (pow2 lookup) | Pre-multiplied, interleaved `d_arr[blk*4+row]` |
| Row alignment | n_rows % 8 == 0 | n_rows % 4 == 0 |

## Repacked tile layout (v1 — simple concat)

Per tile = 4 consecutive Q6K blocks (840 bytes):
```
tile[0..210]   = row0 block (ql[128] + qh[64] + scales[16] + d[2])
tile[210..420] = row1 block
tile[420..630] = row2 block
tile[630..840] = row3 block
```

d_arr is interleaved: `d_arr[blk * 4 + row] = d_q6k * d_q8k`

## File structure

| File | Action | Responsibility |
|------|--------|---------------|
| `kernels/q6k_dot_repacked.ea` | Create | x86 repacked 4-row kernel |
| `kernels/q6k_dot_repacked_arm.ea` | Create | ARM NEON repacked 4-row kernel |
| `src/kernels/ffi_inference_types.rs` | Modify | Add `Q6kDot4RowRepackedFn` type |
| `src/kernels/ffi_inference.rs` | Modify | Load kernel, add wrapper fn |
| `src/inference/repack.rs` | Modify | Add `q6k_repack_4row` (Rust) |
| `src/inference/engine.rs` | Modify | Add `w_down_q6k_repacked` field |
| `src/inference/engine_helpers.rs` | Modify | Call repack for Q6K ffn_down |
| `src/inference/matmul_graph.rs` | Modify | Dispatch repacked in `q6k_matvec_ws` + `matvec_batch_ws` |
| `src/inference/matmul_seq.rs` | Modify | Dispatch repacked in `q6k_matvec` |
| `src/inference/matmul_par.rs` | Modify | Dispatch repacked in `par_q6k_matvec` |
| `src/inference/matmul.rs` | Modify | Pass repacked through `matvec`/`par_matvec`/`matvec_maybe_repacked` |

---

### Task 1: Ea kernel files

**Files:**
- Create: `kernels/q6k_dot_repacked.ea`
- Create: `kernels/q6k_dot_repacked_arm.ea`

- [ ] **Step 1: Copy ARM kernel from autoresearch**

Copy `eacompute/autoresearch/kernels/q6k_dot/kernel_repacked_arm.ea` to `kernels/q6k_dot_repacked_arm.ea`. The kernel exports `q6k_dot_q8k_4row_repacked` with signature:

```
export func q6k_dot_q8k_4row_repacked(
    packed: *restrict i8,
    q8: *restrict i8,
    bsums: *restrict i16,
    out scores: *mut f32,
    n_blocks: i32,
    d_arr: *restrict f32
)
```

- [ ] **Step 2: Write x86 kernel**

Create `kernels/q6k_dot_repacked.ea` — same logic but using `maddubs_i16` + `madd_i16` instead of `vdot_i32`, and `u8x16`/`i8x16` types with the `sbyte()` helper for scale reads. Use the same tile layout (stride-210 between rows within tile). Copy the structure from the existing `kernels/q6k_dot.ea` 4-row kernel but read from one `packed` pointer with offsets `b0=tile, b1=tile+210, b2=tile+420, b3=tile+630`.

The x86 kernel header:
```
#[cfg(x86_64)]

func sbyte(p: *restrict u8, off: i32) -> i32 {
    let v: i32 = to_i32(p[off])
    if v > 127 { return v - 256 }
    return v
}

export func q6k_dot_q8k_4row_repacked(
    packed: *restrict u8,
    q8: *restrict i8,
    bsums: *restrict i16,
    out scores: *mut f32 [cap: 4],
    n_blocks: i32,
    d_arr: *restrict f32
)
```

- [ ] **Step 3: Build to verify kernels compile**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build --release 2>&1 | tail -5
```

`build.rs` auto-discovers new `.ea` files. Verify `q6k_dot_repacked` appears in the kernel list.

- [ ] **Step 4: Commit**

```bash
git add kernels/q6k_dot_repacked.ea kernels/q6k_dot_repacked_arm.ea
git commit -m "feat: Q6K repacked 4-row dot kernels (ARM + x86)"
```

---

### Task 2: FFI wiring

**Files:**
- Modify: `src/kernels/ffi_inference_types.rs`
- Modify: `src/kernels/ffi_inference.rs`

- [ ] **Step 1: Add type alias**

In `ffi_inference_types.rs`, add after the `Q6kDot4RowFn` type:

```rust
pub type Q6kDot4RowRepackedFn = unsafe extern "C" fn(
    *const u8,   // packed (4 × Q6K blocks per tile, 840 bytes)
    *const i8,   // q8_qs
    *const i16,  // q8_bsums
    *mut f32,    // scores [4]
    i32,         // n_blocks
    *const f32,  // d_arr (interleaved: d_arr[blk*4+row])
);
```

- [ ] **Step 2: Add field to KernelTableInference**

In `ffi_inference.rs`, add to the `KernelTableInference` struct:

```rust
pub q6k_dot_q8k_4row_repacked: Q6kDot4RowRepackedFn,
```

- [ ] **Step 3: Load the kernel**

In `init_kernels()`, add:

```rust
let q6k_dot_repacked_lib = load("q6k_dot_repacked")?;
```

And in the `KernelTableInference` initialization:

```rust
q6k_dot_q8k_4row_repacked: std::mem::transmute(sym(&q6k_dot_repacked_lib, b"q6k_dot_q8k_4row_repacked\0")?),
```

Add `q6k_dot_repacked_lib` to the `libs` vec.

- [ ] **Step 4: Add public wrapper**

After the existing `q6k_dot_q8k_4row` wrapper:

```rust
#[allow(clippy::too_many_arguments)]
pub unsafe fn q6k_dot_q8k_4row_repacked(
    packed: *const u8, q8: *const i8, bsums: *const i16,
    scores: *mut f32, n_blocks: i32, d_arr: *const f32,
) {
    (k().q6k_dot_q8k_4row_repacked)(packed, q8, bsums, scores, n_blocks, d_arr)
}
```

- [ ] **Step 5: Build**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build --release 2>&1 | tail -5
```

- [ ] **Step 6: Commit**

```bash
git add src/kernels/ffi_inference.rs src/kernels/ffi_inference_types.rs
git commit -m "feat: FFI wiring for Q6K repacked dot kernel"
```

---

### Task 3: Repack function + LayerWeights

**Files:**
- Modify: `src/inference/repack.rs`
- Modify: `src/inference/engine.rs`
- Modify: `src/inference/engine_helpers.rs`

- [ ] **Step 1: Add `q6k_repack_4row` to repack.rs**

Pure Rust — no Ea kernel needed for this simple memcpy layout:

```rust
/// Repack Q6K weights: interleave 4 consecutive rows into contiguous tiles.
/// Each tile = 4 × 210 = 840 bytes. Total output = (n_rows / 4) tiles.
///
/// # Requirements
/// - `n_rows` must be a multiple of 4.
/// - `n_cols` must be a multiple of 256.
pub fn q6k_repack_4row(src: *const u8, n_rows: usize, n_cols: usize) -> Vec<u8> {
    debug_assert!(n_rows % 4 == 0, "q6k_repack_4row: n_rows ({n_rows}) must be a multiple of 4");
    debug_assert!(n_cols % 256 == 0, "q6k_repack_4row: n_cols ({n_cols}) must be a multiple of 256");

    let n_blocks = n_cols / 256;
    let row_bytes = n_blocks * 210;
    let tile_bytes = 4 * 210;  // 840 bytes per tile
    let n_quads = n_rows / 4;
    let mut dst = vec![0u8; n_quads * n_blocks * tile_bytes];

    for quad in 0..n_quads {
        for blk in 0..n_blocks {
            for r in 0..4usize {
                let src_off = (quad * 4 + r) * row_bytes + blk * 210;
                let dst_off = (quad * n_blocks + blk) * tile_bytes + r * 210;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.add(src_off),
                        dst.as_mut_ptr().add(dst_off),
                        210,
                    );
                }
            }
        }
    }
    dst
}
```

- [ ] **Step 2: Add `w_down_q6k_repacked` to LayerWeights**

In `engine.rs`, add after `w_down_repacked`:

```rust
pub w_down_q6k_repacked: Option<Vec<u8>>,
```

And in the layer construction (where `w_down_repacked: None` is set), add:

```rust
w_down_q6k_repacked: None,
```

- [ ] **Step 3: Add repack call in engine_helpers**

In `engine_helpers.rs`, add a new function `try_repack_q6k`:

```rust
pub(crate) fn try_repack_q6k(
    weight: *const u8,
    dtype: u32,
    n_rows: usize,
    n_cols: usize,
) -> Option<Vec<u8>> {
    if dtype != crate::inference::matmul::GGML_TYPE_Q6_K { return None; }
    if n_rows % 4 != 0 { return None; }
    if n_cols % 256 != 0 { return None; }
    Some(crate::inference::repack::q6k_repack_4row(weight, n_rows, n_cols))
}
```

In `populate_q4k_repacked`, add after the PLE repack lines:

```rust
// Q6K ffn_down repack (4-row tiles)
lw.w_down_q6k_repacked = try_repack_q6k(lw.w_down, lw.w_down_dtype, hidden_dim, ffn_dim);
```

Note: ffn_down shape is `[ffn_dim, hidden_dim]` in GGUF, but matmul calls pass `n_rows=hidden_dim, n_cols=ffn_dim`. The repack function takes `(n_rows, n_cols)` matching the matmul convention: `n_rows=hidden_dim=1536, n_cols=ffn_dim` (6144 or 12288). Check the existing `w_down_repacked` line in `populate_q4k_repacked` to match the convention:

```rust
lw.w_down_repacked = try_repack_q4k(lw.w_down, lw.w_down_dtype, hidden_dim, ffn_dim);
```

So Q6K repack uses the same `(hidden_dim, ffn_dim)` order.

- [ ] **Step 4: Build**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build --release 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```bash
git add src/inference/repack.rs src/inference/engine.rs src/inference/engine_helpers.rs
git commit -m "feat: Q6K 4-row repack at model load time"
```

---

### Task 4: Dispatch repacked kernel in matmul paths

**Files:**
- Modify: `src/inference/matmul_graph.rs` (work-stealing paths)
- Modify: `src/inference/matmul_seq.rs` (sequential path)
- Modify: `src/inference/matmul_par.rs` (parallel thread-pool path)
- Modify: `src/inference/matmul.rs` (dispatch entry points)

The repacked kernel has a **different d_arr format**: interleaved `d_arr[blk*4+row]` instead of 4 separate `d0/d1/d2/d3` arrays. The existing d_scratch extraction code needs to write the interleaved format when calling the repacked kernel.

- [ ] **Step 1: Add repacked dispatch to `q6k_matvec_ws` in matmul_graph.rs**

The function `q6k_matvec_ws` (line ~150 in matmul_graph.rs) currently takes `weight: *const u8`. Add an `Option<&[u8]>` parameter for repacked data. When `Some`, use the repacked kernel:

Change the signature:
```rust
pub fn q6k_matvec_ws(
    weight: *const u8,
    repacked: Option<&[u8]>,  // NEW
    q8: *const i8, q8_d: *const f32, bsums: *const i16,
    output: *mut f32, d_scratch: *mut f32,
    n_rows: usize, n_cols: usize,
    current_chunk: &AtomicI32, ith: usize, nth: usize,
)
```

In the work-stealing loop body, add the repacked path before the existing code:

```rust
if let Some(packed) = repacked {
    let tile_bytes = n_blocks * 840;
    let d_interleaved = my_scratch;
    unsafe {
        // Extract interleaved d_arr for this quad
        for blk in 0..n_blocks {
            for (row_off, r) in [(0, 0), (1, 1), (2, 2), (3, 3)] {
                let w = weight.add((base_row + row_off) * row_bytes + blk * Q6K_BLOCK_BYTES + 208);
                let raw = u16::from_le_bytes([*w, *w.add(1)]);
                *d_interleaved.add(blk * 4 + r) = f16_to_f32_scalar(raw) * *q8_d.add(blk);
            }
        }
        let tile_offset = (chunk as usize) * tile_bytes;
        ffi_inference::q6k_dot_q8k_4row_repacked(
            packed.as_ptr().add(tile_offset),
            q8, bsums,
            output.add(base_row),
            n_blocks as i32,
            d_interleaved,
        );
    }
} else {
    // existing 4-pointer path
}
```

Note: `my_scratch` needs `n_blocks * 4` floats (same size as before, just interleaved differently).

- [ ] **Step 2: Add repacked dispatch to `matvec_batch_ws` Q6K branch**

In the `GGML_TYPE_Q6_K` match arm of `matvec_batch_ws` (line ~479), add the repacked path. This is the batch prefill path.

The function needs a new parameter for the repacked buffer. But `matvec_batch_ws` handles all dtypes and doesn't know about repacked. The cleanest approach: add a separate `q6k_repacked: Option<&[u8]>` parameter.

Actually, looking at the call site in `forward_batch_layer.rs` — `matvec_batch_step` already dispatches repacked Q4K via `Some(p)`. For Q6K ffn_down, it currently passes `None` (no Q4K repack for Q6K weights). We need a different approach since the Q6K repack has a different kernel signature.

The simplest change: in `forward_batch_layer.rs`, add a **separate code path for Q6K repacked ffn_down** that bypasses `matvec_batch_step` entirely:

In `forward_batch_layer.rs` section 8 (FFN down GEMM, around line 547-558), before the `matvec_batch_step` call for w_down:

```rust
// FFN down — check for Q6K repacked first
if let Some(ref q6k_packed) = lw.w_down_q6k_repacked {
    // Q6K repacked path: work-stealing with interleaved tiles
    matmul_graph::q6k_repacked_batch_ws(
        q6k_packed.as_ptr(), lw.w_down, /* for d extraction */
        state.batch_ffn_q8_qs.as_ptr(), state.batch_ffn_q8_d.as_ptr(),
        state.batch_ffn_q8_bsums.as_ptr(),
        state.batch_down.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        hd, ffn_dim, n, hd,
        current_chunk, ith, nth,
    );
} else {
    matvec_batch_step(/* existing call */);
}
```

Add `q6k_repacked_batch_ws` to `matmul_graph.rs` — similar to the existing `matvec_batch_ws` Q6K branch but using repacked tiles.

- [ ] **Step 3: Add `q6k_repacked_batch_ws` to matmul_graph.rs**

New function that processes repacked Q6K tiles with 2D chunking (quads × token chunks):

```rust
#[allow(clippy::too_many_arguments)]
pub fn q6k_repacked_batch_ws(
    packed: *const u8, weight: *const u8, /* original for d extraction */
    batch_q8_qs: *const i8, batch_q8_d: *const f32, batch_q8_bsums: *const i16,
    output: *mut f32, d_scratch: *mut f32,
    n_rows: usize, n_cols: usize, n_tokens: usize, output_stride: usize,
    current_chunk: &AtomicI32, ith: usize, _nth: usize,
) {
    let n_blocks = n_cols / Q6K_BLOCK_SIZE;
    let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
    let qs_stride = n_cols + 12;
    let full_quads = n_rows / 4;
    let tile_bytes = n_blocks * 840;

    let scratch_per = n_blocks * 4;
    let my_scratch = unsafe { d_scratch.add(ith * scratch_per) };

    let tok_chunk_size = 16usize;
    let n_tok_chunks = (n_tokens + tok_chunk_size - 1) / tok_chunk_size;
    let total_chunks = full_quads * n_tok_chunks;

    let mut chunk = ith as i32;
    while (chunk as usize) < total_chunks {
        let quad = (chunk as usize) % full_quads;
        let tok_idx = (chunk as usize) / full_quads;
        let t_start = tok_idx * tok_chunk_size;
        let t_end = (t_start + tok_chunk_size).min(n_tokens);
        let base_row = quad * 4;

        for t in t_start..t_end {
            let q8 = unsafe { batch_q8_qs.add(t * qs_stride) };
            let q8_d = unsafe { batch_q8_d.add(t * n_blocks) };
            let bsums = unsafe { batch_q8_bsums.add(t * n_blocks * 16) };
            let out_ptr = unsafe { output.add(t * output_stride + base_row) };

            unsafe {
                // Extract interleaved d_arr
                for blk in 0..n_blocks {
                    for r in 0..4usize {
                        let w = weight.add((base_row + r) * row_bytes + blk * Q6K_BLOCK_BYTES + 208);
                        let raw = u16::from_le_bytes([*w, *w.add(1)]);
                        *my_scratch.add(blk * 4 + r) = f16_to_f32_scalar(raw) * *q8_d.add(blk);
                    }
                }
                ffi_inference::q6k_dot_q8k_4row_repacked(
                    packed.add(quad * tile_bytes),
                    q8, bsums, out_ptr,
                    n_blocks as i32, my_scratch,
                );
            }
        }

        chunk = current_chunk.fetch_add(1, Ordering::Relaxed);
    }

    // Remainder rows (n_rows % 4), handled by thread 0 with original kernel
    if ith == 0 {
        let base = full_quads * 4;
        for r in 0..(n_rows % 4) {
            let row = base + r;
            for t in 0..n_tokens {
                let q8 = unsafe { batch_q8_qs.add(t * qs_stride) };
                let q8_d = unsafe { batch_q8_d.add(t * n_blocks) };
                let bsums = unsafe { batch_q8_bsums.add(t * n_blocks * 16) };
                unsafe {
                    for blk in 0..n_blocks {
                        let w = weight.add(row * row_bytes + blk * Q6K_BLOCK_BYTES + 208);
                        let raw = u16::from_le_bytes([*w, *w.add(1)]);
                        *my_scratch.add(blk) = f16_to_f32_scalar(raw) * *q8_d.add(blk);
                    }
                    let val = ffi_inference::q6k_dot_q8k(
                        weight.add(row * row_bytes), q8, bsums,
                        n_blocks as i32, my_scratch,
                    );
                    *output.add(t * output_stride + row) = val;
                }
            }
        }
    }
}
```

- [ ] **Step 4: Wire into forward_batch_layer.rs**

In the FFN down section (~line 547), add the Q6K repacked dispatch before the existing `matvec_batch_step`. Replace the current `matvec_batch_step` call for ffn_down with:

```rust
if let Some(ref q6k_buf) = lw.w_down_q6k_repacked {
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait();
    matmul_graph::q6k_repacked_batch_ws(
        q6k_buf.as_ptr(), lw.w_down,
        state.batch_ffn_q8_qs.as_ptr(), state.batch_ffn_q8_d.as_ptr(),
        state.batch_ffn_q8_bsums.as_ptr(),
        state.batch_down.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        hd, ffn_dim, n, hd,
        current_chunk, ith, nth,
    );
} else {
    matvec_batch_step(
        lw.w_down_repacked.as_deref(), lw.w_down_dtype, lw.w_down,
        state.batch_ffn_q8_a.as_ptr(),
        state.batch_ffn_q8_qs.as_ptr(), state.batch_ffn_q8_d.as_ptr(),
        state.batch_ffn_q8_bsums.as_ptr(),
        state.batch_down.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        hd, ffn_dim, n, n_pad, hd,
        current_chunk, ith, nth,
    );
}
```

Wait — the Q6K repacked path doesn't use Q8K repacked tiles (`batch_ffn_q8_a`). It reads directly from `batch_ffn_q8_qs/d/bsums` per-token. This means we need the barrier + `current_chunk.store` before the call but we DON'T need the `repack_q8_for_gemm` step for Q6K layers. However, the repack + `current_chunk.store` already happened earlier (section 7b, lines 476-484). The `matvec_batch_step` for ffn_down resets `current_chunk` too. So the Q6K repacked path just needs `current_chunk.store(nth as i32, Relaxed)` before its call.

Actually, looking more carefully at the flow: section 7b does `repack_q8_for_gemm` and `current_chunk.store` for the GEMM path. Then section 8 calls `matvec_batch_step` which either uses the repacked Q4K GEMM or falls through to `matvec_batch_ws`. The `current_chunk` reset and barrier are already in the right place. For Q6K repacked, we just need to dispatch differently.

Simplify: just check `w_down_q6k_repacked` in the same spot where `matvec_batch_step` is currently called. The barriers and chunk resets are already correct.

- [ ] **Step 5: Update decode paths (matmul_seq.rs, matmul_graph.rs q6k_matvec_ws)**

In `matmul_seq.rs` `q6k_matvec`, add `repacked: Option<&[u8]>` parameter. When `Some`, use the repacked kernel with interleaved d_arr.

In `matmul_graph.rs` `q6k_matvec_ws`, same change.

In `matmul.rs`, update `matvec` and `matvec_maybe_repacked` and `par_matvec` to pass through Q6K repacked data. The Q6K branch in `matvec` currently calls `q6k_matvec(weight, ...)`. It needs to also pass the repacked buffer.

The cleanest approach for the decode paths: add `q6k_repacked: Option<&[u8]>` to the general `matvec`/`par_matvec` signatures, or handle Q6K specially at the call site.

Looking at the call sites:
- `forward_graph.rs` calls `matmul::matvec` and `matmul::matvec_maybe_repacked` for ffn_down
- `forward_attn.rs` calls `matmul::par_matvec_maybe_repacked` for ffn_down

Both pass `lw.w_down_repacked` (Q4K) already. For Q6K layers, `w_down_repacked` is None (Q4K repack fails on Q6K dtype). So `matvec_maybe_repacked` falls through to `matvec` which falls through to `q6k_matvec`.

Simplest approach: in `forward_graph.rs` and `forward_attn.rs`, check `w_down_q6k_repacked` explicitly at the ffn_down call site, similar to what we did for PLE:

```rust
// In forward_graph.rs, ffn_down matvec:
if let Some(ref q6k_buf) = lw.w_down_q6k_repacked {
    matmul::q6k_matvec_repacked(q6k_buf, lw.w_down, ..., n_rows, n_cols);
} else {
    matmul::matvec_maybe_repacked(lw.w_down_dtype, lw.w_down, lw.w_down_repacked.as_deref(), ...);
}
```

Add `q6k_matvec_repacked` to `matmul.rs` that handles the d_arr interleaving + repacked kernel call.

- [ ] **Step 6: Build**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build --release 2>&1 | tail -10
```

- [ ] **Step 7: Commit**

```bash
git add src/inference/matmul_graph.rs src/inference/matmul_seq.rs src/inference/matmul.rs \
        src/inference/forward_batch_layer.rs src/inference/forward_graph.rs \
        src/inference/forward_attn.rs
git commit -m "feat: dispatch Q6K repacked kernel in all matmul paths"
```

---

### Task 5: Verification

- [ ] **Step 1: N=1 bit-exact test**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" \
cargo test --release --test forward_batch_verify -- --nocapture 2>&1 | tail -15
```

If it fails, the accumulation order changed (expected). Update the graph/decode paths to also use repacked for consistency, then re-test. Delete and regenerate the regression snapshot if needed.

- [ ] **Step 2: Regression tests**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" \
cargo test --release --test gemma4_parallel_regression 2>&1 | tail -6
```

If snapshot mismatch: delete `tests/snapshots/gemma4_logits_bos.bin`, re-run twice to capture + verify.

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" \
cargo test --release --test gemma4_verify -- --test-threads=1 2>&1 | tail -15
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" \
cargo test --release --test gemma4_smoke -- --nocapture 2>&1 | tail -10
```

- [ ] **Step 3: Cross-compile and Pi test**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" \
RUSTFLAGS="-C link-args=-Wl,--wrap=pidfd_spawnp -C link-args=-Wl,--wrap=pidfd_getpid" \
cargo build --release --target aarch64-unknown-linux-gnu 2>&1 | tail -3
scp -i ~/.ssh/id_ed25519_pi target/aarch64-unknown-linux-gnu/release/olorin peter@10.46.0.27:~/olorin
ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 \
  'echo -e "Hello\n/quit" | GEMMA4_TIMING=1 timeout 90 ~/olorin --model ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf 2>&1 | tail -30'
```

Expected: `gemm_down` should drop from ~230ms. The 17 Q6K layers should be ~3× faster, the 18 Q4K layers unchanged. Estimated new `gemm_down`: ~230 - 77 ≈ ~153ms.

- [ ] **Step 4: Commit**

```bash
git add tests/snapshots/gemma4_logits_bos.bin
git commit -m "test: update regression snapshot for Q6K repacked accumulation order"
```

---

## Expected Impact

| Metric | Before | After (est.) |
|--------|--------|-------------|
| gemm_down (17 Q6K layers) | ~115ms | ~38ms |
| gemm_down (18 Q4K layers) | ~115ms | ~115ms (unchanged) |
| **gemm_down total** | **230ms** | **~153ms** |
| **Prefill total** | **645ms** | **~568ms** |

## Risks

1. **d_arr interleaving correctness** — the repacked kernel expects `d_arr[blk*4+row]` while the existing code writes 4 separate `d0/d1/d2/d3` arrays. Must verify the extraction loop writes the correct interleaved format.

2. **Remainder rows** — if `hidden_dim % 4 != 0`, the last few rows can't use the repacked kernel. For Gemma 4 E2B, `hidden_dim=1536` which is divisible by 4, so no remainder.

3. **matmul_graph.rs is at 560 lines** — adding `q6k_repacked_batch_ws` (~50 lines) pushes it further over 500. Consider extracting Q6K-specific functions to a separate file if needed.

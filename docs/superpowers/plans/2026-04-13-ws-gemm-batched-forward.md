# Work-Stealing Batched Forward — Match llama.cpp GEMM Threading

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **HARD RULES (apply to ALL agents):**
> - No file exceeds 500 lines. Split before you hit the limit.
> - Every feature proven by end-to-end test. If it's not tested, it doesn't exist.
> - No fake functions. No silent fallbacks.
> - Match llama.cpp exactly first. Same code, same order, same math.
> - eacompute compiler: `/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release/ea`
> - Build: `PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build --release`
> - Branch: `gemma4-batched-prompt-eval`

**Goal:** Make the batched forward pass work-stealing with **constant barrier count** regardless of N, matching llama.cpp's GEMM threading. Each matmul processes all N tokens in one barrier interval with all threads participating.

**Architecture:** For each projection matmul, thread 0 quantizes all N tokens upfront into batch Q8K buffers. After one barrier, all threads work-steal 8-row tiles of the weight matrix — for each claimed tile, process all N tokens using the existing `q4k_8x8_q8k_matvec` kernel. After one barrier, proceed to next op. Barrier count per layer is constant (same as single-token `forward_graph.rs`), independent of N.

**Tech Stack:** Rust, existing `q4k_8x8_q8k_matvec` Ea kernel (proven on x86, ARM disabled separately), existing `SpinBarrier` + `GraphPool`.

**Key insight:** llama.cpp splits GEMM across output rows (weight row tiles), not across tokens. Each thread claims a tile, processes ALL tokens for that tile, claims next tile. This keeps barrier count constant and distributes compute evenly.

---

## How llama.cpp Does Batched GEMM Threading

```
Per projection matmul (e.g., Q = W_q × X):
  1. Thread 0: quantize all N tokens into Q8K format     (small, sequential)
  2. Barrier
  3. All threads: work-steal weight row tiles (8 rows each)
     Per claimed tile:
       For each token t in 0..N:
         matvec_8x8(weight_tile, q8k[t]) → output[t][tile_rows]
     Claim next tile via atomic fetch_add
  4. Barrier
```

Barrier count per matmul = 2 (same as single-token decode).
Total barriers per layer ≈ 12-14 (same as `forward_graph.rs`).
Total barriers per layer does NOT grow with N.

---

## File Map

**Modified files:**
- `src/inference/forward_batch_layer.rs` — rewrite: batch-quant then WS tile loop
- `src/inference/forward.rs` — re-add batch Q8K buffers (sized for max_batch)

**Not touched:**
- Ea kernels — reuse existing `q4k_8x8_q8k_matvec` per-tile
- `src/inference/matmul_graph.rs` — existing WS functions (reference pattern)
- `src/inference/forward_batch.rs` — outer loop unchanged
- `src/inference/forward_graph.rs` — decode path unchanged

---

## Task 1: Re-add batch Q8K buffers to Gemma4State

**Goal:** The work-stealing GEMM needs all N tokens quantized into Q8K format before the barrier. This requires batch-sized Q8K buffers that were removed in the previous plan. Re-add them, but only the ones needed: `batch_q8_qs`, `batch_q8_d`, `batch_q8_bsums` (for hd-dim projections) and `batch_ffn_q8_qs`, `batch_ffn_q8_d`, `batch_ffn_q8_bsums` (for ffn-dim down projection).

**Files:**
- Modify: `src/inference/forward.rs`

- [ ] **Step 1: Add batch Q8K fields back to Gemma4State**

In `src/inference/forward.rs`, after the `batch_ple_signal` field, add:

```rust
    // Batch Q8K buffers for work-stealing GEMM (all N tokens quantized before barrier)
    pub(crate) batch_q8_qs: Vec<i8>,
    pub(crate) batch_q8_d: Vec<f32>,
    pub(crate) batch_q8_bsums: Vec<i16>,
    pub(crate) batch_ffn_q8_qs: Vec<i8>,
    pub(crate) batch_ffn_q8_d: Vec<f32>,
    pub(crate) batch_ffn_q8_bsums: Vec<i16>,
```

- [ ] **Step 2: Allocate in new()**

After the `batch_ple_signal` allocation, add:

```rust
            // Q8K stride: dim + 12 bytes padding per token
            // Sized for max(hd, max_qkv) since Wo quantizes attn_out (n_heads*head_dim)
            let max_q8_dim = max_qkv.max(hd);
            let nb_q8 = max_q8_dim / 256;
            batch_q8_qs: vec![0; (max_q8_dim + 12) * max_batch],
            batch_q8_d: vec![0.0; nb_q8 * max_batch],
            batch_q8_bsums: vec![0; nb_q8 * 16 * max_batch],
            batch_ffn_q8_qs: vec![0; (max_ffn + 12) * max_batch],
            batch_ffn_q8_d: vec![0.0; n_blocks_ffn * max_batch],
            batch_ffn_q8_bsums: vec![0; n_blocks_ffn * 16 * max_batch],
```

Memory cost: ~12 MB for max_batch=512 (vs ~29 MB before — no gemm scratch/repack buffers).

- [ ] **Step 3: Build**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build --release 2>&1 | grep "^error" | head -5
```

- [ ] **Step 4: Commit**

```bash
git add src/inference/forward.rs
git commit -m "feat: re-add batch Q8K buffers for work-stealing GEMM"
```

---

## Task 2: Write `gemm_ws` — work-stealing batched matvec

**Goal:** Write a function that all threads call. Each thread claims 8-row weight tiles via atomic `current_chunk`, and for each claimed tile, runs the matvec kernel for all N tokens. This is the core work-stealing GEMM primitive.

**Files:**
- Modify: `src/inference/matmul_graph.rs` — add `q4k_matvec_8x8_batch_ws` function

- [ ] **Step 1: Add the work-stealing batch matvec function**

At the end of `src/inference/matmul_graph.rs`, add:

```rust
/// Q4K 8×8 repacked batch matvec: work-stealing across output row tiles.
/// Processes all N tokens per claimed tile. Constant barrier count (independent of N).
///
/// Layout:
///   batch_q8_qs: token t's qs at offset t * qs_stride, where qs_stride = n_cols + 12
///   batch_q8_d:  token t's d at offset t * nb
///   batch_q8_bsums: token t's bsums at offset t * nb * 16
///   output: token t's row at offset t * output_stride
///
/// current_chunk must be reset to nth before calling.
#[allow(clippy::too_many_arguments)]
pub fn q4k_matvec_8x8_batch_ws(
    packed: *const u8,
    batch_q8_qs: *const i8,
    batch_q8_d: *const f32,
    batch_q8_bsums: *const i16,
    output: *mut f32,
    n_rows: usize,
    n_cols: usize,
    n_tokens: usize,
    output_stride: usize,
    current_chunk: &AtomicI32,
    ith: usize,
    _nth: usize,
) {
    debug_assert!(n_rows % 8 == 0);
    debug_assert!(n_cols % Q4K_BLOCK_SIZE == 0);

    let n_blocks = n_cols / Q4K_BLOCK_SIZE;
    let tile_bytes = n_blocks * 1152;
    let n_tiles = n_rows / 8;
    let qs_stride = n_cols + 12;
    let pow2 = pow2_table();
    let mut scratch = [0u8; 128];

    let mut chunk = ith as i32;
    while (chunk as usize) < n_tiles {
        let tile = chunk as usize;
        let w_ptr = unsafe { packed.add(tile * tile_bytes) };
        let out_col_off = tile * 8;

        // Process all N tokens for this tile
        for t in 0..n_tokens {
            let q8 = unsafe { batch_q8_qs.add(t * qs_stride) };
            let q8_d = unsafe { batch_q8_d.add(t * n_blocks) };
            let bsums = unsafe { batch_q8_bsums.add(t * n_blocks * 16) };
            let out_ptr = unsafe { output.add(t * output_stride + out_col_off) };
            unsafe {
                ffi_inference::q4k_8x8_q8k_matvec(
                    w_ptr, q8, q8_d, bsums,
                    pow2.as_ptr(), scratch.as_mut_ptr(),
                    out_ptr, 8i32, n_cols as i32,
                );
            }
        }

        chunk = current_chunk.fetch_add(1, Ordering::Relaxed);
    }
}
```

Also add a fallback for non-repacked weights (Q5K/Q6K/non-8x8 Q4K):

```rust
/// Batch matvec fallback for non-repacked weights: work-stealing across 4-row chunks.
/// Each thread claims a 4-row chunk, processes all N tokens for that chunk.
#[allow(clippy::too_many_arguments)]
pub fn matvec_batch_ws(
    dtype: u32,
    weight: *const u8,
    batch_q8_qs: *const i8,
    batch_q8_d: *const f32,
    batch_q8_bsums: *const i16,
    output: *mut f32,
    d_scratch: *mut f32,
    n_rows: usize,
    n_cols: usize,
    n_tokens: usize,
    output_stride: usize,
    current_chunk: &AtomicI32,
    ith: usize,
    nth: usize,
) {
    let n_blocks = n_cols / Q4K_BLOCK_SIZE;
    let qs_stride = n_cols + 12;
    let full_quads = n_rows / 4;
    let pow2 = pow2_table();

    // Per-thread d_scratch for Q6K
    let scratch_per = n_blocks * 4;
    let my_scratch = unsafe { d_scratch.add(ith * scratch_per) };

    let mut chunk = ith as i32;
    while (chunk as usize) < full_quads {
        let base_row = (chunk as usize) * 4;

        for t in 0..n_tokens {
            let q8 = unsafe { batch_q8_qs.add(t * qs_stride) };
            let q8_d = unsafe { batch_q8_d.add(t * n_blocks) };
            let bsums = unsafe { batch_q8_bsums.add(t * n_blocks * 16) };
            let out_ptr = unsafe { output.add(t * output_stride + base_row) };

            match dtype {
                GGML_TYPE_Q4_K => {
                    let row_bytes = n_blocks * Q4K_BLOCK_BYTES;
                    unsafe {
                        ffi_inference::q4k_dot_q8k_4row(
                            weight.add(base_row * row_bytes),
                            weight.add((base_row + 1) * row_bytes),
                            weight.add((base_row + 2) * row_bytes),
                            weight.add((base_row + 3) * row_bytes),
                            q8, bsums, out_ptr,
                            n_blocks as i32, q8_d, pow2.as_ptr(),
                        );
                    }
                }
                GGML_TYPE_Q5_K => {
                    let row_bytes = n_blocks * Q5K_BLOCK_BYTES;
                    unsafe {
                        ffi_inference::q5k_dot_q8k_4row(
                            weight.add(base_row * row_bytes),
                            weight.add((base_row + 1) * row_bytes),
                            weight.add((base_row + 2) * row_bytes),
                            weight.add((base_row + 3) * row_bytes),
                            q8, bsums, out_ptr,
                            n_blocks as i32, q8_d, pow2.as_ptr(),
                        );
                    }
                }
                GGML_TYPE_Q6_K => {
                    let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
                    unsafe {
                        let d0 = my_scratch;
                        let d1 = my_scratch.add(n_blocks);
                        let d2 = my_scratch.add(n_blocks * 2);
                        let d3 = my_scratch.add(n_blocks * 3);
                        for blk in 0..n_blocks {
                            let off = 208;
                            for (row_off, d_ptr) in [(0, d0), (1, d1), (2, d2), (3, d3)] {
                                let w = weight.add((base_row + row_off) * row_bytes + blk * Q6K_BLOCK_BYTES + off);
                                let raw = u16::from_le_bytes([*w, *w.add(1)]);
                                *d_ptr.add(blk) = f16_to_f32_scalar(raw) * *q8_d.add(blk);
                            }
                        }
                        ffi_inference::q6k_dot_q8k_4row(
                            weight.add(base_row * row_bytes),
                            weight.add((base_row + 1) * row_bytes),
                            weight.add((base_row + 2) * row_bytes),
                            weight.add((base_row + 3) * row_bytes),
                            q8, bsums, out_ptr,
                            n_blocks as i32, d0, d1, d2, d3,
                        );
                    }
                }
                _ => panic!("unsupported weight dtype {dtype}"),
            }
        }

        chunk = current_chunk.fetch_add(1, Ordering::Relaxed);
    }

    // Remainder rows (thread 0 only)
    if ith == 0 {
        let base = full_quads * 4;
        for r in 0..(n_rows % 4) {
            let row = base + r;
            for t in 0..n_tokens {
                let q8 = unsafe { batch_q8_qs.add(t * qs_stride) };
                let q8_d = unsafe { batch_q8_d.add(t * n_blocks) };
                let bsums = unsafe { batch_q8_bsums.add(t * n_blocks * 16) };
                let out_ptr = unsafe { output.add(t * output_stride + row) };
                match dtype {
                    GGML_TYPE_Q4_K => {
                        let row_bytes = n_blocks * Q4K_BLOCK_BYTES;
                        unsafe {
                            *out_ptr = ffi_inference::q4k_dot_q8k(
                                weight.add(row * row_bytes), q8, bsums,
                                n_blocks as i32, q8_d, pow2.as_ptr(),
                            );
                        }
                    }
                    GGML_TYPE_Q5_K => {
                        let row_bytes = n_blocks * Q5K_BLOCK_BYTES;
                        unsafe {
                            *out_ptr = ffi_inference::q5k_dot_q8k(
                                weight.add(row * row_bytes), q8, bsums,
                                n_blocks as i32, q8_d, pow2.as_ptr(),
                            );
                        }
                    }
                    GGML_TYPE_Q6_K => {
                        let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
                        unsafe {
                            let d0 = my_scratch;
                            for blk in 0..n_blocks {
                                let w = weight.add(row * row_bytes + blk * Q6K_BLOCK_BYTES + 208);
                                let raw = u16::from_le_bytes([*w, *w.add(1)]);
                                *d0.add(blk) = f16_to_f32_scalar(raw) * *q8_d.add(blk);
                            }
                            *out_ptr = ffi_inference::q6k_dot_q8k(
                                weight.add(row * row_bytes), q8, bsums,
                                n_blocks as i32, d0,
                            );
                        }
                    }
                    _ => panic!("unsupported weight dtype {dtype}"),
                }
            }
        }
    }
}
```

- [ ] **Step 2: Build**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build --release 2>&1 | grep "^error" | head -5
```

- [ ] **Step 3: Commit**

```bash
git add src/inference/matmul_graph.rs
git commit -m "feat: work-stealing batch matvec — all threads, constant barriers, any N"
```

---

## Task 3: Rewrite `forward_batch_layer.rs` — constant barrier count

**Goal:** Replace the per-token barrier loops with batch-quant-then-WS-matvec pattern. Thread 0 quantizes all N tokens, one barrier, all threads work-steal weight tiles across all tokens, one barrier. Barrier count matches `forward_graph.rs` regardless of N.

**Files:**
- Rewrite: `src/inference/forward_batch_layer.rs`

The new pattern for each matmul:

```
// Thread 0: batch quantize all N tokens
if ith == 0 {
    for t in 0..n {
        matmul::quant_input(
            &input[t*dim..(t+1)*dim],
            &mut batch_q8_qs[t*qs_stride..(t+1)*qs_stride],
            &mut batch_q8_d[t*nb..(t+1)*nb],
            &mut batch_q8_bsums[t*nb*16..(t+1)*nb*16],
        );
    }
}
barrier.wait();

// All threads: work-steal tiles across all N tokens
current_chunk.store(nth as i32, Ordering::Relaxed);
barrier.wait();
matvec_batch_step(
    repacked, dtype, weight,
    batch_q8_qs, batch_q8_d, batch_q8_bsums,
    output, d_scratch,
    n_rows, n_cols, n, output_stride,
    current_chunk, ith, nth,
);
barrier.wait();
```

**Barrier count per matmul: 3** (quant barrier + chunk reset barrier + WS barrier). Same as single-token `forward_graph.rs`. Total per layer ≈ 12-14, constant regardless of N.

- [ ] **Step 1: Write the dispatch helper**

At the top of `forward_batch_layer.rs`, add a `matvec_batch_step` that dispatches repacked vs fallback:

```rust
/// Dispatch batch matvec: repacked 8x8 or fallback, work-stealing.
#[inline]
#[allow(clippy::too_many_arguments)]
fn matvec_batch_step(
    repacked: Option<&[u8]>,
    dtype: u32,
    weight: *const u8,
    q8_qs: *const i8,
    q8_d: *const f32,
    q8_bsums: *const i16,
    output: *mut f32,
    d_scratch: *mut f32,
    n_rows: usize,
    n_cols: usize,
    n_tokens: usize,
    output_stride: usize,
    current_chunk: &AtomicI32,
    ith: usize,
    nth: usize,
) {
    match repacked {
        Some(p) => matmul_graph::q4k_matvec_8x8_batch_ws(
            p.as_ptr(), q8_qs, q8_d, q8_bsums, output,
            n_rows, n_cols, n_tokens, output_stride,
            current_chunk, ith, nth,
        ),
        None => matmul_graph::matvec_batch_ws(
            dtype, weight, q8_qs, q8_d, q8_bsums, output, d_scratch,
            n_rows, n_cols, n_tokens, output_stride,
            current_chunk, ith, nth,
        ),
    }
}
```

- [ ] **Step 2: Rewrite the layer function**

Replace the body of `layer_forward_batch` with the constant-barrier pattern. Each projection follows the same 3-barrier cycle: thread-0 batch quant → barrier → chunk reset + barrier → WS batch matvec → barrier.

**Key changes from the per-token version:**
- Quantize loop runs on thread 0 BEFORE the barrier, writing to `batch_q8_qs[t*stride..]`
- Single `matvec_batch_step` call replaces the per-token loop
- Output lands at `batch_q[t * qkv_dim + tile_start..]` using `output_stride = qkv_dim`
- FFN gate+up: use dual batch variant if both Q4K, otherwise two separate batch matvecs
- Barrier count per section matches `forward_graph.rs` exactly

Note: for Wo projection, the input is `batch_attn_out` (dim = n_heads * head_dim), which may differ from `hd`. The batch Q8K buffers must be sized for `max(hd, max_qkv)` — which they are (Task 1 uses `max_q8_dim`).

The full rewrite follows the structure of the current `forward_batch_layer.rs` but replaces every `for t in 0..n { ... barrier ... }` with `batch_quant_all → barrier → ws_batch_matvec → barrier`.

- [ ] **Step 3: Build**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build --release 2>&1 | grep "^error" | head -10
```

- [ ] **Step 4: Line count check**

```bash
wc -l src/inference/forward_batch_layer.rs
```

Must be under 500. If over, split the helper functions into a separate file.

- [ ] **Step 5: Commit**

```bash
git add src/inference/forward_batch_layer.rs
git commit -m "feat: forward_batch_layer constant barrier count — WS GEMM matching llama.cpp"
```

---

## Task 4: N=1 bit-exact test

**Goal:** Verify `forward_batch(&[BOS])` still produces identical logits to `forward_one_graph(BOS)`.

**Files:**
- Read: `tests/forward_batch_verify.rs` (existing)

- [ ] **Step 1: Run the test**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo test --release --test forward_batch_verify -- --nocapture 2>&1 | tail -15
```

Expected: PASS with "forward_batch(N=1) bit-exact match".

**Do NOT proceed until this passes.** If it fails, debug per-layer.

- [ ] **Step 2: Run regression tests**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo test --release --test gemma4_parallel_regression 2>&1 | tail -6
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo test --release --test gemma4_verify -- --test-threads=1 2>&1 | tail -15
```

Both must pass.

---

## Task 5: Wire generate.rs + Pi test

**Goal:** Switch prefill to `forward_batch`, verify on x86 and Pi 5.

**Files:**
- Modify: `src/inference/generate.rs`

- [ ] **Step 1: Replace prefill with forward_batch**

```rust
        // 4. Prefill: batched forward (all prompt tokens at once)
        let mut logits_snapshot = {
            let logits = self.state.forward_batch(&self.model, &tokens, &self.graph_pool);
            logits.to_vec()
        };
```

- [ ] **Step 2: x86 smoke test**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo test --release --test gemma4_smoke -- --nocapture 2>&1 | tail -10
```

- [ ] **Step 3: Cross-compile and Pi test**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" \
RUSTFLAGS="-C link-args=-Wl,--wrap=pidfd_spawnp -C link-args=-Wl,--wrap=pidfd_getpid" \
cargo build --release --target aarch64-unknown-linux-gnu 2>&1 | tail -3

scp -i ~/.ssh/id_ed25519_pi target/aarch64-unknown-linux-gnu/release/olorin peter@10.46.0.27:~/olorin

ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 \
  'echo -e "Hello\n/quit" | timeout 90 ~/olorin --model ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf 2>&1'
```

Must complete within 90s and produce coherent output. If it hangs, the constant-barrier approach still has a bug.

- [ ] **Step 4: Commit**

```bash
git add src/inference/generate.rs
git commit -m "feat: generate.rs uses forward_batch for prefill — constant barriers, WS GEMM"
```

---

## Task 6: Benchmark

**Goal:** Record prefill and decode throughput.

- [ ] **Step 1: Update bench to use forward_batch for prefill**

In `tests/bench_decode_speed.rs`, the prefill section should use `forward_batch`:

```rust
    let t_prefill = Instant::now();
    let logits = state.forward_batch(&model, &prompt_ids, &graph_pool);
    let mut next = argmax(logits);
    let prefill_secs = t_prefill.elapsed().as_secs_f64();
```

- [ ] **Step 2: Run bench**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo test --release --test bench_decode_speed -- --nocapture 2>&1 | tail -30
```

Record: prefill t/s, decode t/s, parallel efficiency.

- [ ] **Step 3: Commit with numbers**

```bash
git add tests/bench_decode_speed.rs
git commit -m "bench: WS GEMM batched prefill baseline

WSL x86 (threads):
  prefill: [XX.X] t/s
  decode:  [XX.X] t/s
  parallel efficiency: [XX]%"
```

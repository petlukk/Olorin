# True GEMM Prefill + Parallel Quantization

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **HARD RULES (apply to ALL agents):**
> - No file exceeds 500 lines. Split before you hit the limit.
> - Every feature proven by end-to-end test. If it's not tested, it doesn't exist.
> - No fake functions. No silent fallbacks.
> - Match llama.cpp exactly first. Same code, same order, same math.
> - eacompute compiler: `/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release/ea`
> - Build: `PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build --release`
> - Branch: `gemma4-gemm-prefill`

**Goal:** Close the 1.6x prefill gap vs llama.cpp by using true GEMM (one kernel call per chunk, weight loaded once) instead of N×matvec, and by parallelizing input quantization across all threads.

**Architecture:** Two changes that match llama.cpp's repack `forward_mul_mat` exactly: (1) All threads quantize input tokens in parallel (stripe by token index), then barrier. (2) Work-stealing GEMM: each thread claims a chunk of output rows, calls `q4k_8x8_q8k_gemm` with all tokens and that row range. Weight data loaded once per chunk instead of N times.

**Tech Stack:** Rust, existing `q4k_8x8_q8k_gemm` Ea kernel (x86 + ARM), existing `q8k_repack_4` kernel, `SpinBarrier` + `GraphPool`.

**Baseline (Pi 5):** prefill 16.69 t/s, decode 6.09 t/s. Target: prefill >25 t/s.

---

## How llama.cpp Does It (repack.cpp:4253-4384)

```
forward_mul_mat(params, op):
  1. ALL threads quantize input in parallel:
     for i11 = ith*4; i11 < ne11; i11 += nth*4:
       quantize_mat(src1[i11..i11+4], wdata[i11..i11+4])
     barrier

  2. thread 0: current_chunk = nth
     barrier

  3. ALL threads work-steal 2D chunks (output_rows × token_planes):
     while current_chunk < nchunk0 * nchunk1:
       (src0_start, src0_end) = row range for this chunk
       (src1_start, src1_end) = token range for this chunk
       if nrows > 3:
         gemm(packed + src0_start, q8_repacked, dst + src0_start, nrows, ncols)
       else:
         gemv per remaining row
       current_chunk = atomic_fetch_add(1)
```

Key: the `gemm` kernel processes ALL tokens against a SUBSET of weight rows in one call. Weight data loaded once from memory, reused across all tokens.

---

## Prerequisite: Fix ARM GEMM scale bug

The ARM GEMM kernel (`q4k_dot_8x8_gemm_arm.ea`) has the same scale bug fixed in the matvec kernel: rows 4-7 use `sc_lo_0` instead of `sc_hi_0`. Must fix before using GEMM on ARM.

---

## File Map

**Modified files:**
- `kernels/q4k_dot_8x8_gemm_arm.ea` — fix scale bug (same fix as matvec kernel)
- `src/inference/matmul_graph.rs` — replace `q4k_matvec_8x8_batch_ws` with `q4k_gemm_8x8_batch_ws`
- `src/inference/forward_batch_layer.rs` — parallel quant + GEMM dispatch
- `src/inference/forward.rs` — add Q8K repack buffers (`batch_q8_a`)

**Not touched:**
- `src/inference/matmul_batch.rs` — existing gemm wrapper (reference only, not reused)
- `src/inference/forward_graph.rs` — decode path unchanged
- Ea kernels (x86) — no changes, already correct

---

## Task 1: Fix ARM GEMM kernel scale bug

**Goal:** Apply the same `sc_hi_0`/`sc_hi_1` fix to `q4k_dot_8x8_gemm_arm.ea` that was applied to `q4k_dot_8x8_arm.ea`.

**Files:**
- Modify: `kernels/q4k_dot_8x8_gemm_arm.ea`

- [ ] **Step 1: Find and fix the scale lines**

In `kernels/q4k_dot_8x8_gemm_arm.ea`, find these lines (around line 221-223):

```
let sumf_lo_47: f32x4 = to_f32(sc_lo_0 .* plo_47)
...
let sumf_hi_47: f32x4 = to_f32(sc_lo_1 .* phi_47)
```

Replace with:

```
let sumf_lo_47: f32x4 = to_f32(sc_hi_0 .* plo_47)
...
let sumf_hi_47: f32x4 = to_f32(sc_hi_1 .* phi_47)
```

Also remove any dead `while j < 4` loop with `// TODO` comment if present (same pattern as the matvec fix).

- [ ] **Step 2: Build (x86 + ARM)**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build --release 2>&1 | grep "^error" | head -5
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" \
RUSTFLAGS="-C link-args=-Wl,--wrap=pidfd_spawnp -C link-args=-Wl,--wrap=pidfd_getpid" \
cargo build --release --target aarch64-unknown-linux-gnu 2>&1 | grep "^error" | head -5
```

- [ ] **Step 3: Cross-compile and run GEMM test on Pi**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" \
RUSTFLAGS="-C link-args=-Wl,--wrap=pidfd_spawnp -C link-args=-Wl,--wrap=pidfd_getpid" \
cargo test --release --target aarch64-unknown-linux-gnu --test gemm_q4k_8x8 --no-run 2>&1 | grep "Executable"

scp -i ~/.ssh/id_ed25519_pi <executable_path> peter@10.46.0.27:~/test_gemm
ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 'rm -rf ~/.olorin/lib/ && timeout 30 ~/test_gemm --nocapture 2>&1 | tail -20'
```

Expected: all GEMM tests pass on ARM.

- [ ] **Step 4: Commit**

```bash
git add kernels/q4k_dot_8x8_gemm_arm.ea
git commit -m "fix: ARM GEMM kernel scale bug — rows 4-7 used wrong scale half (same as matvec fix)"
```

---

## Task 2: Add Q8K repack buffer to Gemma4State

**Goal:** The GEMM kernel needs Q8K input in `block_q8_Kx4` repacked format. Add a buffer sized for `(max_batch/4) * block_q8_Kx4_size` where `block_q8_Kx4_size = nb * 1168`.

**Files:**
- Modify: `src/inference/forward.rs`

- [ ] **Step 1: Add the field**

After the `batch_ffn_q8_bsums` field, add:

```rust
    // Q8K repacked A-side for GEMM (block_q8_Kx4 tiles, groups of 4 tokens)
    pub(crate) batch_q8_a: Vec<u8>,
    pub(crate) batch_ffn_q8_a: Vec<u8>,
```

- [ ] **Step 2: Allocate in new()**

After the `batch_ffn_q8_bsums` allocation, add:

```rust
            let max_q8_dim = max_qkv.max(hd);
            let nb_q8_max = max_q8_dim / 256;
            let nb_ffn_max = max_ffn / 256;
            let q8_a_groups = (max_batch + 3) / 4;
            batch_q8_a: vec![0u8; q8_a_groups * nb_q8_max * 1168],
            batch_ffn_q8_a: vec![0u8; q8_a_groups * nb_ffn_max * 1168],
```

- [ ] **Step 3: Build**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build --release 2>&1 | grep "^error" | head -5
```

- [ ] **Step 4: Commit**

```bash
git add src/inference/forward.rs
git commit -m "feat: add Q8K repack buffers for GEMM (batch_q8_a, batch_ffn_q8_a)"
```

---

## Task 3: Write `q4k_gemm_8x8_batch_ws` — work-stealing GEMM

**Goal:** Replace `q4k_matvec_8x8_batch_ws` with a function that calls the actual GEMM kernel per chunk instead of N×matvec. Each thread claims a chunk of output rows (8-row aligned), calls `q4k_8x8_q8k_gemm` with ALL tokens and that row range.

**Files:**
- Modify: `src/inference/matmul_graph.rs`

- [ ] **Step 1: Add the GEMM work-stealing function**

Replace `q4k_matvec_8x8_batch_ws` with:

```rust
/// Q4K 8×8 repacked batch GEMM: work-stealing across output row tiles.
/// Calls the GEMM kernel per claimed tile — weight data loaded once, all tokens processed.
/// Matches llama.cpp repack.cpp:4241 `gemm(ne00, dst + src0_start, ...)`.
///
/// `q8_a` must be pre-repacked into block_q8_Kx4 format via q8k_repack_4.
/// `nr` = number of tokens (must be multiple of 4, caller zero-pads).
/// `nc` = number of output rows (total, must be multiple of 8).
/// Output: `out[token * output_stride + row]`.
#[allow(clippy::too_many_arguments)]
pub fn q4k_gemm_8x8_batch_ws(
    packed: *const u8,
    q8_a: *const u8,
    output: *mut f32,
    n_inner: usize,
    nc: usize,
    nr: usize,
    output_stride: usize,
    current_chunk: &AtomicI32,
    ith: usize,
    _nth: usize,
) {
    debug_assert!(nc % 8 == 0);
    debug_assert!(n_inner % 256 == 0);
    debug_assert!(nr % 4 == 0);

    let nb = n_inner / 256;
    let tile_bytes = nb * 1152;  // block_q4_Kx8 tile size
    let n_tiles = nc / 8;
    let block_q8_kx4_size = nb * 1168;
    let n_groups = nr / 4;

    let mut scratch = [0u8; 128];

    let mut chunk = ith as i32;
    while (chunk as usize) < n_tiles {
        let tile = chunk as usize;
        let w_ptr = unsafe { packed.add(tile * tile_bytes) };
        let col_start = tile * 8;

        // Call GEMM kernel: nr tokens × 8 output cols
        // Output layout: out[y*4+r][bs] where bs = output_stride
        // We need to write to out[token * output_stride + col_start]
        // The kernel writes out[y*4+r, x*8+c] at out[(y*4+r)*bs + x*8+c]
        // With nc=8 (single tile), x is always 0, so out[(y*4+r)*bs + c]
        // We offset the output pointer to col_start
        unsafe {
            ffi_inference::q4k_8x8_q8k_gemm(
                w_ptr,
                q8_a as *const u8,
                scratch.as_mut_ptr(),
                output.add(col_start),
                output_stride as i32,  // bs = row stride between tokens
                n_inner as i32,        // n = inner dimension
                nr as i32,             // nr = number of Q8K rows (tokens, multiple of 4)
                8i32,                  // nc = 8 output cols (one tile)
            );
        }

        chunk = current_chunk.fetch_add(1, Ordering::Relaxed);
    }
}
```

**Key difference from `q4k_matvec_8x8_batch_ws`:** One GEMM kernel call per tile processes all N tokens. Weight data loaded from memory once. In the matvec version, weight data was loaded N times (once per token).

- [ ] **Step 2: Keep `matvec_batch_ws` as fallback for non-Q4K and non-repacked**

Don't delete `matvec_batch_ws` — it handles Q5K, Q6K, and non-repacked Q4K weights.

- [ ] **Step 3: Build**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build --release 2>&1 | grep "^error" | head -5
```

- [ ] **Step 4: Commit**

```bash
git add src/inference/matmul_graph.rs
git commit -m "feat: q4k_gemm_8x8_batch_ws — true GEMM work-stealing, weight loaded once"
```

---

## Task 4: Parallel quant + GEMM dispatch in forward_batch_layer.rs

**Goal:** Rewrite the batch quantization and matmul dispatch to: (1) all threads quantize in parallel, (2) thread 0 repacks Q8K into block_q8_Kx4, (3) all threads work-steal GEMM tiles. Matching llama.cpp repack.cpp:4296-4384.

**Files:**
- Modify: `src/inference/forward_batch_layer.rs`

- [ ] **Step 1: Write parallel batch_quant helper**

Replace the existing `batch_quant` function (thread-0 only) with a parallel version where all threads participate:

```rust
/// All threads quantize a stripe of tokens in parallel.
/// Thread ith handles tokens [ith*4, ith*4+4), [ith*4+nth*4, ...] etc.
/// Matches llama.cpp repack.cpp:4300.
fn parallel_batch_quant(
    input: &[f32],
    q8_qs: &mut [i8], q8_d: &mut [f32], q8_bsums: &mut [i16],
    dim: usize, n: usize, ith: usize, nth: usize,
) {
    let nb = dim / 256;
    let qs_stride = dim + 12;
    // Each thread handles tokens in strides of nth
    let mut t = ith;
    while t < n {
        matmul::quant_input(
            &input[t * dim..(t + 1) * dim],
            &mut q8_qs[t * qs_stride..(t + 1) * qs_stride],
            &mut q8_d[t * nb..(t + 1) * nb],
            &mut q8_bsums[t * nb * 16..(t + 1) * nb * 16],
        );
        t += nth;
    }
}
```

- [ ] **Step 2: Write Q8K repack helper (thread 0)**

After parallel quant + barrier, thread 0 repacks Q8K into block_q8_Kx4 tiles for the GEMM kernel:

```rust
/// Repack N Q8K-quantized tokens into block_q8_Kx4 tile format for the GEMM kernel.
/// Must be called after parallel_batch_quant, thread 0 only.
fn repack_q8_for_gemm(
    q8_qs: &[i8], q8_d: &[f32], q8_bsums: &[i16],
    q8_a: &mut [u8],
    dim: usize, n_pad: usize,
) {
    let nb = dim / 256;
    let qs_stride = dim + 12;
    let block_q8_kx4_size = nb * 1168;

    for group in 0..(n_pad / 4) {
        let r0 = group * 4;
        // Interleave d values for the 4 rows in this group
        let mut row_d = [0.0f32; 192]; // nb <= 48, 48*4 = 192
        for b in 0..nb {
            for r in 0..4 {
                row_d[b * 4 + r] = q8_d[(r0 + r) * nb + b];
            }
        }
        let dst_off = group * block_q8_kx4_size;
        ffi_inference::q8k_repack_4(
            &q8_qs[(r0) * qs_stride..],
            &q8_qs[(r0 + 1) * qs_stride..],
            &q8_qs[(r0 + 2) * qs_stride..],
            &q8_qs[(r0 + 3) * qs_stride..],
            row_d.as_ptr(),
            &q8_bsums[(r0) * nb * 16..],
            &q8_bsums[(r0 + 1) * nb * 16..],
            &q8_bsums[(r0 + 2) * nb * 16..],
            &q8_bsums[(r0 + 3) * nb * 16..],
            &mut q8_a[dst_off..],
            nb as i32,
        );
    }
}
```

- [ ] **Step 3: Update `matvec_batch_step` dispatch**

Update the dispatch helper to use GEMM when repacked, passing the repacked Q8K buffer:

```rust
fn matvec_batch_step(
    repacked: Option<&[u8]>,
    dtype: u32, weight: *const u8,
    q8_a: *const u8,             // repacked Q8K (for GEMM path)
    q8_qs: *const i8,            // raw Q8K (for fallback path)
    q8_d: *const f32,
    q8_bsums: *const i16,
    output: *mut f32, d_scratch: *mut f32,
    n_rows: usize, n_cols: usize, n_pad: usize, output_stride: usize,
    current_chunk: &AtomicI32, ith: usize, nth: usize,
) {
    match repacked {
        Some(p) => matmul_graph::q4k_gemm_8x8_batch_ws(
            p.as_ptr(), q8_a, output,
            n_cols, n_rows, n_pad, output_stride,
            current_chunk, ith, nth,
        ),
        None => matmul_graph::matvec_batch_ws(
            dtype, weight, q8_qs, q8_d, q8_bsums, output, d_scratch,
            n_rows, n_cols, n_pad / 4, output_stride, // n_tokens for fallback
            current_chunk, ith, nth,
        ),
    }
}
```

Note: GEMM path uses `n_pad` (multiple of 4) as `nr`. Fallback uses `n_pad / 4` ... no, the fallback `matvec_batch_ws` takes `n_tokens` not groups. Pass the actual token count. Let me correct: the fallback should get the original `n` (not padded). The caller needs to pass both `n` and `n_pad`.

Actually simpler: the GEMM path takes `nr = n_pad` (padded to 4). The fallback takes `n_tokens = n` (actual count). The dispatch helper needs both:

```rust
fn matvec_batch_step(
    repacked: Option<&[u8]>,
    dtype: u32, weight: *const u8,
    q8_a: *const u8,
    q8_qs: *const i8, q8_d: *const f32, q8_bsums: *const i16,
    output: *mut f32, d_scratch: *mut f32,
    n_rows: usize, n_cols: usize, n: usize, n_pad: usize, output_stride: usize,
    current_chunk: &AtomicI32, ith: usize, nth: usize,
) {
    match repacked {
        Some(p) => matmul_graph::q4k_gemm_8x8_batch_ws(
            p.as_ptr(), q8_a, output,
            n_cols, n_rows, n_pad, output_stride,
            current_chunk, ith, nth,
        ),
        None => matmul_graph::matvec_batch_ws(
            dtype, weight, q8_qs, q8_d, q8_bsums, output, d_scratch,
            n_rows, n_cols, n, output_stride,
            current_chunk, ith, nth,
        ),
    }
}
```

- [ ] **Step 4: Rewrite each projection's quant+matmul section**

The new pattern for each projection (example: Q projection):

```rust
    // ── 1. Attn norm (thread 0) ──────────────────────────────
    if ith == 0 {
        for t in 0..n {
            ffi_inference::gemma4_rmsnorm(
                state.batch_x[t * hd..].as_ptr(), lw.attn_norm,
                state.batch_x_norm[t * hd..].as_mut_ptr(), hd as i32, model.rms_eps,
            );
        }
        // Zero-pad batch_x_norm for tokens n..n_pad
        for t in n..n_pad {
            state.batch_x_norm[t * hd..(t + 1) * hd].fill(0.0);
        }
    }
    barrier.wait();

    // ── 1b. Parallel quant (all threads) ─────────────────────
    parallel_batch_quant(
        &state.batch_x_norm, &mut state.batch_q8_qs, &mut state.batch_q8_d,
        &mut state.batch_q8_bsums, hd, n_pad, ith, nth,
    );
    barrier.wait();

    // ── 1c. Q8K repack for GEMM (thread 0) ──────────────────
    if ith == 0 {
        repack_q8_for_gemm(
            &state.batch_q8_qs, &state.batch_q8_d, &state.batch_q8_bsums,
            &mut state.batch_q8_a, hd, n_pad,
        );
    }
    barrier.wait();

    // ── 2. Q projection GEMM (work-stealing, all threads) ───
    current_chunk.store(nth as i32, Ordering::Relaxed);
    barrier.wait();
    matvec_batch_step(
        lw.wq_repacked.as_deref(), lw.wq_dtype, lw.wq,
        state.batch_q8_a.as_ptr(),
        state.batch_q8_qs.as_ptr(), state.batch_q8_d.as_ptr(),
        state.batch_q8_bsums.as_ptr(),
        state.batch_q.as_mut_ptr(), state.q6k_d_scratch.as_mut_ptr(),
        qkv_dim, hd, n, n_pad, qkv_dim,
        current_chunk, ith, nth,
    );
    barrier.wait();
```

The same Q8K data is reused for Q, K, V projections (same input = attn_normed x). So quant + repack happens once, then Q/K/V each just do the GEMM.

For Wo: re-quant + re-repack from `batch_attn_out` (different dim = `n_heads * head_dim`).
For FFN gate/up: re-quant + re-repack from `batch_x_norm` (dim = `hd`), use `batch_ffn_q8_a` repack buffer. Two separate GEMMs (gate, up).
For FFN down: re-quant + re-repack from `batch_gate` (dim = `ffn_dim`), use `batch_ffn_q8_a`.

- [ ] **Step 5: Build and check line count**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build --release 2>&1 | grep "^error" | head -10
wc -l src/inference/forward_batch_layer.rs
```

Must be under 500 lines.

- [ ] **Step 6: Commit**

```bash
git add src/inference/forward_batch_layer.rs
git commit -m "feat: parallel quant + true GEMM dispatch in forward_batch_layer

All threads quantize in parallel (striped by token index).
Thread 0 repacks Q8K into block_q8_Kx4 for GEMM.
Work-stealing GEMM: one kernel call per tile, weight loaded once."
```

---

## Task 5: N=1 bit-exact test + regression

**Goal:** Verify `forward_batch(&[BOS])` still matches `forward_one_graph(BOS)`.

**Files:**
- Read: `tests/forward_batch_verify.rs`

- [ ] **Step 1: Run bit-exact test**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo test --release --test forward_batch_verify -- --nocapture 2>&1 | tail -10
```

Expected: PASS.

**Do NOT proceed if this fails.** The GEMM kernel must produce identical results to the matvec path for N=1 (where nr=4 after padding, nc=8 per tile).

- [ ] **Step 2: Regression suite**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo test --release --test gemma4_parallel_regression 2>&1 | tail -4
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo test --release --test gemma4_verify -- --test-threads=1 2>&1 | tail -4
```

---

## Task 6: Pi 5 smoke test + benchmark

**Goal:** Verify on ARM and measure speedup.

- [ ] **Step 1: Cross-compile**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" \
RUSTFLAGS="-C link-args=-Wl,--wrap=pidfd_spawnp -C link-args=-Wl,--wrap=pidfd_getpid" \
cargo build --release --target aarch64-unknown-linux-gnu 2>&1 | tail -3
```

- [ ] **Step 2: Deploy and smoke test**

```bash
ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 'pgrep -af olorin | grep -v pgrep || echo clean'
scp -i ~/.ssh/id_ed25519_pi target/aarch64-unknown-linux-gnu/release/olorin peter@10.46.0.27:~/olorin
ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 \
  'rm -rf ~/.olorin/lib/ && echo -e "Hello\n/quit" | timeout 90 ~/olorin --model ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf 2>&1'
```

Must complete in 90s with coherent output.

- [ ] **Step 3: Run benchmark on Pi**

```bash
# Cross-compile bench
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" \
RUSTFLAGS="-C link-args=-Wl,--wrap=pidfd_spawnp -C link-args=-Wl,--wrap=pidfd_getpid" \
cargo test --release --target aarch64-unknown-linux-gnu --test bench_decode_speed --no-run 2>&1 | grep "Executable"

scp -i ~/.ssh/id_ed25519_pi <path> peter@10.46.0.27:~/bench_decode
ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 \
  'rm -rf ~/.olorin/lib/ && timeout 600 ~/bench_decode --nocapture 2>&1 | tail -30'
```

Record prefill t/s, decode t/s. Target: prefill >25 t/s (was 16.69 t/s).

- [ ] **Step 4: Commit with numbers**

```bash
git commit --allow-empty -m "bench: true GEMM prefill baseline

Pi 5 (4 threads):
  prefill: [XX.X] t/s  (was 16.69 t/s, target >25)
  decode:  [XX.X] t/s  (was 6.09 t/s, should be unchanged)"
```

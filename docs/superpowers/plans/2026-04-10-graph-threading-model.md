# Graph-Based Threading Model — Match llama.cpp Exactly

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Olorin's per-op mutex/condvar dispatch (~380 dispatches/token, 170ms) with llama.cpp's graph-loop threading model (1 dispatch/token, target 153ms) for decode parity on Pi 5.

**Architecture:** All threads loop through the entire forward pass together. Each op splits work by `ith/nth`. Atomic spin-barrier between ops. Work-stealing via atomic `current_chunk` for matmul. Threads sleep between tokens via condvar, spin during token processing.

**Tech Stack:** Rust, atomics (Ordering::Acquire/Release), std::hint::spin_loop, no external deps.

---

## Background: How llama.cpp Does It

### Thread lifecycle (ggml-cpu.c:2962)
```c
// Each thread runs this for the entire graph:
for (node_n = 0; node_n < n_nodes; node_n++) {
    ggml_compute_forward(&params, node);   // each thread computes its slice
    ggml_barrier(threadpool);              // spin-wait for all threads
}
```

### Barrier (ggml-cpu.c:562)
```c
void ggml_barrier(threadpool) {
    // Last thread to arrive resets counter and signals others
    n = atomic_fetch_add(&n_barrier, 1);
    if (n == n_threads - 1) {
        atomic_store(&n_barrier, 0);
        atomic_fetch_add(&n_barrier_passed, 1);  // release
        return;
    }
    // Others spin until n_barrier_passed changes
    while (n_barrier_passed == old_value) { cpu_relax(); }
}
```

### Matmul work distribution (ggml-cpu.c:1404)
```c
// Work-stealing: each thread starts with chunk `ith`, then grabs next
int current_chunk = ith;
while (current_chunk < total_chunks) {
    compute_one_chunk(ir0_start..ir0_end, ir1_start..ir1_end);
    current_chunk = atomic_fetch_add(&threadpool->current_chunk, 1);
}
```

### Scalar ops (norm, add, scale)
```c
// Simple range split: thread ith handles rows [ith..n, step nth]
for (i01 = ith; i01 < ne01; i01 += nth) { ... }
```

### Key: threads NEVER sleep during a token. Sleep only between tokens (condvar).

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/inference/threadpool.rs` | **Rewrite** | Graph-loop pool: spawn threads, graph dispatch, spin-barrier, work-stealing |
| `src/inference/graph.rs` | **Create** | Op list builder: encode forward pass as op sequence, each op is a fn(ith, nth, &SharedCtx) |
| `src/inference/forward.rs` | **Modify** | `forward_one` builds op list, dispatches to graph pool |
| `src/inference/forward_attn.rs` | **Modify** | `layer_forward` → `layer_ops()` that pushes ops instead of executing them |
| `src/inference/forward_attn_heads.rs` | **Modify** | Attention/norms become graph ops with ith/nth splitting |
| `src/inference/matmul.rs` | **Modify** | Matmul ops use work-stealing (atomic chunk counter) instead of pool.run() |
| `src/inference/mod.rs` | **Modify** | Add `pub mod graph;` |
| `tests/gemma4_verify.rs` | **Modify** | Update to use new pool API (run existing tests to verify no regression) |

---

### Task 1: Spin-barrier primitive

**Files:**
- Modify: `src/inference/threadpool.rs`

- [ ] **Step 1: Write the failing test**

```rust
// tests/threadpool_test.rs
#[test]
fn test_spin_barrier_4_threads() {
    use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
    use std::sync::Arc;

    let n_threads = 4;
    let n_barrier = Arc::new(AtomicI32::new(0));
    let n_barrier_passed = Arc::new(AtomicI32::new(0));

    // Simulate 4 threads hitting barrier 100 times
    let handles: Vec<_> = (0..n_threads).map(|_| {
        let nb = Arc::clone(&n_barrier);
        let nbp = Arc::clone(&n_barrier_passed);
        std::thread::spawn(move || {
            for _ in 0..100 {
                // llama.cpp barrier logic
                let old_passed = nbp.load(Ordering::Relaxed);
                let n = nb.fetch_add(1, Ordering::SeqCst);
                if n == n_threads - 1 {
                    nb.store(0, Ordering::Relaxed);
                    nbp.fetch_add(1, Ordering::SeqCst);
                } else {
                    while nbp.load(Ordering::Relaxed) == old_passed {
                        std::hint::spin_loop();
                    }
                }
                std::sync::atomic::fence(Ordering::SeqCst);
            }
        })
    }).collect();

    for h in handles { h.join().unwrap(); }
}
```

- [ ] **Step 2: Run test to verify it passes (this is a reference impl test)**

Run: `cargo test --test threadpool_test -- --nocapture`

- [ ] **Step 3: Implement SpinBarrier in threadpool.rs**

```rust
// At top of threadpool.rs
use std::sync::atomic::{AtomicI32, Ordering};

pub(crate) struct SpinBarrier {
    n_threads: i32,
    n_barrier: AtomicI32,
    n_barrier_passed: AtomicI32,
}

impl SpinBarrier {
    pub fn new(n_threads: usize) -> Self {
        Self {
            n_threads: n_threads as i32,
            n_barrier: AtomicI32::new(0),
            n_barrier_passed: AtomicI32::new(0),
        }
    }

    /// Spin-barrier matching llama.cpp ggml_barrier() exactly.
    #[inline]
    pub fn wait(&self) {
        if self.n_threads == 1 { return; }

        let old_passed = self.n_barrier_passed.load(Ordering::Relaxed);
        let n = self.n_barrier.fetch_add(1, Ordering::SeqCst);

        if n == self.n_threads - 1 {
            // Last thread — reset and signal
            self.n_barrier.store(0, Ordering::Relaxed);
            self.n_barrier_passed.fetch_add(1, Ordering::SeqCst);
            return;
        }

        // Spin until last thread signals
        while self.n_barrier_passed.load(Ordering::Relaxed) == old_passed {
            std::hint::spin_loop();
        }
        std::sync::atomic::fence(Ordering::SeqCst);
    }
}
```

- [ ] **Step 4: Test SpinBarrier directly**

```rust
// tests/threadpool_test.rs — add:
#[test]
fn test_spin_barrier_struct() {
    let barrier = olorin::inference::threadpool::SpinBarrier::new(4);
    // ... same logic but using barrier.wait()
}
```

- [ ] **Step 5: Commit**

```bash
git add src/inference/threadpool.rs tests/threadpool_test.rs
git commit -m "feat: SpinBarrier matching llama.cpp ggml_barrier()"
```

---

### Task 2: Graph op representation

**Files:**
- Create: `src/inference/graph.rs`
- Modify: `src/inference/mod.rs`

- [ ] **Step 1: Define Op trait and OpList**

An "op" is a function that takes `(ith, nth, ctx)` where ctx is a shared mutable context. In llama.cpp each op is a node in a compute graph. We represent this as a Vec of function pointers.

```rust
// src/inference/graph.rs
//! Op-list for graph-loop threading. Each op is a fn(ith, nth)
//! executed by all threads, with spin-barrier between ops.

/// A single graph operation. All threads call this with their ith/nth.
/// The function must split work internally by ith/nth.
pub type GraphOp = Box<dyn Fn(usize, usize) + Send + Sync>;

/// Ordered list of ops for one forward pass.
pub struct OpList {
    pub ops: Vec<GraphOp>,
}

impl OpList {
    pub fn new() -> Self {
        Self { ops: Vec::with_capacity(64) }
    }

    pub fn push(&mut self, op: impl Fn(usize, usize) + Send + Sync + 'static) {
        self.ops.push(Box::new(op));
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }
}
```

- [ ] **Step 2: Add `pub mod graph;` to mod.rs**

- [ ] **Step 3: Verify it compiles**

Run: `cargo build --release`

- [ ] **Step 4: Commit**

```bash
git add src/inference/graph.rs src/inference/mod.rs
git commit -m "feat: graph op-list for graph-loop threading"
```

---

### Task 3: Graph-loop thread pool

**Files:**
- Modify: `src/inference/threadpool.rs`

This is the core change. Replace mutex/condvar dispatch with:
1. Worker threads sleep (condvar) between tokens
2. Dispatcher wakes all threads with a graph (OpList)
3. All threads loop through all ops, spin-barrier between each
4. When done, threads signal completion and go back to sleep

- [ ] **Step 1: Implement GraphPool alongside existing ThreadPool**

Keep the old ThreadPool (rename to `ThreadPool`) and add `GraphPool`:

```rust
use super::graph::OpList;
use std::sync::atomic::{AtomicI32, AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

pub struct GraphPool {
    barrier: &'static SpinBarrier,
    // Shared between dispatcher and workers
    ops_ptr: &'static AtomicU64,      // pointer to current OpList
    n_ops: &'static AtomicUsize,
    generation: &'static AtomicU64,
    remaining: &'static AtomicUsize,
    shutdown: &'static AtomicBool,
    // Work-stealing for matmul
    pub(crate) current_chunk: &'static AtomicI32,
    // Sleep between tokens
    wake_mutex: &'static Mutex<()>,
    wake_cond: &'static Condvar,
    workers: Vec<std::thread::JoinHandle<()>>,
    n_threads: usize,
}
```

Worker loop:
```rust
// Each worker thread:
loop {
    // Sleep until new graph
    wait_for_generation(&wake_mutex, &wake_cond, &generation, &shutdown);

    // Get OpList
    let ops = unsafe { &*(ops_ptr as *const OpList) };

    // Execute all ops with barriers (like llama.cpp graph loop)
    for i in 0..n_ops {
        ops.ops[i](tid, n_threads);
        if i + 1 < n_ops {
            barrier.wait();
        }
    }

    // Signal completion
    remaining.fetch_sub(1, Ordering::AcqRel);
}
```

Dispatcher:
```rust
pub fn execute(&self, ops: &OpList) {
    // Store ops pointer + count
    self.ops_ptr.store(ops as *const _ as u64, ...);
    self.n_ops.store(ops.len(), ...);
    self.remaining.store(self.n_threads, ...);

    // Wake all threads
    self.generation.fetch_add(1, ...);
    self.wake_cond.notify_all();

    // Wait for completion (condvar, not spin — we're between tokens)
    while self.remaining.load(...) != 0 {
        std::hint::spin_loop();
    }
}
```

- [ ] **Step 2: Test with a trivial 3-op graph**

```rust
#[test]
fn test_graph_pool_basic() {
    let pool = GraphPool::new();
    let counter = AtomicUsize::new(0);
    let mut ops = OpList::new();
    ops.push(|ith, nth| { /* op1 */ });
    ops.push(|ith, nth| { /* op2 */ });
    ops.push(|ith, nth| { /* op3 */ });
    pool.execute(&ops);
    // verify all ops ran on all threads
}
```

- [ ] **Step 3: Commit**

```bash
git add src/inference/threadpool.rs
git commit -m "feat: GraphPool with spin-barrier graph loop"
```

---

### Task 4: Convert matmul to ith/nth + work-stealing

**Files:**
- Modify: `src/inference/matmul.rs`

Each matmul becomes a graph op that takes `(ith, nth)` and uses atomic `current_chunk` for work-stealing, exactly like llama.cpp's `ggml_compute_forward_mul_mat`.

- [ ] **Step 1: Add work-stealing matmul functions**

For each quant type (Q4K, Q5K, Q6K), add a `_graph` variant:

```rust
/// Q4K matvec as graph op — work-stealing via atomic current_chunk.
/// Matches llama.cpp ggml_compute_forward_mul_mat_one_chunk pattern.
pub(crate) fn q4k_matvec_graph(
    weight: *const u8, input_qs: *const i8, input_d: *const f32,
    input_bsums: *const i16, output: *mut f32,
    n_rows: usize, n_cols: usize,
    current_chunk: &AtomicI32,
    ith: usize, nth: usize,
) {
    let n_blocks = n_cols / Q4K_BLOCK_SIZE;
    let row_bytes = n_blocks * Q4K_BLOCK_BYTES;
    let full_quads = n_rows / 4;
    let pow2 = pow2_table();

    // Work-stealing: start at ith, grab next via atomic
    let mut chunk = ith as i32;
    while (chunk as usize) < full_quads {
        let base_row = (chunk as usize) * 4;
        unsafe {
            ffi_inference::q4k_dot_q8k_4row(
                weight.add(base_row * row_bytes),
                weight.add((base_row + 1) * row_bytes),
                weight.add((base_row + 2) * row_bytes),
                weight.add((base_row + 3) * row_bytes),
                input_qs, input_bsums,
                output.add(base_row),
                n_blocks as i32, input_d, pow2.as_ptr(),
            );
        }
        chunk = current_chunk.fetch_add(1, Ordering::Relaxed);
    }

    // Remainder rows: only thread 0
    if ith == 0 {
        let base = full_quads * 4;
        for i in 0..(n_rows % 4) {
            let row = base + i;
            unsafe {
                *output.add(row) = ffi_inference::q4k_dot_q8k(
                    weight.add(row * row_bytes), input_qs, input_bsums,
                    n_blocks as i32, input_d, pow2.as_ptr(),
                );
            }
        }
    }
}
```

Same pattern for Q5K and Q6K.

- [ ] **Step 2: Test work-stealing matmul matches existing par_matvec**

```rust
#[test]
fn test_q4k_graph_vs_par() {
    // Same weights + input, compare results
}
```

- [ ] **Step 3: Commit**

---

### Task 5: Convert scalar ops (norm, add, scale, rope, gelu) to ith/nth

**Files:**
- Modify: `src/inference/forward_attn.rs`
- Modify: `src/inference/forward_attn_heads.rs`

Each scalar op becomes a function that takes `(ith, nth)` and processes its slice. Matching llama.cpp's `for (i01 = ith; i01 < ne01; i01 += nth)` pattern.

For ops on small data (norm on 1536 floats, rope on 256×8=2048 floats): only thread 0 runs them. Matching llama.cpp where small ops have `if (ith != 0) return;`.

- [ ] **Step 1: Implement ith/nth variants**

```rust
// RMSNorm: only thread 0 (data too small to split)
fn rmsnorm_op(ith: usize, _nth: usize, x: *const f32, w: *const f32, out: *mut f32, n: i32, eps: f32) {
    if ith != 0 { return; }
    ffi_inference::gemma4_rmsnorm(x, w, out, n, eps);
}

// Vec add: only thread 0
fn vec_add_op(ith: usize, _nth: usize, a: *const f32, b: *const f32, out: *mut f32, n: i32) {
    if ith != 0 { return; }
    ffi_inference::vec_add_f32(a, b, out, n);
}

// Attention: split by heads (like current code)
fn attention_op(ith: usize, nth: usize, /* all attn params */) {
    let per = (n_heads + nth - 1) / nth;
    let h_start = ith * per;
    let h_end = ((ith + 1) * per).min(n_heads);
    for h in h_start..h_end { /* Q·K, softmax, V·scores */ }
}
```

- [ ] **Step 2: Test each op variant produces same results**

- [ ] **Step 3: Commit**

---

### Task 6: Build op-list from forward_one

**Files:**
- Modify: `src/inference/forward.rs`
- Modify: `src/inference/forward_attn.rs`

Convert `forward_one` from "execute ops sequentially" to "build OpList, then execute via GraphPool".

- [ ] **Step 1: Create `build_forward_ops()` that returns OpList**

```rust
pub fn build_forward_ops(&mut self, model: &Gemma4Model, token_id: u32) -> OpList {
    let mut ops = OpList::new();

    // Pre-loop: embed + scale (thread 0 only)
    ops.push(|ith, _| { if ith != 0 { return; } /* embed + scale */ });

    // PLE Phase A (thread 0 only)
    ops.push(|ith, _| { if ith != 0 { return; } /* prepare_ple */ });

    // Per-layer ops
    for il in 0..model.n_layers {
        self.push_layer_ops(&mut ops, model, il);
    }

    // Post-loop: final norm + output matmul + softcap
    ops.push(|ith, _| { if ith != 0 { return; } /* final norm */ });
    ops.push(|ith, nth| { /* output matmul — work-stealing */ });
    ops.push(|ith, _| { if ith != 0 { return; } /* softcap */ });

    ops
}
```

- [ ] **Step 2: `push_layer_ops()` pushes per-layer ops**

Each layer pushes ~8 ops matching the current layer_forward steps:
1. attn_norm + quant (thread 0)
2. Q matmul (work-stealing)
3. Q norm + rope (thread 0)
4. K/V matmul (work-stealing) — only for KV layers
5. K norm + V norm + K rope + cache store (thread 0)
6. Attention (split by heads)
7. Wo matmul (work-stealing)
8. post_attn_norm + residual (thread 0)
9. ffn_norm (thread 0)
10. gate+up matmul (work-stealing)
11. gelu (thread 0)
12. down matmul (work-stealing)
13. post_ffn_norm + residual (thread 0)
14. PLE ops (thread 0 + work-stealing for matmul)
15. out_scale (thread 0)

Total: ~15 ops per KV layer, ~12 per shared layer ≈ 490 ops per token, 490 barriers.

But barriers are spin (nanoseconds) vs old condvar (microseconds). 490 × ~100ns = ~49µs overhead vs old 380 × ~10µs = ~3.8ms.

- [ ] **Step 3: Wire forward_one to use GraphPool**

```rust
pub fn forward_one(&mut self, model: &Gemma4Model, token_id: u32, pool: &GraphPool) -> &[f32] {
    let ops = self.build_forward_ops(model, token_id);
    pool.execute(&ops);
    self.cache.advance();
    &self.logits
}
```

- [ ] **Step 4: Run full test suite to verify no regression**

- [ ] **Step 5: Commit**

---

### Task 7: Deploy and measure on Pi

**Files:** none (just measurement)

- [ ] **Step 1: Cross-compile and deploy**

```bash
PATH="..." RUSTFLAGS="..." cargo build --release --target aarch64-unknown-linux-gnu
scp ... peter@10.46.0.27:~/olorin
```

- [ ] **Step 2: Run with GEMMA4_TIMING=1**

```bash
ssh peter@pi 'echo -e "Hello\n/quit" | GEMMA4_TIMING=1 ~/olorin --model ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf 2>&1 | grep timing'
```

Expected: ~155 ms/token (from 170ms), ~6.4 tok/s (from 5.9).

- [ ] **Step 3: Run llama-bench for comparison**

```bash
ssh peter@pi '/tmp/llama.cpp/build/bin/llama-bench -m ... -t 4 -n 16 -p 0 -r 1'
```

Target: within 5% of llama.cpp (6.5 tok/s).

- [ ] **Step 4: Commit measurement results**

---

### Task 8: Remove old ThreadPool

**Files:**
- Modify: `src/inference/threadpool.rs` — delete old ThreadPool
- Modify: all files that reference ThreadPool → GraphPool

- [ ] **Step 1: Search and replace all usages**
- [ ] **Step 2: Run full test suite**
- [ ] **Step 3: Commit**

---

## Risk Analysis

**Lifetime/safety:** OpList captures raw pointers to model weights and state buffers. These are valid for the duration of `forward_one` (same as today). The `execute()` call is synchronous — all ops complete before it returns.

**Correctness:** Every op must produce identical results to the current code. The test suite (18 tests) verifies this. Run after every task.

**Spin on 4 cores:** Unlike pure spin-pool (which failed at 1640ms), this approach has threads doing useful work on every op. They only spin during the barrier (~100ns between ops). Inactive threads for small ops (`if ith != 0 return`) immediately hit the next barrier — negligible waste.

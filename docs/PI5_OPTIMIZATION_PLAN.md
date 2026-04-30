# Pi 5 Decode Optimization Plan

Snapshot 2026-04-28. Target: Pi 5 (Cortex-A76, glibc 2.36, 17 GB/s LPDDR4X-4267).
Decode is **memory-bandwidth-bound** at ~21 GB/s effective.
The optimization frontier is therefore **bytes per decoded token**, not FLOPs.

Current decode after last optimization: **7.49 t/s, 133.57 ms/tok** (Q4K-embed variant).

## Status

- ✅ **gemma4_gelu exp_poly_f32 swap** shipped on `iq3-eval` (`c144a47`).
  - Kernel-level: 2.23× speedup (n=12288).
  - Prompt-eval: 2.00× (compute-bound — full benefit).
  - Decode: 1.012× (bandwidth-bound — kernel speedup mostly hidden).
  - Correctness validated: all 17 `gemma4_verify` tests pass; smoke output bit-identical to baseline; `forward_batch_verify` bit-exact 262144 logits; ~3% logits L2 drift vs llama.cpp (vs 0.5% baseline) — within reference test tolerance.
  - A/B harness: `tests/bench_gemma4_gelu.rs` (`57f8c55`).
  - Pi 5 binaries kept at `~/bench_gelu_libm` and `~/bench_gelu_exp_poly`.

- ✅ **eacompute `feat/i8mm-intrinsics` 287ce17** — pulled, built, integrated.
  - New intrinsics live: `f32x4_from_scalars` / `f32x8_from_scalars` (gather workaround), `--fp16` flag + `f16x4`/`f16x8` types, `exp_poly_f32`.

- ✅ **token_embd Q6K → Q4K (output head reduction)** shipped on `iq3-eval` (`5fffa70`, 2026-04-28).
  - Decode: **6.68 → 7.49 t/s = +12.1%** (beat predicted ~10% ceiling).
  - TTFT: 145.62 → 133.05 ms = −8.6%.
  - RSS: **3075 → 2440 MB = −636 MB** (mostly from dropping the Q6K-only repack/d_arr heap copies in `engine.rs:401-415` for non-Q6K dtypes).
  - Custom requant tool: `tests/requant_token_embd_q6k_to_q4k.rs` — port of llama.cpp `make_qkx2_quants` + `quantize_row_q4_K_ref`. Surgical: only `token_embd.weight` changes, every other tensor byte-identical.
  - New variant on Pi: `~/.olorin/models/gemma-4-e2b-it-Q4_K_M-q4kembed.gguf` (3203 MB). Run with `OLORIN_MODEL_PATH=...q4kembed.gguf`.
  - Dispatch wiring: new `dequant::embed_lookup(weight, dtype, ...)` dispatcher + `q4k_embed_lookup`. Output-head matmul side already routed correctly (engine.rs:401 guard).
  - Validation: 17/17 gemma4_verify, smoke coherent, forward_batch_verify bit-exact 262144 logits.
  - Why decode beat the bandwidth math (~7%): Q4K matvec via i8mm intrinsics outperforms the Q6K-repacked output head it replaces. The 15ms-net-win Q6K repack note in `engine.rs:399` is **wrong** in retrospect — vanilla Q4K matvec is faster.

## What we measured today (the surprise)

`tests/bf16_inventory.rs` (`dump_all_dtypes`) on `gemma-4-e2b-it-Q4_K_M.gguf`:

| Dtype | Count | MB | % total |
|-------|-------|-----|---------|
| Q6K | 53 | 2,407 | 73.2% |
| Q4K | 194 | 755 | 23.0% |
| Q5K | 70 | 97 | 3.0% |
| BF16 | 1 | 26 | 0.8% |
| F32 (norms) | 283 | 1.1 | 0.03% |
| **Total** | 601 | **3,287 MB** | |

**Two embedding tables are 65.5% of all bytes:**

| Tensor | Dtype | Shape | MB |
|--------|-------|-------|-----|
| `per_layer_token_embd.weight` | Q6K | [8960, 262144] | 1,838 |
| `token_embd.weight` (tied to output head) | Q6K | [1536, 262144] | 315 |
| `per_layer_model_proj.weight` | BF16 | [1536, 8960] | 26 |

**Bytes ≠ bandwidth.** Embedding tables are *row lookups* during decode (~8 KB/token total, negligible).
The actual per-token bandwidth is dominated by the transformer body (~1.1 GB) plus the output-head matmul through the tied `token_embd` table (~315 MB) ≈ 1.4 GB/token.

## Ruled out

| Item | Why ruled out |
|------|---------------|
| BF16 PLE → INT8 | Only 26 MB BF16 in model; 13 MB savings = 0.42% decode bandwidth. Not worth 1–2 weeks. |
| Quantizing `per_layer_token_embd` harder | 1.84 GB target on disk but ~0% decode bandwidth (1 row read/token). Helps cold-start RSS only. |
| F32 norms quantization | Total F32 norms = 1.1 MB. Nothing to optimize. |
| Rust toolchain pin (1.83) for cross-compile | Tested — still pulls GLIBC_2.39 weak symbols. The actual fix is RUSTFLAGS wrap (see below). |

## Re-prioritized queue (post 2026-04-29)

| # | Lever | Mechanism | Win | Effort |
|---|-------|-----------|-----|--------|
| ~~0~~ | ~~**Output head reduction**~~ | ~~Q6K token_embd → Q4K via custom requantize.~~ | ~~+12.1% decode~~ | DONE (5fffa70) |
| ~~1~~ | ~~**Q3_K port for Q4K bucket**~~ | ~~Selective Q3K on FFN tensors (1-row dot), then Q3Kx8 prefill GEMM (Tier 2 C).~~ | ~~+13.4% prefill, +14.1% decode, −24.7% RSS vs Q4K_M~~ | DONE (4821351 — q3kffnimpl ship) |
| ~~1~~ | ~~**Vocab pruning**~~ | INVALIDATED. Path A (frequency-based) failed Apr 21: top-50K coverage 95.56% vs 99.9% needed (commit f3ebcd7). Path B (Cauchy-Schwarz norm-based) attempted 2026-04-29 PM, **Phase 1 GATE FAILED at 0.0% safe-skip rate** across hot_n ∈ {30000, 100000} on 1550 decode steps × 25 prompts. Bound `‖hidden‖ × max_cold_norm` runs 20-50× looser than actual `embed_row · hidden`; row-norm distribution is flat in long tail (max_cold_norm only drops 4.7% from 1.279 → 1.219 going 30k → 100k hot_n). **Output-head pruning as a class is dead** without categorically different approach (LSH approximate argmax, learned hot-set, or column-wise early-termination matmul). | NEGATIVE result | DONE (reverted) |
| ~~2~~ | ~~**Q3Kx8 dual-matvec**~~ | ATTEMPTED 2026-04-29, REVERTED. Bit-exact correct (parity test passed 12,288 elements), but **decode regressed −2.85%** (7.73 → 7.51 t/s avg). Root cause: Q3Kx8 tiled layout is **1168 B/super-block (146 B/row)** vs raw Q3K's 110 B/row — **+33% weight-stream bytes per row**. Q8K activation (~1.5 KB) is L1d-resident at decode, so the "halved Q8K bandwidth" hypothesis was structurally wrong: there's no DRAM bandwidth to amortize. Decode is weight-bandwidth-bound, and the tiled layout makes weights bigger. **Lesson**: the Q3Kx8 tiled layout is a prefill-only artifact; do NOT reuse it for decode. | NEGATIVE result | DONE (reverted) |
| 3 | **f16 KV cache** with `--fp16` build | Keep KV f16 end-to-end, no f32 round-trip per attention/RoPE/RMSNorm. Activation axis. | Tens of MB at long context; scales with seq | 1–2 wk |
| ~~4~~ | ~~**i8mm/smmla port**~~ | NOT APPLICABLE to Pi 5. Cortex-A76 is ARMv8.2-A (dotprod-yes, i8mm-no — i8mm is ARMv8.6-A optional). The runtime HWCAP2_I8MM check in `ffi_inference/mod.rs:98` correctly returns false on Pi 5 → falls back to dotprod libs. Earlier note here that "Cortex-A76 has i8mm" was wrong. Open work for A78+ targets only. | n/a on Pi 5 | DELETED |
| 5 | **Q6K → Q5K downgrade on Q6K layers** | 11× 14.77 MB ffn_down + others, ~162 MB target. Requires custom requantize (llama-quantize ignores overrides — see eabrain note 2026-04-24, same gotcha as Q4K-embed). Template: `tests/requant_q4k_to_q3k.rs`. | ~3% if accuracy holds | 1–2 wk + tool |
| 6 | **imatrix-based Q3K** | Lets `attn_*` tensors safely join the Q3K subset (~30 MB more decode bandwidth). Requires running calibration on representative data. Doesn't change prefill. | ~1–2% decode | 1–2 wk + calibration |
| ~~7~~ | ~~**Q3K matvec_8x8 (decode 8x8 single-Q8)**~~ | INVALIDATED by #2 finding above. Item #7 rests on the same "exploit resident Q3Kx8 repack for decode" assumption that just failed empirically — the layout is 33% bigger per row, decode is weight-bandwidth-bound, this can't help. The single-matrix variant has no Q8K-load amortization either. | NEGATIVE prediction | DELETED |
| ~~8~~ | ~~**Memory bloat cleanup**~~ | INVALIDATED by #2 finding. The plan was to drop raw Q3K once #7 existed; investigation showed raw Q3K is what makes decode efficient (smaller working set). Dropping it would regress decode further. The +328 MB RSS for keeping both layouts is the price of having both fast prefill (tiled) and fast decode (raw). | n/a | DELETED |
| 9 | **Q5K → Q4K** | 97 MB → 76 MB | ~0.6% | <1 wk |
| FUTURE | **3-bit-preserving Q3Kx8 decode layout** | Speculative replacement for failed #2/#7. Build a new Q3Kx8 tile that preserves the raw 3-bit `qs` + 1-bit `hmask` form (~208 B/sb vs current 1168 B/sb; ~76% of raw Q3K's 880 B-equivalent — net BANDWIDTH WIN). Kernel-side becomes harder (3-bit unpack inline), but on a bandwidth-bound workload that's the right trade. Substantial new kernel work. | Maybe ~3-5% decode if it works | 2–3 wk + R&D |

## Outstanding tech debt

- `src/inference/generate.rs` at **502 lines** (2 over the 500-line cap). Trivial trim or small split.
- `q3k_dot_q8k_4row` symbol + Q3kDot4RowFn FFI: exported but unused at dispatch sites (kept for cross-arch dispatch parity per q4k pattern). Could be removed.
- Q3Kx8 debug tools (`tests/q3k_repack_scalar_gemm.rs`, `tests/q3k_pi_scalar_vs_kernel.rs`): used during bring-up to bisect kernel vs layout bugs. Keep as future debugging template OR delete to reduce noise — the durable parity gate is `tests/gemm_q3k_8x8.rs`.
- Plan-doc/memory drift in earlier sessions claimed "Q4Kx8 uses i8mm smmla" — it doesn't. Both Q4K and Q3K 8×8 GEMMs use `vdot_lane_i32` (dotprod). Update memory line 121 if the Pi5 plan note resurfaces. (Memory file already corrected in this session.)

## How to resume tomorrow

### Cross-compile recipe (Olorin → Pi 5)

```bash
cd /mnt/c/Users/Peter.lukka/Desktop/DEV/Olorin
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" \
RUSTFLAGS="-C link-args=-Wl,--wrap=pidfd_spawnp -C link-args=-Wl,--wrap=pidfd_getpid" \
cargo build --release --target aarch64-unknown-linux-gnu --test <test_name>
```

The wraps drop GLIBC_2.39 weak `pidfd_*` symbols that Pi's glibc 2.36 cannot resolve.
Verify before SCPing: `aarch64-linux-gnu-objdump -T <binary> | grep "GLIBC_2\.\(3[6-9]\|4[0-9]\)"` should be empty.

### Pi access

`ssh pi` (alias) → `peter@10.46.0.27`, key `~/.ssh/id_ed25519_pi`.

### Re-run benchmarks at any commit

```bash
# A/B kernel-level
git checkout 57f8c55 && cargo build ... --test bench_gemma4_gelu  # baseline
git checkout c144a47 && cargo build ... --test bench_gemma4_gelu  # swap

# A/B end-to-end decode
cargo build ... --test bench_decode_speed

# Correctness
cargo build ... --test gemma4_verify --test gemma4_smoke --test forward_batch_verify
```

### Validation discipline

After any non-bit-exact kernel change, run `gemma4_verify` + `gemma4_smoke` + `forward_batch_verify` on Pi 5 *before* queuing the next optimization. Today's swap was validated this way; future changes should be too.

### Memory pointers

eabrain `[architecture]` 2026-04-27 — corrected gguf byte composition + re-prioritized queue.
eabrain `[pattern]` 2026-04-27 — cross-compile RUSTFLAGS recipe (corrected version, supersedes earlier wrong toolchain-pin advice).
eabrain `[decision]` 2026-04-27 — gelu swap correctness validation.
eabrain `[note]` 2026-04-27 — Pi 5 A/B decode bench results.
eabrain `[note]` 2026-04-28 — Q4K-embed Pi 5 bench (+12.1% decode, −636 MB RSS).
Filesystem feedback: `~/.claude/projects/-mnt-c-Users-Peter-lukka-Desktop-DEV/memory/`.

---

## Next Up: Q3_K port for Q4K bucket (~5–6% decode ceiling, 2–3 wk)

**TL;DR for the next session.** Branch is `iq3-eval @ 5fffa70` (origin synced). Q4K-embed
shipped today. Pi 5 has the new GGUF and validation binaries. The custom requant tool
pattern is established (`tests/requant_token_embd_q6k_to_q4k.rs`); reuse it for Q3_K.
The big new chunk of work is the Q3_K kernels themselves — neither Olorin nor eacompute
has any.

### Block layout (canonical, from llama.cpp `ggml-common.h`)

```c
// 110 bytes per 256 elements; 3.4375 bpw
typedef struct {
    uint8_t hmask[QK_K/8]; // 32 bytes — high bit per element (256 bits)
    uint8_t qs[QK_K/4];    // 64 bytes — low 2 bits per element (4 vals/byte)
    uint8_t scales[12];    // 16 sub-blocks of 16 elements, 6-bit scales packed
    ggml_half d;           //  2 bytes — super-block scale
} block_q3_K;              // 110 bytes total
```

Dequant rule: `value = d * scale[sub_block] * (q3 - 4)`. Where `q3` is the assembled
3-bit value from `(qs >> shift & 0x3) | ((hmask & bit) ? 4 : 0)`. **Note:** scale-only,
**no `min` per sub-block** (unlike Q4K). The `-4` shifts `q3 ∈ [0..7]` to symmetric `[-4..3]`.
16 sub-blocks of 16 elements (Q4K had 8 sub-blocks of 32) — finer-grained scaling.

### Reference implementations to port (llama.cpp paths confirmed 2026-04-28)

- `quantize_row_q3_K_ref` at `ggml/src/ggml-quants.c:1167`
- `dequantize_row_q3_K`   at `ggml/src/ggml-quants.c:1243`
- `vec_dot_q3_K_q8_K`     in the cpu backend (search `ggml/src/ggml-cpu/`)
- `block_q3_K`            at `ggml/src/ggml-common.h:305`

### Concrete first actions (in order)

1. **Read** `kernels/q4k_dot_arm.ea` and `kernels/q4k_dot_8x8_arm.ea` as templates.
   The Q4K kernels do roughly: load `qs`/`hmask`, unpack to 8-bit, dot with `q8k`
   activations, scale-and-accumulate per sub-block. Q3_K's dot kernel is the same
   shape minus the `min`/`dmin` term — strictly simpler.

2. **Write `kernels/q3k_dot.ea` + `kernels/q3k_dot_arm.ea`** mirroring the Q4K
   structure. Hard rule from `CLAUDE.md`: no scalar fallback, every kernel SIMD.
   Q3_K's tighter sub-block size (16 not 32) means 8-wide loops per sub-block
   instead of full vector lanes — adapt accordingly. There's no llama.cpp i8mm
   path for Q3_K yet; codebook-free 3-bit packs nicely into `vshl`/`vand` chains.

3. **Wire FFI** in `src/kernels/ffi_inference.rs` (mirror `q4k_dot_q8k`,
   `q4k_dot_q8k_4row`). Add `Q3kDot*Fn` types in `ffi_inference_types.rs`.

4. **Wire matmul** in `src/inference/matmul.rs` (`q3k_matvec`) and
   `src/inference/matmul_graph.rs` (`q3k_matvec_ws`). Add `GGML_TYPE_Q3_K = 11`
   constant to `matmul.rs`. Update the dispatch in `matvec_ws` and the equivalent
   prefill batch dispatcher.

5. **Update the loader** in `src/inference/engine_helpers.rs` so layer weights with
   `dtype == GGML_TYPE_Q3_K` flow through the right repack (or none — the 8x8 batched
   path is a later optimization). The first cut can skip repack entirely.

6. **Custom requant tool**: `tests/requant_q4k_to_q3k.rs`. Reuse the structure from
   `tests/requant_token_embd_q6k_to_q4k.rs` — same GGUF read/write skeleton (the
   `gguf.rs` accessors `meta_end` / `raw()` / `tensor_names` are already in place).
   The Q4K → f32 dequant path can crib from `dequantize_row_q4_K` (or just call
   `q4k_embed_lookup` once you generalize the row-width assertion). f32 → Q3_K
   port via `quantize_row_q3_K_ref`.

   Decision: requant *all 194 Q4K tensors* in one pass, OR an opt-in subset (e.g.
   `ffn_gate`/`ffn_up` only — the biggest)? Start with all 194 to maximize the
   bandwidth win, accept some accuracy regression risk, then trim selectively if
   `gemma4_verify` flags drift outside tolerance.

7. **Build the Q3_K-bucket variant**:
   `~/.olorin/models/gemma-4-e2b-it-Q4_K_M-q3kbucket.gguf` (or stack: take the
   q4kembed.gguf as input so we get both wins).

8. **Validation discipline (3 legs, established 2026-04-28):**
   - `gemma4_smoke` with `OLORIN_MODEL_PATH=...q3kbucket.gguf` — coherent reply.
   - `forward_batch_verify` — must stay bit-exact 262144 logits (internal parity,
     unaffected by quant noise as long as both paths use the same dispatch).
   - `gemma4_verify` — 17/17 with same loose-bound asserts. The `llama.cpp:` reference
     numbers in `eprintln!`s are Q6K-baseline anchors; Q3_K-bucket variant will
     diverge from them — that's expected and not asserted.

9. **A/B bench**: `bench_decode_speed` baseline vs the q3kbucket variant. Expect
   ~5–6% decode improvement from ~178 MB/token bandwidth saved on the transformer
   body. If the win is much smaller, suspect the Q3_K kernel itself is slower than
   Q4K's i8mm path on Pi 5 (Cortex-A76 has no native 3-bit dot — pure NEON only).

### Pitfalls anticipated

- **Q3_K decode kernel may be slower than Q4K** on Cortex-A76 because there's no
  hardware 3-bit dot product, just bit-extraction overhead. The bandwidth saving
  has to outweigh this. If the kernel-level bench (n=12288 dot) is more than ~25%
  slower than `q4k_dot`, the end-to-end win could disappear.
- **Accuracy**: 3-bit is meaningfully lossier than 4-bit. May need to keep
  `attn_v` and `ffn_down` (Q4K-bucket members not in the Q4K target — they're
  actually Q6K in the source) as Q6K, OR exempt specific layers based on
  sensitivity (`adaptive_quant.rs` recipe). Run `gemma4_verify`'s logit drift
  comparison after the first cut to decide.
- **Dispatch hole**: the engine doesn't currently know about Q3_K. Forgetting to
  add the GGML_TYPE_Q3_K constant + dispatch arms in `matmul_graph.rs:340` etc.
  will produce a runtime panic at the first Q3_K matvec call rather than a build
  error (because the `match` is on `u32`).
- **Q3_K vs Q3_K_M aliasing**: in `llama-quantize`'s help, `Q3_K` is "alias for
  Q3_K_M". The underlying GGML dtype is `GGML_TYPE_Q3_K = 11` — that's what we
  produce in the requant tool. Don't get confused by recipe names.

### Reusable infrastructure shipped 2026-04-28

- `src/inference/gguf.rs` accessors: `meta_end: u64`, `tensor_names: Vec<String>`,
  `pub fn raw(&self) -> &[u8]`. The requant tool needs all three.
- `src/inference/dequant.rs::embed_lookup(weight, dtype, ...)` dispatcher pattern.
  Extend with `GGML_TYPE_Q3_K` arm if the embed table ever becomes Q3_K (probably
  won't — but the layer matmul side is what we touch instead).
- `tests/requant_token_embd_q6k_to_q4k.rs` skeleton — copy as `requant_q4k_to_q3k.rs`,
  swap the inner Q4K logic for Q3_K, swap the target dtype in the new tensor info.
- `OLORIN_MODEL_PATH` env-var convention is now in `gemma4_verify`,
  `gemma4_smoke`, `forward_batch_verify`, `bench_decode_speed`.

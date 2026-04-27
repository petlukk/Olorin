# Pi 5 Decode Optimization Plan

Snapshot 2026-04-27. Target: Pi 5 (Cortex-A76, glibc 2.36, 17 GB/s LPDDR4X-4267).
Decode is **memory-bandwidth-bound** at ~21 GB/s effective (~1.4 GB/token, 146 ms/token, 6.84 t/s).
The optimization frontier is therefore **bytes per decoded token**, not FLOPs.

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

## Re-prioritized queue

| # | Lever | Mechanism | Decode ceiling | Effort |
|---|-------|-----------|----------------|--------|
| 1 | **Output head reduction** | Vocab prune (262144→domain subset) OR Q6K head→Q4K via custom requantize. Tied to `token_embd`, so one change = head shrinks AND input embed lookup shrinks. | ~10% | 1 wk vocab prune; 1–2 wk requantize |
| 2 | **Q3_K port for Q4K bucket** | 755 MB → ~566 MB. Codebook-free 3-bit per `q4k_*_arm.ea` template. | ~5–6% | 2–3 wk |
| 3 | **Q6K → Q5K downgrade on layer Q6K** | 11× 14.77 MB ffn_down + others, ~162 MB target. Requires custom requantize (llama-quantize ignores overrides — see eabrain note 2026-04-24). | ~3% if accuracy holds | 1–2 wk + tool |
| 4 | **f16 KV path** with `--fp16` build | Keep KV f16 end-to-end, no f32 round-trip per attention/RoPE/RMSNorm. Activation axis. | Tens of MB at long context; scales with seq | 1–2 wk |
| 5 | Q5K → Q4K | 97 MB → 76 MB | ~0.6% | <1 wk |

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
Filesystem feedback: `~/.claude/projects/-mnt-c-Users-Peter-lukka-Desktop-DEV/memory/`.

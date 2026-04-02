# Performance Analysis — Olorin vs llama.cpp (2026-04-02)

## Hardware
- Raspberry Pi 5 (Cortex-A76, 4 cores @ 2.4GHz)
- LPDDR4X measured bandwidth: 14.3 GB/s
- ISA: ARMv8.2-A + DOTPROD (no I8MM, no SVE)

## Model
- Llama 3.2 3B Instruct Q4_K_M (1.87 GB)
- h=3072, kv=512, f=8192, head_dim=128, 28 layers

## Current Performance

|              | Olorin  | llama.cpp | Gap   |
|--------------|---------|-----------|-------|
| Decode       | 231ms/tok (4.3 tok/s) | 174ms/tok (5.7 tok/s) | 1.33× |
| Prefill (29) | 72ms/tok (13.9 tok/s) | 35ms/tok (28.5 tok/s) | 2.06× |

## Isolated Kernel Benchmark

Branch `bench/q4k-dot-isolate`: same Q4K+Q8K data, 12 blocks, 100K iterations.

```
olorin (Eä → LLVM):  ~230 ns/call
llama.cpp (GCC -O3): ~220 ns/call
median ratio: 1.05×
```

**Eä compiler produces near-identical NEON code to GCC.** Kernel quality is NOT the bottleneck.

## Decode Critical Path (1 token)

```
total: 224ms
├── kernel work (in dispatches):  215ms  (96.0%)
├── serial overhead:                6.7ms (3.0%)
│   ├── rmsnorm × 3
│   ├── quant_f32_q8k × 3
│   ├── bias + rope
│   └── kv_append (TurboQuant)
├── dispatch tax:                   1.3ms (0.6%)
│   └── 113 dispatches × 11µs condvar
└── other:                          1.0ms (0.4%)

Dispatches per layer: 4 (QKV split3 + Wo + gate_up_silu + down)
```

## Prefill Critical Path (29 tokens)

```
total: 2090ms (72ms/tok)
├── kernel work (in dispatches): 1980ms  (94.7%)
├── serial overhead:              111ms  (5.3%)
│   ├── attention per-token loop
│   ├── KV transpose (token→head major)
│   └── bias + rope per token
├── dispatch tax:                   3.8ms (0.2%)
│   └── 393 dispatches × 10µs condvar
└── other:                          ~0ms

Dispatches per layer: 14 (norm+quant + Q + K + V + quant_Wo + Wo + vecadd + norm+quant + gate_up + quant_down + down + vecadd + attention)
```

## Where the 1.33× Decode Gap Lives

Dispatch overhead is 0.6% — NOT the bottleneck.
Serial overhead is 3.0% — NOT the bottleneck.
96% of time is inside kernel calls — but Olorin does MORE WORK per token than llama.cpp.

### Gap Breakdown

| Source | Estimated cost | % of 224ms |
|--------|---------------|-----------|
| Kernel 5% slower (isolated) | ~10ms | 4.5% |
| **Scale pre-comp (f16→f32) separate pass** | **~28ms** | **12.5%** |
| KV TurboQuant rotation | ~1.1ms | 0.5% |
| Q8K quant extra passes (3×) | ~1ms | 0.5% |
| Serial overhead | ~6.7ms | 3.0% |
| Dispatch tax | ~1.3ms | 0.6% |
| **Total identified overhead** | **~48ms** | **~22%** |
| Remaining (≈ llama.cpp baseline) | ~176ms | ~78% |

176ms pure kernel ≈ llama.cpp's 174ms. Gap explained.

### Scale Pre-computation: The Biggest Single Issue

Olorin pre-computes `d_arr[blk] = f16_to_f32(q4_d) * q8_d` in Rust (`unpack_d()`) BEFORE calling the kernel. This is done per 4-row group, per GEMM call.

llama.cpp reads f16 d/dmin INSIDE the kernel, from the same cache line as the nibble data — zero extra memory traffic.

Olorin's separate pass:
- 4 rows × 12 blocks × 2 f16 reads = 96 conversions per group
- 768 groups per Q matmul (h=3072) → ~150µs
- 7 matmuls/layer × 28 layers → ~28ms/token

### Fix Options (Priority Order)

1. **Read f16 d/dmin inside kernel** — eliminate `unpack_d()` entirely. Kernel already has weight pointer; just read bytes 0-3 per block inline. Requires eacompute `f16_to_f32` intrinsic or `ptr_as` cast.

2. **Pre-compute scales once per layer** — cache all row scales at layer start, reuse across GEMM calls. Eliminates redundant `unpack_d` in tiled GEMM.

3. **Simpler KV cache** — replace TurboQuant (FWHT rotation + Q4 quant) with direct Q8_0 write (like llama.cpp). Saves ~1ms/token.

4. **Fuse Q8K quantization with rmsnorm** — eliminate separate quant passes. Single kernel: norm → quant in one pass. Saves ~1ms/token.

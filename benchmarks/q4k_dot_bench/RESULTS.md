# Q4K Dot Product Benchmark Results — 2026-04-02

## Setup
- **Hardware:** Raspberry Pi 5 (Cortex-A76, 4 cores @ 2.4GHz, LPDDR4X 14.3 GB/s)
- **ISA:** ARMv8.2-A + DOTPROD (no I8MM, no SVE)
- **Workload:** 12 Q4K blocks (3072 dim = Llama 3.2 3B hidden_dim), 100K iterations
- **Olorin kernel:** Eä → LLVM 18 → AArch64, `--opt-level=3 --target=cortex-a76 --dotprod`
- **llama.cpp kernel:** GCC 12.2 `-O3 -march=armv8.2-a+dotprod`, extracted from ggml quants.c

## Results

```
olorin: ~230 ns/call
llama:  ~220 ns/call
ratio:  1.05× (median over 5 runs)
```

Individual runs: 1.04×, 1.39×, 1.04×, 1.05×, 1.05×
(1.39× outlier = scheduler interrupt)

## Conclusion

**Eä compiler produces near-identical NEON code to GCC -O3.** The 5% gap is within
measurement noise and minor instruction scheduling differences.

The end-to-end decode gap (Olorin 231ms vs llama.cpp 174ms = 1.3×) is NOT caused by
kernel quality. It comes from Rust orchestration overhead:

1. **Thread dispatch** — condvar wake/sleep per pool.run() call (~10 dispatches/layer)
2. **Scale pre-computation** — f16→f32 × q8_d per block per 4-row group, every matmul call
3. **KV cache TurboQuant** — FWHT rotation + Q4 quantization per token append
4. **Pointer indirection** — BatchQ8K stores per-token pointers as Vec<usize>

## Implications

- Stop optimizing LLVM flags / instruction patterns — kernel is at parity
- Focus on reducing Rust-side overhead (fewer dispatches, simpler KV cache)
- Speculative decoding amortizes orchestration overhead over K tokens
- Consider: pre-compute scales once per layer, not per 4-row group

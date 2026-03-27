# Olorin Performance — Why It's Fast

## The Pipeline

Every message through Olorin hits this path:

```
input → safety scan → command router → recall → vault → inference → output
```

Each stage uses SIMD kernels compiled by [Eä](https://github.com/petlukk/eacompute) that fit in L1/L2 cache. No allocations in the hot path. No syscalls. No framework overhead. The HTTP server is raw TCP with manual parsing — no Actix, no Axum, no Tokio in the request path.

## Stage Breakdown

### Safety Scan — `fused_safety` kernel

Three SIMD scanners fused into a single pass over the input bytes:
- **Byte classifier** — validates input encoding
- **Leak scanner** — detects secrets (API keys, tokens, credentials)
- **Injection scanner** — catches prompt injection attempts

On Pi 5: **15 µs** for 1 KB input (0.07 GB/s). The entire safety check completes before a single network round-trip.

### Command Router — `command_router` kernel

SIMD hash-based lookup with 2-stage verification. Routes 27 slash commands.

On Pi 5: **10 ns/call** — faster than a single L3 cache miss (~40 ns on A76). At 1.2 million dispatches per 12 ms, the router is effectively free.

### Recall — JL projection + SIMD cosine search

The recall system stores conversation fragments as embeddings and retrieves the most relevant ones for context injection.

**Dimensionality reduction:** 256-dim embeddings are projected to 64-dim using the Johnson-Lindenstrauss lemma — a Walsh-Hadamard Transform (FWHT) with random sign-flips, implemented as the `turbo_rotate` + `jl_project` kernels. This preserves pairwise distances within ε with high probability while cutting memory and compute by 4x.

**Search:** `batch_cosine` computes similarity between the query and all stored vectors using NEON dot product instructions. `top_k` selects the best matches with branchless selection.

On Pi 5 (1024 stored vectors):
- **Insert:** 3.6 µs/vector
- **Recall (top-5):** 52.5 µs/query
- **JL projection:** 3.3 µs per 256→64 dim reduction
- **Memory:** 256 KB for 1024 vectors (256 bytes each) — fits in L2

### Vault — ChaCha20 encryption (`eachacha`)

SIMD-accelerated ChaCha20 stream cipher with interleaved quarter-rounds. All vault operations (persist conversations, recall fragments) are encrypted at rest.

Kernel: `libchacha20.so` via the `eachacha` crate.

### Inference — Cougar engine

**BitNet b1.58:** Ternary weights {-1, 0, +1} mean matrix multiplication becomes pure addition and subtraction. ARM NEON `i8dot` kernels with pre-converted signed i8 weights (no runtime XOR). 10 NEON kernels with full x86 parity.

**Q4K/Q6K:** 4-bit quantized Llama/Qwen models using fused dequantize + dot product kernels.

On Pi 5 (ARM Cortex-A76 @ 2.4 GHz):
- **BitNet 2B I2_S:** ~16 tok/s
- **Qwen2.5 1.5B Q4_K_M:** ~9 tok/s
- **Llama 3.2 3B Q4_K_M:** ~4 tok/s

## Why It Fits In Cache

ARM Cortex-A76 (Raspberry Pi 5): 64 KB L1d, 512 KB L2, 2 MB shared L3.

| Data | Size | Cache Level |
|------|------|-------------|
| Query vector (64-dim) | 256 B | L1d |
| ChaCha20 state | 64 B | L1d |
| Command router hash table | <1 KB | L1d |
| SIMD kernel code | <4 KB each | L1i |
| JL vectors (1024 × 64-dim) | 256 KB | L2 |

Everything except the vector store fits in L1. The vector store fits in L2. No main memory access in the hot path — only cache hits. This is why a $80 single-board computer can run the entire agent pipeline in microseconds.

## Numbers (Raspberry Pi 5, ARM A76 @ 2.4 GHz)

```
─── safety (fused_safety — SIMD byte classifier + leak + injection, single pass) ───
  1024 B input, 10000 iterations
  Per call: 15093 ns
  Throughput: 0.07 GB/s

─── router (command_router — SIMD hash lookup, 2-stage verified) ───
  12 commands × 100000 iterations
  Per call: 10 ns
  Total calls: 1200000

─── recall (JL 256→64, NEON batch_cosine, 1024 vecs = 256 KB) ───
  Per insert: 3.6 us
  Per recall (top-5 from 1024): 52.5 us
  Memory: 256 KB for 1024 vectors (256 B/vec)

─── search (search kernel — NEON/AVX batch dot, branchless top-k) ───
  Per search: 39.5 µs  (1024 × 64-dim vectors)
  Throughput: 25K searches/s

─── jl (turbo_rotate FWHT + sign-flip → jl_project 256→64) ───
  Per projection: 3271 ns
  Throughput: 0.3M projections/s
```

## Run It Yourself

```
olorin --interactive
/bench all
```

Or in the web UI, open a REPL tile (Alt+T) and type `/bench all`.

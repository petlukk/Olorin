# Architecture

Olorin is a single Rust crate with two runtime dependencies (`libc` +
`libloading`). Everything else — the model loader, the BPE tokenizer, the
JSON parser, the WhatsApp transport, the web UI server — is hand-built.

The compute is in **Ea SIMD kernels** (`.ea` source files, cross-compiled to
`.so` shared objects by `build.rs`, embedded into the binary via
`include_bytes!`, and extracted to `~/.olorin/lib/{version}/` on first run).
The Rust code is the orchestrator: it loads kernels into a `KernelTable` at
startup and calls into them via FFI wrappers. Zero scalar fallbacks; every
compute path is a kernel.

## Source layout

```
olorin/
  src/
    core/           The Brain
      router.rs       The Olorin Pipe — sole entry/exit point
      safety.rs       Fused SIMD safety scan (score-based, multi-language)
      shell_guard.rs  Shell command classifier
      anthropic.rs    Cloud fallback (curl subprocess)
      dispatch.rs     Command routing + intent classification
      handlers.rs     LLM turn handling + output guard
      tool_parse.rs   Streaming tool_call detector
      llm.rs          Message types, Gemma 4 chat formatting

    inference/      The Engine
      generate.rs     Public API — prompt in, text out
      engine.rs       Gemma 4 model loading (GGUF)
      forward.rs      Gemma 4 forward pass + PLE phase-A
      forward_graph.rs  Graph-threaded decode (spin-barrier)
      forward_batch.rs  Batched prefill forward
      cache.rs        F16 KV-cache (sliding window + shared layers)
      matmul.rs       Q4K/Q5K/Q6K/BF16 matmul dispatchers
      matmul_graph.rs Work-stealing parallel GEMV/GEMM
      threadpool.rs   SpinBarrier + GraphPool (matches llama.cpp)
      gguf.rs         GGUF format parser
      tokenizer.rs    BPE tokenizer

    storage/        The Vault
      vault.rs        Encrypted append-only storage (ChaCha20-Poly1305 AEAD)
      crypto.rs       ChaCha20 encrypt/decrypt + Poly1305 MAC
      search.rs       FusedSearcher (zero-exposure)
      secure.rs       SecureBuffer (mlock + SIMD zeroize)
      json.rs         Recursive descent JSON parser
      key.rs          Key derivation

    interface/      The Gates
      terminal.rs     REPL (stdin/stdout)
      server.rs       Web UI + WhatsApp (std::net, TCP_NODELAY SSE)
      exec.rs         Process spawning (fork/exec)
      pty.rs          PTY management for web terminal

    kernels/        The Arsenal
      ffi.rs          KernelTable + core/safety/storage wrappers
      ffi_inference.rs  Inference FFI (Q4K/Q6K/BF16 dot, GEMM, RoPE, etc.)

    tools/          20 built-in tools
    recall.rs       VectorStore (JL-projected embeddings)

  kernels/          64 Ea SIMD kernel source files (flat) — 42 logical kernels
                    with ARM variants
  web/chat.html     Chat UI (Catppuccin themed, embedded in binary)
  tests/            60 test files, 318 tests
```

## Gemma 4 E2B inference

Text-only. Q4K weights, Q6K embeddings. 35-layer transformer with:

- **Sliding window (512 tok) + global attention** alternating 4:1
- **Shared KV cache** — last N layers reuse KV from earlier layers
- **Per-Layer Embeddings (PLE)** — GELU-gated residual signal per layer
- **Dual RoPE** — standard for sliding, proportional for global
- **GeGLU FFN** — GELU-gated feed-forward

All compute in Ea SIMD kernels: Q4K 8×8 repacked GEMV for decode, work-stealing
GEMM for prefill, hot-vocab (32K logits) for fast sampling.

## Kernel infrastructure

64 Ea SIMD kernel source files (42 logical kernels with ARM variants) compiled
by `build.rs` into shared objects (ARM NEON + dotprod on aarch64, SSE/AVX2 on
x86_64).

Kernels are embedded in the binary via `include_bytes!` and extracted to
`~/.olorin/lib/{version}/` on first run. Version is a content hash.

Key kernels:

- `q4k_dot_8x8_arm.ea` — Q4K 8-row tiled GEMV with vdot_i32
- `q4k_dot_8x8_dual_arm.ea` — Fused gate+up dual GEMV (shared Q8K input)
- `q4k_dot_8x8_gemm_arm.ea` — Q4K 8×8 GEMM for batched prefill
- `q6k_gemm_arm.ea` — Q6K GEMM with vdot_lane_i32
- `bf16_matvec_arm.ea` — BF16 dot with 4-token register tile
- `fused_safety.ea` — Single-pass injection + leak detection
- `chacha20_search_v2.ea` — Decrypt-and-search in SIMD registers
- `csv_scan.ea` — CSV structural scan (commas + newlines) for runes
- `jsonl_struct.ea` — JSON Lines structural scan (5-bit mask:
  newlines/quotes/colons/commas/backslashes)
- `log_level_scan.ea` — Multi-keyword severity scan with word boundaries +
  line count + ERROR/FATAL position recording (cross-arch, no `_arm.ea`
  variant)

## Runtime layout

```
~/.olorin/
  lib/{version}/  # Extracted SIMD kernels (.so)
  vault/default/  # Encrypted conversations (ChaCha20-Poly1305 AEAD)
  models/         # GGUF model files
```

## Stability contract

The v1 runes (`eacrunch`, `eajson`, `eaparquet`, `ealog`, `eatime`, `eadiff`),
the `RuneOutput v1` JSON schema, and the `--json` chaining contract are
stable. They will not change without a v2.0 bump. See [`CHANGELOG.md`](../CHANGELOG.md)
for what's stable vs out-of-scope for the v1 commitment.

## Hard rules (enforced)

- **No file exceeds 500 lines.** Split before you hit the limit.
- **Every feature proven by end-to-end test.** If it's not tested, it doesn't
  exist.
- **No fake functions. No silent fallbacks.** No `todo!()`, no
  `// TODO`/`// HACK`/`// placeholder`.
- **Delete, don't comment.** Dead code gets removed.
- **SIMD first.** Every compute path is an Ea kernel; zero scalar fallbacks.

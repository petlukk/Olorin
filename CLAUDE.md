# Olorin — The Wakeful Mind in Ea

Single-binary AI agent. Gemma 4 E2B SIMD inference, encrypted vault, tool-use, Web UI + WhatsApp.

## eabrain (use first, grep second)

Run `eabrain status` at the start of every session/task. Before grepping for an Eä symbol or assuming a kernel/intrinsic doesn't exist, run `eabrain search <name>` and `eabrain ref <name>`. After editing `.ea` files, run `eabrain index` to refresh the index. Save cross-session findings via `eabrain remember`.

**Limitation:** eabrain indexes `.ea` kernel source files but **not** eacompute's Rust intrinsic definitions. If `eabrain ref` returns nothing for an intrinsic name, grep `/root/dev/eacompute/src/typeck/intrinsics*.rs` and `/root/dev/eacompute/src/codegen/simd*.rs` directly before concluding the intrinsic doesn't exist.

## Build

Ea compiler required in PATH:
```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build
```

## Hard Rules

- **No file exceeds 500 lines.** Split before you hit the limit.
- **Every feature proven by end-to-end test.** If it's not tested, it doesn't exist.
- **No fake functions.** No silent fallbacks.
- **No premature features.** Don't build what isn't needed yet.
- **Delete, don't comment.** Dead code gets removed.

## Ea SIMD Rules (for Claude and all subagents)

- **SIMD is priority.** Every compute operation must be an Ea kernel. Zero scalar fallbacks.
- **Triple-check eacompute before concluding an intrinsic doesn't exist.** Check `src/typeck/intrinsics*.rs`, `src/codegen/simd*.rs`, `CHANGELOG.md`, `README.md`, and `tests/`. If you checked all five and it's genuinely not there, stop and report — Peter will build it if needed.
- **Never replace an Ea kernel with Rust scalar code.** Olorin is the Ea showcase. Every kernel demonstrates what eacompute can do.
- **Ea has full if/else.** Don't use workarounds. Write natural control flow.
- **Ea load is type-directed.** `let v: i16x4 = load(ptr, 0)` gives a 64-bit load. No `load_narrow` needed.
- **When in doubt, it should be a kernel.** f16 conversion, attention, RoPE, GELU, RMSNorm — all Ea kernels.
- **Match llama.cpp exactly first.** Get correctness before optimizing. No creativity in the forward pass until output matches.

## The Ea Way

- **SIMD-first.** If a problem can be solved with a kernel, solve it that way.
- **Cache-resident.** Hot path fits in L1d/L2.
- **Zero-exposure.** Plaintext never lingers. Encrypt immediately, search encrypted.
- **Pre-allocate, reuse.** Structs own their buffers. Allocate once, reuse forever.
- **Kernel in KernelTable.** All SIMD kernels loaded once via OnceLock at startup.
- **True Zero deps.** Only libc + libloading. Everything else hand-built.
- **Minimal binary.** Single executable, all kernels embedded via include_bytes!.

## Architecture

Single crate, single binary. No workspace.

| Module | Role |
|--------|------|
| `core/router.rs` | Master Router — The Olorin Pipe (sole entry/exit) |
| `core/safety.rs` | Fused safety pipeline |
| `core/shell_guard.rs` | Shell command classifier |
| `core/anthropic.rs` | Cloud fallback via curl subprocess |
| `core/llm.rs` | Message types, ChatML formatting |
| `core/dispatch.rs` | Command routing + intent classification |
| `core/handlers.rs` | LLM turn handling + message building |
| `core/tool_parse.rs` | Streaming tool_call XML detector |
| `inference/engine.rs` | Gemma 4 model loading (GGUF) |
| `inference/forward.rs` | Gemma 4 forward pass |
| `inference/cache.rs` | F16 KV-cache (sliding window + shared layers) |
| `inference/matmul.rs` | Q4K/Q6K matmul wrappers |
| `inference/gguf.rs` | GGUF format parser |
| `inference/tokenizer.rs` | BPE tokenizer |
| `inference/generate.rs` | Public inference API — prompt in, text out |
| `inference/threadpool.rs` | Work-stealing thread pool |
| `storage/vault.rs` | Encrypted conversation storage |
| `storage/crypto.rs` | ChaCha20 encrypt/decrypt |
| `storage/search.rs` | FusedSearcher — zero-exposure vault search |
| `storage/secure.rs` | SecureBuffer (Ghost-Buster: mlock + SIMD zeroize) |
| `storage/json.rs` | Minimal JSON scanner (no serde) |
| `storage/key.rs` | Vault key derivation + xxhash |
| `interface/terminal.rs` | REPL |
| `interface/server.rs` | Web UI + WhatsApp gateway (std::net, no tokio) |
| `interface/exec.rs` | Process spawning (fork/exec) |
| `kernels/ffi.rs` | KernelTable + core/safety/storage FFI wrappers |
| `kernels/ffi_inference.rs` | Inference FFI wrappers |
| `recall.rs` | VectorStore (session embeddings, SecureBuffer-backed) |
| `tools/` | 19 built-in tools |

## The Olorin Pipe

All input follows: Safety -> Slash -> Intent -> Recall -> Infer -> Vault save -> Response.
Recall only sees post-safety sanitized input. All responses saved encrypted to vault. No exceptions.
All tests in `tests/` — zero `#[cfg(test)]` in `src/`.

## Kernel Infrastructure

All kernels in flat `kernels/` dir. `build.rs` auto-discovers, compiles, generates `KernelId` enum.
Embedded via `include_bytes!`, extracted to `~/.olorin/lib/{version}/`.
`KernelTable` stored in `OnceLock`. All FFI wrappers in `ffi.rs` and `ffi_inference.rs`.

## Key Patterns

**FusedSearcher** (vault search): ~23 KB scratch, chacha20_search_v2 kernel, decrypt+search in SIMD registers
**SecureBuffer** (Ghost-Buster): mlock'd memory, SIMD-zeroed on Drop via zeroize kernel
**VectorStore** (recall): JL-projected byte-histogram embeddings, ring buffer, SecureBuffer scratch

All follow: `new()` -> own buffers -> `&mut self` methods reuse them.

## Gemma 4 E2B Inference

- **Text-only.** No vision/audio encoders.
- **Q4K weights.** Q6K for embeddings.
- **Sliding window (512 tok) + global attention** alternating 4:1.
- **Shared KV cache** — last N layers reuse KV from earlier layers.
- **Per-Layer Embeddings (PLE)** — residual signal per decoder layer.
- **Dual RoPE** — standard for sliding, proportional for global.
- **GeGLU FFN** — GELU-gated feed-forward.
- **Verify against llama.cpp** — layer-by-layer L2-norm comparison.

## Security

- **Vault key**: XOR-obfuscated in .rodata, derived at runtime via hardware ID
- **SecureBuffer**: mlock + SIMD-zeroed Drop for all sensitive data
- **Recall**: Embeddings zeroed via SecureBuffer after use
- **All input untrusted**: Web, WhatsApp, REPL all pass through full safety pipeline

## SIMD Alignment

Ea kernels that take `*const i32` / `*mut i32` require 4-byte alignment. Use `Vec<i32>` not `Vec<u8>` for scratch buffers passed as i32 pointers.

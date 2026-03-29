# Olorin — The Wakeful Mind in Eä

Single-binary AI agent. BitNet SIMD inference, encrypted vault, tool-use, Web UI + WhatsApp.

## Build

Eä compiler required in PATH:
```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build
```

## Hard Rules

- **No file exceeds 500 lines.** Split before you hit the limit.
- **Every feature proven by end-to-end test.** If it's not tested, it doesn't exist.
- **No fake functions.** No silent fallbacks.
- **No premature features.** Don't build what isn't needed yet.
- **Delete, don't comment.** Dead code gets removed.

## The Eä Way

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
| `core/handlers.rs` | LLM turn handling + output guard |
| `core/tool_parse.rs` | Streaming tool_call XML detector |
| `inference/engine.rs` | BitNet/Llama model loading |
| `inference/forward.rs` | BitNet forward pass |
| `inference/forward_llama.rs` | Llama forward pass |
| `inference/cache.rs` | TurboQuant KV-cache |
| `inference/matmul.rs` | I2S matmul |
| `inference/gguf.rs` | GGUF format parser |
| `inference/tokenizer.rs` | BPE tokenizer |
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
| `kernels/ffi_inference.rs` | Inference + KV-cache FFI wrappers |
| `inference/generate.rs` | Public inference API — prompt in, text out |
| `recall.rs` | VectorStore (session embeddings, SecureBuffer-backed) |
| `tools/` | 19 built-in tools |

## The Olorin Pipe

All input follows: Safety -> Slash -> Intent -> Recall -> Infer -> Guard -> Vault save -> Response.
Recall only sees post-safety sanitized input. All responses saved encrypted to vault. No exceptions.
All tests in `tests/` — zero `#[cfg(test)]` in `src/`.

## Kernel Infrastructure

All kernels in flat `kernels/` dir. `build.rs` auto-discovers, compiles, generates `KernelId` enum.
Embedded via `include_bytes!`, extracted to `~/.olorin/lib/{version}/`.
`KernelTable` stored in `OnceLock`. All FFI wrappers in `ffi.rs` and `ffi_inference.rs`.

## Key Patterns

**FusedSearcher** (vault search): ~23 KB scratch, chacha20_search_v2 kernel, decrypt+search in SIMD registers
**SecureBuffer** (Ghost-Buster): mlock'd memory, SIMD-zeroed on Drop via zeroize kernel
**EakvCache** (KV-cache): TurboQuant quantization, fused attention kernels
**VectorStore** (recall): JL-projected byte-histogram embeddings, ring buffer, SecureBuffer scratch

All follow: `new()` -> own buffers -> `&mut self` methods reuse them.

## Security

- **Vault key**: XOR-obfuscated in .rodata, derived at runtime via hardware ID
- **SecureBuffer**: mlock + SIMD-zeroed Drop for all sensitive data
- **Recall**: Embeddings zeroed via SecureBuffer after use
- **All input untrusted**: Web, WhatsApp, REPL all pass through full safety pipeline

## SIMD Alignment

Ea kernels that take `*const i32` / `*mut i32` require 4-byte alignment. Use `Vec<i32>` not `Vec<u8>` for scratch buffers passed as i32 pointers.

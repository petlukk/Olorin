# Olorin

> The Wakeful Mind in Ea

Single-binary AI agent. Gemma 4 E2B inference via Ea SIMD kernels,
encrypted vault, 20 tools, Web UI and WhatsApp bridge. Two dependencies.

```
./olorin --model ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf --serve
./olorin --model ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf --interactive
./olorin --serve --whatsapp
```

## The Olorin Pipe

Every message follows the same path. No sidechannels. No exceptions.

```
         REPL / Web UI / WhatsApp
                   |
                   v
        core::router::dispatch()
                   |
        1. Safety Scan ---------> BLOCK
        2. Slash Command? ------> /tools direct
        3. Intent Router? ------> kernel (calc/time/cpu)
        4. Recall --------------> session + vault search
           (sanitized input only, never raw)
        5. Inference -----------> Gemma 4 local or Anthropic cloud
        6. Output Guard --------> truncate/block
                   |
                   v
        storage::vault::append()  (ChaCha20 encrypted)
                   |
                   v
              Response
```

## Architecture

Single crate. No workspace. Two dependencies.

```toml
[dependencies]
libc = "0.2"
libloading = "0.8"
```

Everything else is hand-built:

```
olorin/
  src/
    core/           The Brain
      router.rs       The Olorin Pipe — sole entry/exit point
      safety.rs       Fused SIMD safety scan
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
      vault.rs        Encrypted append-only storage
      crypto.rs       ChaCha20 encrypt/decrypt
      search.rs       FusedSearcher (zero-exposure)
      secure.rs       SecureBuffer (mlock + SIMD zeroize)
      json.rs         Recursive descent JSON parser
      key.rs          Key derivation + xxHash64

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

  kernels/          59 Ea SIMD kernel source files (flat)
  web/chat.html     Chat UI (Catppuccin themed, embedded in binary)
  tests/            33 test files, 213 tests
```

## Gemma 4 E2B Inference

Text-only. Q4K weights, Q6K embeddings. 35-layer transformer with:

- **Sliding window (512 tok) + global attention** alternating 4:1
- **Shared KV cache** — last N layers reuse KV from earlier layers
- **Per-Layer Embeddings (PLE)** — GELU-gated residual signal per layer
- **Dual RoPE** — standard for sliding, proportional for global
- **GeGLU FFN** — GELU-gated feed-forward

All compute in Ea SIMD kernels. Q4K 8x8 repacked GEMV for decode,
work-stealing GEMM for prefill. Hot-vocab (32K logits) for fast sampling.

## Performance

Measured on Raspberry Pi 5 (4 cores, LPDDR4X):

| | Olorin | llama.cpp | Delta |
|---|---|---|---|
| Decode | **7.7 tok/s** | 6.4 tok/s | +20% |
| Prefill | 26.7 tok/s | 27.4 tok/s | -3% |

Gemma 4 E2B IT (Q4_K_M, 3.3 GB). Olorin's decode advantage comes from
hot-vocab sampling, fused dual gate+up GEMV, and graph-loop threading
tuned for 4-core ARM.

## The Vault

Every conversation is encrypted at rest using ChaCha20.

```
Write:  message --> ChaCha20 encrypt --> vault.bin (append-only)
                                     --> byte histogram --> index

Search: query --> histogram --> cosine similarity vs index --> ranked blocks
             --> FusedSearcher: decrypt+search in SIMD registers
             --> only matched context lines returned

Read:   block --> xxHash64 verify --> ChaCha20 decrypt
```

The FusedSearcher (`chacha20_search_v2` kernel) decrypts in SIMD registers,
searches in-register, and returns only matched context lines. ~95% of block
content never exists as plaintext.

## Security

- **Vault key**: XOR-obfuscated seed in .rodata, derived at runtime via hardware ID
- **SecureBuffer**: `mlock` + SIMD-zeroed on Drop for all sensitive data
- **All input untrusted**: every channel passes full safety pipeline

## Tools

20 built-in tools, routed by SIMD command parser:

| Tool | Description |
|---|---|
| `/calc <expr>` | SIMD expression evaluator |
| `/shell <cmd>` | Guarded shell execution (safety-scanned) |
| `/http <url>` | HTTP GET via curl |
| `/read <path>` | Read file contents |
| `/write <path> <content>` | Write file |
| `/ls [path]` | Directory listing |
| `/grep <pattern> [path]` | Search files |
| `/git <cmd>` | Git operations |
| `/json <action> <input>` | JSON keys/get/pretty |
| `/memory <action> [key] [val]` | In-memory key-value store |
| `/time` | Current local time |
| `/cpu` | CPU info, memory, load |
| `/tokens <text>` | Byte/word/token count |
| `/bench <target>` | Benchmark (safety, router, recall, vault, search) |
| `/weather <city>` | Weather lookup |
| `/translate <lang> <text>` | Translation |
| `/define <word>` | Dictionary lookup |
| `/summarize <text>` | Text summarization |
| `/remind <time> <msg>` | Reminder |
| `/teleport <query>` | Vault search + decrypt |

## Kernel Infrastructure

59 Ea SIMD kernel source files compiled by `build.rs` into shared objects
(ARM NEON + dotprod on aarch64, SSE/AVX2 on x86_64).

Kernels are embedded in the binary via `include_bytes!` and extracted to
`~/.olorin/lib/{version}/` on first run. Version is a content hash.

Key kernels:
- `q4k_dot_8x8_arm.ea` — Q4K 8-row tiled GEMV with vdot_i32
- `q4k_dot_8x8_dual_arm.ea` — Fused gate+up dual GEMV (shared Q8K input)
- `q4k_dot_8x8_gemm_arm.ea` — Q4K 8x8 GEMM for batched prefill
- `q6k_gemm_arm.ea` — Q6K GEMM with vdot_lane_i32
- `bf16_matvec_arm.ea` — BF16 dot with 4-token register tile
- `fused_safety.ea` — Single-pass injection + leak detection
- `chacha20_search_v2.ea` — Decrypt-and-search in SIMD registers

## Building

Ea compiler required in PATH:

```bash
PATH="/path/to/eacompute/target/release:$PATH" cargo build --release
```

### Cross-compile for Raspberry Pi

```bash
PATH="/path/to/eacompute/target/release:$PATH" \
RUSTFLAGS="-C link-args=-Wl,--wrap=pidfd_spawnp -C link-args=-Wl,--wrap=pidfd_getpid" \
cargo build --release --target aarch64-unknown-linux-gnu
```

### Runtime layout

```
~/.olorin/
  lib/{version}/  # Extracted SIMD kernels (.so)
  vault/default/  # Encrypted conversations
  models/         # GGUF model files
```

### Model setup

```bash
# Gemma 4 E2B IT Q4K (recommended)
cp gemma-4-e2b-it-Q4_K_M.gguf ~/.olorin/models/

./olorin --model ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf --serve
```

Without a model, Olorin runs tools and slash-commands. Set `ANTHROPIC_API_KEY`
for cloud inference as fallback.

## Project Stats

| Metric | Value |
|---|---|
| Rust source | 14,347 lines |
| Ea kernel source | 11,000 lines (59 files) |
| Test lines | 5,552 (33 files, 213 tests) |
| Dependencies | 2 (libc, libloading) |
| Release binary (ARM) | 3.4 MB (all kernels embedded) |
| Max file size | 500 lines (enforced) |

# Olorin

> The Wakeful Mind in Ea

A single-binary AI agent with two dependencies. Local BitNet SIMD inference,
encrypted conversation vault, 19 tools, Web UI and WhatsApp bridge.

```
./olorin --serve                    # Web UI on port 8080
./olorin --interactive              # Terminal REPL
./olorin --serve --whatsapp         # Web + WhatsApp bridge
./olorin --model bitnet --serve     # Explicit BitNet model
```

## The Olorin Pipe

Every message — REPL, Web UI, WhatsApp — follows the same path.
No sidechannels. No exceptions.

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
        5. Inference -----------> BitNet local or Anthropic cloud
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
olorin1/
  src/
    core/           The Brain
      router.rs       The Olorin Pipe — sole entry/exit point
      safety.rs       Fused SIMD safety scan
      shell_guard.rs  Shell command classifier
      anthropic.rs    Cloud fallback (curl subprocess)
      dispatch.rs     Command routing + intent classification
      handlers.rs     LLM turn handling + output guard
      tool_parse.rs   Streaming tool_call detector
      llm.rs          Message types, ChatML formatting
      session.rs      Session state

    inference/      The Engine
      generate.rs     Public API — prompt in, text out
      engine.rs       Model struct + GGUF weight loading
      forward.rs      BitNet forward pass
      forward_llama.rs  Llama forward pass
      cache.rs        TurboQuant Q4 KV-cache
      matmul.rs       I2S SIMD matmul
      matmul_q4k.rs   Q4K matmul
      matmul_q6k.rs   Q6K matmul
      gguf.rs         GGUF format parser
      tokenizer.rs    BPE tokenizer

    storage/        The Vault
      vault.rs        Encrypted append-only storage
      crypto.rs       ChaCha20 encrypt/decrypt
      search.rs       FusedSearcher (zero-exposure)
      secure.rs       SecureBuffer (mlock + SIMD zeroize)
      json.rs         Recursive descent JSON parser
      key.rs          Key derivation + xxHash64

    interface/      The Gates (dumb — calls router only)
      terminal.rs     REPL (stdin/stdout)
      server.rs       Web UI + WhatsApp (std::net, epoll)
      exec.rs         Process spawning (fork/exec)

    kernels/        The Arsenal
      ffi.rs          KernelTable + core/safety/storage wrappers
      ffi_inference.rs  Inference + KV-cache wrappers

    tools/          19 built-in tools
    recall.rs       VectorStore (JL-projected embeddings)

  kernels/          61 Ea SIMD kernel source files (flat)
  bridge/           Go WhatsApp bridge (whatsmeow)
  web/chat.html     Chat UI (embedded in release binary)
  tests/            14 test files, 101 tests
```

## The Vault

Every conversation is encrypted at rest using ChaCha20. No plaintext
ever lingers in memory.

```
Write:  message --> ChaCha20 encrypt --> vault.bin (append-only)
                                     --> byte histogram --> index

Search: query --> histogram --> cosine similarity vs index --> ranked blocks
             --> FusedSearcher: decrypt+search in SIMD registers
             --> only matched context lines returned
             --> plaintext never exists as contiguous buffer

Read:   block --> xxHash64 verify --> ChaCha20 decrypt (explicit /teleport only)
```

The FusedSearcher (`chacha20_search_v2` kernel) decrypts in SIMD registers,
searches in-register, zeroes the sliding window, and returns only matched
context lines. ~95% of block content never exists as plaintext.

## Security

- **Vault key**: XOR-obfuscated seed in .rodata, derived at runtime via hardware ID (`/etc/machine-id`)
- **SecureBuffer**: `mlock` + SIMD-zeroed on Drop for all sensitive data (recall scratch, key buffers)
- **Token wipe**: inference zeroes all token buffers after generation
- **All input untrusted**: every channel passes full safety pipeline (fused SIMD scan + leak detection + shell guard)
- **Recall isolation**: vault search only sees post-safety sanitized input

## Tools

19 built-in tools, routed by SIMD command parser:

| Tool | Description |
|---|---|
| `/calc <expr>` | SIMD expression evaluator (fixed-point, replaces python3) |
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
| `/bench <target>` | Benchmark (safety, router, recall, vault, search, fused) |
| `/weather <city>` | Weather lookup |
| `/translate <lang> <text>` | Translation |
| `/define <word>` | Dictionary lookup |
| `/summarize <text>` | Text summarization |
| `/remind <time> <msg>` | Reminder |

## Kernel Infrastructure

61 Ea SIMD kernel source files compiled by `build.rs` into ~39 shared objects
(architecture-dependent: AVX2 on x86, NEON on ARM).

`build.rs` auto-generates a `KernelId` enum with one variant per kernel.
`KernelTable` is an array indexed by `KernelId as usize` — no HashMap,
no string lookup. Loaded once into `OnceLock` at startup.

Kernels are embedded in the binary via `include_bytes!` and extracted to
`~/.olorin/lib/{version}/` on first run. Version is a content hash —
old kernels never shadow new ones.

## Performance

Measured on x86-64 (AMD Ryzen 7 1700, AVX2).

| Operation | Metric |
|---|---|
| Safety scan (fused SIMD) | < 1 us per message |
| Command routing (SIMD hash) | < 500 ns |
| Vault write (encrypt + index) | ~15 us per 4KB block |
| Vault search (histogram cosine) | ~2 us per entry |
| BitNet inference (2B, I2S) | ~6.4 tok/s decode, 48 tok/s prefill |
| Compile time | ~5 seconds |
| Startup to ready | < 50 ms (no model) |

## Building

Ea compiler required in PATH:

```bash
PATH="/path/to/eacompute/target/release:$PATH" cargo build --release
```

The release binary includes all 39 compiled SIMD kernels embedded.
No prebuilt kernels, no runtime downloads.

### Runtime layout

```
~/.olorin/
  lib/{version}/  # Extracted SIMD kernels (.so), content-hash versioned
  vault/default/  # Encrypted conversations (vault.bin)
  models/         # GGUF model files
```

### Model setup

```bash
# BitNet b1.58 2B (recommended — smallest, fastest)
cp ggml-model-i2_s.gguf ~/.olorin/models/

# Or Llama 3.2 3B Q4K
cp Llama-3.2-3B-Instruct-Q4_K_M.gguf ~/.olorin/models/

# Or specify any path
./olorin --model /path/to/model.gguf --interactive
```

Without a model, Olorin runs tools and slash-commands. Set `ANTHROPIC_API_KEY`
for cloud inference as fallback.

## Project Stats

| Metric | Value |
|---|---|
| Source lines | 10,711 |
| Test lines | 924 |
| Dependencies | 2 (libc, libloading) |
| Ea kernel sources | 61 files |
| Compiled kernels (x86) | 39 shared objects |
| Tests | 101 |
| Release binary | 1.5 MB (incl. 39 embedded kernels) |
| Max file size | 485 lines (500 limit enforced) |

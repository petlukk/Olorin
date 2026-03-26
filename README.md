# Olorin

> The Wakeful Mind in Ea

A unified AI agent in a single 388KB binary. Local-first inference via BitNet SIMD,
encrypted conversation vault, tool-use, and seamless session handoff between Web UI
and WhatsApp.

```
./olorin --serve                    # Web UI on port 8080
./olorin --interactive              # Terminal REPL
./olorin --serve --whatsapp         # Web + WhatsApp bridge
./olorin --model bitnet --serve     # Explicit BitNet model
```

## Architecture

```
                        ┌──────────────────────────────────┐
                        │          olorin-cli (bin)         │
                        │   args, model resolution, REPL   │
                        └──────────┬───────────────────────┘
                                   │
                        ┌──────────▼───────────────────────┐
                        │         olorin-core (lib)         │
                        │                                   │
                        │  ┌─────────┐  ┌───────────────┐  │
                        │  │  Agent  │  │  Safety Layer  │  │
                        │  │ Router  │  │ fused_safety   │  │
                        │  │         │  │ leak_scanner   │  │
                        │  │ /help   │  │ shell_guard    │  │
                        │  │ /calc   │  │ sanitizer      │  │
                        │  │ /shell  │  └───────────────┘  │
                        │  │ /http   │                      │
                        │  │ /teleport  ┌───────────────┐  │
                        │  └─────────┘  │    Vault       │  │
                        │               │ ChaCha20 enc   │  │
                        │  ┌─────────┐  │ histogram idx  │  │
                        │  │ Recall  │  │ xxHash64 cksum │  │
                        │  │ cosine  │  └───────────────┘  │
                        │  │ +recency│                      │
                        │  └─────────┘  ┌───────────────┐  │
                        │               │  Channels      │  │
                        │               │ web (HTTP/SSE) │  │
                        │               │ whatsapp (Go)  │  │
                        │               └───────────────┘  │
                        └──────────┬───────────────────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                     │
    ┌─────────▼──────┐  ┌─────────▼──────┐  ┌──────────▼─────┐
    │ cougar-engine  │  │   eachacha     │  │     eakv       │
    │ BitNet/Llama   │  │ ChaCha20 SIMD  │  │ Q4 KV cache    │
    │ GGUF loader    │  │ searchable enc │  │ compression    │
    │ tokenizer      │  └────────────────┘  └────────────────┘
    │ SIMD matmul    │
    └────────────────┘           ┌────────────────┐
                                 │    eastat      │
                                 │ CSV/log stats  │
                                 │ SIMD parsing   │
                                 └────────────────┘

    ───────────────── 39 SIMD kernels (Ea) ─────────────────
    Compiled to .so, embedded via include_bytes!, extracted
    to ~/.olorin/lib/ at first run. Zero runtime deps.
```

## The Ea Stack

Six Rust crates in a Cargo workspace, backed by 49 Ea kernel source files
(39 compiled for x86, ARM sources ready for cross-compilation).

| Crate | Role | Kernels |
|---|---|---|
| **olorin-cli** | Binary entry point, arg parsing, REPL | -- |
| **olorin-core** | Agent router, safety, vault, recall, tools, channels | 8 (byte_classifier, command_router, fused_safety, json_scanner, leak_scanner, sanitizer, search, search_avx512) |
| **cougar-engine** | BitNet/Llama inference, GGUF, tokenizer, SIMD matmul | 11 (bitnet_*, q4k_*, q6k_*, rope) |
| **eachacha** | ChaCha20 encryption with SIMD search | 4 (chacha20, chacha20_fused, chacha20_search, chacha20_search_v2) |
| **eakv** | Q4 KV cache quantization and compression | 12 (quantize, dequantize, fused_attention, fused_k_score*, fused_v_sum*, validate) |
| **eastat** | SIMD CSV/log statistics | 4 (csv_layout, csv_parse, csv_scan, csv_stats) |

All kernels are written in [Ea](https://github.com/peteole/ea-compiler), a SIMD-first
language that compiles to native AVX2/AVX-512/NEON shared objects.

## The Vault

Conversations are encrypted at rest in `~/.olorin/vault/` using ChaCha20.
Each conversation is an append-only sequence of 4KB blocks.

```
Write path:
  message  -->  ChaCha20 encrypt  -->  4KB block  -->  vault file
                                   -->  histogram  -->  index entry
                                   -->  xxHash64   -->  integrity check

Search path (no decryption):
  query  -->  byte histogram  -->  cosine similarity vs index  -->  ranked results

Read path:
  block  -->  xxHash64 verify  -->  ChaCha20 decrypt  -->  plaintext
```

The histogram index lets you search across encrypted conversations without ever
decrypting them. Each index entry stores a 256-byte frequency vector computed
from the plaintext before encryption.

## /teleport

Seamless session handoff between Web UI and WhatsApp:

```
[Web UI]  /teleport whatsapp
          │
          ├── saves SessionToken to ~/.olorin/session.json
          │   (vault_id, seq_len, context_window_start, model, TTL=24h)
          │
          └── WhatsApp bridge picks up token, resumes context
              from the vault, continues conversation
```

The session token includes enough state to reconstruct the conversation window
without re-sending the full history. Tokens expire after 24 hours.

## Tools

The agent can call 20 built-in tools, routed by a SIMD command parser:

| Tool | Description |
|---|---|
| `/calc <expr>` | Arithmetic evaluation |
| `/shell <cmd>` | Guarded shell execution (safety-scanned) |
| `/http <url>` | HTTP GET with response summary |
| `/read <path>` | Read file contents |
| `/write <path>` | Write file |
| `/ls <path>` | Directory listing |
| `/grep <pattern>` | Search files |
| `/git <cmd>` | Git operations |
| `/json <query>` | JSON extraction |
| `/memory <note>` | Persistent memory store |
| `/time` | Current time |
| `/cpu` | CPU info and load |
| `/tokens <text>` | Token count |
| `/bench <expr>` | Benchmark a tool call |
| `/teleport <target>` | Session handoff |

Shell commands pass through a multi-layer safety pipeline before execution:
`fused_safety` (SIMD pattern scan) -> `leak_scanner` (secrets detection) ->
`shell_guard` (allowlist/blocklist) -> `sanitizer` (output scrubbing).

## Performance

Measured on x86-64 (AMD Ryzen, AVX2). The hotpath -- safety scan, command routing,
and recall lookup -- fits in ~30KB of L1 instruction cache.

| Operation | Metric |
|---|---|
| Safety scan (fused_safety kernel) | < 1 us for typical message |
| Command routing (SIMD) | < 500 ns per command |
| Vault write (encrypt + index) | ~15 us per 4KB block |
| Vault search (histogram cosine) | ~2 us per entry |
| BitNet inference (2B params) | ~7 tok/s on x86 |
| Binary size (release, LTO) | 388 KB |
| Startup to ready | < 50 ms (no model) |

## Building

```bash
cargo build --release
```

The release binary lands at `target/release/olorin-cli` (388KB with LTO).

Prebuilt x86 kernels are included in `kernels/prebuilt/x86/` -- no Ea compiler
needed for a standard build. To recompile kernels from source:

```bash
# Requires: ea-compiler (pip install ea-compiler)
ea compile kernels/olorin/*.ea -o kernels/prebuilt/x86/
ea compile kernels/cougar/*.ea -o kernels/prebuilt/x86/
ea compile kernels/eachacha/*.ea -o kernels/prebuilt/x86/
ea compile kernels/eakv/*.ea -o kernels/prebuilt/x86/
ea compile kernels/eastat/*.ea -o kernels/prebuilt/x86/
```

### Runtime layout

```
~/.olorin/
  lib/           # extracted SIMD kernels (.so)
  vault/         # encrypted conversations
  models/        # GGUF model files
  session.json   # /teleport session token
```

### Model setup

Place a GGUF model in `~/.olorin/models/`:

```bash
# BitNet 2B (recommended)
cp ggml-model-i2_s.gguf ~/.olorin/models/

# Or Llama 3.2 3B Q4
cp Llama-3.2-3B-Instruct-Q4_K_M.gguf ~/.olorin/models/

# Or specify any path
./olorin --model /path/to/model.gguf --interactive
```

## Project Stats

| Metric | Value |
|---|---|
| Rust source lines | ~19,000 |
| Ea kernel sources | 49 files |
| Compiled kernels (x86) | 23 shared objects |
| Workspace crates | 6 |
| Tests | 360 |
| Commits | 30 |
| Release binary | 388 KB |

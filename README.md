# Olorin

> Deterministic SIMD analyst with LLM narration

Olorin is built around one principle: **SIMD kernels and tools do the real
work; the language model is a presentation layer, not an answer engine.**
That makes it categorically different from chat-first agents — Olorin's
job is to give you reproducible analysis, then let the model phrase it
in plain English.

## Why Olorin?

- **Local** — Your data never leaves the machine. ChaCha20-encrypted at rest.
- **Deterministic** — Same input produces the same kernel output, every
  time. The LLM is only invoked to phrase what the kernels already
  computed; it doesn't invent facts.
- **Fast** — SIMD kernels summarize MB-scale data in milliseconds. On
  a 100K-row transactions CSV: **eacrunch in 50 ms vs pandas in 580 ms
  (11× faster end-to-end)**, using 4× less RAM. See
  [`benchmarks/results.md`](benchmarks/results.md) for the full numbers.
- **Honest scope** — Single binary, two dependencies (libc + libloading),
  3.8 MB release on ARM. Runs on a Raspberry Pi 5.

## Try it

```
./olorin                                              # interactive REPL
echo '/rune eajson ~/access.log.jsonl' | ./olorin     # one-shot summarization
./olorin --serve                                      # HTTP + web UI on :8080
./olorin --strict                                     # LLM disabled (~25ms startup)
```

The `/rune` commands ([Runes](#runes--simd-tool-calls)) are where Olorin's
SIMD-first architecture shines. They run a kernel pass over a file
(milliseconds for MBs of data) and the model narrates the result in 1-2
sentences. No Python, no pandas, no cloud.

`--strict` disables the LLM entirely: no model load, no narration, only
deterministic dispatch (slash commands, intent router, kernels, runes,
recall). Useful for fast CLI use (`./olorin --strict` starts in ~25ms
vs ~25s with the model) and for security-conscious deployments that
need a categorical "this binary will never call an LLM" guarantee.

`--audit <path>` writes a JSON Lines log of every dispatch turn:

```
{"ts_ms":1778136026682,"turn":1,"phase":"input","input_len":5}
{"ts_ms":1778136026682,"turn":1,"phase":"command","wall_us":6}
{"ts_ms":1778136026682,"turn":2,"phase":"input","input_len":28}
{"ts_ms":1778136026682,"turn":2,"phase":"rune_with_narration","wall_us":107,"narration":true}
{"ts_ms":1778136026682,"turn":3,"phase":"input","input_len":11}
{"ts_ms":1778136026682,"turn":3,"phase":"strict_refused","wall_us":8}
```

Two events per turn (input received + dispatch result with phase + timing).
Captures only metadata — input lengths, phase names, microsecond timing —
not the input text or rune output content. The log itself never leaks
the data it's meant to protect. Combine with `--strict` for a
provable "this binary did/didn't invoke an LLM" record.

## The Olorin Pipe

LLM is the last step, not the first. Every message walks this dispatch
order; the model only fires if no deterministic path matched.

```
         REPL / Web UI / WhatsApp
                   |
                   v
        core::router::dispatch()
                   |
        1. Safety Scan ---------> BLOCK            \
        2. Slash Command? ------> /tools direct     |  deterministic
        3. Intent Router? ------> kernel match      |  paths first
        4. Recall --------------> session + vault   |
           (sanitized input only)                  /
        5. LLM (last resort) ---> Gemma 4 local or Anthropic cloud
        6. Output Guard --------> truncate/block
                   |
                   v
        storage::vault::append()  (ChaCha20 encrypted)
                   |
                   v
              Response
```

Steps 1–4 are pure SIMD kernels and tool dispatch — no LLM involvement.
Step 5 only fires when nothing else matched; most rune calls and tool
invocations never touch the model.

## Runes — SIMD tool calls

Runes let Olorin reason over data that is bigger than the model's context
window. Gemma 4 2B has ~128K tokens of context; a modest bank statement
can be 50 MB of text. A kernel summarizes the file in sub-second time,
then the model narrates the summary in one or two plain-English sentences.

Each rune is one SIMD-first kernel + a thin Rust orchestrator. Output
is wrapped in `<rune_output untrusted="true">...</rune_output>` and
runs through the inbound safety scan before reaching the LLM turn —
file-derived bytes are always treated as data, never instructions.

### eacrunch — CSV summarizer

```
/rune eacrunch ~/Downloads/statement.csv
```

```
rows: 1247
columns: 4
date (text): 340 unique; top values: 2024-01-15, 2024-02-10, 2024-03-05
category (text): 8 unique; top values: groceries, food, rent
amount (number): count=1247, mean=46.70, min=1.00, max=1850.00, sum=58235.00
merchant (text): 42 unique; top values: Coop, ICA, SL
```

### eajson — JSON Lines summarizer

```
/rune eajson ~/Downloads/access.log.jsonl
```

```
rows: 1000
keys: 7 (+12 high-cardinality keys suppressed)
ts (timestamp): 1000 unique of 1000; range: 2026-05-06T08:00:00Z .. 2026-05-06T08:42:13Z
status (number): count=1000, mean=232.10, min=200.00, max=503.00, sum=232100.00
method (text): 4 unique; top values: GET, POST, HEAD
src_ip (text): 47 unique; top values: 1.2.3.4, 10.0.0.5, 192.168.1.10
http.user_agent (text): 12 unique; top values: curl/7.68, Mozilla/5.0, Nikto
cached (bool): true=623, false=377
MESSAGE (text): 8 unique; top values: GET /index, POST /api/auth, GET /admin
```

eajson handles real systemd / container / web-server log shapes:
nested objects flatten to `parent.child` keys, byte-array MESSAGE fields
(systemd's binary format) decode as UTF-8, ISO-8601 timestamp fields
report a `min..max` range, and high-cardinality noise (cursors, sequence
IDs) is suppressed with a count notice. Escape sequences in strings
(`\"`, `\\`, etc.) are correctly handled by the kernel via a 5th match
character (backslash) and an odd-run filter in the orchestrator.

### eaparquet — Parquet metadata summarizer

```
/rune eaparquet ~/data/transactions.parquet
```

```
rows: 10000000
columns: 4 (across 78 row groups)
id (number): values=10000000, min=1, max=10000000, nulls=0
category (text): values=10000000, nulls=12 [byte-array column; min/max not decoded]
amount (number): values=10000000, min=0.50, max=9999.99, nulls=0
is_recurring (bool): values=10000000, nulls=0
```

Reads only the file footer — Parquet writers pre-compute per-column
min/max/null_count at write time and store them in the metadata. The
rune walks the footer (Thrift compact decoder, scalar — no SIMD path
exists for variable-length encodings) and aggregates per-column
statistics across row groups via the `f64_stats` SIMD kernel. For a
file with N row groups and C columns, that's `3*C` kernel calls
each doing an N-element f64x2 reduction — real SIMD work that scales
with file size.

**Limit**: column-data SIMD decoding (PLAIN/RLE/dictionary encoding +
snappy/gzip/zstd decompression) is out of scope for v1. Statistics
must be present in the file metadata (most modern writers include
them by default). Primitive types only: BOOLEAN/INT32/INT64/FLOAT/DOUBLE
get min/max; BYTE_ARRAY (strings) and INT96 (legacy timestamp) are
reported by type but their stats are left absent.

### Real public data to try it on

- **Synthetic fixtures** — `tests/fixtures/runes/{tiny.csv,tiny.jsonl}` in this repo. Small, good for a smoke test.
- **systemd journal** — `journalctl -o json -n 1000 > /tmp/log.jsonl` then `/rune eajson /tmp/log.jsonl` — real local data, no setup.
- **US Bank Transaction Categories v2** — 68K real transaction descriptions, MIT-licensed (CSV):
  https://huggingface.co/datasets/DoDataThings/us-bank-transaction-categories-v2
- **NYC TLC Yellow Taxi trip records** — millions of rows per month, permissive (CSV):
  https://catalog.data.gov/dataset/2023-yellow-taxi-trip-data

### Limits

- Max input: 4 GB (2 GB for the `csv_scan` / `jsonl_struct` kernels in this version — bumping to i64 is a planned follow-up).
- Path allowlist: `~` and `/tmp` only. Symlinks escaping the allowlist are rejected at open time.
- Output cap: 32 KB summary (truncated with a `[...truncated N bytes]` marker at a UTF-8-safe boundary).
- eacrunch: unquoted CSV only; CRLF line endings tolerated (trailing `\r` trimmed per field).
- eajson: top-level scalars only — nested objects flatten one level deep (`http.status`); deeper nesting and arrays-of-objects are skipped. Mixed-type keys (number on one line, string on another) collapse to `(mixed)` with no stats. Text top-N capped at 10K cardinality.
- eaparquet: metadata-only — column data is never decoded. Statistics must be present in the file footer (most modern writers include them). BYTE_ARRAY (string) and INT96 (legacy timestamp) min/max are not decoded. Flat schemas only; nested groups (LIST/MAP/STRUCT children) are skipped from the column list.
- Narration: the model gets a token budget of ~1248 prompt + 768 decode. Outputs over that skip narration with a clear notice — the kernel summary is shown either way.

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

  kernels/          64 Ea SIMD kernel source files (flat) — 42 logical kernels with ARM variants
  web/chat.html     Chat UI (Catppuccin themed, embedded in binary)
  tests/            60 test files, 318 tests
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

### Runes vs pandas (x86 WSL, transactions CSV)

| Rows | File | eacrunch (`--strict`) | pandas | Speedup |
|------|------|----------------------:|-------:|--------:|
| 10,000 | 341 KB | **0.02 s** | 0.56 s | **28×** |
| 100,000 | 3.4 MB | **0.05 s** | 0.58 s | **11×** |
| 1,000,000 | 34 MB | **0.45 s** | 0.87 s | **1.9×** |

Cold-start one-shot wall-clock — pandas's ~500 ms Python+import startup
dominates at small sizes; at 1M rows pandas finally amortizes its
overhead and the gap narrows. eacrunch uses 1-4× less RAM at small
sizes (no full DataFrame materialization). Reproduce with
`bash benchmarks/bench.sh`; full commentary + caveats in
[`benchmarks/results.md`](benchmarks/results.md).

### Gemma 4 inference

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

64 Ea SIMD kernel source files (42 logical kernels with ARM variants)
compiled by `build.rs` into shared objects (ARM NEON + dotprod on
aarch64, SSE/AVX2 on x86_64).

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
- `csv_scan.ea` — CSV structural scan (commas + newlines) for runes
- `jsonl_struct.ea` — JSON Lines structural scan (5-bit mask: newlines/quotes/colons/commas/backslashes)

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
| Rust source | 17,660 lines |
| Ea kernel source | 12,568 lines (64 files, 42 logical kernels) |
| Test lines | 9,466 (60 files, 318 tests) |
| Dependencies | 2 (libc, libloading) |
| Release binary (ARM) | 3.8 MB (all kernels embedded) |
| Max file size | 500 lines (enforced) |

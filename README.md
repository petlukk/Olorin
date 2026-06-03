# Olorin

[![CI](https://github.com/petlukk/Olorin/actions/workflows/ci.yml/badge.svg)](https://github.com/petlukk/Olorin/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Deterministic SIMD analyst with LLM narration

Olorin is built around one principle: **SIMD kernels and tools do the real
work; the language model is a presentation layer, not an answer engine.**
That makes it categorically different from chat-first agents — Olorin's job
is to give you reproducible analysis, then let the model phrase it in plain
English.

## Why Olorin?

- **Local** — Your data never leaves the machine. ChaCha20-Poly1305 AEAD
  encrypted at rest.
- **Deterministic** — Same input produces the same kernel output, every time.
  The LLM is invoked only to phrase what the kernels already computed; it
  doesn't invent facts.
- **Fast** — SIMD kernels summarize MB-scale data in milliseconds. eacrunch
  is 11× faster end-to-end than pandas on a 100K-row CSV; eatime's
  `timestamp_scan` kernel scans at 6.34 GB/s on x86. See [`benchmarks/results.md`](benchmarks/results.md).
- **Honest scope** — Single binary, two dependencies (libc + libloading),
  ~2 MB release on x86_64 / ~4.3 MB on ARM. Runs on a Raspberry Pi 5.

Current version: **v2.0.7** — Windows x86_64 returns to the release
matrix, cross-compiled from Ubuntu via MinGW-w64; `olorin.exe` is
published alongside the Linux x86_64 and aarch64 binaries on every
release. The vault key remains Argon2id-derived from a user passphrase
+ per-vault salt (arc #3, shipped in v2.0.0). The rune family, the
`RuneOutput v1` schema, and the `--json` chaining contract remain
stable. **Upgrading from v1.x or pre-v2.0.0:** v3 vaults are not
read-compatible with earlier formats — delete `~/.olorin/vault/default/`
(or `%USERPROFILE%\.olorin\vault\default\` on Windows) and you'll be
prompted for a fresh passphrase on next run. See
[`CHANGELOG.md`](CHANGELOG.md) for the full history.

## Try it

```
./olorin                                              # interactive REPL
echo '/rune eajson ~/access.log.jsonl' | ./olorin     # one-shot summarization
./olorin --serve                                      # HTTP + web UI on :8080
./olorin --strict                                     # LLM disabled (~25ms startup)
```

`--strict` disables the LLM entirely: no model load, no narration, only
deterministic dispatch. Starts in ~25 ms vs ~25 s with the model. Useful for
fast CLI use and for security-conscious deployments that need a categorical
"this binary will never call an LLM" guarantee. `--audit <path>` writes a
JSON Lines log of every dispatch turn — metadata only (phase, timing, input
length), never content.

## The Olorin Pipe

LLM is the **last** step, not the first. Every message walks this dispatch
order; the model only fires if no deterministic path matched.

```
         REPL / Web UI / WhatsApp
                   |
                   v
        core::router::dispatch()
                   |
        1. Safety Scan ---------> BLOCK            \
        2. Slash Command? ------> /tools or /rune     |  deterministic
        3. Intent Router? ------> kernel match      |  paths first
        4. Recall --------------> session + vault   |
           (sanitized input only)                  /
        5. LLM (last resort) ---> Gemma 4 local or Anthropic cloud
        6. Output Guard --------> truncate/block
                   |
                   v
        storage::vault::append()  (ChaCha20-Poly1305 AEAD)
                   |
                   v
              Response
```

Steps 1–4 are pure SIMD kernels and tool dispatch — no LLM involvement. Most
rune calls and tool invocations never touch the model.

## Runes — the differentiator

A *rune* is a SIMD-first command that turns megabyte-to-gigabyte scale raw
data into a small structured summary in sub-second time. The model never sees
the raw bytes — only the kernel's output, which it phrases in one or two
plain-English sentences.

This is what separates Olorin from chat-first agents. A standard LLM agent
hits its context window the moment you ask it about a real log file. Olorin
runs a kernel pass over the file (milliseconds for MBs of data) and gives
the model a compressed structural summary it can actually reason about. No
Python, no pandas, no cloud.

```
/rune eatime ~/gharchive.log
```

`eatime` bucketizes every ISO-8601 / RFC3339 timestamp in a file by hour-of-day
(or weekday) in one SIMD pass. It matches `YYYY-MM-DDTHH:MM:SS` **anywhere** in a
line — so it works on JSON logs, container/k8s output, and `journalctl -o
short-iso`, but **not** space-separated syslog or Apache timestamps (no `T`
anchor). Grab a real input — a few hours of public GitHub events:

```bash
curl -s https://data.gharchive.org/2015-01-01-{12,16,20}.json.gz | gunzip > ~/gharchive.log
grep -om1 '"created_at":"[^"]*"' ~/gharchive.log     # "created_at":"2015-01-01T12:00:01Z"
```

```
> /rune eatime ~/gharchive.log          # Raspberry Pi 5 Model B, aarch64
bytes:       72.00 MB
timestamps:  68140
scan:        27 ms

hour-of-day:
  11:00          681  ( 1.00%)
  12:00        13750  (20.18%)   <- 12:00 archive
  13:00          669  ( 0.98%)
  ...
  16:00        20270  (29.75%)   <- 16:00 archive
  ...
  20:00        22179  (32.55%)   <- 20:00 archive
  21:00          645  ( 0.95%)
  ...
peak: 20:00 (22179 timestamps)
```

The three spikes are the hours pulled from the archive (the events' own
`created_at`); the smaller background counts are timestamps embedded in the event
payloads — repo, comment, and actor times spanning the rest of the day. A nice
correctness signal that it's bucketing real data, not file order.

**72 MB scanned in 27 ms on a Pi 5 (~20 ms warm).** End-to-end throughput tracks
timestamp *density*: this log carries ~1 timestamp per KB so the scan dominates
(~2.7 GB/s); on a dense every-line log (~1 per 70 B) the per-match bookkeeping
takes over (~1.4 GB/s). The bare `timestamp_scan` **kernel**, benchmarked in
isolation on a dense fixture, hits **6.34 GB/s on Ryzen 7700X (SSE2)** and **1.80
GB/s on Pi 5 (NEON)** — see
[`benchmarks/timestamp_scan_bench.c`](benchmarks/timestamp_scan_bench.c).

Add `--json` for the stable schema that pipes into other runes (e.g. `eadiff`):

```bash
$ /rune eatime --json ~/gharchive.log
{"rune":"eatime","source":{"bytes":75501657,"format":"iso8601"},
 "totals":{"rows":68140,"scan_us":24826},
 "categories":[{"name":"00:00","count":528}, … {"name":"20:00","count":22179}]}
```

Six v1 runes:

- **`eacrunch`** — CSV summarizer (rows, columns, per-column stats + top-N)
- **`eajson`** — JSON Lines summarizer (handles systemd / container / web-server shapes)
- **`eaparquet`** — Parquet metadata (per-column min/max/null_count from the footer)
- **`ealog`** — log severity scanner (DEBUG/INFO/WARN/ERROR/FATAL + sample lines)
- **`eatime`** — ISO-8601 timestamp histogram (hour-of-day or weekday buckets)
- **`eadiff`** — structural delta between any two `--json` rune outputs

Each rune also accepts `--json` for piping into another rune. See
**[`docs/runes.md`](docs/runes.md)** for the full catalog with per-rune
samples, limits, and the chaining contract.

## The Vault

Every conversation is encrypted at rest using ChaCha20-Poly1305 AEAD.

```
Write:  message --> ChaCha20-Poly1305 seal --> vault.bin (append-only)
                                           --> byte histogram --> index

Search: query --> histogram --> cosine similarity vs index --> ranked blocks
             --> Poly1305 verify each candidate block
             --> FusedSearcher: decrypt+search in SIMD registers
             --> only matched context lines returned

Read:   block --> Poly1305 verify --> ChaCha20 decrypt
```

The FusedSearcher (`chacha20_search_v2` kernel) decrypts in SIMD registers,
searches in-register, and returns only matched context lines. ~95% of block
content never exists as plaintext.

## Security & threat model

**What Olorin protects against:**

- **File-system theft of the vault — including hardware theft.** Every
  block is ChaCha20-Poly1305 AEAD with the tag verified before decrypt.
  The vault key is Argon2id-derived (64 MiB, t=3, p=1) from a user
  passphrase + a per-vault salt; the salt is stored next to the vault
  but is useless on its own. An attacker who exfiltrates
  `~/.olorin/vault/` — or the entire laptop — cannot read or modify any
  conversation without also knowing the passphrase. Tampered bytes are
  rejected at load time, never silently surfaced as garbage.
- **Plaintext lingering in process memory.** The FusedSearcher decrypts and
  searches inside SIMD registers; ~95% of block content never exists as
  plaintext. SecureBuffer wraps all sensitive data with `mlock` (no swap-out)
  and SIMD-zeroes on Drop.
- **File content posing as instructions.** Rune output is wrapped in
  `<rune_output untrusted="true">`; agent read/write/grep route through a
  sensitive-subtree denylist; the shell tool classifier blocks exfil paths
  textually before any policy-mode check.
- **Basic-to-mid prompt injection.** Score-based multi-language (EN + SV)
  inbound matcher with two-form normalization (alphanumeric + alnum-only)
  catches keyword bypasses, punctuation/spacing obfuscation, and word-variant
  attacks. All input — REPL, Web UI, WhatsApp — passes the same pipeline.

**What it does *not* yet protect against:**

- **A weak or leaked passphrase.** Argon2id at 64 MiB / t=3 raises the
  per-guess cost dramatically, but a dictionary-strength passphrase is
  still recoverable by a determined attacker with the vault file. Pick a
  passphrase the way you'd pick a master password for a password manager.
  Losing the passphrase means losing the vault — there is no recovery
  path, by design.
- **Sophisticated prompt injection.** Adversarial paraphrasing and
  out-of-distribution languages can slip past keyword + score matching. A
  full-ML classifier is a future project, not on the current roadmap.
- **Code execution on the host.** Olorin is a local binary you choose to run.
  There's no sandboxing of the agent against an attacker who already has
  shell access to the host.
- **Side-channel attacks.** `poly1305_verify` uses constant-time comparison,
  but Olorin is not formally hardened against timing, cache, or
  speculative-execution side channels.

**Designed for:** a single user on their own machine, protecting conversation
history and analysis from file-system-level theft (lost laptop, leaked backup,
cloud-sync mishap). The passphrase + Argon2id flow extends this to hardware
theft as well, provided the passphrase is strong. **Not designed for:**
multi-user systems, hostile-network adversaries with host access, or scenarios
where the passphrase itself is compromised.

## Interfaces

Three ways to talk to the same `DispatchContext`. All three walk the same
Olorin Pipe — same SIMD kernels, same vault, same audit log.

- **Terminal REPL** (default) — `./olorin` or `./olorin --strict`
- **Web UI** — `./olorin --serve`, then open `http://127.0.0.1:8080`.
  Catppuccin-themed chat UI embedded in the binary at compile time; streams
  responses token-by-token via TCP_NODELAY SSE.
- **WhatsApp** — `/teleport` from REPL or web UI launches a Go bridge as a
  subprocess; scan the QR with WhatsApp on your phone, then any message to
  the linked number gets dispatched through the same Pipe.

## Install

Prebuilt binaries are published per release for **Linux x86_64**,
**Linux aarch64** (Raspberry Pi 5), and **Windows x86_64**.

**Linux / WSL** — open a terminal and run:

```bash
curl -fsSL https://raw.githubusercontent.com/petlukk/Olorin/main/scripts/install.sh | sh
```

**Windows** — open a PowerShell window (`Win+X` → "Windows PowerShell"
or "Terminal") and run:

```powershell
iwr -useb https://raw.githubusercontent.com/petlukk/Olorin/main/scripts/install.ps1 | iex
```

> Both commands download a small installer script and execute it. The
> URLs intentionally serve plain text — clicking them in a browser
> just shows the source. Run them in a shell.

The installer downloads the latest release binary, verifies its SHA256
against the published `SHA256SUMS`, optionally prompts for an
`ANTHROPIC_API_KEY` (cloud fallback when no local model is loaded),
and optionally fetches the WhatsApp `/teleport` bridge. Cloud-fallback
and bridge are both opt-in; the core binary is ~2 MB on x86 / ~4.3 MB
on ARM with all SIMD kernels embedded. Olorin reads `~/.olorin/env` at
startup, so the key written by the installer is picked up without any
shell-rc plumbing.

> **Windows note:** `wa-bridge.exe` is currently not shipped — the
> bridge depends on `mattn/go-sqlite3` (CGo), pending a swap to
> `modernc.org/sqlite` (pure Go). The `olorin.exe` core binary is
> fully functional; WhatsApp `/teleport` simply isn't available on
> Windows yet.

Drop a GGUF model into `~/.olorin/models/` to use local inference:

```bash
cp gemma-4-e2b-it-Q4_K_M.gguf ~/.olorin/models/
olorin --model ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf --serve
```

Without a model, Olorin runs tools, slash-commands, and runes
deterministically; cloud fallback fires only if `ANTHROPIC_API_KEY` is
set.

## Building from source

For contributors. Requires the [Ea compiler](https://github.com/petlukk/eacompute)
on `PATH`:

```bash
PATH="/path/to/eacompute/target/release:$PATH" cargo build --release
```

Cross-compile for Raspberry Pi:

```bash
PATH="/path/to/eacompute/target/release:$PATH" \
RUSTFLAGS="-C link-args=-Wl,--wrap=pidfd_spawnp -C link-args=-Wl,--wrap=pidfd_getpid" \
cargo build --release --target aarch64-unknown-linux-gnu
```

To build the WhatsApp bridge separately (requires Go 1.25+):

```bash
cd bridge && go build -trimpath -ldflags='-s -w' -o ../wa-bridge
```

See [`docs/architecture.md`](docs/architecture.md) for the full source
layout, kernel inventory, and runtime contracts.

## What isn't in the binary

The 2 MB / 2-dep figure isn't aspirational — it's the result of deliberately
not pulling in the things most LLM agents depend on.

| Layer | Common choice | Olorin |
|---|---|---|
| Async runtime | `tokio` | `std::net` + work-stealing pool ([`inference/threadpool.rs`](src/inference/threadpool.rs)) |
| HTTP server | `axum`, `actix`, `hyper`, `warp` | hand-rolled in [`interface/server.rs`](src/interface/server.rs) |
| SSE streaming | `axum::response::sse` | raw `text/event-stream` frames |
| Web terminal | `xterm.js` (~700 KB JS) | canvas + cell-grid emulator in [`web/chat.html`](web/chat.html) (527 lines total) |
| HTTP client | `reqwest` | `curl` subprocess for cloud fallback ([`core/anthropic.rs`](src/core/anthropic.rs)) |
| JSON | `serde` | minimal hand-rolled scanner ([`storage/json.rs`](src/storage/json.rs)) |
| Crypto | `ring`, `rustcrypto` | ChaCha20-Poly1305 + Argon2id as Ea SIMD kernels |
| Tokenizer | `tokenizers` (Hugging Face) | hand-rolled BPE ([`inference/tokenizer.rs`](src/inference/tokenizer.rs)) |
| GGUF parser | `ggml` / `llama-cpp` bindings | hand-rolled ([`inference/gguf.rs`](src/inference/gguf.rs)) |
| Inference engine | `candle`, `burn`, `llama.cpp` | hand-rolled Gemma 4 forward pass in Rust + Ea ([`inference/forward.rs`](src/inference/forward.rs)) |

The runtime dependencies are `libc` and `libloading` — the latter only used
to `dlopen` the embedded Ea SIMD kernels at startup. The release binary
contains every SIMD kernel embedded via `include_bytes!`, the web UI
embedded via `include_str!`, and the tokenizer / GGUF parser / forward pass
inline. No CDN fetches, no runtime dependency resolution, no missing-DLL
surprises.

## Project stats

| Metric | Value |
|---|---|
| Rust source | 23,786 lines (122 files) |
| Ea kernel source | 14,286 lines (71 files, 49 logical kernels) |
| Tests | 15,966 lines (98 files, 582 tests) |
| Runtime dependencies | 2 (libc, libloading) |
| Release binary | 2.0 MB on x86_64 / 4.3 MB on ARM (all kernels embedded) |
| Max file size | 500 lines for Rust + tests (no exceptions); 2 Ea kernels exceed it |

## License

MIT. See [`LICENSE`](LICENSE).

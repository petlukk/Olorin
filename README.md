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
  is 11× faster end-to-end than pandas on a 100K-row CSV; eatime scans
  timestamps at 6.34 GB/s on x86. See [`benchmarks/results.md`](benchmarks/results.md).
- **Honest scope** — Single binary, two dependencies (libc + libloading),
  3.8 MB release on ARM. Runs on a Raspberry Pi 5.

Current version: **v2.0.0** — the vault key is now Argon2id-derived from a
user passphrase + per-vault salt (arc #3 of the security follow-up queue).
The rune family, the `RuneOutput v1` schema, and the `--json` chaining
contract remain stable. **Breaking change:** v2.0 vaults are not
read-compatible with v1.2.x — delete `~/.olorin/vault/default/` and you'll
be prompted for a fresh passphrase. See [`CHANGELOG.md`](CHANGELOG.md) for
the full history.

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
/rune eatime ~/var/log/app.log
```

```
bytes:       1.2 GB
timestamps:  47123891
scan:        184 ms

hour-of-day:
  00:00     1230891  ( 2.61%)
  ...
  06:00     8421902  (17.87%)  <-- peak
  07:00     6122891  (12.99%)
  ...
  23:00     1320012  ( 2.80%)
```

One SIMD pass over 1.2 GB. 47M timestamps bucketed in 184 ms. **6.34 GB/s on
Ryzen, 1.80 GB/s on Pi 5 ARM NEON** — 14× faster than awk, 29× faster than
pandas at small sizes.

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

**Linux / macOS shell:**

```bash
curl -fsSL https://raw.githubusercontent.com/petlukk/Olorin/main/scripts/install.sh | sh
```

**Windows PowerShell:**

```powershell
iwr -useb https://raw.githubusercontent.com/petlukk/Olorin/main/scripts/install.ps1 | iex
```

The installer downloads the latest release binary, optionally prompts for
an `ANTHROPIC_API_KEY` (cloud fallback when no local model is loaded),
and optionally fetches the WhatsApp `/teleport` bridge. Cloud-fallback
and bridge are both opt-in; the core binary is ~2 MB on x86 / ~4.3 MB on
ARM with all SIMD kernels embedded. Olorin reads `~/.olorin/env` at
startup, so the key written by the installer is picked up without any
shell-rc plumbing.

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

## Project stats

| Metric | Value |
|---|---|
| Rust source | 23,722 lines |
| Ea kernel source | 14,286 lines (71 files, 49 logical kernels) |
| Test lines | 15,899 (101 files, 571 tests) |
| Dependencies | 2 (libc, libloading) |
| Release binary (ARM) | 4.3 MB (all kernels embedded) |
| Max file size | 500 lines (enforced) |

## License

MIT. See [`LICENSE`](LICENSE).

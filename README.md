# Olorin

[![CI](https://github.com/petlukk/Olorin/actions/workflows/ci.yml/badge.svg)](https://github.com/petlukk/Olorin/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)




https://github.com/user-attachments/assets/be3fc1b8-c997-476b-8d4e-356f2986e45d





> Deterministic SIMD analyst with LLM narration

Olorin is built around one principle: **SIMD kernels and tools do the real
work; the language model is a presentation layer, not an answer engine.**
That makes it categorically different from chat-first agents — Olorin's job
is to give you reproducible analysis, then let the model phrase it in plain
English. 1 GB NASA-HTTP access log (July 1995) is SIMD-scanned in 789 ms, a real traffic spike flagged on the 13th, and narrated by the local model — zero cloud, on a Raspberry Pi.

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
  ~2.4 MB release on x86_64 / ~5.2 MB on ARM. Runs on a Raspberry Pi 5.

Prebuilt for Linux x86_64, Linux aarch64 (Pi 5), and Windows x86_64. See
[`CHANGELOG.md`](CHANGELOG.md) for version history; the `RuneOutput v1`
schema and `--json` chaining contract are stable across releases.

## Try it

```
./olorin                                          # interactive REPL
./olorin rune eajson ~/access.log.jsonl           # one-shot: rune output to stdout
./olorin rune eatime --bucket series --json x.log > out.json   # clean JSON for jq/matplotlib
./olorin report app.log deploys.csv -o report.html             # self-contained HTML report
./olorin palantir --alert /var/log/app.log --daemon            # predictive log watcher (background)
./olorin --serve                                  # HTTP + web UI on :8080
./olorin --strict                                 # LLM disabled (~25ms startup)
```

`olorin rune <name> [args…]` runs a single rune non-interactively and writes
only its answer to stdout — no banner, no model load — so `--json` output
pipes straight into a file for downstream tools.

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

A *rune* is a SIMD-first command that turns megabyte-to-gigabyte raw data into
a small structured summary in sub-second time. The model never sees the raw
bytes — only the kernel's output, which it phrases in a sentence or two.

This is what separates Olorin from chat-first agents. A standard LLM agent hits
its context window the moment you point it at a real log file. Olorin runs a
kernel pass over the file (milliseconds for MBs), then hands the model a
compressed structural summary it can actually reason about. No Python, no
pandas, no cloud.

**Find the moment a log broke.** `eatime --bucket series` builds a chronological
histogram (auto-width to ~120 buckets) and flags the buckets where the event
rate broke from a robust **median/MAD** baseline — a large spike can't inflate
its own threshold and hide. Then the on-board model narrates the SIMD-detected
numbers it cannot fabricate. Here it is on a **Raspberry Pi 5** reading an
Apache access log (CLF auto-detected):

```
> /rune eatime --bucket series ~/access.log
timestamps:  1292
scan:        716 µs                             # SIMD scan, NEON
peak bucket: 2026-06-11T14:08:36 (342 timestamps)
anomalies:   2 spike(s) detected
  2026-06-11T13:08:36 count=79  (4.5× baseline 18)
  2026-06-11T14:08:36 count=342 (19.5× baseline 18)

  Significant spikes in activity were detected, with one window showing nearly
  twenty times the normal volume — during the afternoon of June 11th.
```

"Nearly twenty times" is the measured 19.5×; "afternoon of June 11th" is the
14:08 bucket. eatime auto-detects **ISO-8601** (both `T`- and space-separated —
Postgres/Python-logging/OpenStack), **classic syslog** (`Jun 14 15:16:01` —
Linux `/var/log`), **Apache/nginx CLF**, and **JSON / ndjson** (an ISO string,
or a numeric Unix epoch under `ts`/`time`/`timestamp`/`@timestamp` — Go zap,
pino — with the s/ms/µs unit inferred from magnitude; a SIMD kernel each);
`--bucket hour|weekday|series` picks the view.

**Find *why* your service died.** Point `eacorrelate` at several timestamped
logs — a deploy log, an app log, an nginx access log (its **5xx** read as an
error stream) — and it doesn't just correlate them, it assembles the answer
into one ordered **incident timeline**:

```
> olorin rune eacorrelate deploy.log app.log access.log
incident timeline (confidence 0.91):
  Deployment at 14:02
  -> app errors rise 4 minutes later (r=0.93)
  -> request traffic drops 12 minutes later (anomaly 0.91)
```

It finds the cascade's root — a discrete trigger like a deploy, or the first
break — anchors on it, and orders the fallout by lag: a co-spiking stream that
*rises* (carrying its Pearson `r`) or a stream that *falls* (a downward
anomaly). `confidence` is the weakest link. The wording stays strictly temporal
— "errors rise 4 minutes later", never "the deploy *caused* it" — a correlation
over too few buckets is rejected (the guard that killed a real NASA
`+654 h, r=1.00` artifact), and when nothing correlates, no timeline appears.
Nobody forwards a correlation matrix; everybody forwards "why did my service
die?". Built entirely on the differentially-proven correlation engine — no model
involved.

**Charts, in the terminal and the browser.** Drop a timestamped log into the web
UI — or run the rune in the REPL — and Olorin renders the rate over time as a
block-bar chart with spikes flagged, from a single SIMD-downsampled (`col_reduce`)
renderer shared by both surfaces:

```
 600                             ▇
 420                             █
     ─▁─▁──▁─▁───▁▁───▁▁───▁▁───▁█───▁─▁──▁─▁──▁─▁───▁▁───▁▁─
 240 ████████████████████████████████████████████████████████
  60 ████████████████████████████████████████████████████████
     ────────────────────────────────────────────────────────
     08:00             11:10             14:25          17:45
     median (300)
```

**Stable JSON, pipeable.** `--json` emits a single-line `RuneOutput v1` object —
the schema that chains into other runes (`eadiff`) and feeds matplotlib/jq via
the `olorin rune … > out.json` one-shot:

```bash
$ olorin rune eatime --json ~/gharchive.log
{"rune":"eatime","source":{"bytes":75501657,"format":"iso8601"},
 "totals":{"rows":68140,"scan_us":24826},
 "categories":[{"name":"00:00","count":528}, … {"name":"20:00","count":22179}]}
```

The bare `timestamp_scan` kernel hits **6.34 GB/s on Ryzen 7700X (SSE2)** and
**1.80 GB/s on Pi 5 (NEON)** in isolation — see
[`benchmarks/results.md`](benchmarks/results.md).

**Proven correct, and fuzzed.** Every rune is differentially validated against
the standard tool for its format — pandas, pyarrow, a real SQLite engine, numpy
— on real public files at scale (a 3M-row NYC-taxi dataset, a 558 MB GitHub
Archive dump, 196 MB of NASA access logs), with zero count mismatches. The rune
parsers, the encrypted vault's crash-recovery, and the tokenizer are
continuously fuzzed — mutation fuzzing plus randomized on-disk crash-state
injection — on **both x86 and Raspberry Pi NEON**, with every standing harness
paired with a negative control that proves it can actually fail. Bit-identical
output across architectures is a CI gate.

Eight runes:

- **`eacrunch`** — CSV summarizer (rows, columns, per-column stats + top-N); SQL-style `GROUP BY` (`--by`/`--agg`) and `WHERE` (`--where`) row filtering
- **`eajson`** — JSON Lines summarizer (systemd / container / web-server shapes); nested objects flatten to dotted keys up to `--depth N` (default 4)
- **`eaparquet`** — Parquet metadata (per-column min/max/null_count from the footer; INT96 timestamps decode to ISO, DECIMAL to scaled value)
- **`ealog`** — log severity scanner (DEBUG/INFO/WARN/ERROR/FATAL + sample lines)
- **`eatime`** — timestamp histogram (ISO-8601 + classic syslog + Apache/nginx CLF); hour-of-day, weekday, or chronological `series` buckets with robust spike detection + charts
- **`easql`** — SQL-dump summarizer (`pg_dump` / `mysqldump`): dialect, table count, per-table row + column counts
- **`eacorrelate`** — cross-file lag correlation → **incident timeline**: drop 2–8 timestamped files and get the ordered story ("deploy at 14:02 → errors rise +4 min → traffic drops +12 min, confidence 0.91")
- **`eadiff`** — structural delta between any two `--json` rune outputs

See **[`docs/runes.md`](docs/runes.md)** for the full catalog with per-rune
samples, limits, and the chaining contract.

**Shareable reports.** `olorin report <files…> -o out.html` runs the same
deterministic pipeline (a rune per file, `eacorrelate` across them) and writes
**one self-contained HTML file** — inline charts, zero external assets, zero
JavaScript — that opens anywhere and prints cleanly. In the web UI, a
"📄 download report" link appears under every file-drop analysis.

## Palantír — the predictive log watch

A *palantír* reads a log the way runes read a file — SIMD scan, no model in the
loop — but **continuously**, and it calls the incident *before* the error storm.
It's the first new subsystem since runes.

Most cascades have a trigger: a deploy, a restart, a config push. Palantír learns
the lag from the file's own history — the gap between past triggers and the
errors that followed — and when a fresh trigger arrives it **predicts** the
cascade with an ETA, then watches the window to **confirm** it or stand down. You
get the warning during the propagation delay, not after the pager fires:

```
⚠  PALANTÍR  14:02:11  trigger detected — cascade predicted by 14:02:31 (historical lag), watching 45s…
🔴 PALANTÍR  14:02:33  CASCADE CONFIRMED — 36 error(s), 20s after the trigger
✓  PALANTÍR  14:02:56  window clear — no cascade 45s after the trigger
```

It detects two incident shapes:

- **Triggered** — a deploy/restart line followed by the predicted error cascade
  (above). Confirmed when enough errors land inside the window; stood down with a
  `✓ window clear` when they don't.
- **Rate anomaly** — no trigger, just errors creeping up. A streaming
  **median/MAD** over per-bucket error counts catches leaks and slow degradation,
  with warmup, streak, and cooldown gating to keep false positives down.

The detector reuses the rune format engine — ISO-8601, classic syslog, Apache/
nginx CLF (5xx as an error stream), JSON string/numeric levels — so it needs no
new kernel.

**Run it.** Foreground for a quick look, or `--daemon` to detach (double-fork +
setsid; the watcher outlives the launching shell, the web-UI terminal, and the
server itself):

```bash
olorin palantir --alert /var/log/app.log --daemon     # start, detached
olorin palantir --status                               # list watchers + last alert
olorin palantir --stop --name app.log                  # stop one
```

Tune detection with `--sensitivity low|med|high`. Route alerts with repeatable
`--notify stdout|webhook:URL|exec:CMD` — POST a Slack/PagerDuty-shaped JSON, or
run any command with the alert details in its environment. State lives under
`~/.olorin/palantir/<name>.{pid,json,log}`; the `.json` snapshot is the contract
the web UI and chat read.

**From the web UI.** The 🔮 badge in the heartbeat bar shows how many watchers are
running and how many are alerting — **green** while watching, **red** during a
live incident. Click it to list each watcher with its last alert, start a new one
from the "watch a file…" field, or stop one with ✕. Web-started watchers are
read-only and refuse alert sinks (no `exec:`/`webhook:` over the wire); their path
guard permits `$HOME`, `/tmp`, and read-only `/var/log` — kept separate from the
agent's file-tool guard, so it widens nothing else.

**Chat-aware.** While an incident is live, the local model senses it: the Pipe
injects a short `<recent_observations>` block, so you can ask "what's going on?"
and get an answer grounded in the actual watcher state — and only during an
incident, so quiet runs add nothing to the prompt. Pi-NEON gated end-to-end, like
the runes.

**Proven against a service that has incidents.** The runes and the palantír are
exercised by an **incident lab** (`benchmarks/incident-lab/`): a leaderboard API
with a deploy-introduced bug whose DB pool fails first, serves cached data
through a propagation delay, then cascades into HTTP 500s — logged in all nine
rune formats. A deterministic simulator freezes five scenarios as committed
fixtures, and a CI gate asserts both that a bad deploy yields the cascade
timeline *and* that the controls — a healthy baseline and a clean deploy under
realistic diurnal, bursty traffic — raise no false alarm. Detecting an incident
is half the job; not crying wolf on a normal evening is the other half.

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

### Exposing the Web UI on a network

`--serve` binds `127.0.0.1` by default — local-only, no authentication, the
right default for a single user on their own machine. To reach the UI from
another device (a phone, a laptop on the LAN), bind a non-loopback address.
Olorin is **fail-closed** here: a non-loopback bind *requires* an auth token,
or the server refuses to start — so the tool-running endpoints are never
exposed unauthenticated by accident.

Set the token (and the bind address) in `~/.olorin/env`, which Olorin loads at
startup:

```bash
printf 'OLORIN_AUTH_TOKEN=%s\nOLORIN_BIND=0.0.0.0\n' "$(openssl rand -hex 32)" >> ~/.olorin/env
chmod 600 ~/.olorin/env
grep OLORIN_AUTH_TOKEN ~/.olorin/env   # the token you'll put in the URL
```

Then `./olorin --serve` and, from the other device, open
`http://<host-lan-ip>:8080/?token=<token>` **once**. That sets an `HttpOnly`
cookie, and every later request — page load, SSE stream, terminal WebSocket —
carries it automatically. Requests without the token get `401`; the check is
constant-time and runs before any dispatch, so it covers every endpoint.

For a throwaway session, pass them inline instead of persisting:

```bash
OLORIN_AUTH_TOKEN=$(openssl rand -hex 32) OLORIN_BIND=0.0.0.0 ./olorin --serve
```

Notes: restart `--serve` after editing the env file (a running process won't
reload it); an empty `OLORIN_AUTH_TOKEN` counts as "no token" and is refused on
a non-loopback bind; loopback binds (the default) need no token. The token in
`~/.olorin/env` is plaintext — `chmod 600` keeps it owner-only.

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
and bridge are both opt-in; the core binary is ~2.4 MB on x86 / ~5.2 MB
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

The 2.4 MB / 2-dep figure isn't aspirational — it's the result of deliberately
not pulling in the things most LLM agents depend on.

| Layer | Common choice | Olorin |
|---|---|---|
| Async runtime | `tokio` | `std::net` + work-stealing pool ([`inference/threadpool.rs`](src/inference/threadpool.rs)) |
| HTTP server | `axum`, `actix`, `hyper`, `warp` | hand-rolled in [`interface/server.rs`](src/interface/server.rs) |
| SSE streaming | `axum::response::sse` | raw `text/event-stream` frames |
| Web terminal | `xterm.js` (~700 KB JS) | canvas cell-grid emulator over a hand-rolled WebSocket ([`web/chat.html`](web/chat.html) + [`interface/ws.rs`](src/interface/ws.rs)) |
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
| Rust source | 31,613 lines (150 files) |
| Ea kernel source | 16,103 lines (83 files, 60 logical kernels) |
| Tests | 24,862 lines (163 files, 887 tests) |
| Runtime dependencies | 2 (libc, libloading) |
| Release binary | 2.4 MB on x86_64 / 5.2 MB on ARM (all kernels embedded) |
| Max file size | 500 lines for Rust + tests (no exceptions); 2 Ea kernels exceed it |

## License

MIT. See [`LICENSE`](LICENSE).

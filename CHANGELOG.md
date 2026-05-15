# Changelog

All notable changes to Olorin. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project
uses [semver](https://semver.org/) at the minor level. Each release
is tagged in git as `vX.Y.Z` and listed below in reverse-chronological
order.

## [1.1.1] — 2026-05-15

Doc-only patch.  README architecture overview and "The Vault" diagram
brought up to date with the v1.1.0 AEAD format.  No code change; no
on-disk format change; v1.1.0 vaults open unchanged.

### Fixed

- README's "The Vault" diagram still showed `block --> xxHash64 verify
  --> ChaCha20 decrypt` for the read path.  Now reads `Poly1305 verify
  --> ChaCha20 decrypt` to match the v1.1.0 AEAD format.
- README intro to "The Vault" still said "encrypted at rest using
  ChaCha20" — updated to "ChaCha20-Poly1305 AEAD" so first-time readers
  see the authenticated half.
- README write-path diagram updated from "ChaCha20 encrypt" to
  "ChaCha20-Poly1305 seal" (matches `src/storage/aead.rs::seal`).
- README search-path diagram now shows the per-block Poly1305 verify
  gate that runs before each `FusedSearcher` candidate — surfacing the
  verify-then-search behavior added in v1.1.0.
- README architecture tree's `storage/key.rs` entry no longer advertises
  `xxHash64`.  The function survives as an internal helper used only by
  the v1→v2 migration verifier; it is no longer part of the live vault
  read or write flow.

## [1.1.0] — 2026-05-14

Vault crypto upgrade: per-block ChaCha20-Poly1305 AEAD with binding AAD,
a Poly1305 tag over the header + index, constant-time tag compare on
open, and verify-then-search in the fused decrypt path.  v1 vaults
auto-migrate to v2 on first open by a v1.1.0 binary, with the original
preserved as `vault.bin.v1.bak`.

### Added

- **Vault format v2 (ChaCha20-Poly1305 AEAD).** Per-block integrity now
  uses Poly1305 (RFC 8439) over `aad || ciphertext` instead of v1's
  xxhash check.  The AAD binds `key_id || version || nonce_counter ||
  timestamp || histogram`, so a swapped or copied block fails open even
  if the attacker controls both halves.
- **Header MAC.** The 64-byte v2 header carries a `header_tag` field
  that Poly1305-MACs `header[0..46] || serialized_index` using a
  domain-separated nonce (the high bit of the 4-byte counter slot is
  reserved for this domain — see the tightened wrap guard below).  Any
  byte flipped inside the MAC region — `block_count`, `index_offset`,
  `key_id`, `nonce_seed_8`, `header_rewrites`, or any index entry —
  fails open with `Error::Vault("vault header or index has been
  tampered")`.
- **Auto-migration of v1 vaults.** On first open by a v1.1.0 binary,
  v1 vaults are streamed, xxhash-verified, re-encrypted as v2 blocks,
  and atomically renamed into place.  The original is preserved as
  `vault.bin.v1.bak`.  A corrupt v1 fails before any backup or write,
  so users can't lose data to a silent migration error.
- **Constant-time tag verification.** `poly1305_verify` exported by
  both x86 and ARM kernels runs OR-reduce + branchless `is_zero` on
  the 16-byte XOR difference.  No early-exit; no byte-position
  timing leak.
- **Verify-then-search.** `Vault::search` MAC-verifies each candidate
  block before handing it to `FusedSearcher`.  A tampered block drops
  silently — no block index, no partial line, no signal reaches the
  result set.
- `src/storage/aead.rs` — `seal` / `open` / `verify` (verify-only,
  no decrypt) for ChaCha20-Poly1305 with AAD.
- `kernels/poly1305.ea` + `kernels/poly1305_arm.ea` — Poly1305 SIMD
  kernel, 5×26-bit radix using `wmul_u64_lo` / `wmul_u64_hi`.
- Cross-arch bit-identity golden at `tests/fixtures/aead_golden.bin`
  (1040 bytes = 1024 ct + 16 tag).  x86 ≡ ARM NEON.

### Changed

- **Vault block counter wrap tightened** from `u32::MAX` to
  `0x80000000`.  The high bit of the 4-byte counter slot is now
  reserved for the header-MAC nonce domain, so an honest counter must
  never set it.  At one append per second this still gives ~68 years
  before exhaustion.
- `IndexEntry.xxhash` is now `_reserved` — on-disk layout preserved
  (zeros written, ignored on read), so a v1→v2 migration can stream
  entries without shifting offsets.
- `FusedSearcher::search` now takes a `ctr_init: i32` parameter
  (v1 vaults passed 0, v2 passes 1 — counter 0 is the Poly1305 OTK).
- `Vault::read_encrypted_block` returns `(ct, tag, nonce)` instead of
  `(ct, nonce)`; only `Vault::search` uses it.

### Removed (breaking on-disk format)

- v1 vault format.  v1 vaults auto-migrate to v2 on first open;
  external tooling that parsed v1 directly must update.
- `Vault::last_block_hash()` — xxhash slot is gone.  Per-block AEAD
  tag is a stronger and constant-time integrity check.

### Security findings closed

- v1's xxhash-on-plaintext was non-cryptographic — an attacker who
  could write to `vault.bin` could substitute matching plaintext +
  precomputed xxhash and read it back without detection.  v2's
  Poly1305 tag is key-dependent so this attack now requires the key.
- v1 had no header integrity — `block_count`, `nonce_seed`, or index
  bytes could be tampered with at rest and the vault would open
  normally.  v2's `header_tag` closes this.

## [1.0.0] — 2026-05-12

Marks the rune family as production-ready. **No new features over
0.9.4** — this is a stability commitment, not a feature release.
Promotes the contracts that solidified over the v0.8.x and v0.9.x
lines to "stable" status.

### Stable contracts

The following surfaces are now stable and will not change without
a corresponding 2.0 bump:

- **`RuneOutput` v1 schema** and its compact JSONL wire format
  (`schema_version: 1`, `rune`, `source`, `totals`, `fields[]`,
  `categories[]`, `samples[]`, `error`). The v1 shape carried six
  runes and every `FieldKind` variant without a field addition;
  the design is settled.
- **`--json` mode behavior**: structured output is emitted
  verbatim (no `<rune_output>` wrap, no `[timing: …]` footer, no
  LLM narration). Refusal paths emit JSON too. Safety scan runs
  on the raw bytes regardless of format.
- **Six rune names and their core argument shape**: `eacrunch`,
  `eajson`, `eaparquet`, `ealog`, `eatime`, `eadiff`. Each accepts
  `[--json] <path>`; `eatime` also accepts `--bucket hour|weekday`;
  `eadiff` takes two paths.
- **Cross-rune chaining semantics**: match by exact name across
  `fields[]` and `categories[]`. Numeric deltas signed in
  `numeric.mean`. Bool fields split into paired
  `<col>.true_delta` / `<col>.false_delta`. Timestamp emits
  `<col>.unique_delta` + `<col>.min_shift_s` / `<col>.max_shift_s`.
  Text emits `<col>.unique_delta` plus per-value
  `<col>:<value>.count_delta` and `[appeared in top]` /
  `[disappeared from top]` markers. Asymmetric structural changes
  use `[appeared] <name>` / `[disappeared] <name>` Mixed markers.
  Categories use `+<name>` / `-<name>` for symmetric deltas.

### Out of scope for the 1.0 commitment

These exist in the codebase but are not promised stable:

- Internal kernel function signatures, the FFI loader, and the
  `KernelTable` shape. Private to the implementation.
- Specific throughput numbers (measured but workload-dependent).
- The Gemma 4 inference stack. Still evolving on a separate
  cadence; the 1.0 promise covers the rune family only.
- Web UI / WhatsApp gateway. Present and functional but not
  audited to the same depth as the rune dispatch path.

### Final state at 1.0

- 6 runes
- 1 cross-arch SIMD kernel family (csv_scan, jsonl_struct,
  log_level_scan, timestamp_scan, f32_stats, f64_stats), all
  validated on x86 SSE2 and ARM NEON
- 77 test suites, 423 passing tests, 0 failures
- 0 build warnings
- No file over the 500-LOC cap except the two documented
  chacha20 fused decrypt+search kernels (Ea has no module
  system; monolithic is required)

## [0.9.4] — 2026-05-12

### Fixed

- `--json` REPL dispatch contract: `/rune <name> --json …` now emits
  the structured JSONL answer verbatim. Previously the answer was
  wrapped in `<rune_output untrusted="true">` tags (broke
  parseability), suffixed with a `[timing: Nµs]` footer (broke
  JSONL line semantics), and routed through the LLM for narration
  (defeated the user's machine-output intent). The safety scan
  still runs on the raw JSON bytes — prompt-injection patterns
  inside file-derived string values stay blocked regardless of
  format.

### Changed

- `RuneResult` gains a `structured: bool` field set by each rune's
  `run()` when `--json` is in the args. Threads through
  `wrap_rune_result` (skips wrap), `build_narration_prompt` (returns
  `None`), and the REPL dispatch path (omits footer).

## [0.9.3] — 2026-05-12

### Added

- `eadiff` now emits Timestamp range-shift seconds: alongside the
  existing `<col>.unique_delta`, Timestamp diffs emit
  `<col>.min_shift_s` and `<col>.max_shift_s` carrying the signed
  second-deltas of the range endpoints. A range moving forward by
  one day reads as `+86400.00`. Garbage ISO strings silently omit
  the shift fields instead of crashing or emitting fake values.
- `eadiff` now emits Text top-N value comparison: per-value entries
  for each Text column. Values present in both top-N with different
  counts emit as `<col>:<value>.count_delta`; values appearing in
  only one side emit as `[appeared in top] <col>:<value>` /
  `[disappeared from top] <col>:<value>` Mixed markers.
- `benchmarks/timestamp_scan_bench.c` — C-driver throughput
  benchmark for the eatime kernel. Measured **6.34 GB/s on Ryzen
  7700X (SSE2)** and **1.80 GB/s on Pi 5 Cortex-A76 (NEON)** on a
  100 MB synthetic log.

### Changed

- `tests/runes_eadiff.rs` split at the FieldKind axis to keep both
  files under the 500-LOC cap: core dispatch + asymmetric + chain
  tests in the original file, per-Kind diff modes in
  `tests/runes_eadiff_kinds.rs`. Shared helpers extracted to
  `tests/common/eadiff_helpers.rs`.

## [0.9.2] — 2026-05-12

### Added

- `eadiff` handles every `FieldKind`. Number deltas unchanged. Bool
  fields split into paired `<col>.true_delta` / `<col>.false_delta`.
  Timestamp and Text emit `<col>.unique_delta`. Asymmetric keys (in
  one input but not the other) emit `[appeared] <name>` /
  `[disappeared] <name>` with kind=Mixed. Mismatched kinds emit
  `[kind-changed] <name>`. Same encoding works for `fields[]` and
  `categories[]`.
- `eatime --bucket weekday`: 7-bucket Mon..Sun histogram via
  Zeller's congruence on the year/month/day digits the kernel
  already extracted. Same `timestamp_scan` pass, different scalar
  post-processing. Default `--bucket hour` keeps the v0.9.1
  24-slot output bit-identical.

## [0.9.1] — 2026-05-12

### Added

- New rune `eatime`: ISO-8601 hour-of-day histogram. New SIMD
  kernel `timestamp_scan.ea` emits byte offsets where
  `YYYY-MM-DDT` occurs in a file; Rust extracts HH and buckets.
  All 24 hour-of-day slots always emitted (deterministic for
  downstream `eadiff`). Single `.ea` source; cross-arch via
  structural-anchor + scalar-validate idiom (no `movemask`, no
  `sat_sub`). Bit-exact bucket counts on x86 SSE2 and ARM NEON.
- New rune `eadiff`: structural delta between two prior `--json`
  rune outputs. Match-by-name across `fields[]` and `categories[]`.
  Generic — the same code handles `eatime × eatime`,
  `eacrunch × eacrunch`, `eaparquet × eaparquet`, etc. — never
  branches on the source rune.
- `src/kernels/ffi_data.rs` extracted from `ffi.rs` (data-plane
  kernel wrappers: csv_scan, jsonl_struct_scan, log_level_scan,
  timestamp_scan, f32_stats, f64_stats). Re-exported so existing
  call sites continue to compile.

## [0.9.0] — 2026-05-12

The structured-output release. Every rune now produces a stable
`RuneOutput` v1 contract that downstream runes (and shell pipes)
can consume mechanically.

### Added

- `RuneOutput` v1 schema in `src/runes/output.rs`. Cross-rune JSON
  contract with three orthogonal axes — `fields[]` (per-column),
  `categories[]` (bucketed counts), `samples[]` (exemplar
  records) — plus `source`, `totals`, and `error`. NaN/Inf
  serialize as `null`; round-trips through
  `to_json` / `from_json` via `storage::json` (zero new deps).
- `--json` flag on all four existing runes (eacrunch, eajson,
  ealog, eaparquet). Refusal paths also emit JSON so chained
  downstream runes get a parseable failure rather than free-form
  text.

### Changed

- All four existing runes migrated to use `RuneOutput` as the
  source of truth. Both the JSON form and the legacy text form are
  rendered from the same in-memory `RuneOutput` — they cannot
  drift. Zero schema fields added during the migration; the v1
  shape was sized correctly from the union of existing rune
  outputs.
- `src/runes/eajson.rs` split into `eajson.rs` + the new
  `eajson_aggregate.rs` to keep both files under the 500-LOC cap.

### Fixed

- `open_capped` `NotFound` error now reads `"file not found"`
  instead of the debug-format `"open failed: NotFound"`. Latent
  in all four migrated runes; surfaced by the structured-output
  failure-path tests.
- Deterministic top-N text ordering in `eacrunch` and `eajson`:
  same-count values now sort by count desc, value asc. Previously
  HashMap-seed-dependent across runs.

## [0.8.5] — 2026-05-11

### Fixed

- Windows rune allowlist: `std::fs::canonicalize` on Windows
  returns the verbatim form (`\\?\C:\…`), which
  `Path::starts_with` treats as a distinct prefix from the
  non-verbatim home. Now canonicalizes `home` too — no-op on Unix.
  Latent since v0.8.1 because no rune was invoked in the
  windows-port acceptance criteria.

## [0.8.4] — 2026-05-11

### Fixed

- Kernel extraction is now atomic: write-to-tmp + rename pattern
  with per-PID/per-thread tmp paths. `cargo test` parallel mode
  works again — previously concurrent processes raced on
  `~/.olorin/lib/{version}/lib<name>.so` writes.

## [0.8.3] — 2026-05-11

### Added

- `ealog` now records byte offsets of the first 5 high-severity
  (ERROR / FATAL) matches and surfaces sample lines in the rune
  output. Kernel extended with a `(out_positions, max_positions,
  out_n_positions, scratch)` signature pair using the same
  store-mask-to-scratch + scalar-walk pattern as `csv_scan` and
  `jsonl_struct`. `reduce_add` early-out keeps the no-match common
  case scratch-free; Pi 5 NEON perf unchanged at 0.43 GB/s.

## [0.8.2] — 2026-05-11

### Added

- New rune `ealog`: SIMD log severity scanner. Counts word-bounded
  `DEBUG / INFO / WARN / ERROR / FATAL` plus newline bytes in one
  pass. Cross-arch single `.ea` source; validated bit-exact on x86
  SSE2 (Ryzen 7700X) and ARM NEON (Pi 5 Cortex-A76). Measured 0.70
  GB/s on Ryzen, 0.43 GB/s on Pi 5. The
  `reduce_add`-on-u8x16-with-`select` idiom (no `movemask`) is the
  template future cross-arch byte-scanning kernels follow.

## [0.8.1] — 2026-05-11

### Added

- Windows parity release. Olorin now builds and runs natively on
  Windows: inference (mmap GGUF + 35-layer forward + 16-thread
  futex pool), embedded ConPTY terminal panel (input + output +
  resize), web chat + SSE, slash commands (`/cpu`, `/weather`,
  `/time`, `/sh`), vault key derivation machine-bound (HKLM
  MachineGuid). 22 commits from `windows-port` branch
  rebase-merged via the repository's first PR.

### Changed

- Cross-platform abstractions added under `src/platform/`: every
  libc-equivalent syscall behind a thin wrapper (mmap, futex,
  home_dir, hwid, sysinfo) or behind a trait + `_unix.rs` /
  `_windows.rs` backends (Spawner, PtyBackend). Result:
  `cargo check --target x86_64-pc-windows-gnu` produces a finite,
  mechanical TODO list of cfg-gated unimplemented backends.

## [0.8.0] — 2026-05-07

### Added

- Repositioned as "deterministic SIMD analyst with LLM narration".
  README rewritten to lead with the runes (kernel-first) story.
- `--strict` CLI mode: disables the LLM entirely. No model load,
  no narration. Starts in ~25 ms vs ~25 s with the model. Useful
  for fast one-shot CLI use and security-conscious deployments
  that need a categorical "this binary will never call an LLM"
  guarantee.
- `--audit <path>`: writes a JSON Lines log of every dispatch
  turn. Two events per turn (input received + dispatch result
  with phase + microsecond timing). Captures metadata only — no
  input text or rune output content.
- New rune `eaparquet`: Parquet footer summarizer. Reads
  metadata only (never decodes column data). Per-column min/max
  via `f64_stats` SIMD reduction across row groups. Milliseconds
  even on multi-GB files.
- `benchmarks/` driver: eacrunch vs pandas on 10K / 100K / 1M
  transaction CSVs. Numbers in
  [`benchmarks/results.md`](benchmarks/results.md).

### Changed

- 4 over-cap source files split (`router`, `router_tools`,
  `server`, `parquet`); helper modules extracted
  (`thrift_compact`, `router_streaming`).

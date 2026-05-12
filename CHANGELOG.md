# Changelog

All notable changes to Olorin. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project
uses [semver](https://semver.org/) at the minor level. Each release
is tagged in git as `vX.Y.Z` and listed below in reverse-chronological
order.

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

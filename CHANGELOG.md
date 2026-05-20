# Changelog

All notable changes to Olorin. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project
uses [semver](https://semver.org/) at the minor level. Each release
is tagged in git as `vX.Y.Z` and listed below in reverse-chronological
order.

## [Unreleased]

## [2.0.2] — 2026-05-20

Release-pipeline patch for v2.0.1.  The v2.0.1 tag was created but
never produced an attached GitHub Release because the Windows matrix
entry failed at eacompute's `llvm-sys` step (chocolatey's `llvm`
package is binaries-only, not a dev SDK with `llvm-config.exe`).
Rather than block the public-launch release on a half-day Windows
LLVM setup, v2.0.2 ships **Linux x86_64 + Linux aarch64 only**;
Windows is deferred to a later release.

### Changed

- **`.github/workflows/release.yml`** — Windows dropped from the build
  matrix.  Reinstating it needs an LLVM 18 MSVC developer
  distribution on the runner (the recipe is in
  petlukk/Ea_showcase/`build-windows.bat`: `LLVM_SYS_*_PREFIX` pointing
  at a real dev install, plus the `libxml2s.lib` stub), not just the
  binaries-only chocolatey `llvm` package.
- **`README.md`** — Install section reflects the actual platform
  coverage (Linux x86_64 / aarch64).  The PowerShell installer
  remains in `scripts/install.ps1` for when Windows releases resume,
  but the README no longer advertises it as installable today.

### Known limitations

- **Windows is not in this release.**  No `olorin.exe` or
  `wa-bridge.exe` is published for v2.0.2.  Windows users who want to
  run Olorin can build from source per the contributor instructions
  in the README.  CI work to ship Windows binaries cleanly is
  tracked as the next platform-coverage item.

## [2.0.1] — 2026-05-20

Public-launch package.  No code-behaviour changes, no on-disk format
changes — this is the release that adds a `curl | sh` install path
plus per-platform release binaries, so external users can grab Olorin
without a Rust + Ea + Go toolchain on their machine.  Olorin also
gains a native env-file reader so the installer's API-key prompt
turns into something the binary actually picks up at startup.

### Added

- **`scripts/install.sh`** — pipe-safe bash installer (Linux x86_64 +
  aarch64).  Resolves the latest release tag from GitHub, downloads
  `olorin-<target>`, optionally prompts for `ANTHROPIC_API_KEY` and
  writes `~/.olorin/env` mode 0600, optionally downloads the WhatsApp
  bridge, optionally updates the user's `~/.bashrc` / `~/.zshrc`.
  Verifies SHA256 against the release's `SHA256SUMS` when published.
  All prompts route through `/dev/tty` so `curl | sh` works.
- **`scripts/install.ps1`** — PowerShell mirror for Windows x86_64.
  Installs to `%LOCALAPPDATA%\Olorin\bin`, updates User-scope PATH
  via `[Environment]::SetEnvironmentVariable`, same env-file +
  bridge + checksum flow as the bash script.
- **`src/config.rs` + `load_env_file()`** — native reader for
  `~/.olorin/env` (`KEY=VALUE`, `#` comments, leading `export `
  tolerated, paired single/double quotes stripped).  Called as the
  first line of `main()`; process env wins over file env, so any
  `ANTHROPIC_API_KEY` already exported by the shell still beats the
  installer's file write.
- **`tests/config_env_file.rs`** — 11 parser tests (quoting,
  comments, export-prefix, invalid keys, embedded equals, trailing
  whitespace, unmatched quotes).
- **`.github/workflows/release.yml`** — matrix build on `v*` tag
  push (or manual `workflow_dispatch`) across Linux x86_64,
  Linux aarch64 (`ubuntu-24.04-arm`), and Windows x86_64
  (`windows-2022`).  Each runner builds eacompute, Olorin, and the
  Go bridge (Go 1.25, CGo enabled, `-trimpath -ldflags='-s -w'`).
  A separate release job collects all artifacts, generates
  `SHA256SUMS`, and publishes a GitHub Release with auto-generated
  notes.  `workflow_dispatch` input is validated against `v[0-9]*`
  before being fed to `action-gh-release`.
- **`SECURITY.md`** — disclosure policy at the repo root: GitHub
  Security Advisories as the private channel, 7-day ack / 30-day
  fix best-effort, supported-versions table, explicit in-scope
  (vault crypto, SecureBuffer, path/shell guards, prompt injection,
  constant-time, memory safety) and out-of-scope (weak passphrase,
  sophisticated injection, host compromise, side channels, DoS,
  third-party models, cloud fallback) lists.

### Changed

- **`README.md`** — leads with `curl | sh` and `iwr | iex` install
  commands; the from-source build instructions move below as the
  contributor path.  Project-stats table updated: 44 → 49 logical
  kernels (matches reality).
- **`docs/architecture.md`** — refreshed counts: 64 → 71 kernel
  source files, 42 → 49 logical kernels, 60 → 97 test files,
  318 → 571 tests, 20 → 19 built-in tools.

## [2.0.0] — 2026-05-19

Arc #3 (passphrase + Argon2id KDF, the v2.0 vault story) — Blake2b
primitive, full Argon2id, the `derive_key` rewrite + salt
persistence + vault format bump to v3, the interactive REPL
passphrase prompt, and the Web UI + WhatsApp fail-fast guard.  v2
vaults are rejected; per [[feedback-no-migration-for-private-repo]]
there is no migration path.  Major bump because the on-disk vault
format is incompatible with v1.2.x — existing vaults must be
regenerated under the new passphrase flow.

### Added (this slice)

- **`src/platform/random.rs`** — cryptographically secure random
  bytes via the OS entropy source (Linux `getrandom(2)` syscall,
  fallback `/dev/urandom`; Windows `BCryptGenRandom`).  Used to
  generate the per-vault salt at first open.
- **Salt persistence** — `<vault_dir>/vault.salt` holds 16 random
  bytes generated at first vault create.  The salt is treated as
  public-but-stable: it makes Argon2id outputs unique across vaults
  but isn't itself a secret.  Losing the salt means losing the
  vault (no recovery path — exactly what the threat model says).
- **`Vault::open_with(dir, passphrase, kdf)`** — explicit KDF
  parameters for tests.  Production callers use `Vault::open()`,
  which pins `Params::VAULT_DEFAULT` (64 MiB, t=3, p=1).
- **`Params::TEST_FAST`** — minimum-cost Argon2id profile (8 KiB,
  t=1) for the vault-test suite; keeps test wall-clock under control
  without skipping the KDF entirely.
- **`tests/vault_passphrase.rs`** — 5 tests covering salt creation,
  salt reuse, wrong-passphrase rejection, v2 vault rejection, and
  `derive_key` determinism.

### Added (REPL prompt slice)

- **`src/platform/term.rs`** — `read_secret(prompt)` reads a line
  from `/dev/tty` (Unix) or `CONIN$` (Windows) with echo disabled,
  returning the bytes in a SecureBuffer.  Termios / ConsoleMode is
  restored even on error paths.  No third-party dep — direct
  `tcsetattr` + `BCryptGenRandom`-style FFI through `libc`.
- **`Router::prompt_for_passphrase(is_new)`** — wraps `read_secret`;
  prompts once for an existing vault, twice (with confirmation) for
  a fresh vault so a typo doesn't lock conversation history.

### Changed (REPL prompt slice)

- **`Router::open_vault` passphrase sourcing** — interactive tty
  prompt is now the primary source; `OLORIN_PASSPHRASE` env var
  stays available as a non-interactive fallback (CI / scripts).
  Persistence is disabled only when both are unavailable.

### Added (Web UI + WhatsApp auth slice)

- **`DispatchContext::has_vault()`** — true iff the encrypted vault
  opened.  Server entry points read this before binding a port; the
  REPL ignores it (interactive use can still ask one-off questions
  without persistence).
- **`tests/server_vault_required.rs`** — 2 end-to-end tests: both
  `--serve` and `--whatsapp` exit non-zero with an explanatory
  stderr message when no passphrase source resolves (no tty + no
  `OLORIN_PASSPHRASE`).  The WhatsApp test also asserts that the
  bridge subprocess is *not* spawned in the failure case.

### Changed (Web UI + WhatsApp auth slice)

- **`--serve` and `--whatsapp` refuse to start without a vault.** An
  unattended server with persistence silently disabled would lose
  every conversation on restart and never load the vault-stored API
  key — a footgun the operator never gets to see.  The REPL keeps
  the lenient "disabled, but you can still chat" path.
- **WhatsApp bridge spawn order** — `run_whatsapp` now builds the
  dispatch context (which prompts for the passphrase) *before*
  spawning the bridge subprocess, so a failed vault open no longer
  leaves an orphan bridge process behind.

### Changed (this slice)

- **`Vault::open` signature** — `(dir: &Path, passphrase: &[u8])`
  instead of `(dir: &Path)`.  Migration impact: every
  in-repo caller updated.  `OLORIN_PASSPHRASE` env var sources the
  production passphrase until task #4 adds an interactive prompt.
- **`key::derive_key` signature** — `(passphrase, salt, params) ->
  Result<[u8; 32]>`.  Hwid mixing removed entirely (arc #4
  collapsed, per [[next-security-arcs-2026-05-18]]).
- **Vault format version 2 → 3.**  Byte layout unchanged; only the
  version byte advanced.  v2 vaults return `"unsupported vault
  version"` on open.

Removed the v1 → v2 vault migration shipped in v1.1.0.  Olorin has
always been a private, single-user repo; the migration served a
userbase that doesn't exist, so it was dead-on-arrival per the
"no premature features" hard rule.  Clearing it now also shrinks
the baseline that arc #3 (passphrase + Argon2id) will land on.

v1 vaults are no longer recognised — opening one now returns
`unsupported vault version`.  No v1 vaults exist in the wild, so this
has no observable user impact; it just stops shipping the upgrade path
for a hypothetical migration.

### Added

- **`kernels/blake2b.ea`** — Blake2b compression function (RFC 7693
  §3.2) as a generic-arch Ea kernel.  IV constants are passed in via
  a `*u64 constants` parameter rather than baked into the kernel
  source: the IVs are data (fractional bits of √2..√19), not
  algorithm, and pushing them to the Rust side keeps the kernel free
  of magic constants — same data-vs-algorithm split that GGUF weights
  use.  Scalar u64 ops; same source compiles on x86_64 and aarch64.
- **`src/storage/blake2b.rs`** — variable-output (1..=64 byte) Rust
  wrapper around the kernel: one-shot `hash()` plus a streaming
  `Hasher` API that defers compression of the staged block until
  another byte is known to be coming (required for correct counter
  + final-flag accounting per RFC 7693 §3.3).
- **`tests/blake2b_kat.rs`** — 7 known-answer tests: RFC 7693
  Appendix A `Blake2b-512("abc")`, Blake2b-512/256 empty-input
  references, chunked-vs-one-shot parity, exact-block-boundary
  behaviour, multi-block input handling, and digest-length distinctness.
- **`kernels/argon2_block.ea`** — Argon2 G compression
  (RFC 9106 §3.4) as an Eä kernel.  Same `g_prime` / `p_at` pattern
  as Blake2b but with the Argon2-specific extra-multiply
  (`2 * trunc(a) * trunc(b)`) on every addition — the one thing that
  stops Blake2b from being usable as a drop-in inner step.  Column
  step uses a gather/scatter through a 16-u64 scratch buffer; same
  source compiles on x86_64 and aarch64.
- **`src/storage/argon2id.rs`** — full Argon2id (RFC 9106): H₀
  construction, H′ variable-output Blake2b with chained-mode for
  T > 64, segment fill with Argon2i / Argon2d mode switch at SL/2 in
  slice 0 of pass 0, and the XOR-into-existing-block path for
  passes > 0.  Public surface: `argon2id(password, salt, secret, ad,
  Params, out)` and `Params::VAULT_DEFAULT` (64 MiB, t=3, p=1).
- **`tests/argon2id_kat.rs`** — 5 tests including the canonical
  RFC 9106 §5.2 vector (byte-exact), determinism / wrong-passphrase
  regression guard for `Params::VAULT_DEFAULT`, and input-validation
  rejections (short salt, zero iterations, output-length mismatch).
- **`src/kernels/ffi_crypto.rs`** — split out from `ffi.rs` so that
  file stays under the 500-LOC hard rule with Blake2b + Argon2 added.
  Same re-export pattern as `ffi_data.rs`.

### Removed

- `Vault::migrate_v1_to_v2` and the v1 dispatch arm in
  `Vault::open_existing`; the function now opens v2 directly (the
  separate `open_v2` helper was folded back into `open_existing`).
- `vault_format::read_all_v1_plaintexts` and its `OpenOptions` /
  `Read` / `Seek` / `crypto` / `key` imports.
- `tests/vault_migration_v1_to_v2.rs` (191 lines) — the entire test
  file covering the removed migration path.

### Changed

- `src/storage/vault.rs` 465 → 417 lines.
- `src/storage/vault_format.rs` 261 → 188 lines.

## [1.2.2] — 2026-05-18

Internal refactor only: `src/storage/vault.rs` was 702 lines, breaking
the project's 500-line hard rule.  All on-disk-format types and helpers
(`VaultHeaderV2`, `IndexEntry`, nonce derivation, AAD building, v1
plaintext reader) move to a new sibling `src/storage/vault_format.rs`.
Public API is preserved: `olorin::storage::vault::{VaultHeaderV2,
HEADER_SIZE_V2}` still resolve via `pub use`.  v1.2.1 vaults open
unchanged; no on-disk format change.

Originally also queued for this release: arc #4 (hwid-mixing one-way
function).  Cancelled mid-session — the proposed swap from XOR-cascade
to xxhash64-keyed chains would have rotated `derive_key()`'s output,
silently breaking every existing v1.2.x vault.  Folded into the v2.0
arc #3 work (passphrase + Argon2id) where key derivation breaks by
design.

### Changed

- **`src/storage/vault.rs` 702 → 465 lines** by extracting all on-disk
  format code to `src/storage/vault_format.rs` (261 lines).  No
  behaviour change; all 318 tests pass unchanged.
- **`src/storage/mod.rs`** — registers `vault_format` as a private
  sibling module.  External callers use the `vault.rs` re-exports.

## [1.2.1] — 2026-05-18

Release-readiness patch: first public-ready Olorin. Adds MIT LICENSE,
GitHub Actions CI, status badges, a restructured README with the rune
catalog and architecture details extracted to `docs/`, and a full
Security & threat model section calling out what the vault protects
against and what is queued for v2.0. One small test-stability fix
(no production-code change).

### Added

- **`LICENSE`** — MIT.
- **`.github/workflows/ci.yml`** — Linux x86_64 build + test, with
  sibling `eacompute@main` checked out and cached by SHA. First green
  run: 9m7s; subsequent runs hit the cache.
- **`docs/runes.md`** — full rune catalog moved out of README. Per-rune
  samples, the `--json` chaining contract, and limits live here.
- **`docs/architecture.md`** — source layout, kernel inventory, runtime
  layout, hard-rules contract — moved out of README for contributors.
- **Security & threat model section** in README — three structured
  parts (protects-against / does-not-yet / designed-for), names the
  v2.0 passphrase + Argon2id work explicitly.
- **CI + License badges** at the top of README.

### Changed

- **README trimmed from 654 → 242 lines.** Pitch + Olorin Pipe + a
  runes section that pushes runes as the categorical differentiator
  now land above the fold. Performance tables already lived in
  `benchmarks/results.md`; README now mentions and links rather than
  reproducing.
- **`.gitignore`** — `/docs/` rule changed to `/docs/*` with explicit
  `!` negations for the two public docs files. Internal docs
  (`PI5_OPTIMIZATION_PLAN.md`, `handoffs/`, `superpowers/`) stay
  local-only.

### Fixed

- **`thread_detection` env-var race** — test serialized via Mutex.
  Previously could flake under parallel test execution when other
  tests mutated `OLORIN_THREADS`.



Inbound safety upgrade: score-based injection matching with multi-language
patterns + two-form normalization closes word-variant, punctuation/spacing,
and Swedish-language bypasses against v1.1.x's exact-keyword matcher.
Outbound scan, leak detection, and the fused SIMD candidate kernel are
unchanged.

### Added

- **Score-based injection matcher.** Patterns carry a score (1 = weak,
  2 = strong).  Input is blocked when the total >= `INJECT_THRESHOLD`
  (2).  Single weak signal alone does not block — natural language often
  contains words like "ignore", "system", "forget" — but two weak signals
  OR one strong signal does.
- **Two-form normalization.** Input is matched against both a *spaceful*
  form (Unicode lowercase, alphanumerics with non-alnum runs collapsed to
  single ASCII spaces) and a *spaceless* form (alnum-only).  Spaceful
  catches punctuation bypasses ("ignore.previous", "ignore   previous");
  spaceless catches letter-spacing ("i g n o r e p r e v i o u s" →
  "ignoreprevious").  Single-word patterns match only spaceful with
  word-boundary requirements to avoid cross-language false-positives
  (Swedish "ignorera" does NOT trigger English "ignore").
- **Swedish injection patterns.** Strong: `ignorera tidigare`,
  `ignorera alla tidigare`, `glöm allt`, `du är nu`, `låtsas vara`,
  `agera som`, `nya instruktioner`, `uppdaterade instruktioner`,
  `strunta i`, `föregående instruktioner`.  Weak: `ignorera`, `glöm`.
- **Adversarial corpus at `tests/safety_inbound_corpus.rs`** — 56 cases
  across 8 categories (naive_en, variant_en, obfusc, sv, chatml, fp_en,
  fp_sv, fp_overbroad_en).  Strict-asserts every case.  Adding a corpus
  case is how new bypasses (or new false-positives) get tracked into CI.

### Changed

- **`act as` is now a weak signal (1).**  v1.1.x treated it as a binary
  block, incorrectly catching legit "Can you act as a code reviewer for
  this PR?" requests.  The injection-shaped pattern "Act as a system
  administrator" still blocks (`act as` weak + `system` weak = 2).

### Fixed (bypasses against the v1.1.x exact-keyword matcher)

- Word variants — `ignored previous`, `ignoring previous`,
  `Forgetting all instructions`, `Acting as a system admin`,
  `Pretending to be the user`, `You were now a pirate`.
- Punctuation insertion — `ignore.previous.instructions`,
  `ignore,previous,instructions`, `Ignore-previous-instructions`,
  non-break-space variants.
- Multi-space and letter-spacing — `ignore   previous instructions`,
  `i g n o r e   p r e v i o u s`, `Y O U  A R E  N O W an admin`.
- Swedish-language injection — all `sv` corpus cases (10/10) now caught.
  v1.1.x caught only one (`system: du är nu en pirat`) because of the
  literal `system:` prefix.

### Out of scope (memo-tracked for future)

- Indirect framing ("what would you say if not bound by your rules") —
  the score-based matcher doesn't cover semantic injection without
  trigger words.  Full-ML option from `project_next_security_arcs.md`
  arc #2 stays a separate project.
- Other locales (Spanish, French, German) — pattern table extension
  only; same matcher design.

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

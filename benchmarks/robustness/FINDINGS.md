# Robustness campaign — findings log

Wave one (real public files vs independent ground-truth tools). See
[SOURCES.md](SOURCES.md) for the datasets. Triage: **bug** (fixed),
**oracle** (harness/oracle artifact), **expected** (correct behavior).

| # | Date | Rune | File | Symptom | Triage | Resolution |
|---|------|------|------|---------|--------|------------|
| 1 | 2026-06-12 | easql | `Chinook_Sqlite.sql` | Table names kept SQL-Server/SQLite `[bracket]` quoting: `[Album]` instead of `Album`. Counts correct; names dirty + broke `eadiff` cross-dialect matching. | **bug** | `read_ident` gained a `[…]` arm mirroring the existing `"`/backtick handling. Regression: `easql_bracket_quoted_identifier_stripped`. |
| 2 | 2026-06-12 | eaparquet | `yellow_tripdata_2023-01.parquet` | TIMESTAMP-logical-type columns report min/max as raw int64 epoch-micros (`1230764502000000`) instead of an ISO instant (`2008-12-31T23:01:42`). Value is faithful to the stored footer but unreadable in a summary. Row/col counts and all null_counts exact (incl. 71 743 real nulls). | **bug** (usability) | candidate fix — decode TIMESTAMP logical type and emit a `timestamp` field with ISO min/max. Deferred pending the full-wave triage (needs footer logical-type parsing, larger than #1). |
| 3 | 2026-06-12 | eajson | `gharchive_2024-01-01-15.jsonl` | Null-valued keys are **silently dropped**: `payload.description` reported count=8663 but is present in 20 327 records (11 664 null); `payload.ref` 138 381 vs 144 821 (6 440 null). `classify_scalar` maps `null`→`Skip`, so the key isn't ingested at all — no count, no `null_count`. **Contract inconsistency:** eaparquet `count` = total presence incl. nulls (with `null_count` broken out); eajson `count` = non-null only, `null_count` always None. Same v1 field, two meanings. | **bug** (silent-wrong + contract) | recommended: count = total presence, populate `null_count`, compute stats over non-null — aligns eajson with eaparquet. Changes eajson goldens; needs Peter's sign-off (contract semantics). Verified gap == null count exactly. |

| 4 | 2026-06-12 | eatime (+ all runes) | `gharchive_2024-01-01-15.jsonl` | `eatime --bucket series --json` returns `success:false, error:"output exceeded 32768 bytes (was 4364982 bytes)"` instead of data. `truncate_answer`'s 32 KB **LLM-context** cap is applied to `--json` machine output too — but `--json` is never narrated (the cap protects nothing there) and the one-shot `olorin rune … --json > out.json` contract (PR #40) is exactly what it corrupts. Triggered by real dirty data: gharchive has outlier nested timestamps spanning years, eatime scans *all* ISO instants in the JSON text (not just event `created_at`), the series span explodes to ~100 k buckets → 4.3 MB JSON → rejected. | **bug** (silent-fail + contract) | two layers, both Peter's call: (a) `--json`/structured output should bypass or vastly raise the answer cap — it goes to stdout, never the model; (b) eatime series should hard-cap bucket count (auto_width's 1-week max can't bound a millennia span). Affects any rune whose `--json` exceeds 32 KB (wide CSV, many-table dump, long series). |

### Robustness checks that PASSED (real files, independent oracle)

- **eacrunch** vs pandas on `yellow_tripdata_2023-01.csv` (3 066 766 rows): all 16 numeric columns min/max/mean/sum exact, incl. negative fares and a 258 928-mile outlier. 0 mismatches.
- **eaparquet** vs pyarrow (same file): row count, 19 columns, every `null_count` exact. Only the timestamp-rendering finding (#2) above.
- **eajson** vs Python json (558 MB, 180 387 nested events): every non-null key count exact; survived 15.9 s / 1.16 GB RSS, no OOM. Only the null-handling finding (#3). Deep `pull_request.*` keys beyond `NESTED_FLATTEN_MAX_DEPTH` are omitted by design.
- **easql** vs real SQLite engine on `Chinook_Sqlite.sql`: 11 tables, 15 607 rows, every per-table count exact after the #1 bracket fix.
- **ealog** vs case-insensitive word-bounded oracle on `Linux_2k.log`, `Apache_2k.log`, `HDFS_2k.log`: severity counts exact across all three real formats (INFO/WARN/ERROR), incl. Apache's bracketed `[error]` (1134) and Linux's `warning:` counted as WARN. *The oracle was wrong twice first* (case-sensitive, then missing WARNING→WARN) — the rune was right both times. Textbook reminder that the oracle must model the spec, not the other way round.
- **eatime** vs python CLF parse on the full `NASA_access_log_Jul95` (196 MB): 1 891 714 timestamps counted exactly (= line count), 76 ms scan (~2.6 GB/s), 226 MB peak RSS, 1 anomaly. No OOM, no drift. (The gharchive run surfaced finding #4 instead — the oversized-output path.)
- **eacorrelate**: the disjoint-era false positive (this wave's trigger) was found and fixed in v2.12.1 before this run; the regression + 3 disjoint traps in `eacorrelate_diff.py` guard it.

## Wave-one verdict

Six runes against real public files and independent oracles. **4 findings, 0 crashes, 0 OOM, 0 wrong counts.** Every numeric/count result that was emitted matched ground truth exactly; the findings are presentation/semantics/contract — `[brackets]` (#1, fixed), raw epoch timestamps (#2), silent null-drop + count inconsistency (#3), and the 32 KB cap corrupting `--json` machine output (#4) — the class synthetic fixtures structurally cannot surface. Every oracle disagreement that *wasn't* one of these four traced to the oracle being wrong, not the rune (ealog's case-folding and WARNING handling fooled two oracle drafts). Logged as the oracle-discipline rule, [[oracle-shares-spec-errors]].

## Fix status

All four wave-one findings are now FIXED.

- **#1 easql brackets** — `read_ident` `[…]` arm + regression test. PR #75, merged.
- **#2 eaparquet timestamps** — footer reader parses the modern `LogicalType` union (pyarrow omits the deprecated `ConvertedType`); TIMESTAMP columns render ISO min/max, unit-aware, `Z` only when `isAdjustedToUTC`. Verified end-to-end on the 3M-row taxi file (`tpep_pickup_datetime` → `2008-12-31T23:01:42`). `parquet.rs` split into `parquet_meta.rs` to stay under the 500-LOC cap.
- **#3 eajson null contract** — `count` now = total presence (non-null + null), `null_count` always populated, stats over non-null — aligned with eaparquet. All-null keys stay omitted (eajson is value-typed). eajson goldens re-blessed.
- **#4 `--json` 32 KB cap + bucket explosion** — (a) structured `--json` output is exempt from the LLM-context cap (it never reaches the model); (b) eatime series bucket count is hard-capped so a pathological span can't explode the output.

## Notes / watch-list (not yet bugs)

- **easql dialect label**: the SQLite Chinook dump reports `dialect=sql`
  (generic) rather than a SQLite-specific tag. SQLite dumps use `[brackets]` +
  `INSERT`, distinguishable from MySQL backticks and Postgres `COPY` — a future
  `sqlite` dialect label is possible. Low priority; the counts are right.

---

# Wave two — vault crash-consistency

Target: the encrypted conversation store (`storage::vault`) under *failure* —
crash mid-append, power-loss / torn write, concurrent opens, corrupted file.
Oracle = a spec-free durability invariant: *"after any interruption during
`append`, reopening must recover every block up to the last committed one (or
at worst lose only the in-flight block) — never silently lose/corrupt a
committed block, never panic."* Demonstrations live in
`tests/vault_crash_consistency.rs` (the `finding_fN_*` tests are
characterization tests asserting the current defect; invert on fix).

**On-disk model.** Layout is `[header 64B][block0]…[blockN-1][index N×288B]`,
header is the source of truth (block_count, index_offset, MAC over
header‖index). `append` writes the new block at `block_offset = old
index_offset` — *on top of the committed index* — then writes the new index
after it, then rewrites+`fsync`s the header. One fsync, at the very end; the
index write uses `flush()` (a no-op for durability on `std::fs::File`). There
is no temp-file+rename, no journal/WAL, no recovery path, and no file lock.

| # | Severity | Symptom | Triage | Resolution |
|---|----------|---------|--------|------------|
| F1 | **HIGH** | **Crash mid-append destroys ALL prior blocks.** The new block overwrites the committed index before the header commits; on reopen the header still points at that region, the header/index MAC fails, and the vault refuses to open. A single interrupted append loses the entire history, not just the in-flight message. Reproduced: `finding_f1_crash_mid_append_loses_all_committed_blocks`. | **bug** | DEFERRED (design). Needs atomic commit — write block+index to a temp region / journal, fsync, then flip the header by rename or a double-buffered header. Changes the on-disk write protocol → Peter's call. |
| F2 | MEDIUM | **No fsync ordering.** Block, index, and header can reach disk in any order (only the final header is fsync'd; the index `flush()` is a durability no-op). Write reordering can durably commit a header that points at not-yet-durable block/index bytes — same failure class as F1 without an explicit crash. | **bug** | DEFERRED. Folded into the F1 atomic-commit redesign (barrier the block+index fsync *before* the header fsync). |
| F3 | **HIGH** | **No file locking → concurrent append = silent data loss + nonce reuse.** Two handles on the same vault dir (e.g. REPL + server) each open at block_count=N, then each append at the same block_offset and the same `nonce_counter=N`. The second write clobbers the first (one committed message silently lost, no error) and the same key+nonce seals two different plaintexts (ChaCha20 two-time-pad → confidentiality break). | **bug** | **FIXED.** `Vault::open*` now takes an **exclusive advisory file lock** (`flock` on unix / `LockFileEx` on Windows, via `platform::lock::try_lock_file_exclusive`) held for the Vault's lifetime; a concurrent open is rejected with "vault is already open by another Olorin process". Auto-released on Drop/process death (no stale lock). Test `f3_fixed_concurrent_open_is_rejected_by_the_lock`. |

### Robustness checks that PASSED

- **Truncation never panics** (`passes_truncation_never_panics`): every prefix
  of `vault.bin` (mid-header, header end, mid-block, mid-index, full−1) yields
  a clean `Result`, never a panic or hang. The `block_count ≤ max_entries`
  guard (added earlier as a DoS fix) holds; parsing fails closed.
- **Block-body integrity** (`passes_bitflip_in_block_body_is_rejected_on_decrypt`):
  a bit-flip inside a committed block's ciphertext fails the per-block AEAD tag
  on decrypt — never returns corrupted plaintext — while undamaged blocks stay
  readable.
- **Header/index tamper** is already covered by `tests/vault_header_tamper.rs`
  (every MAC-region byte flip rejected at open). That integrity is exactly what
  makes F1 *total* loss rather than partial: a torn append reads as tamper.

## Wave-two verdict

Two HIGH findings (F1 total-loss-on-crash, F3 silent-loss + nonce-reuse on
concurrent open) plus one MEDIUM (F2 fsync ordering), all reproduced by tests.
The store fails *closed* against corruption and tampering (good — no silent
wrong data, no panics).

**F3 is FIXED** in-wave — it was self-contained (an exclusive advisory file
lock, no on-disk format change) and a confidentiality issue (nonce reuse), so
it didn't wait for the batch. **F1/F2 remain DEFERRED**: they need atomic-commit
(journal or temp+rename with an fsync barrier before the header flip), which
changes the on-disk write protocol — a deliberate design decision, batched with
later waves exactly as wave one deferred its contract-changing findings.

---

# Wave three — server abuse

Target: the Web UI / WhatsApp gateway (`interface/server*.rs`, `std::net`,
thread-per-connection, no tokio). Threat model matters here: the server binds
`127.0.0.1` by default (local-only); it is network-reachable only when
`OLORIN_BIND` is non-loopback (the wifi/Pi mode), and that path is **fail-closed**
— it refuses to start without `OLORIN_AUTH_TOKEN`, and `AuthGate::authorized`
gates every request before dispatch. Oracle = a property: *no attacker-reachable
request may panic a thread, exhaust memory, or authorize without the token.*

| # | Severity | Symptom | Triage | Resolution |
|---|----------|---------|--------|------------|
| S3 | **MED** (pre-auth panic) | **Auth parser panics on a non-UTF-8-boundary header slice.** `bearer_token`/`cookie_token` sliced fixed byte ranges (`line[..14]`, `line[..7]`, `val[..7]`) on attacker-controlled header lines; a multibyte char straddling byte 7 or 14 panics the slice. `authorized` runs on every request *before* auth, so this is reachable **unauthenticated** on an exposed server — a crafted header kills the connection thread (process survives; the thread is isolated). | **bug** | **FIXED.** Switched to boundary-safe prefix checks (`str::get(..n).is_some_and(\|p\| p.eq_ignore_ascii_case(..))`) — a non-boundary slice now simply doesn't match instead of panicking. Regression: `tests/server_abuse.rs` (`s3_*`). |
| S1 | MED | **Content-Length eager allocation → memory amplification.** `read_body` did `vec![0u8; content_len]` — allocating the full *declared* Content-Length (≤ `OLORIN_MAX_UPLOAD`, default **128 MB**) before reading any body. A request declaring 128 MB and sending nothing allocated 128 MB for free; with S2 (unbounded threads), N such requests → OOM. | **bug** | **FIXED.** `read_body` now grows with the bytes that actually arrive (64 KB chunks, bounded by Content-Length) and stops at EOF/timeout — a lying length costs only what it sends. The `OLORIN_MAX_UPLOAD` cap still rejects an over-cap declared length up front. Tests `s1_*` (real socket). |
| S2 | MED | **Unbounded thread-per-connection.** The accept loop spawned a 16 MB-stack thread per connection with no cap / pool; thread-spawn + header read happen *pre-auth*, so an unauthenticated flood (when exposed) exhausts threads/address space. | **bug** | **FIXED.** The accept loop now caps in-flight connections (`OLORIN_MAX_CONN`, default 64) via an atomic counter + a `ConnGuard` that releases the slot on drop (even on panic); beyond the cap it returns `503` and closes without spawning. Loop not unit-reachable → verified by inspection + Pi gate. |

### Robustness checks that PASSED

- **Auth gate logic is solid.** Constant-time token compare (`ct_eq`), fail-closed
  on a non-loopback bind without a token, unparseable bind host treated as
  non-loopback, all three credential channels checked (Bearer / cookie / query),
  bootstrap cookie reflects the *configured* token (no reflection), `HttpOnly` +
  `SameSite=Strict`. Covered by `tests/server_auth_gate.rs`; S3 was a parsing
  panic, not an auth-logic bypass.
- **Body cap + read timeout exist.** `OLORIN_MAX_UPLOAD` rejects an over-cap
  declared length outright, and a 10 s socket read timeout bounds slowloris on
  both the header read and the body read (the thread can't hang forever).
- **Malformed request heads don't bypass or crash dispatch.** Non-UTF-8 request
  bytes return early; missing method/path use safe defaults; `parse_content_length`
  is `unwrap_or(0)` (overflow/garbage → 0).

## Wave-three verdict

The network-facing design is **fail-closed and the auth logic is sound** — the
real exposure only exists in the opt-in `OLORIN_BIND` mode. One genuine
pre-auth defect (S3, a parser panic on malformed input) is **FIXED in-wave**
(small, self-contained, reachable unauthenticated — same posture as wave two's
F3 lock fix). The two availability findings (S1 memory amplification, S2
unbounded threads) are now also **FIXED** — incremental body read + an
accept-loop concurrency cap — in the batch-fix pass.

---

# Wave four — inference limits

Target: the Gemma 4 forward pass under inputs that exceed its fixed context
window (`max_seq_len`, 2048 in production). Oracle = a property: *no input may
panic, OOM, or hang the inference path — an over-window prompt must be clamped
or refused, never crash.* Demonstrated model-free at the KV-cache layer
(`tests/inference_limits.rs`), so it runs in CI without the GGUF.

| # | Severity | Symptom | Triage | Resolution |
|---|----------|---------|--------|------------|
| W1 | **HIGH** | **No context-length guard in `Engine::generate` → over-window prompt panics.** `generate` tokenized and called `forward_batch(&tokens)` with **no check** that `tokens.len() ≤ max_seq_len`. In `KvCache::store_batch` a Global layer writes at `pos = seq_len + t` with no bound; past `max_seq_len` the `kb[cache_off..cache_off + stride]` slice range is out of bounds → **panic** (Rust's bounds check prevents memory corruption, but the thread dies — REPL: process crash; server: connection thread). Only the **narration** callers budgeted via `count_prompt_tokens`; the **chat** path (`router_streaming` `:94`, `router.rs` `:420`) and the **tool-call follow-up** (`router_toolcall` `:75/:77`, raw tool output) were **unguarded**. Pasting a long message (~2048+ tokens, not adversarial) triggered it. | **bug** | **FIXED.** `generate` now applies a central `decode_budget(n_prompt, max_tokens, max_seq_len)`: it refuses a window-filling prompt with a clean `Err` (callers already handle it) and bounds the decode loop to `max_seq_len − n_prompt`, so neither prefill nor decode can write past the cache — for every caller, not just the ones that pre-budget. Pure budget fn unit-tested (`w1_fix_decode_budget_guards_the_window`); raw overflow still documented by the cache-layer tests. Smart history-trimming (so long conversations keep working rather than erroring) is a follow-on. Guard wiring wants the Pi gate (model-gated). |

### Robustness checks that PASSED

- **Bounds check prevents corruption.** The overflow is a *safe* slice-range
  panic, not an out-of-bounds write — no UB, no silent KV corruption.
- **Sliding-window layers can't overflow.** They index `(seq_len + t) %
  window_size`, so they ring-wrap and absorb any token count; the overflow is
  specific to Global layers (which hold the full sequence). Test
  `sliding_window_layer_wraps_and_never_overflows`.
- **Narration is already guarded.** `router_tools` / `router_streaming` skip
  narration with a clear notice when the prompt exceeds
  `NARRATION_MAX_PROMPT_TOKENS` — the budget pattern W1's other callers lack.
- **Empty prompt is handled** — `generate` returns a clean `Err` when the
  tokenized prompt is empty, rather than proceeding.

## Wave-four verdict

One HIGH finding: the context-window budget is enforced per-caller (only
narration does it) instead of centrally, so the chat and tool-follow-up paths
panic on an over-window prompt — triggerable by an ordinary long paste. The
crash is *safe* (a bounds-checked panic, not memory corruption), and the fix is
a small central guard in `generate`, but it carries a policy choice (refuse vs
trim history) and can't be CI-verified without the model, so it is **DEFERRED**
to the batch-fix pass. Scope note: this wave covered the context-overflow /
KV-cache-exhaustion vector; adversarial **tokenizer** fuzzing (pathological byte
sequences, huge single tokens) is a remaining watch-list item, not yet swept.

---

# Deferred backlog (for the prioritized batch-fix pass)

After all four discovery waves, **one fix remains** — it changes the vault's
on-disk write protocol, so it's the last and most carefully gated:

- **vault F1/F2** — atomic append (journal or temp+rename with an fsync barrier
  before the header flip) so a crash mid-append can't lose committed history.

Already fixed: easql `[brackets]` (#1), eaparquet timestamps (#2), eajson null
contract (#3), `--json` cap + eatime buckets (#4) — all shipped in v2.13.0;
vault concurrent lock (F3); server auth-parser panic (S3); inference context
guard (W1); **server body-read S1 + connection-cap S2**. Remaining: vault F1/F2.

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

- **#1 easql brackets** — FIXED this wave (`read_ident` `[…]` arm + regression test).
- **#2 / #3 / #4** — confirmed, root-caused, deferred for a batched fix: each changes output goldens or contract semantics (`count` meaning, `--json` size policy, timestamp rendering) and wants a deliberate design decision rather than a reflexive patch.

## Notes / watch-list (not yet bugs)

- **easql dialect label**: the SQLite Chinook dump reports `dialect=sql`
  (generic) rather than a SQLite-specific tag. SQLite dumps use `[brackets]` +
  `INSERT`, distinguishable from MySQL backticks and Postgres `COPY` — a future
  `sqlite` dialect label is possible. Low priority; the counts are right.

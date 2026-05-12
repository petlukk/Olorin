# Benchmark: eacrunch vs pandas

**Goal**: measure Olorin's `/rune eacrunch` against pandas's
`pd.read_csv() + describe() + value_counts()` on synthetic transaction
CSVs at 10K, 100K, and 1M rows. Both produce equivalent per-column
reports (numeric stats + top-3 text values).

**Environment**: WSL2 Ubuntu 24.04, Ryzen, Python 3.12.3 + pandas 2.3.3.
Olorin built with `cargo build --release` (LTO + opt-level=z).
Olorin invoked with `--strict` so model-load time is excluded from the
comparison — both are doing kernel/library + analyze + print.

**Method**: each tool runs as a single process, end-to-end:
- Olorin: `echo '/rune eacrunch <path>\n/quit' | ./olorin --strict`
- Pandas: `python3 bench_pandas.py <path>`

Wall-clock measured by `/usr/bin/time -v`. Run `bash benchmarks/bench.sh`
from the repo root to reproduce.

## Results

| Rows | File size | eacrunch (--strict) | pandas | Speedup | eacrunch RSS | pandas RSS |
|------|----------:|--------------------:|-------:|--------:|-------------:|-----------:|
| 10,000 | 341 KB | **0.02 s** | 0.56 s | **28×** | 6 MB | 107 MB |
| 100,000 | 3.4 MB | **0.05 s** | 0.58 s | **11×** | 27 MB | 116 MB |
| 1,000,000 | 34 MB | **0.45 s** | 0.87 s | **1.9×** | 236 MB | 212 MB |

(All times include cold-start: process spawn → first byte → process
exit. Full one-shot wall-clock as a user actually experiences it.)

## What's actually being measured

eacrunch (Olorin, `--strict`):
1. Load embedded SIMD kernels from cache (~5 ms first run, ~0 ms cached)
2. Open vault (~5 ms)
3. csv_scan SIMD pass — finds commas + newlines
4. Per-column type sniff (32-row sample)
5. f32_stats SIMD reduction for numeric columns
6. HashMap top-N for text columns
7. Format output, write to stdout

pandas:
1. Python interpreter startup (~50 ms)
2. `import pandas` (~400 ms)
3. `pd.read_csv` — full DataFrame construction
4. `describe()` per numeric column (mean/min/max + percentiles we don't use)
5. `value_counts().head(3)` per text column
6. Format + print

## What this tells us

**Olorin wins decisively at small-to-medium sizes** because pandas pays
a ~500 ms fixed cost (Python startup + pandas import) before any data
work happens. For files under 1M rows, that fixed cost dominates pandas's
runtime. Olorin's kernel extraction is in the single-digit milliseconds.

**At 1M rows the gap narrows to ~2×** because pandas finally amortizes
its startup over real work, and eacrunch's per-row Rust orchestration
(field-walk, f32 parse, HashMap lookups for text top-N) is no longer
dwarfed by the SIMD scan. The structural scan is still ~10× faster than
pandas's CSV parser, but the orchestrator becomes the bottleneck — same
shape as csv_scan vs Rust's `split(',')`.

**Memory**: Olorin uses 1-4× less RAM at small sizes because pandas
always materializes a full DataFrame; eacrunch streams. At 1M rows the
position arrays from `csv_scan` (sized at worst-case `len`) dominate
and Olorin uses slightly *more* RAM than pandas (236 MB vs 212 MB).
This is a known v2 lever — narrower position-array sizing or chunked
processing.

## What this leaves out (be honest)

1. **DuckDB**: not measured. DuckDB's `SELECT count, AVG, MIN, MAX FROM
   read_csv(...)` would likely beat pandas at 1M rows (column-store
   vectorized execution) and might compete with eacrunch. Worth adding.

2. **Output equivalence**: pandas's `describe()` includes percentiles
   (25/50/75) we don't compute. eacrunch outputs unique counts and
   top-3 frequency for text — pandas's `value_counts().head(3)` is the
   closest equivalent. Both produce a useful per-column summary; not
   strictly identical work, but comparable user-facing output.

3. **Narration**: this comparison is kernel-only. Olorin without
   `--strict` adds 5-10 s on x86 (model load) + 5-10 s (narration
   generation) for the LLM-narrated answer. The pandas equivalent is
   "copy `df.describe()` to ChatGPT" — pandas (0.56 s) + ChatGPT
   roundtrip (5-15 s wall-clock including human copy-paste). Hard to
   measure rigorously without a fixed-API harness; not attempted here.

4. **Larger files**: 10M and 100M rows would test where eacrunch's
   SIMD truly shines vs where pandas's columnar internals would catch
   up. Out of scope for this run; interesting follow-up.

## Reproducing

```bash
# From repo root, after `cargo build --release`:
bash benchmarks/bench.sh
```

Generates synthetic CSVs in `/tmp/olorin_bench/` (idempotent — keeps
files between runs). Output format matches the table above. Edit
`benchmarks/gen_synthetic.py` to change schema; edit
`benchmarks/bench.sh` to add new sizes or tools.

---

# Benchmark: eatime vs awk vs pandas

**Goal**: measure `/rune eatime` against two everyday alternatives for
"bucket log timestamps by hour-of-day" — `awk` (canonical Unix tool)
and `pandas` (the Python default for log analysis). All three produce
the same 24-slot hour-of-day histogram from a synthetic log file with
ISO-8601 timestamps prefixed to every line.

**Environment**: WSL2 Ubuntu 24.04, Ryzen 7700X, Python 3.12 + pandas
2.3, GNU awk 5.x. Olorin built with `cargo build --release` and
invoked via `--strict` so the LLM is excluded from the wall-clock.

**Method**: each tool runs as a single process, end-to-end. Wall-clock
and peak RSS measured by `/usr/bin/time -v`. Run
`bash benchmarks/bench_eatime.sh` from the repo root to reproduce.

## Results

| Tool   | Size    |   Wall time |    Peak RSS |
|--------|---------|------------:|------------:|
| eatime | 10 MB   |  **0.04 s** |       15 MB |
| awk    | 10 MB   |    0.15 s   |        4 MB |
| pandas | 10 MB   |    1.15 s   |      110 MB |
| eatime | 100 MB  |  **0.10 s** |      109 MB |
| awk    | 100 MB  |    1.48 s   |        4 MB |
| pandas | 100 MB  |    0.52 s   |      111 MB |

| Comparison        | 10 MB     | 100 MB    |
|-------------------|-----------|-----------|
| eatime vs awk     | **3.8×**  | **14.8×** |
| eatime vs pandas  | **28.8×** |  **5.2×** |

The fixtures: 145,635 lines at 10 MB and 1,456,355 lines at 100 MB,
every line beginning with `2026-MM-DDTHH:MM:SS` plus realistic log
content. `bench_eatime.sh` generates them deterministically.

## What this tells us

**Against awk, the win grows with size.** awk is single-threaded
scalar — every byte is touched by the regex engine. eatime's
structural-anchor SIMD filter (3 `.==` lane masks per 16-byte chunk)
trims ~95% of positions before any scalar work. At 100 MB the gap is
~15×; at 10 MB it's smaller because both tools spend a similar fixed
overhead on process startup.

**Against pandas, the curve flips.** pandas pays ~500 ms of fixed
cost (Python startup + pandas import + `pd.to_datetime` setup) before
any data work. At 10 MB that startup dominates and eatime wins ~29×.
At 100 MB pandas finally amortizes its setup, narrowing the gap to
~5×. Still a real win, but the curve says "pandas catches up on big
files" — same shape as the eacrunch vs pandas comparison above.

**Memory**: awk is constant ~4 MB regardless of size (true streaming).
eatime uses ~file-size memory because `open_capped` reads the file
into a `Vec<u8>` rather than mmap-streaming — known follow-up lever.
pandas materializes the full Series + parsed datetime column, so
RSS is ~110 MB at both sizes.

## What this leaves out

1. **DuckDB** could likely beat pandas at 100 MB with its columnar
   vectorized execution. Not measured. Same caveat as the eacrunch
   bench above.
2. **Sort stability**: awk's hash iteration order is undefined
   without `gawk --posix`; the bench script pipes through `sort` to
   match eatime's deterministic Mon..Sun / 00..23 ordering. Output
   contents identical; format strings differ slightly.
3. **Kernel-only throughput** (not end-to-end): the
   `timestamp_scan` kernel measured separately at **6.34 GB/s on
   Ryzen SSE2** and **1.80 GB/s on Pi 5 NEON** (see
   `benchmarks/timestamp_scan_bench.c`). The end-to-end wall-clock
   above includes process spawn, file read into `Vec<u8>`, JSON
   serialization, and writeout — the kernel itself is ~6× faster
   than the end-to-end pipeline implies.

## A note on eadiff

There is no comparable Unix tool for "compute a structural delta
between two prior `--json` rune outputs." The closest competitor is a
hand-rolled Python script that loads two JSON files and walks them.
At realistic rune-output sizes (~1 KB each), wall-clock is dominated
by process startup on both sides (~25 ms olorin `--strict`, ~50 ms
Python). The eadiff value-add isn't speed; it's "deterministic
structural delta with zero dependencies on the consumer side, output
is itself a chainable RuneOutput." For users who want to chain rune
runs without writing custom Python per pipeline step, eadiff is the
one-line answer; speed is incidental.

## Reproducing the eatime bench

```bash
# From repo root, after `cargo build --release`:
bash benchmarks/bench_eatime.sh
```

Generates synthetic logs in `/tmp/olorin_bench/` (idempotent). Edit
`benchmarks/gen_log_fixture.py` to change shape; edit
`benchmarks/bench_eatime.sh` to add sizes or tools.

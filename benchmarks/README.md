# benchmarks/

Reproducible measurements of Olorin's runes against external tooling.

## Currently included

- **`gen_synthetic.py`** — generates a synthetic transaction CSV at a
  given row count (deterministic seed; same input every run).
- **`bench_pandas.py`** — pandas equivalent of `/rune eacrunch`:
  `pd.read_csv()` + per-column type detection + numeric stats +
  text-column top-3 frequency. Designed to produce a comparable
  output report.
- **`bench.sh`** — driver that generates 10K/100K/1M-row CSVs and
  runs both tools end-to-end, capturing wall-clock + peak RSS.
- **`results.md`** — measured numbers + commentary + caveats.

## Running

```bash
# From repo root:
cargo build --release
bash benchmarks/bench.sh
```

Outputs a markdown-friendly table to stdout. Inputs cached in
`/tmp/olorin_bench/` between runs.

## Adding a new tool

1. Write a runner script (Python, shell, whatever) that takes a path
   and produces an analogous report on stdout.
2. Add a `measure "your-tool" <command>` line to `bench.sh`.
3. Re-run; results print alongside existing tools.

## Open follow-ups

- DuckDB comparison (`SELECT count, AVG, MIN, MAX FROM read_csv(...)`)
- 10M / 100M-row tests (currently capped at 1M)
- Narration TTFT comparison (`olorin` non-strict vs pandas+ChatGPT)
- JSONL benchmark (eajson vs `jq` + Python streaming)

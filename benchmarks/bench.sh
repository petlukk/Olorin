#!/usr/bin/env bash
# Driver for the eacrunch vs pandas comparison.
#
# Generates synthetic transaction CSVs at three sizes (10K, 100K, 1M rows)
# and runs each tool against each, capturing wall-clock and peak RSS via
# /usr/bin/time -v. Output is a markdown-friendly table on stdout.
#
# Run from the repo root:
#   bash benchmarks/bench.sh
#
# Requires: python3, pandas, ./target/release/olorin (built strict-aware).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH_DIR="$REPO_ROOT/benchmarks"
OLORIN="$REPO_ROOT/target/release/olorin"

if [[ ! -x "$OLORIN" ]]; then
    echo "olorin binary not found at $OLORIN — build first with cargo build --release" >&2
    exit 1
fi

# Generate inputs (only if missing — keep them stable across runs).
mkdir -p /tmp/olorin_bench
for n in 10000 100000 1000000; do
    out="/tmp/olorin_bench/tx_${n}.csv"
    if [[ ! -f "$out" ]]; then
        python3 "$BENCH_DIR/gen_synthetic.py" "$n" "$out"
    fi
done

# Time helper: prints wall-clock seconds and max RSS in KB on one line.
# Uses /usr/bin/time -v which writes to stderr; we redirect to a temp.
measure() {
    local label="$1"; shift
    local tmp; tmp=$(mktemp)
    /usr/bin/time -v "$@" >/dev/null 2>"$tmp"
    local wall rss
    wall=$(awk -F': ' '/Elapsed.*wall clock/ {print $NF}' "$tmp")
    rss=$(awk -F': ' '/Maximum resident set size/ {print $NF}' "$tmp")
    rm -f "$tmp"
    printf "%-28s  wall=%-10s  rss=%s KB\n" "$label" "$wall" "$rss"
}

echo "## eacrunch vs pandas — synthetic transactions CSV"
echo
echo "Cold-start one-shot (full process: import/load + analyze + print)."
echo "wall clock = mm:ss.ms or h:mm:ss; rss = peak resident memory (KB)."
echo

for n in 10000 100000 1000000; do
    csv="/tmp/olorin_bench/tx_${n}.csv"
    bytes=$(stat -c %s "$csv" 2>/dev/null || stat -f %z "$csv")
    echo "### ${n} rows (${bytes} bytes)"
    echo

    # Olorin in strict mode — kernel only, no model load, no narration.
    measure "olorin --strict eacrunch" \
        bash -c "echo -e '/rune eacrunch ${csv}\n/quit' | $OLORIN --strict"

    # Pandas equivalent — pd.read_csv + describe + value_counts.
    measure "pandas (read + describe)" \
        python3 "$BENCH_DIR/bench_pandas.py" "$csv"

    echo
done

#!/usr/bin/env bash
# Driver for the eatime vs awk vs pandas comparison.
#
# Generates synthetic logs at 10 MB and 100 MB, runs each tool
# producing a 24-slot hour-of-day histogram, captures wall-clock and
# peak RSS via /usr/bin/time -v. Markdown-friendly table on stdout.
#
# Run from the repo root:
#   bash benchmarks/bench_eatime.sh
#
# Requires: python3, pandas, ./target/release/olorin (built with
# `cargo build --release` — strict mode disables the LLM so model
# load is excluded from the comparison).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH_DIR="$REPO_ROOT/benchmarks"
OLORIN="$REPO_ROOT/target/release/olorin"

if [[ ! -x "$OLORIN" ]]; then
    echo "olorin binary not found at $OLORIN — build first" >&2
    exit 1
fi

mkdir -p /tmp/olorin_bench
for mb in 10 100; do
    out="/tmp/olorin_bench/log_${mb}mb.log"
    if [[ ! -f "$out" ]]; then
        python3 "$BENCH_DIR/gen_log_fixture.py" "$mb" "$out"
    fi
done

run() {
    local label="$1"; shift
    local size="$1"; shift
    local out_file
    out_file="$(mktemp)"
    /usr/bin/time -v -o "$out_file" "$@" >/dev/null 2>&1 || true
    local secs rss
    secs=$(awk '/Elapsed \(wall/ { split($NF, a, ":"); if (length(a)==3) print a[1]*3600+a[2]*60+a[3]; else print a[1]*60+a[2] }' "$out_file")
    rss=$(awk '/Maximum resident set size/ { print $NF/1024 }' "$out_file")
    rm -f "$out_file"
    printf "| %-8s | %-7s | %8.3f s | %6.0f MB |\n" "$label" "$size" "$secs" "$rss"
}

echo
echo "## eatime vs awk vs pandas — hour-of-day histogram"
echo
echo "| Tool     | Size    | Wall time  |   RSS    |"
echo "|----------|---------|-----------:|---------:|"
for mb in 10 100; do
    log="/tmp/olorin_bench/log_${mb}mb.log"
    size_str="${mb} MB"

    # eatime via REPL stdin (--strict skips model load). After v0.9.4
    # the JSON answer is emitted verbatim so stdout is parseable.
    run "eatime"  "$size_str"  bash -c "echo '/rune eatime --json $log' | $OLORIN --strict"

    # awk: extract HH from leading YYYY-MM-DDTHH:..., count, sort, print.
    run "awk"     "$size_str"  bash -c "awk 'match(\$0, /^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}/) { hour[substr(\$0,12,2)]++ } END { for (h in hour) printf \"%s:00 %d\\n\", h, hour[h] }' $log | sort"

    # pandas: read_csv → str.extract → to_datetime → dt.hour.value_counts.
    run "pandas"  "$size_str"  python3 "$BENCH_DIR/bench_eatime_pandas.py" "$log"
done
echo

#!/usr/bin/env bash
# Run every wave-one differential against the fetched real datasets.
# Needs: a release olorin binary, python3 with pandas + pyarrow (sqlite3,
# json, gzip are stdlib). Fetch the data first with ./fetch.sh.
#
# Usage: run_all.sh [path-to-olorin-binary]   (default ./target/release/olorin)
set -uo pipefail
cd "$(dirname "$0")"
BIN="${1:-../../target/release/olorin}"
DATA="data"

if [ ! -x "$BIN" ]; then echo "no olorin binary at $BIN (build --release first)"; exit 2; fi
if [ ! -d "$DATA" ] || [ -z "$(ls -A "$DATA" 2>/dev/null)" ]; then
  echo "no data — run ./fetch.sh first"; exit 2
fi

# CSV view of the parquet for eacrunch (one-time, ~300 MB).
if [ -f "$DATA/yellow_tripdata_2023-01.parquet" ] && [ ! -f "$DATA/yellow_tripdata_2023-01.csv" ]; then
  echo "converting parquet -> csv for eacrunch …"
  python3 -c "import pyarrow.parquet as pq; pq.read_table('$DATA/yellow_tripdata_2023-01.parquet').to_pandas().to_csv('$DATA/yellow_tripdata_2023-01.csv', index=False)"
fi

fail=0
for rune in easql eaparquet eacrunch eajson; do
  echo "======================================================================"
  echo "  $rune"
  echo "======================================================================"
  python3 "diff_${rune}.py" "$BIN" "$DATA" || fail=1
  echo
done

echo "======================================================================"
echo "  ealog (3 real formats) — inline oracle"
echo "======================================================================"
python3 ealog_check.py "$BIN" "$DATA" || fail=1

echo
echo "======================================================================"
echo "  eatime (NASA CLF, scale) — inline oracle"
echo "======================================================================"
python3 eatime_check.py "$BIN" "$DATA" || fail=1

echo
[ "$fail" = 0 ] && echo "ALL CLEAN" || echo "FINDINGS — see output above and FINDINGS.md"
exit $fail

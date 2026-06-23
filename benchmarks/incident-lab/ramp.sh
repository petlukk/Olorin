#!/bin/bash
# No-trigger incident: the 500 rate climbs gradually (a leak/degradation) with NO
# deploy line — exercises the palantír rate-anomaly detector live.
#
#   ramp.sh [DURATION=60] [RATE=30] [WORKERS=6] [RAMP=40]
set -e
BASE="$(cd "$(dirname "$0")" && pwd)"; cd "$BASE"
DUR="${1:-60}"; RATE="${2:-30}"; WORKERS="${3:-6}"; RAMP="${4:-40}"
rm -f access.log app.log db.log db.syslog app.jsonl pino.jsonl syslog.log apache.log hdfs.log system.log deploy.log deploy.syslog
echo v1.0.0 > version
RAMP="$RAMP" python3 app.py & SVC=$!
trap 'kill $SVC 2>/dev/null || true' EXIT
sleep 1
echo "[ramp] no-trigger 500 ramp over ${RAMP}s, ${DUR}s traffic @ ~${RATE} req/s..."
python3 loadgen.py "$DUR" "$RATE" "$WORKERS"
kill $SVC 2>/dev/null || true; trap - EXIT
echo "[ramp] done:"; wc -l system.log access.log
echo "--- app ERRORs in system.log: $(grep -c ' ERROR ' system.log) | deploy lines: $(grep -c deploy system.log) ---"

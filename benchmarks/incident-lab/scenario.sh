#!/bin/bash
# Run one live incident: healthy traffic -> deploy -> (cascade | nothing).
#
#   scenario.sh [HEALTHY=25] [BROKEN=25] [RATE=30] [WORKERS=6] [LAG=20] [KIND=bad]
#
# KIND=bad  deploys v1.1.0 (the str-slice bug) -> db-pool lead -> lagged 500
#           cascade. The headline: a palantír on system.log predicts at the deploy.
# KIND=good deploys v1.0.1 (still healthy) -> the control. Nothing should fire;
#           a watcher must NOT raise a false alarm on a clean deploy.
set -e
BASE="$(cd "$(dirname "$0")" && pwd)"; cd "$BASE"
HEALTHY="${1:-25}"; BROKEN="${2:-25}"; RATE="${3:-30}"; WORKERS="${4:-6}"; LAG="${5:-20}"; KIND="${6:-bad}"
TARGET=v1.1.0; [ "$KIND" = good ] && TARGET=v1.0.1

rm -f access.log app.log db.log db.syslog app.jsonl pino.jsonl syslog.log apache.log hdfs.log system.log deploy.log deploy.syslog
echo v1.0.0 > version

BUG_LAG="$LAG" python3 app.py & SVC=$!
trap 'kill $SVC 2>/dev/null || true' EXIT
sleep 1
echo "[scenario:$KIND] healthy traffic on v1.0.0 (${HEALTHY}s @ ~${RATE} req/s, diurnal)..."
python3 loadgen.py "$HEALTHY" "$RATE" "$WORKERS"
echo "[scenario:$KIND] >>> deploy $TARGET <<<"
./deploy.sh "$TARGET"
echo "[scenario:$KIND] traffic after the deploy (${BROKEN}s)..."
python3 loadgen.py "$BROKEN" "$RATE" "$WORKERS"
kill $SVC 2>/dev/null || true; trap - EXIT
echo "[scenario:$KIND] done:"
wc -l access.log $([ -f db.log ] && echo db.log) system.log
db_errs=$(grep -c ERROR db.log 2>/dev/null || true)
echo "--- 5xx in access.log: $(grep -cE '\" 5[0-9][0-9] ' access.log || true) | db errors: ${db_errs:-0} | lag=${LAG}s kind=${KIND} ---"

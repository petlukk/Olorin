#!/bin/bash
# Record a deploy and flip the running version (app.py reads it live). The deploy
# lines match logfmt.deploy_lines so the live and simulated logs are identical in
# shape — the ISO deploy.log + unified system.log (real era) and BSD deploy.syslog
# (syslog era, which needs its own same-era trigger to anchor a syslog incident).
set -e
BASE="$(cd "$(dirname "$0")" && pwd)"
NEW="${1:?usage: deploy.sh <version>}"
OLD="$(cat "$BASE/version" 2>/dev/null || echo none)"
TS="$(date -u +%Y-%m-%dT%H:%M:%S)"
SHA="$(head -c3 /dev/urandom | od -An -tx1 | tr -d ' \n')"
echo "${TS}Z deploy leaderboard $OLD -> $NEW service=incident-lab actor=ci commit=$SHA" >> "$BASE/deploy.log"
echo "$TS INFO deploy leaderboard $OLD -> $NEW commit=$SHA" >> "$BASE/system.log"
STS="$(date -u +"%b %d %H:%M:%S")"
echo "$STS pi-host deploy[$$]: released $OLD -> $NEW commit=$SHA" >> "$BASE/deploy.syslog"
echo "$NEW" > "$BASE/version"
echo "deployed $OLD -> $NEW"

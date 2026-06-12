#!/usr/bin/env bash
# Fetch every robustness-campaign dataset into ./data/ (gitignored).
# Public, stable, direct downloads — no keys, no logins. See SOURCES.md.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data
cd data

# name  url
fetch() {
  local name="$1" url="$2"
  if [ -f "$name" ]; then
    echo "have   $name"
    return
  fi
  echo "fetch  $name"
  curl -fsSL --retry 3 -o "$name.part" "$url"
  mv "$name.part" "$name"
}

# 1. NASA-HTTP (ITA / LBNL) — Apache CLF, July 1995
fetch NASA_access_log_Jul95.gz "https://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz"
[ -f NASA_access_log_Jul95 ] || gunzip -k NASA_access_log_Jul95.gz

# 2. GH Archive — one hour of GitHub events, JSONL
fetch 2024-01-01-15.json.gz "https://data.gharchive.org/2024-01-01-15.json.gz"
[ -f gharchive_2024-01-01-15.jsonl ] || zcat 2024-01-01-15.json.gz > gharchive_2024-01-01-15.jsonl

# 3. NYC TLC Yellow Taxi — Parquet, Jan 2023
fetch yellow_tripdata_2023-01.parquet "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"

# 4. Chinook — SQL dumps, two dialects
fetch Chinook_Sqlite.sql     "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
fetch Chinook_PostgreSql.sql "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_PostgreSql.sql"

# 5. Loghub — three real log formats
fetch Linux_2k.log  "https://raw.githubusercontent.com/logpai/loghub/master/Linux/Linux_2k.log"
fetch Apache_2k.log "https://raw.githubusercontent.com/logpai/loghub/master/Apache/Apache_2k.log"
fetch HDFS_2k.log   "https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log"

echo
echo "fetched into $(pwd):"
ls -lh

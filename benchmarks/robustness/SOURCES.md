# Robustness campaign — real-world data sources

Wave one tests every rune against **real, messy, public files** — not
synthetic fixtures — and diffs the output against an **independent** ground-truth
tool. The point is to catch *spec* errors, not just implementation drift: a
differential harness written from the rune's own spec can only confirm the rune
matches itself (the v2.12.0 eacorrelate disjoint-era bug passed 15/15 against
exactly such an oracle and was found instead by a real NASA log). So every
oracle here is a different, independently-maintained implementation — pandas,
pyarrow, SQLite, the Python stdlib — never a re-statement of Olorin's algorithm.

We hunt three failure classes: **panics** (crash on valid input), **silent-wrong**
(plausible output that disagrees with ground truth), and **OOM / runaway** (a
file that should summarize in MB and seconds instead exhausts memory or time).

## Replicate

```bash
# fetch every dataset into benchmarks/robustness/data/ (gitignored, ~250 MB total)
benchmarks/robustness/fetch.sh

# run the differentials (needs python3 with pandas + pyarrow; sqlite3/json are stdlib)
benchmarks/robustness/run_all.sh
```

No API keys, no logins — every source below is a public, stable, direct download.

## Sources

| # | Dataset | File | Format | Runes exercised | Ground-truth oracle |
|---|---------|------|--------|-----------------|---------------------|
| 1 | NASA-HTTP server log (Jul 1995) | `NASA_access_log_Jul95.gz` | Apache CLF | eatime, eacorrelate | Python `datetime` (independent CLF parse) |
| 2 | GH Archive (one hour of GitHub events) | `2024-01-01-15.json.gz` | JSON Lines | eajson, eatime | Python `json` (stdlib) |
| 3 | NYC TLC Yellow Taxi (Jan 2023) | `yellow_tripdata_2023-01.parquet` | Parquet | eaparquet, eacrunch | pyarrow (metadata), pandas (CSV stats) |
| 4 | Chinook sample DB | `Chinook_Sqlite.sql`, `Chinook_PostgreSql.sql` | SQL dump | easql | Python `sqlite3` (load + `COUNT(*)`) |
| 5 | Loghub Linux/Apache/HDFS logs | `Linux_2k.log`, `Apache_2k.log`, `HDFS_2k.log` | syslog / access / app | ealog, eatime | Python regex counts |

### 1. NASA-HTTP — The Internet Traffic Archive (LBNL)
- **URL:** <https://ita.ee.lbl.gov/traces/NASA_access_log_Jul95.gz>
- **What:** every HTTP request to NASA's Kennedy Space Center web server, 1–31 Jul 1995 (~1.9 M lines, ~20 MB uncompressed). Apache Common Log Format with `[dd/Mon/yyyy:hh:mm:ss -0500]` timestamps.
- **Why it bites:** real CLF (the format `eatime --format clf` and the file-drop sniffer must handle), a 31-day span (multi-day x-axis labelling), and genuine diurnal traffic with real spikes. This is the file that surfaced the eacorrelate disjoint-era bug.
- **License:** freely available for research; see the ITA terms on the LBNL site.

### 2. GH Archive
- **URL:** <https://data.gharchive.org/2024-01-01-15.json.gz> (any `YYYY-MM-DD-H.json.gz`)
- **What:** every public GitHub event in one UTC hour as newline-delimited JSON (~100 k+ deeply-nested objects, ~20 MB gz → ~100 MB+). `created_at` is ISO-8601.
- **Why it bites:** deeply nested + heterogeneous JSON (the per-key sniffer must not choke on missing/variant keys), large object count, and ISO timestamps for eatime. Already referenced in the README's `--json` example.
- **License:** GitHub public event data, CC-BY-4.0 (gharchive.org).

### 3. NYC TLC Trip Record Data
- **URL:** <https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet>
- **What:** ~3 M yellow-taxi trips, Jan 2023 (~50 MB Parquet). 19 columns: timestamps, floats, ints, strings.
- **Why it bites:** a real Parquet footer with per-column statistics across several row groups (eaparquet reads min/max/null_count from it); converted to CSV it stress-tests eacrunch's numeric stats at 3 M rows. Columns include nulls and negative fares (real data is dirty).
- **License:** NYC Taxi & Limousine Commission, public domain.

### 4. Chinook database
- **URL (SQLite):** <https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql>
- **URL (PostgreSQL):** <https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_PostgreSql.sql>
- **What:** the canonical sample music store — 11 tables, 15 607 rows. Two dialects so easql's dialect detection and per-table attribution are checked on both.
- **Why it bites:** the SQLite dump uses `INSERT INTO "Table" VALUES(...)`, the Postgres dump uses `COPY` blocks — different shapes the same rune must count. Ground truth = load into an actual SQLite engine and `COUNT(*)` per table.
- **License:** MIT (github.com/lerocha/chinook-database).

### 5. Loghub log collection (LogPAI)
- **URLs:**
  - <https://raw.githubusercontent.com/logpai/loghub/master/Linux/Linux_2k.log>
  - <https://raw.githubusercontent.com/logpai/loghub/master/Apache/Apache_2k.log>
  - <https://raw.githubusercontent.com/logpai/loghub/master/HDFS/HDFS_2k.log>
- **What:** 2 000-line samples of real production logs — Linux syslog, Apache error log, Hadoop HDFS. Each carries genuine severity tokens and timestamps in a *different* layout.
- **Why it bites:** ealog's word-bounded severity counting must agree with a regex oracle across three real formats (not the tidy synthetic ladder), and eatime must sniff three different timestamp grammars. The full Loghub corpus (logpai/loghub) has 16 systems if deeper coverage is wanted.
- **License:** research use; cite LogPAI/loghub (the collection is widely used in log-analysis papers).

## Findings log

Each run appends to `benchmarks/robustness/FINDINGS.md` — one row per
divergence with the file, the rune, what Olorin said, what the oracle said, and
the triage (real bug / oracle artifact / expected). Confirmed bugs become E2E
regression fixtures so they can never silently return.

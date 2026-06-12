#!/usr/bin/env python3
"""easql differential: per-table row counts from `olorin rune easql --json`
vs an ACTUAL SQLite engine that executes the dump and runs COUNT(*).

This is the strongest oracle in the suite — not a re-implementation of
easql's sweep, but a real database loading the same bytes. Run on the
SQLite Chinook dump (the Postgres dump uses COPY, which sqlite3 can't
execute, so it's checked structurally instead).

Usage: diff_easql.py <olorin-binary> <data-dir>
"""
import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

# Runes only read under ~ or /tmp (the file-drop allowlist; the CLI inherits
# it). Stage each dataset into /tmp, exactly as a real web/REPL file-drop does.
STAGE = Path("/tmp/olorin_robustness")


def stage(path):
    STAGE.mkdir(exist_ok=True)
    dst = STAGE / Path(path).name
    if not dst.exists():
        shutil.copy(path, dst)
    return dst


def olorin_easql(binary, path):
    out = subprocess.run(
        [binary, "rune", "easql", "--json", str(stage(path))],
        capture_output=True, text=True, timeout=120,
    )
    if out.returncode != 0:
        raise SystemExit(f"olorin exited {out.returncode} on {path}:\n{out.stderr}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def sqlite_truth(sql_path):
    """Execute the dump in a real in-memory SQLite DB, return {table: rows}."""
    con = sqlite3.connect(":memory:")
    script = Path(sql_path).read_text(encoding="utf-8", errors="replace")
    con.executescript(script)
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [r[0] for r in cur.fetchall()]
    counts = {}
    for t in tables:
        cur.execute(f'SELECT COUNT(*) FROM "{t}"')
        counts[t] = cur.fetchone()[0]
    con.close()
    return counts


def main():
    binary, data = sys.argv[1], Path(sys.argv[2])
    findings = []

    sqlite_dump = data / "Chinook_Sqlite.sql"
    truth = sqlite_truth(sqlite_dump)
    got = olorin_easql(binary, sqlite_dump)

    # Normalize identifier quoting on both sides so the diff compares row
    # counts, not delimiter style ([Album] vs "Album" vs `Album` vs Album).
    def unquote(s):
        return s.strip('[]"`')

    olorin_tables = {unquote(c["name"]): c["count"] for c in got.get("categories", [])}
    truth = {unquote(k): v for k, v in truth.items()}

    truth_total = sum(truth.values())
    olorin_total = got["totals"]["rows"]
    print(f"Chinook_Sqlite.sql: olorin dialect={got.get('source',{}).get('format','?')} "
          f"tables olorin={len(olorin_tables)} sqlite={len(truth)} "
          f"rows olorin={olorin_total} sqlite={truth_total}")

    all_tables = sorted(set(truth) | set(olorin_tables))
    for t in all_tables:
        o = olorin_tables.get(t)
        s = truth.get(t)
        mark = "ok" if o == s else "MISMATCH"
        if o != s:
            findings.append((t, o, s))
        print(f"  {t:<20} olorin={str(o):>8} sqlite={str(s):>8}  {mark}")

    print()
    if findings:
        print(f"FINDINGS ({len(findings)}):")
        for t, o, s in findings:
            print(f"  table {t}: olorin={o} sqlite={s} (delta {None if o is None or s is None else o - s})")
        sys.exit(1)
    print("easql vs SQLite engine: 0 mismatches")


if __name__ == "__main__":
    main()

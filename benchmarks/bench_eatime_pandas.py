#!/usr/bin/env python3
"""Pandas equivalent of `olorin /rune eatime`: read a log file, extract
the leading ISO-8601 timestamp from each line, bucket by hour-of-day,
print a 24-slot histogram. Matches eatime's --bucket hour semantics.

Usage:
    python3 bench_eatime_pandas.py <path>
"""
import sys
import pandas as pd

def main() -> None:
    path = sys.argv[1]
    # Read all lines as a single Series, extract the timestamp prefix,
    # parse to datetime, bucket by hour-of-day.
    s = pd.read_csv(
        path, header=None, sep="\n", names=["line"],
        engine="python", quoting=3,  # QUOTE_NONE — log content can contain "
    )["line"]
    ts = pd.to_datetime(
        s.str.extract(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})", expand=False),
        errors="coerce",
    )
    counts = ts.dt.hour.value_counts().sort_index()
    for h in range(24):
        print(f"{h:02d}:00 {int(counts.get(h, 0))}")

if __name__ == "__main__":
    main()

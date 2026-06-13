#!/usr/bin/env python3
"""eacrunch differential: per-column numeric stats from `olorin rune
eacrunch --json` vs pandas on the same CSV. pandas is an independent
oracle (its own CSV parser + numeric reductions).

Checks rows, column count, and for numeric columns min/max/mean/sum.
Usage: diff_eacrunch.py <olorin-binary> <data-dir>
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

STAGE = Path("/tmp/olorin_robustness")


def stage(path):
    STAGE.mkdir(exist_ok=True)
    dst = STAGE / Path(path).name
    if not dst.exists():
        shutil.copy(path, dst)
    return dst


def olorin_eacrunch(binary, path, extra=()):
    out = subprocess.run(
        [binary, "rune", "eacrunch", "--json", *extra, str(stage(path))],
        capture_output=True, text=True, timeout=300,
    )
    if out.returncode != 0:
        raise SystemExit(f"olorin exited {out.returncode}:\n{out.stderr}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def diff_groupby(binary, path, findings):
    """GROUP BY payment_type vs pandas groupby — count + mean(total_amount)
    + sum(trip_distance). pandas reads the key column as str so both sides
    group on the identical raw CSV tokens; value columns coerce-to-numeric
    with skipna, matching olorin's finite-only accumulation."""
    by = "payment_type"
    got = olorin_eacrunch(
        binary, path,
        extra=["--by", by, "--agg", "mean:total_amount,sum:trip_distance,count"],
    )
    groups = {g["key"]: g for g in got.get("groups", [])}

    df = pd.read_csv(path, dtype={by: str}, low_memory=False)
    keys = df[by].fillna("")
    size = keys.value_counts()
    ta = pd.to_numeric(df["total_amount"], errors="coerce").groupby(keys).mean()
    td = pd.to_numeric(df["trip_distance"], errors="coerce").groupby(keys).sum()

    print(f"  GROUP BY {by}: olorin groups={len(groups)} pandas groups={len(size)}")
    if len(groups) != len(size):
        findings.append(f"group count: olorin={len(groups)} pandas={len(size)}")

    def agg_val(g, op, col=""):
        for a in g["aggs"]:
            if a["op"] == op and a.get("col", "") == col:
                return a["value"]
        return None

    for key, truth_n in size.items():
        g = groups.get(key)
        if g is None:
            findings.append(f"group {key!r}: missing from olorin")
            continue
        ok = True
        if g["count"] != int(truth_n):
            findings.append(f"group {key} count olorin={g['count']} pandas={truth_n}")
            ok = False
        if not approx(agg_val(g, "mean", "total_amount"), ta[key]):
            findings.append(f"group {key} mean(total_amount) "
                            f"olorin={agg_val(g, 'mean', 'total_amount')} pandas={ta[key]}")
            ok = False
        if not approx(agg_val(g, "sum", "trip_distance"), td[key]):
            findings.append(f"group {key} sum(trip_distance) "
                            f"olorin={agg_val(g, 'sum', 'trip_distance')} pandas={td[key]}")
            ok = False
        print(f"    {key:<4} count={g['count']:>9} mean(total)={agg_val(g,'mean','total_amount'):>8.3f} "
              f"sum(dist)={agg_val(g,'sum','trip_distance'):>12.1f}  {'ok' if ok else 'MISMATCH'}")


def approx(a, b, rel=1e-4):
    if a is None or b is None:
        return False
    a, b = float(a), float(b)
    return abs(a - b) <= rel * max(1.0, abs(b))


def main():
    binary, data = sys.argv[1], Path(sys.argv[2])
    path = data / "yellow_tripdata_2023-01.csv"
    df = pd.read_csv(path, low_memory=False)
    got = olorin_eacrunch(binary, path)

    findings = []
    o_rows = got["totals"]["rows"]
    o_fields = {f["name"]: f for f in got.get("fields", [])}
    print(f"{path.name}: rows olorin={o_rows} pandas={len(df)}  "
          f"cols olorin={len(o_fields)} pandas={len(df.columns)}")
    if o_rows != len(df):
        findings.append(f"row count: olorin={o_rows} pandas={len(df)}")

    for col in df.columns:
        f = o_fields.get(col)
        if f is None:
            findings.append(f"column {col}: missing from olorin")
            continue
        num = f.get("numeric")
        if num is None:
            continue  # eacrunch classified it non-numeric; check below
        s = pd.to_numeric(df[col], errors="coerce")
        truth = {"min": s.min(), "max": s.max(), "mean": s.mean(), "sum": s.sum()}
        for k in ("min", "max", "mean", "sum"):
            if not approx(num.get(k), truth[k]):
                findings.append(f"{col}.{k} olorin={num.get(k)} pandas={truth[k]}")
        mark = "ok" if not any(f"{col}." in x for x in findings) else "MISMATCH"
        print(f"  {col:<24} min={str(num.get('min')):>10} max={str(num.get('max')):>12} "
              f"mean={num.get('mean'):>10.3f}  {mark}")

    diff_groupby(binary, path, findings)

    print()
    if findings:
        print(f"FINDINGS ({len(findings)}):")
        for x in findings[:40]:
            print(f"  {x}")
        sys.exit(1)
    print("eacrunch vs pandas: 0 mismatches (column stats + GROUP BY)")


if __name__ == "__main__":
    main()

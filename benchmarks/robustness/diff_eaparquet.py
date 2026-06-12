#!/usr/bin/env python3
"""eaparquet differential: per-column metadata from `olorin rune eaparquet
--json` vs pyarrow reading the same Parquet footer. pyarrow is the
reference Parquet implementation — a genuinely independent oracle.

Checks: row count, column count, and per-column min/max/null_count
aggregated across row groups (what eaparquet reads from the footer).

Usage: diff_eaparquet.py <olorin-binary> <data-dir>
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

STAGE = Path("/tmp/olorin_robustness")


def stage(path):
    STAGE.mkdir(exist_ok=True)
    dst = STAGE / Path(path).name
    if not dst.exists():
        shutil.copy(path, dst)
    return dst


def olorin_eaparquet(binary, path):
    out = subprocess.run(
        [binary, "rune", "eaparquet", "--json", str(stage(path))],
        capture_output=True, text=True, timeout=180,
    )
    if out.returncode != 0:
        raise SystemExit(f"olorin exited {out.returncode}:\n{out.stderr}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def pyarrow_truth(path):
    md = pq.read_metadata(path)
    cols = {}
    for c in range(md.num_columns):
        name = md.schema.column(c).name
        mn, mx, nulls, have_stats = None, None, 0, False
        for rg in range(md.num_row_groups):
            st = md.row_group(rg).column(c).statistics
            if st is None:
                continue
            have_stats = True
            nulls += st.null_count or 0
            if st.has_min_max:
                mn = st.min if mn is None else min(mn, st.min)
                mx = st.max if mx is None else max(mx, st.max)
        cols[name] = {"min": mn, "max": mx, "nulls": nulls, "stats": have_stats}
    return md.num_rows, md.num_columns, cols


def approx(a, b):
    if a is None or b is None:
        return a == b
    try:
        return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return str(a) == str(b)


def main():
    binary, data = sys.argv[1], Path(sys.argv[2])
    path = data / "yellow_tripdata_2023-01.parquet"
    rows, ncols, truth = pyarrow_truth(path)
    got = olorin_eaparquet(binary, path)

    findings = []
    o_rows = got["totals"]["rows"]
    o_fields = {f["name"]: f for f in got.get("fields", [])}
    print(f"yellow_tripdata_2023-01.parquet: rows olorin={o_rows} pyarrow={rows}  "
          f"cols olorin={len(o_fields)} pyarrow={ncols}")
    if o_rows != rows:
        findings.append(f"row count: olorin={o_rows} pyarrow={rows}")
    if len(o_fields) != ncols:
        findings.append(f"column count: olorin={len(o_fields)} pyarrow={ncols}")

    for name, t in truth.items():
        f = o_fields.get(name)
        if f is None:
            findings.append(f"column {name}: missing from olorin output")
            continue
        nc = f.get("null_count")
        if nc is not None and nc != t["nulls"]:
            findings.append(f"{name}.null_count olorin={nc} pyarrow={t['nulls']}")
        num = f.get("numeric")
        if num and t["stats"] and t["min"] is not None:
            if not approx(num.get("min"), t["min"]):
                findings.append(f"{name}.min olorin={num.get('min')} pyarrow={t['min']}")
            if not approx(num.get("max"), t["max"]):
                findings.append(f"{name}.max olorin={num.get('max')} pyarrow={t['max']}")
        mark = "ok" if name in o_fields else "MISSING"
        print(f"  {name:<24} nulls olorin={str(nc):>8} pyarrow={t['nulls']:>8}  {mark}")

    print()
    if findings:
        print(f"FINDINGS ({len(findings)}):")
        for x in findings:
            print(f"  {x}")
        sys.exit(1)
    print("eaparquet vs pyarrow: 0 mismatches")


if __name__ == "__main__":
    main()

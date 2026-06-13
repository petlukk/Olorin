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


def diff_synthetic_types(binary, findings):
    """Write a DECIMAL + INT96 parquet with pyarrow and check eaparquet
    against it — the taxi file has neither type. DECIMAL footer stats are
    present (compared numerically); INT96 stats are omitted by pyarrow
    (undefined sort order), so we only assert the column is labeled a
    timestamp."""
    import datetime, decimal
    import pyarrow as pa

    path = STAGE / "synthetic_types.parquet"
    STAGE.mkdir(exist_ok=True)
    t = pa.table({
        "price": pa.array([decimal.Decimal("19.99"), decimal.Decimal("1234.50"),
                           decimal.Decimal("-7.25"), decimal.Decimal("0.01")],
                          pa.decimal128(10, 2)),
        "event_time": pa.array([datetime.datetime(2023, 1, 15, 8, 30),
                                datetime.datetime(2023, 6, 1, 12, 0),
                                datetime.datetime(2022, 12, 31, 23, 59, 59),
                                datetime.datetime(2023, 3, 1, 0, 0)], pa.timestamp("ns")),
    })
    pq.write_table(t, path, use_deprecated_int96_timestamps=True)

    got = olorin_eaparquet(binary, path)
    f = {x["name"]: x for x in got.get("fields", [])}

    md = pq.read_metadata(path)
    st = md.row_group(0).column(md.schema.names.index("price")).statistics
    num = (f.get("price") or {}).get("numeric")
    if not num:
        findings.append("synthetic price: no numeric stats from olorin")
    else:
        if not approx(num.get("min"), st.min):
            findings.append(f"price.min olorin={num.get('min')} pyarrow={st.min}")
        if not approx(num.get("max"), st.max):
            findings.append(f"price.max olorin={num.get('max')} pyarrow={st.max}")
    print(f"  DECIMAL price: olorin min={num and num.get('min')} max={num and num.get('max')} "
          f"pyarrow min={st.min} max={st.max}")

    ev = f.get("event_time", {})
    print(f"  INT96 event_time: olorin kind={ev.get('kind')} (pyarrow omits INT96 stats)")
    if ev.get("kind") != "timestamp":
        findings.append(f"event_time kind olorin={ev.get('kind')} expected=timestamp")


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

    diff_synthetic_types(binary, findings)

    print()
    if findings:
        print(f"FINDINGS ({len(findings)}):")
        for x in findings:
            print(f"  {x}")
        sys.exit(1)
    print("eaparquet vs pyarrow: 0 mismatches (taxi + DECIMAL/INT96)")


if __name__ == "__main__":
    main()

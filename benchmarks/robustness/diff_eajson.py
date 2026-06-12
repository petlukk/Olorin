#!/usr/bin/env python3
"""eajson differential: per-key presence counts from `olorin rune eajson
--json` vs Python's json, which flattens nested objects to dotted paths
and counts how many records contain each key. Independent stdlib oracle.

eajson flattens nested OBJECTS (actor.id) but does not descend into
arrays — the oracle mirrors that so the comparison is apples-to-apples.

Usage: diff_eajson.py <olorin-binary> <data-dir>
"""
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

STAGE = Path("/tmp/olorin_robustness")


def stage(path):
    STAGE.mkdir(exist_ok=True)
    dst = STAGE / Path(path).name
    if not dst.exists():
        shutil.copy(path, dst)
    return dst


def olorin_eajson(binary, path):
    out = subprocess.run(
        [binary, "rune", "eajson", "--json", str(stage(path))],
        capture_output=True, text=True, timeout=300,
    )
    if out.returncode != 0:
        raise SystemExit(f"olorin exited {out.returncode}:\n{out.stderr}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def flatten_keys(obj, prefix=""):
    """Dotted paths for nested objects; arrays counted as a leaf, not
    descended (mirrors eajson)."""
    keys = set()
    for k, v in obj.items():
        path = f"{prefix}{k}"
        if isinstance(v, dict):
            keys |= flatten_keys(v, path + ".")
        else:
            keys.add(path)
    return keys


def python_truth(path):
    counts = Counter()
    n = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            n += 1
            for key in flatten_keys(obj):
                counts[key] += 1
    return n, counts


def main():
    binary, data = sys.argv[1], Path(sys.argv[2])
    path = data / "gharchive_2024-01-01-15.jsonl"
    n, truth = python_truth(path)
    got = olorin_eajson(binary, path)

    findings = []
    o_rows = got["totals"]["rows"]
    o_fields = {f["name"]: f["count"] for f in got.get("fields", [])}
    print(f"{path.name}: records olorin={o_rows} python={n}  "
          f"keys olorin={len(o_fields)} python={len(truth)}")
    if o_rows != n:
        findings.append(f"record count: olorin={o_rows} python={n}")

    # Compare counts for every key olorin reported (its key set may be a
    # subset if it caps; report keys it MISSED that python saw a lot of).
    for k, oc in sorted(o_fields.items()):
        tc = truth.get(k)
        mark = "ok" if oc == tc else "MISMATCH"
        if oc != tc:
            findings.append(f"{k}: olorin={oc} python={tc}")
        print(f"  {k:<28} olorin={oc:>8} python={str(tc):>8}  {mark}")

    missed = [(k, c) for k, c in truth.most_common() if k not in o_fields]
    if missed:
        print(f"\n  keys python saw but olorin omitted (top 10):")
        for k, c in missed[:10]:
            print(f"    {k:<28} python={c}")

    print()
    if findings:
        print(f"FINDINGS ({len(findings)}):")
        for x in findings[:40]:
            print(f"  {x}")
        sys.exit(1)
    print("eajson vs python json: 0 count mismatches")


if __name__ == "__main__":
    main()

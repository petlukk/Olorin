#!/usr/bin/env python3
"""eatime differential: timestamp count vs an independent regex parse of
ISO-8601 and CLF instants. Usage: eatime_check.py <binary> <data-dir>"""
import json, re, shutil, subprocess, sys
from pathlib import Path

STAGE = Path("/tmp/olorin_robustness")
ISO = re.compile(rb'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}')
CLF = re.compile(rb'\[\d{2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2}')


def stage(p):
    STAGE.mkdir(exist_ok=True)
    d = STAGE / Path(p).name
    if not d.exists(): shutil.copy(p, d)
    return d


def olorin(binary, path):
    out = subprocess.run([binary,"rune","eatime","--bucket","series","--json",str(stage(path))],
                         capture_output=True, text=True, timeout=180)
    if out.returncode: raise SystemExit(f"olorin {out.returncode}: {out.stderr}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def main():
    binary, data = sys.argv[1], Path(sys.argv[2])
    fail = 0
    for f in ["NASA_access_log_Jul95", "gharchive_2024-01-01-15.jsonl"]:
        path = data / f
        if not path.exists(): continue
        raw = open(path,"rb").read()
        ref = max(len(ISO.findall(raw)), len(CLF.findall(raw)))
        got = olorin(binary, path)["totals"]["rows"]
        ok = got == ref
        print(f"{f:<32} olorin={got} regex={ref}  {'ok' if ok else 'MISMATCH'}")
        if not ok: fail = 1
    sys.exit(fail)


if __name__ == "__main__":
    main()

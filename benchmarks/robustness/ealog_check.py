#!/usr/bin/env python3
"""ealog differential: severity counts vs a case-insensitive, word-bounded
oracle modelling ealog's actual rule (delimiters: space tab nl cr [ ] " : ;
plus start/end of buffer; WARNING counts as WARN, CRITICAL as FATAL-ish).
Three real Loghub formats. Usage: ealog_check.py <binary> <data-dir>"""
import json, shutil, subprocess, sys
from pathlib import Path

STAGE = Path("/tmp/olorin_robustness")
DELIM = set(b' \t\n\r[]":;')
# leaf token -> bucket (ealog folds WARNING->WARN, CRITICAL->FATAL family)
TOKENS = {"DEBUG":"DEBUG","INFO":"INFO","WARN":"WARN","WARNING":"WARN",
          "ERROR":"ERROR","FATAL":"FATAL","CRITICAL":"CRITICAL"}


def stage(p):
    STAGE.mkdir(exist_ok=True)
    d = STAGE / Path(p).name
    if not d.exists(): shutil.copy(p, d)
    return d


def olorin(binary, path):
    out = subprocess.run([binary,"rune","ealog","--json",str(stage(path))],
                         capture_output=True, text=True, timeout=120)
    if out.returncode: raise SystemExit(f"olorin {out.returncode}: {out.stderr}")
    o = json.loads(out.stdout.strip().splitlines()[-1])
    return {c["name"]: c["count"] for c in o.get("categories", [])}


def truth(path):
    data = open(path,"rb").read(); low = data.lower()
    counts = {}
    def boundary(b): return b is None or b in DELIM
    for tok, bucket in TOKENS.items():
        t = tok.lower().encode(); i = 0; n = 0
        while True:
            j = low.find(t, i)
            if j < 0: break
            before = data[j-1] if j>0 else None
            after  = data[j+len(t)] if j+len(t) < len(data) else None
            if boundary(before) and boundary(after): n += 1
            i = j+1
        counts[bucket] = counts.get(bucket, 0) + n
    return counts


def main():
    binary, data = sys.argv[1], Path(sys.argv[2])
    fail = 0
    for f in ["Linux_2k.log","Apache_2k.log","HDFS_2k.log"]:
        path = data / f
        if not path.exists(): continue
        o = olorin(binary, path); t = truth(path)
        keys = ["DEBUG","INFO","WARN","ERROR","FATAL"]
        og = {k:o.get(k,0) for k in keys}; tg = {k:t.get(k,0) for k in keys}
        ok = og == tg
        print(f"{f:<16} olorin={og} oracle={tg}  {'ok' if ok else 'MISMATCH'}")
        if not ok: fail = 1
    sys.exit(fail)


if __name__ == "__main__":
    main()

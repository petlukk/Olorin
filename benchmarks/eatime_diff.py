#!/usr/bin/env python3
"""Differential gate: eatime --bucket series bucket counts vs an
independent regex+pandas grouping on the SAME real file. Validates the
chronological bucketization (the part most prone to off-by-one / epoch
bugs), independent of the detection heuristic."""
import re, json, sys
import pandas as pd
from datetime import datetime

LOG = sys.argv[1]
EATIME_JSON = sys.argv[2]
EPOCH0 = datetime(2000, 1, 1)
NICE = [1, 5, 10, 30, 60, 300, 600, 1800, 3600, 21600, 43200, 86400, 604800]

def auto_width(span):
    if span <= 0:
        return 1
    raw = span // 120
    for w in NICE:
        if w >= raw:
            return w
    return NICE[-1]

raw = open(LOG, 'rb').read().decode('utf-8', 'replace')
pat = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}')
epochs = []
for m in pat.finditer(raw):
    try:
        dt = datetime.strptime(m.group(0), '%Y-%m-%dT%H:%M:%S')  # rejects out-of-range
    except ValueError:
        continue  # mirrors eatime's iso_bytes_to_seconds returning None
    epochs.append(int((dt - EPOCH0).total_seconds()))

epochs.sort()
mn, mx = epochs[0], epochs[-1]
width = auto_width(mx - mn)
n = (mx - mn) // width + 1
# Independent grouping via pandas.
s = pd.Series([(e - mn) // width for e in epochs])
vc = s.value_counts()
ref_counts = [int(vc.get(i, 0)) for i in range(n)]

j = json.load(open(EATIME_JSON))
ea_counts = [c['count'] for c in j['categories']]

ok = True
if len(ea_counts) != n:
    print(f"FAIL: bucket count differs: eatime={len(ea_counts)} ref={n}"); ok = False
elif ea_counts != ref_counts:
    diffs = [(i, ea_counts[i], ref_counts[i]) for i in range(n) if ea_counts[i] != ref_counts[i]]
    print(f"FAIL: {len(diffs)} bucket(s) differ, first 5: {diffs[:5]}"); ok = False

total_ref = len(epochs)
total_ea = j['totals']['rows']
if total_ref != total_ea:
    print(f"NOTE: total differs eatime={total_ea} ref={total_ref} "
          f"(ok if message bodies contain extra ISO stamps)")

if ok:
    print(f"PASS: {total_ea} timestamps, {n} buckets @ {width}s — "
          f"eatime counts bit-identical to pandas/regex grouping")
    anoms = j.get('anomalies', [])
    print(f"      anomalies reported: {len(anoms)}")
    for a in anoms:
        # ratio/score are JSON null on a median=0 baseline (infinite ratio);
        # the schema emits non-finite floats as null by design.
        rstr = f"{a['ratio']:.2f}x" if isinstance(a.get('ratio'), (int, float)) else "inf"
        sstr = f"{a['score']:.1f}" if isinstance(a.get('score'), (int, float)) else "n/a"
        print(f"        {a['bucket']} count={a['count']} ratio={rstr} z={sstr} baseline={a['baseline']}")
sys.exit(0 if ok else 1)

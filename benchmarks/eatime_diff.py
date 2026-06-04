#!/usr/bin/env python3
"""Differential gate: eatime --bucket series bucket counts vs an
independent regex+pandas grouping on the SAME real file. Validates the
chronological bucketization (the part most prone to off-by-one / epoch
bugs), independent of the detection heuristic. Handles both ISO-8601 and
Common Log Format inputs, picking the dominant grammar like eatime does."""
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
MONTHS = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun",
     "jul", "aug", "sep", "oct", "nov", "dec"])}

def iso_epochs(text):
    out = []
    for m in re.finditer(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', text):
        try:
            dt = datetime.strptime(m.group(0), '%Y-%m-%dT%H:%M:%S')
        except ValueError:
            continue  # mirrors iso_bytes_to_seconds returning None
        out.append(int((dt - EPOCH0).total_seconds()))
    return out

def clf_epochs(text):
    out = []
    pat = re.compile(r'\[(\d{2})/([A-Za-z]{3})/(\d{4}):(\d{2}):(\d{2}):(\d{2})')
    for d, mon, y, hh, mm, ss in pat.findall(text):
        month = MONTHS.get(mon.lower())
        if month is None:
            continue
        try:
            dt = datetime(int(y), month, int(d), int(hh), int(mm), int(ss))
        except ValueError:
            continue
        out.append(int((dt - EPOCH0).total_seconds()))
    return out

# Pick the dominant grammar, mirroring eatime's detect_format.
iso, clf = iso_epochs(raw), clf_epochs(raw)
epochs = clf if len(clf) > len(iso) else iso
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

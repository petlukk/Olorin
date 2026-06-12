#!/usr/bin/env python3
"""Differential oracle for eacorrelate: an independent numpy
re-implementation of the full pipeline (ISO scan -> ERROR substream ->
shared grid -> lag sweep -> per-window Pearson, positive-only, active-
window gated -> top-3), compared against the real `olorin rune
eacorrelate --json` binary on randomized scenarios with planted lags,
independent-noise controls, and disjoint-era traps.

Usage: python3 benchmarks/eacorrelate_diff.py [path-to-olorin-binary]
Exits non-zero on any mismatch (lag, finding set, threshold decision,
or score drift > 2e-3).
"""

import json
import re
import subprocess
import sys
import random
from pathlib import Path

import numpy as np

# Mirror src/runes/eacorrelate.rs + stream.rs
NICE_WIDTHS = [1, 5, 10, 30, 60, 300, 600, 1800, 3600, 21600, 43200, 86400, 604800]
TARGET_BUCKETS = 512
MAX_LAG_BUCKETS = 128
SCORE_THRESHOLD = 0.5
MIN_EVENTS = 3
TOP_K = 3

ISO_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")
SEV_RE = re.compile(r'(?:^|[ \t\[\]":])(?:ERROR|FATAL)(?:[ \t\[\]":]|$)')
EPOCH_2026_06_11 = 1781136000  # not needed absolutely; relative seconds suffice


def stamp(secs: int) -> str:
    return f"2026-06-11T{secs // 3600:02d}:{(secs % 3600) // 60:02d}:{secs % 60:02d}"


def iso_to_secs(ts: str) -> int:
    h, m, s = int(ts[11:13]), int(ts[14:16]), int(ts[17:19])
    return h * 3600 + m * 60 + s  # all scenarios stay inside one day


def auto_width(span: int) -> int:
    if span <= 0:
        return 1
    raw = span // TARGET_BUCKETS
    for w in NICE_WIDTHS:
        if w >= raw:
            return w
    return NICE_WIDTHS[-1]


def extract_streams(path: Path):
    """(name, epochs) streams: all ISO stamps, plus the ERROR/FATAL
    subset attributed to the last stamped position at or before the
    match — same rule as error_substream() in the rune."""
    text = path.read_text()
    positions = [(m.start(), iso_to_secs(m.group())) for m in ISO_RE.finditer(text)]
    all_epochs = [e for _, e in positions]
    err_epochs = []
    for m in SEV_RE.finditer(text):
        prior = [e for (p, e) in positions if p <= m.start()]
        if prior:
            err_epochs.append(prior[-1])
    streams = []
    if len(all_epochs) >= MIN_EVENTS:
        streams.append((path.name, all_epochs))
        if len(err_epochs) >= MIN_EVENTS:
            streams.append((f"{path.name} (errors)", err_epochs))
    return streams


def reference_findings(files):
    streams = []  # (name, file_idx, epochs)
    for idx, f in enumerate(files):
        for name, epochs in extract_streams(f):
            streams.append((name, idx, epochs))
    if len({i for _, i, _ in streams}) < 2:
        return []
    all_epochs = [e for _, _, ep in streams for e in ep]
    gmin, gmax = min(all_epochs), max(all_epochs)
    span = gmax - gmin
    if span <= 0:
        return []
    width = auto_width(span)
    n = span // width + 1
    max_lag = min(MAX_LAG_BUCKETS, n - 1)

    css = []
    for name, idx, epochs in streams:
        counts = np.zeros(n, dtype=np.float64)
        for e in epochs:
            counts[(e - gmin) // width] += 1.0
        css.append(None if counts.var() == 0.0 else counts)

    findings = []
    for i in range(len(streams)):
        for j in range(i + 1, len(streams)):
            if streams[i][1] == streams[j][1] or css[i] is None or css[j] is None:
                continue
            a, b = css[i], css[j]
            best_lag, best_score = 0, 0.0
            for lag in range(-max_lag, max_lag + 1):
                if lag >= 0:
                    wa, wb = a[lag:], b[: n - lag]
                else:
                    wa, wb = a[: n + lag], b[-lag:]
                # Per-window Pearson (invariant to the binary's z-scoring),
                # gated on both windows being ACTIVE — same spec as the rune.
                if wa.sum() < MIN_EVENTS or wb.sum() < MIN_EVENTS:
                    continue
                if wa.std() <= 0.0 or wb.std() <= 0.0:
                    continue
                r = float(np.corrcoef(wa, wb)[0, 1])
                if r > best_score:
                    best_lag, best_score = lag, r
            # Positive-only, same spec as the rune (negative rate
            # correlation across files = the disjoint-era artifact).
            if best_score < SCORE_THRESHOLD:
                continue
            fi, fj, lag = (i, j, best_lag) if best_lag >= 0 else (j, i, -best_lag)
            findings.append({
                "stream_a": streams[fi][0],
                "stream_b": streams[fj][0],
                "lag_seconds": lag * width,
                "score": best_score,
                "events_a": len(streams[fi][2]),
                "events_b": len(streams[fj][2]),
                "width_seconds": width,
            })
    findings.sort(key=lambda f: (-f["score"], f["stream_a"], f["stream_b"]))
    return findings[:TOP_K]


def run_olorin(binary, files):
    args = [binary, "rune", "eacorrelate", "--json"] + [str(f) for f in files]
    out = subprocess.run(args, capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        sys.exit(f"olorin exited {out.returncode}: {out.stderr}")
    return json.loads(out.stdout.strip().splitlines()[-1])


# ── scenario generators ─────────────────────────────────────────────────────

def gen_planted(rng, tmp, case):
    """Deploys + a log whose ERROR bursts trail each deploy by a fixed lag."""
    lag = rng.choice([60, 120, 240, 600])
    n_deploys = rng.randint(3, 6)
    burst = rng.randint(4, 30)
    deploys = sorted(rng.sample(range(3600, 7 * 3600, 60), n_deploys))
    log = tmp / f"diff_{case}_errors.log"
    csv = tmp / f"diff_{case}_deploys.csv"
    lines = [f"{stamp(m * 60)} INFO heartbeat" for m in range(0, 8 * 60 + 1)]
    for d in deploys:
        lines += [f"{stamp(d + lag)} ERROR upstream timeout #{k}" for k in range(burst)]
    log.write_text("\n".join(lines) + "\n")
    csv.write_text("time,event\n" + "".join(f"{stamp(d)},deploy\n" for d in deploys))
    return [log, csv]


def gen_independent(rng, tmp, case):
    """Two unrelated scatters — the control: no finding may appear."""
    files = []
    for tag in ("a", "b"):
        secs = sorted(rng.randrange(0, 8 * 3600) for _ in range(rng.randint(80, 300)))
        f = tmp / f"diff_{case}_{tag}.log"
        f.write_text("".join(f"{stamp(s)} INFO event\n" for s in secs))
        files.append(f)
    return files


def gen_disjoint(rng, tmp, case):
    """Two diurnal-ish logs in NON-overlapping windows of the day — the
    wild NASA-slices trap: must produce no finding (positive-only +
    active-window gate)."""
    files = []
    for tag, (lo, hi) in (("a", (0, 8 * 3600)), ("b", (14 * 3600, 23 * 3600))):
        lines = []
        for s in range(lo, hi, 60):
            burst = 3 if ((s // 3600) % 2 == 0) else 1
            lines += [f"{stamp(s)} INFO svc-{tag} event {k}" for k in range(burst)]
        f = tmp / f"diff_{case}_{tag}.log"
        f.write_text("\n".join(lines) + "\n")
        files.append(f)
    return files


def main():
    binary = sys.argv[1] if len(sys.argv) > 1 else "target/debug/olorin"
    tmp = Path("/tmp/olorin_eacorrelate_diff")
    tmp.mkdir(exist_ok=True)
    rng = random.Random(20260611)

    failures = 0
    cases = (
        [("planted", gen_planted)] * 10
        + [("independent", gen_independent)] * 5
        + [("disjoint", gen_disjoint)] * 3
    )
    for case_no, (kind, gen) in enumerate(cases):
        files = gen(rng, tmp, case_no)
        got = run_olorin(binary, files).get("correlations", [])
        want = reference_findings(files)

        ok = len(got) == len(want)
        if ok:
            for g, w in zip(got, want):
                ok &= (
                    g["stream_a"] == w["stream_a"]
                    and g["stream_b"] == w["stream_b"]
                    and g["lag_seconds"] == w["lag_seconds"]
                    and g["width_seconds"] == w["width_seconds"]
                    and g["events_a"] == w["events_a"]
                    and g["events_b"] == w["events_b"]
                    and abs(g["score"] - w["score"]) <= 2e-3
                )
        status = "ok" if ok else "MISMATCH"
        print(f"case {case_no:02d} [{kind:11s}] olorin={len(got)} ref={len(want)} {status}")
        if not ok:
            failures += 1
            print(f"  olorin: {json.dumps(got, indent=2)}")
            print(f"  ref:    {json.dumps(want, indent=2)}")

    print(f"\n{len(cases) - failures}/{len(cases)} cases match")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

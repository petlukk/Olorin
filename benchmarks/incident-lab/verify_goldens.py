#!/usr/bin/env python3
"""Verify the frozen incident-lab goldens — the deterministic gate.

Three checks, any failure exits non-zero (the repo's diff-oracle convention):

  1. Determinism / no-drift. Regenerate every scenario from its pinned seed and
     start, and diff against the committed goldens byte-for-byte. Catches an
     accidental simulator change that wasn't re-frozen, and proves the goldens
     are reproducible.
  2. Detection. Run `olorin rune eacorrelate` on the bad-deploy incident streams
     and assert it produces an incident timeline with the leading-indicator lag
     (db-pool errors -> access-log 5xx, a positive lag).
  3. Controls (the point of the lab). Run the same correlation on `quiet` and
     `good-deploy` and assert NO incident timeline appears — a healthy baseline
     and a healthy deploy must not raise a false alarm.

Usage: python3 verify_goldens.py [path-to-olorin-binary]
"""
import datetime
import subprocess
import sys
import tempfile
from pathlib import Path

import simulate
import logfmt

HERE = Path(__file__).resolve().parent
GOLDENS = HERE / "goldens"

# Must match how goldens/ was frozen (see README "Regenerating the goldens").
PARAMS = dict(seed=1, start="2026-06-15T09:00:00", duration=120, rate=6.0)
INCIDENT_STREAMS = ["deploy.log", "db.log", "access.log"]


def fail(msg):
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def regenerate(scen, out):
    start = datetime.datetime.strptime(PARAMS["start"], "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=datetime.timezone.utc)
    streams = simulate.simulate(scen, PARAMS["seed"], start, PARAMS["duration"], PARAMS["rate"])
    out.mkdir(parents=True, exist_ok=True)
    for name in logfmt.STREAMS:
        (out / name).write_text("".join(streams[name]))


def check_determinism(work):
    # Regenerate into a /tmp working dir (also the path the runes will read from:
    # the rune path guard only allows ~ or /tmp, and the repo lives elsewhere).
    for scen in simulate.SCENARIOS:
        regenerate(scen, work / scen)
        for name in logfmt.STREAMS:
            got = (work / scen / name).read_bytes()
            want_path = GOLDENS / scen / name
            if not want_path.exists():
                fail(f"missing committed golden {scen}/{name}")
            if got != want_path.read_bytes():
                fail(f"golden drift in {scen}/{name} — re-freeze with the pinned params")
    print("  [1/3] determinism: all 5 scenarios regenerate byte-identical to goldens ✓")


def eacorrelate(binary, work, scen):
    paths = [str(work / scen / s) for s in INCIDENT_STREAMS]
    out = subprocess.run([binary, "rune", "eacorrelate", *paths],
                         capture_output=True, text=True)
    return out.stdout


def check_detection(binary, work):
    out = eacorrelate(binary, work, "bad-deploy")
    if "incident timeline" not in out:
        fail(f"bad-deploy produced no incident timeline:\n{out}")
    # The leading-indicator cascade: db-pool errors lead, access-log 5xx follow by
    # a positive lag. The timeline renders this across two lines ("db.log spike …"
    # then "-> access.log (errors) rises N seconds later"), so assert on the whole
    # timeline block, not a single line.
    timeline = out[out.index("incident timeline"):]
    if not ("db.log" in timeline and "access.log" in timeline and "later" in timeline):
        fail(f"bad-deploy timeline lacks the db->access leading-indicator lag:\n{out}")
    print("  [2/3] detection: bad-deploy yields the db->access cascade timeline ✓")


def check_controls(binary, work):
    for scen in ("quiet", "good-deploy"):
        out = eacorrelate(binary, work, scen)
        if "incident timeline" in out:
            fail(f"FALSE POSITIVE: {scen} raised an incident timeline:\n{out}")
    print("  [3/3] controls: quiet + good-deploy raise no false incident ✓")


def main():
    binary = sys.argv[1] if len(sys.argv) > 1 else str(
        HERE.parent.parent / "target" / "release" / "olorin")
    if not Path(binary).exists():
        fail(f"olorin binary not found at {binary} (build with `cargo build --release`)")
    print(f"verifying incident-lab goldens against {binary}")
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="olorin-lab-") as td:
        work = Path(td)
        check_determinism(work)
        check_detection(binary, work)
        check_controls(binary, work)
    print("incident-lab goldens: OK")


if __name__ == "__main__":
    main()

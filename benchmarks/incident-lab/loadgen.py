#!/usr/bin/env python3
"""Drive GET traffic at a realistic, time-varying rate for the live lab.

  loadgen.py [DURATION=30] [RATE=30] [WORKERS=6]

Unlike a flat load generator, the request rate follows a diurnal sine envelope
with per-second log-normal jitter and the occasional benign micro-burst (a flash
crowd that is NOT an incident). That variation is the baseline a rate-anomaly
detector must ride out without crying wolf — so the live lab tests the same
"don't false-alarm on normal swings" property the deterministic goldens do.

Most traffic hits /api/leaderboard (the endpoint that breaks after the bad
deploy), so a buggy version turns most of it into real HTTP 500s. Workers back
off slightly on error, modelling clients giving up -> a mild traffic dip.
"""
import math
import os
import random
import sys
import threading
import time
import urllib.request

PORT = os.environ.get("PORT", "8099")
DUR = int(sys.argv[1]) if len(sys.argv) > 1 else 30
RATE = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
WORKERS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
PATHS = ["/api/leaderboard/global?limit=100"] * 7 + ["/api/health", "/api/categories", "/api/unknown"]

START = time.time()
counts = {"ok": 0, "err": 0}
lock = threading.Lock()


def rate_now(rng):
    """Current target req/s: diurnal envelope (two cycles over the run) times
    seeded log-normal jitter, with an occasional micro-burst."""
    t = time.time() - START
    diurnal = 1.0 + 0.55 * math.sin(2 * math.pi * 2.0 * t / max(1, DUR))
    noise = math.exp(rng.gauss(0.0, 0.22))
    burst = rng.uniform(2.0, 3.5) if rng.random() < 0.01 else 1.0
    return max(1.0, RATE * diurnal * noise * burst)


def worker(wid):
    rng = random.Random(wid * 7919 + 1)
    ok = err = 0
    while time.time() - START < DUR:
        path = rng.choice(PATHS)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=2).read()
            ok += 1
            backoff = 1.0
        except Exception:
            err += 1
            backoff = 1.5  # give up a little when the server is erroring
        # Per-worker spacing so the WORKERS threads sum to ~rate_now().
        time.sleep(WORKERS / rate_now(rng) * backoff)
    with lock:
        counts["ok"] += ok
        counts["err"] += err


threads = [threading.Thread(target=worker, args=(i,)) for i in range(WORKERS)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print(f"loadgen done: {counts['ok']} ok, {counts['err']} errors")

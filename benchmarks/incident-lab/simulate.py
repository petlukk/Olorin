#!/usr/bin/env python3
"""Deterministic incident simulator for the Olorin incident-lab.

Given a scenario, a seed, and a fixed start time, this writes the full
multi-format log set (see `logfmt.py`) for one incident — with NO real HTTP
server, threads, or sleeps. Same seed + same start -> byte-identical output, so
the result can be frozen as a golden fixture and asserted on in CI.

What makes it "closer to reality" than a flat load generator:

  * Diurnal + bursty baseline traffic. The request rate follows a sine envelope
    with seeded log-normal jitter and the occasional benign micro-burst (a flash
    crowd that is NOT an incident). This is the variation a rate-anomaly detector
    must ride out without crying wolf — the thing a flat baseline can never test.
  * A response_time on every request (nginx `$request_time`, `rt=<ms>ms`, JSON
    `rt_ms`), so latency-only incidents have a signal to carry.
  * Five scenarios, including the two controls that prove the *absence* of false
    positives:

      quiet           healthy baseline only           -> nothing should fire
      good-deploy     a deploy that stays healthy      -> prediction must NOT alarm
      bad-deploy      deploy -> db-pool lead -> lagged -> eacorrelate timeline;
                      500 cascade -> recovery deploy      palantir predict+confirm,
                                                          then stand down on recovery
      latency-degr.   p99 creeps up, zero errors,      -> latency-only incident
                      no trigger                          (the response_time signal)
      rate-ramp       500 rate climbs, no trigger      -> rate-anomaly leak

Usage:
    python3 simulate.py <scenario> [--seed N] [--start ISO8601]
                        [--duration SECONDS] [--out DIR]
"""
import argparse
import datetime
import math
import os
import sys

import logfmt

SCENARIOS = ("quiet", "good-deploy", "bad-deploy", "latency-degradation", "rate-ramp")

# Request mix: leaderboard dominates (the endpoint that breaks); a little health,
# categories, and an unknown path that 404s.
ENDPOINTS = (
    ["/api/leaderboard/global?limit=100"] * 7
    + ["/api/health", "/api/categories", "/api/unknown"]
)
PID = 4242  # fixed so goldens are stable


def poisson(rng, lam):
    """Knuth's algorithm on a seeded RNG — deterministic Poisson draw."""
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k, p = 0, 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def lognorm(rng, median_ms, sigma=0.5):
    """Log-normal latency around a median, in integer milliseconds."""
    return max(1, int(median_ms * math.exp(rng.gauss(0.0, sigma))))


def rate_at(rng, t, duration, base):
    """Requests/second at simulated second `t`: a diurnal sine envelope (two full
    cycles over the run) times seeded log-normal burst noise. A benign micro-
    burst occasionally multiplies the rate for a few seconds — a flash crowd the
    detector must not mistake for an incident."""
    diurnal = 1.0 + 0.55 * math.sin(2 * math.pi * 2.0 * t / max(1, duration))
    noise = math.exp(rng.gauss(0.0, 0.22))
    burst = 1.0
    if rng.random() < 0.015:           # ~1.5% of seconds kick off a micro-burst
        burst = rng.uniform(2.0, 3.5)
    return max(0.0, base * diurnal * noise * burst)


def status_and_latency(rng, scen, t, duration, deploy_at, lag, recover_at, path):
    """Return (status, response_time_ms, db_error) for one request in `scen` at
    simulated second `t`. `db_error` is True when this request should also emit a
    leading db-pool error line (the cascade's leading signal)."""
    leaderboard = path.startswith("/api/leaderboard")
    if path.startswith("/api/unknown"):
        return 404, lognorm(rng, 8), False

    if scen == "bad-deploy" and leaderboard and deploy_at <= t < recover_at:
        # db pool fails from the deploy onward — the leading signal, every hit.
        since = t - deploy_at
        if since < lag:
            # Degraded: still serving cached rows (200) but slower than healthy.
            return 200, lognorm(rng, 60, 0.6), True
        # Cache drained: the str-slice bug fires -> 500, and a code exception
        # fails fast (low latency), as real exceptions do.
        return 500, lognorm(rng, 6, 0.4), True

    if scen == "rate-ramp" and leaderboard:
        # No trigger: 500 probability climbs linearly to ~60% across the run.
        p = 0.6 * min(1.0, t / max(1, duration))
        if rng.random() < p:
            return 500, lognorm(rng, 6, 0.4), False
        return 200, lognorm(rng, 22), False

    if scen == "latency-degradation" and leaderboard:
        # No errors at all; the median response time creeps from 20ms to ~400ms
        # over the run, with a handful of 504 timeouts only at the very end.
        frac = t / max(1, duration)
        median = 20 + 380 * frac
        if frac > 0.9 and rng.random() < 0.05:
            return 504, lognorm(rng, 2000, 0.3), False
        return 200, lognorm(rng, median, 0.45), False

    # Healthy baseline (quiet, good-deploy, and pre/post-incident windows):
    # overwhelmingly 200, a rare transient 500 that self-heals (real services
    # blip — the detector must tolerate single stray errors).
    if leaderboard and rng.random() < 0.0008:
        return 500, lognorm(rng, 6, 0.4), False
    return 200, lognorm(rng, 20), False


def simulate(scen, seed, start, duration, base_rate):
    import random
    rng = random.Random(seed)
    streams = {name: [] for name in logfmt.STREAMS}

    deploy_at = duration // 2 if scen in ("good-deploy", "bad-deploy") else -1
    lag = 20
    recover_at = deploy_at + 45 if scen == "bad-deploy" else 10 ** 9

    def emit_deploy(sec, old, new):
        dt = start + datetime.timedelta(seconds=sec)
        sha = "".join(rng.choice("0123456789abcdef") for _ in range(6))
        for fname, line in logfmt.deploy_lines(dt, PID, old, new, sha).items():
            streams[fname].append(line)

    if deploy_at >= 0:
        emit_deploy(0, "none", "v1.0.0")  # the healthy version is "deployed" at t0

    for t in range(duration):
        if t == deploy_at:
            emit_deploy(t, "v1.0.0", "v1.1.0")  # the bad (or good) deploy
        if scen == "bad-deploy" and t == recover_at:
            emit_deploy(t, "v1.1.0", "v1.1.1")  # the fix

        n = poisson(rng, rate_at(rng, t, duration, base_rate))
        for i in range(n):
            # Spread events across the second so timestamps are ordered/distinct.
            dt = start + datetime.timedelta(seconds=t, microseconds=int(1e6 * i / max(1, n)))
            path = rng.choice(ENDPOINTS)
            status, rt_ms, db_err = status_and_latency(
                rng, scen, t, duration, deploy_at, lag, recover_at, path)
            if db_err:
                msg = "db pool exhausted: could not acquire connection for leaderboard query"
                streams["db.log"].append(logfmt.db_iso_line(dt, "ERROR", msg))
                streams["db.syslog"].append(logfmt.db_syslog_line(dt, PID, "ERROR", msg))
            nbytes = 220 if status == 200 else (9 if status >= 500 else 9)
            for fname, line in logfmt.request_lines(dt, PID, "10.0.0.7", path, status, nbytes, rt_ms).items():
                streams[fname].append(line)
            if status >= 500 and rng.random() < 0.25:
                # A sampled exception+traceback in the app/system stream, as a
                # real handler logs on a 500.
                for ln in (f"Exception on {path}: TypeError: slice indices must be integers",
                           "  rows = BOARD[:limit]"):
                    el = logfmt.app_line(dt, "ERROR", ln)
                    streams["app.log"].append(el)
                    streams["system.log"].append(el)
    return streams


def main():
    ap = argparse.ArgumentParser(description="Deterministic incident-lab simulator")
    ap.add_argument("scenario", choices=SCENARIOS)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--start", default="2026-06-15T09:00:00",
                    help="ISO-8601 UTC start time (default 2026-06-15T09:00:00)")
    ap.add_argument("--duration", type=int, default=600, help="simulated seconds")
    ap.add_argument("--rate", type=float, default=30.0, help="base requests/second")
    ap.add_argument("--out", default=None, help="output dir (default ./<scenario>)")
    args = ap.parse_args()

    start = datetime.datetime.strptime(args.start, "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=datetime.timezone.utc)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), args.scenario)
    os.makedirs(out, exist_ok=True)

    streams = simulate(args.scenario, args.seed, start, args.duration, args.rate)
    for name in logfmt.STREAMS:
        with open(os.path.join(out, name), "w") as f:
            f.writelines(streams[name])
    total = sum(len(v) for v in streams.values())
    errs = sum(1 for ln in streams["access.log"] if '" 5' in ln)
    print(f"[simulate] {args.scenario}: {total} lines across {len(logfmt.STREAMS)} streams, "
          f"{errs} 5xx, -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

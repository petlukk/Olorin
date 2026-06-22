#!/usr/bin/env python3
"""Live leaderboard API with a deploy-introduced bug — the real-time half of the
incident-lab (the deterministic half is `simulate.py`).

Healthy until version >= v1.1.0; then `/api/leaderboard` slices the board with a
string limit -> TypeError -> HTTP 500. The DB pool fails first (the leading
signal); the API serves cached data in degraded mode for `BUG_LAG` seconds before
the cache drains and requests start 500ing, so access-log 5xx lag behind db-log
errors — a real propagation delay.

Every request is logged in all formats via `logfmt` (shared with the simulator),
each carrying a synthetic response_time. The version is read live from the
`version` file, so `deploy.sh` can flip it under a running server. Point a
palantír at `system.log` and watch it predict the cascade at the deploy.

Env: PORT (8099), BUG_LAG (20), RAMP (0 = off; >0 ramps the 500 rate with no
deploy trigger — the rate-anomaly exercise).
"""
import datetime
import http.server
import math
import os
import random
import threading
import time
import traceback

import logfmt

BASE = os.path.dirname(os.path.abspath(__file__))
VERSION = os.path.join(BASE, "version")
PID = os.getpid()
LOG_LOCK = threading.Lock()
RNG = random.Random(0xC0FFEE)  # synthetic latency only — not correctness-bearing

LAG = float(os.environ.get("BUG_LAG", "20"))
RAMP = float(os.environ.get("RAMP", "0"))
START = time.time()
BOARD = [{"name": f"player{i}", "score": 100 - i} for i in range(20)]


def now():
    return datetime.datetime.now(datetime.timezone.utc)


def write(streams):
    """Append a dict {filename: line} to disk under the shared lock."""
    with LOG_LOCK:
        for fname, line in streams.items():
            with open(os.path.join(BASE, fname), "a") as f:
                f.write(line)


def latency_ms(status):
    # Synthetic: a code exception fails fast; a healthy/degraded request is slower.
    median = 6 if status >= 500 else 22
    return max(1, int(median * math.exp(RNG.gauss(0.0, 0.5))))


def version():
    try:
        return open(VERSION).read().strip()
    except FileNotFoundError:
        return "v1.0.0"


def bug_elapsed():
    try:
        return time.time() - os.path.getmtime(VERSION)
    except OSError:
        return 0.0


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        status, body, dt = 200, b"", now()
        try:
            if self.path.startswith("/api/leaderboard"):
                limit = "10"
                if "limit=" in self.path:
                    limit = self.path.split("limit=")[1].split("&")[0]  # a STRING
                if RAMP > 0:
                    p = max(0.0, min(1.0, (time.time() - START - RAMP) / RAMP))
                    rows = BOARD[:limit] if RNG.random() < p else BOARD[:int(limit)]
                elif version() >= "v1.1.0":
                    write({"db.log": logfmt.db_iso_line(dt, "ERROR",
                              "db pool exhausted: could not acquire connection for leaderboard query"),
                           "db.syslog": logfmt.db_syslog_line(dt, PID, "ERROR",
                              "db pool exhausted: could not acquire connection for leaderboard query")})
                    rows = BOARD[:10] if bug_elapsed() < LAG else BOARD[:limit]
                else:
                    rows = BOARD[:int(limit)]
                body = repr({"leaderboard": rows}).encode()
            elif self.path.startswith("/api/health"):
                body = b'{"status":"ok"}'
            elif self.path.startswith("/api/categories"):
                body = b'["arcade","puzzle","racing"]'
            else:
                status, body = 404, b"not found"
        except Exception as e:
            status, body = 500, b"internal server error"
            for ln in ([f"Exception on {self.path}: {type(e).__name__}: {e}"]
                       + traceback.format_exc().rstrip().splitlines()):
                write({"app.log": logfmt.app_line(dt, "ERROR", ln),
                       "system.log": logfmt.app_line(dt, "ERROR", ln)})
        try:
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            pass
        write(logfmt.request_lines(dt, PID, self.client_address[0], self.path,
                                   status, len(body), latency_ms(status)))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8099"))
    write({"system.log": logfmt.app_line(now(), "INFO",
           f"Started leaderboard service {version()} on :{port}")})
    http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()

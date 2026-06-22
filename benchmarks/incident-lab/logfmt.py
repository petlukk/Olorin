#!/usr/bin/env python3
"""Multi-format log line rendering — one place for every format the Olorin
runes detect, shared by the live HTTP server (`app.py`) and the deterministic
simulator (`simulate.py`).

A real service rarely emits one log format; it emits whatever each library it
links chose. The lab mirrors that spread so the runes are exercised against the
shapes they'll meet in the wild:

  access.log  nginx-style CLF, status visible (5xx is the error stream), with a
              trailing request_time field (nginx `$request_time`, seconds)
  app.log     ISO-8601 app log, INFO/ERROR, `rt=<ms>ms` on each request line
  db.log      ISO-8601 DB-pool errors — the *leading* signal in a cascade
  db.syslog   the same DB cascade in BSD syslog (a different timestamp era)
  app.jsonl   ndjson, STRING level + numeric epoch-millis `time` (zap/zerolog)
  pino.jsonl  ndjson, NUMERIC level (50=error/30=info) + `time` (pino/bunyan)
  syslog.log  classic BSD syslog `MMM DD HH:MM:SS host app[pid]: LEVEL msg`
  apache.log  Apache error log `[Www Mmm DD HH:MM:SS YYYY] [sev] msg`
  hdfs.log    Hadoop `YYMMDD HHMMSS pid LEVEL component: msg`

Each renderer takes a `datetime` (UTC, timezone-aware) so the caller controls
the clock — wall-clock for the live server, a fixed simulated clock for the
deterministic goldens.
"""
import datetime
import json

# The set of newline-terminated log files a full run produces. Keys are the
# logical streams; values are the on-disk filenames.
STREAMS = [
    "access.log", "app.log", "db.log", "db.syslog", "app.jsonl",
    "pino.jsonl", "syslog.log", "apache.log", "hdfs.log", "system.log",
    "deploy.log", "deploy.syslog",
]


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def clf_ts(dt):
    return dt.strftime("%d/%b/%Y:%H:%M:%S +0000")


def syslog_ts(dt):
    # Classic BSD syslog stamp: `MMM DD HH:MM:SS` (yearless — a fixed reference
    # era, deliberately distinct from the ISO streams above).
    return dt.strftime("%b %d %H:%M:%S")


def apache_ts(dt):
    return dt.strftime("[%a %b %d %H:%M:%S %Y]")


def hdfs_ts(dt):
    return dt.strftime("%y%m%d %H%M%S")


def access_line(dt, ip, path, status, nbytes, rt_ms):
    # nginx CLF + trailing `$request_time` in seconds (3 dp), as nginx logs it.
    rt_s = f"{rt_ms / 1000.0:.3f}"
    return (f'{ip} - - [{clf_ts(dt)}] "GET {path} HTTP/1.1" {status} {nbytes} '
            f'"-" "loadgen" {rt_s}\n')


def app_line(dt, level, msg, rt_ms=None):
    suffix = f" rt={rt_ms}ms" if rt_ms is not None else ""
    return f"{iso(dt)} {level} {msg}{suffix}\n"


def db_iso_line(dt, level, msg):
    return f"{iso(dt)} {level} {msg}\n"


def db_syslog_line(dt, pid, level, msg):
    return f"{syslog_ts(dt)} pi-host db[{pid}]: {level} {msg}\n"


def json_line(dt, level, msg, **extra):
    # zap/zerolog convention: STRING level, numeric epoch MILLIS under `time`.
    rec = {"level": level, "time": int(dt.timestamp() * 1000), "msg": msg}
    rec.update(extra)
    return json.dumps(rec, sort_keys=True) + "\n"


def pino_line(dt, levelnum, pid, msg, **extra):
    # pino/bunyan convention: NUMERIC level (50=error, 30=info).
    rec = {"level": levelnum, "time": int(dt.timestamp() * 1000), "pid": pid, "msg": msg}
    rec.update(extra)
    return json.dumps(rec, sort_keys=True) + "\n"


def syslog_line(dt, pid, level, msg):
    return f"{syslog_ts(dt)} pi-host app[{pid}]: {level} {msg}\n"


def apache_line(dt, is_err, msg):
    sev = "error" if is_err else "notice"
    return f"{apache_ts(dt)} [{sev}] {msg}\n"


def hdfs_line(dt, level, msg):
    return f"{hdfs_ts(dt)} {level} dfs.DataNode: {msg}\n"


def deploy_lines(dt, pid, old, new, sha):
    """The deploy trigger, rendered into every stream that anchors a cascade:
    ISO deploy.log + the unified system.log (real era), and BSD deploy.syslog
    (syslog era). Returns a dict {filename: line}."""
    return {
        "deploy.log":
            f"{iso(dt)}Z deploy leaderboard {old} -> {new} "
            f"service=incident-lab actor=ci commit={sha}\n",
        "system.log":
            f"{iso(dt)} INFO deploy leaderboard {old} -> {new} commit={sha}\n",
        "deploy.syslog":
            f"{syslog_ts(dt)} pi-host deploy[{pid}]: released {old} -> {new} "
            f"commit={sha}\n",
    }


def request_lines(dt, pid, ip, path, status, nbytes, rt_ms):
    """Every per-request log line for one request, as a dict {filename: line}.
    The app.log/system.log INFO line is emitted only for non-errors (matching
    the live server, which logs the exception+traceback separately on a 500)."""
    is_err = status >= 500
    out = {
        "access.log": access_line(dt, ip, path, status, nbytes, rt_ms),
        "app.jsonl": json_line(dt, "error" if is_err else "info",
                               f"{status} GET {path}", path=path, status=status,
                               rt_ms=rt_ms),
        "pino.jsonl": pino_line(dt, 50 if is_err else 30, pid,
                                f"{status} GET {path}", path=path, status=status,
                                rt_ms=rt_ms),
        "syslog.log": syslog_line(dt, pid, "ERROR" if is_err else "INFO",
                                  f"{status} GET {path}"),
        "apache.log": apache_line(dt, is_err, f"mod_wsgi: {status} GET {path}"),
        "hdfs.log": hdfs_line(dt, "ERROR" if is_err else "INFO",
                              f"{status} GET {path}"),
    }
    if not is_err:
        line = app_line(dt, "INFO", f"{status} GET {path}", rt_ms=rt_ms)
        out["app.log"] = line
        out["system.log"] = line
    return out

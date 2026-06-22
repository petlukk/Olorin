# incident-lab

A small service that *has incidents*, so Olorin's log analysis — the runes
(`eatime`, `eacorrelate`) and the **palantír** predictive watcher — can be
exercised against data shaped like what a real on-call engineer sees, instead of
hand-written log lines.

It is a leaderboard API with a deploy-introduced bug. A bad deploy exhausts the
DB pool (the *leading* signal); the API serves cached data in degraded mode for a
propagation delay, then the cache drains and requests start returning HTTP 500.
Every request is logged in **all nine formats the runes detect** (nginx CLF,
ISO-8601 app log, BSD syslog, Apache error, HDFS, zap/zerolog + pino/bunyan JSON),
each carrying a `response_time`.

## Two halves

| | `simulate.py` (deterministic) | `app.py` + `*.sh` (live) |
|---|---|---|
| **Clock** | fixed simulated time | wall-clock |
| **Output** | byte-identical per seed | fresh each run |
| **Speed** | instant (no server/sleeps) | real-time |
| **Use** | frozen goldens, CI, offline rune tests | watch a palantír fire in real time |

Both render through the same `logfmt.py`, so the live and simulated logs are
identical in shape — only the clock differs.

## Scenarios

Five, chosen to test detection **and the absence of false positives** — the
property a flat load generator can't measure:

| Scenario | Shape | What it proves |
|---|---|---|
| `quiet` | healthy baseline only (diurnal + bursty traffic, the rare self-healing blip) | nothing fires |
| `good-deploy` | a deploy that stays healthy | the prediction path does **not** false-alarm on a clean deploy |
| `bad-deploy` | deploy → db-pool lead → lagged 500 cascade → recovery deploy | `eacorrelate` builds the timeline; palantír predicts + confirms, then stands down |
| `latency-degradation` | p99 response_time creeps up, zero errors, no trigger | a latency-only incident — the signal lives in `response_time` |
| `rate-ramp` | 500 rate climbs, no deploy trigger | the rate-anomaly detector catches a leak |

**Realistic baseline traffic** is the core upgrade: the request rate follows a
diurnal sine envelope with seeded log-normal jitter and occasional benign
micro-bursts (a flash crowd that is *not* an incident). A detector is only
trustworthy if it rides that variation out without crying wolf — so `quiet` and
`good-deploy` are first-class fixtures, not afterthoughts.

## Running

Deterministic (no server):

```bash
python3 simulate.py bad-deploy --out /tmp/inc      # writes the full log set
olorin rune eacorrelate /tmp/inc/deploy.log /tmp/inc/db.log /tmp/inc/access.log
```

Live (real HTTP, watch a palantír react in real time):

```bash
olorin palantir --alert ./system.log --daemon      # in this directory
./scenario.sh                  # bad deploy → cascade (the headline)
./scenario.sh 25 25 30 6 20 good   # the control: a clean deploy, nothing fires
./ramp.sh                      # no-trigger 500 leak (rate-anomaly)
```

The runes' path guard allows only `~` or `/tmp`, so point them at logs under
those roots (the deterministic flow above writes to `/tmp`).

## Frozen goldens + the gate

`goldens/` holds the five scenarios frozen at a pinned seed and start time
(`seed=1`, `start=2026-06-15T09:00:00`, `duration=120`, `rate=6`). The gate
regenerates them, diffs byte-for-byte (determinism / no-drift), then runs the
real binary and asserts the `bad-deploy` cascade timeline appears **and** the
`quiet`/`good-deploy` controls raise no false incident:

```bash
cargo build --release
python3 benchmarks/incident-lab/verify_goldens.py
```

Exits non-zero on any drift, missed detection, or false positive.

### Regenerating the goldens

Only when the simulator's output intentionally changes — re-freeze in the same
commit as the change (the gate fails otherwise):

```bash
cd benchmarks/incident-lab
for s in quiet good-deploy bad-deploy latency-degradation rate-ramp; do
  python3 simulate.py "$s" --seed 1 --start 2026-06-15T09:00:00 \
    --duration 120 --rate 6 --out goldens/"$s"
done
(cd goldens && find . -name '*.log' -o -name '*.jsonl' -o -name '*.syslog' \
   | sort | xargs sha256sum > SHA256SUMS)
```

Determinism relies on CPython's seeded `random` (Mersenne Twister). The goldens
reproduce byte-for-byte across CPython 3.11 and 3.12 (verified x86 ↔ Pi NEON); a
far-future interpreter change could in principle drift the bytes, which the gate
would catch loudly. The CI gate (`tests/incident_lab_goldens.rs`) asserts the
detection + controls directly on the committed logs, so it needs no Python at
all.

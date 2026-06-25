# Changelog

All notable changes to Olorin. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project
uses [semver](https://semver.org/) at the minor level. Each release
is tagged in git as `vX.Y.Z` and listed below in reverse-chronological
order.

## [Unreleased]

## [3.16.5] — 2026-06-25

### Fixed

- **Web terminal no longer leaves residue when recalling a shorter command.**
  The terminal's ANSI emulator handled erase-to-EOL (`CSI K`) and type-over but
  not DCH (`CSI n P` — Delete Character). `readline` emits DCH when recalling a
  *shorter* history entry over a longer one (`\x1b[2P`) and when backspacing
  mid-line (`\x1b[1P`), so the deleted glyphs lingered in the grid — recalling
  `ls` over `lscpu` rendered as a mix of the two even though the command
  executed correctly. The emulator now implements DCH (shift the line tail left,
  blank the freed cells, cursor unmoved). Only DCH is added — a byte capture of
  `readline` showed it rewrites-and-deletes rather than emitting ICH/ECH in
  these flows, so those sequences stay unimplemented.
- **Chat composer attach/send buttons now have breathing room.** The two buttons
  sat flush against each other; added an 8px gap to the input row.

## [3.16.4] — 2026-06-24

### Fixed

- **Rune narration now names the concrete finding.** The narration system prompt
  told the model to "avoid repeating raw numbers verbatim", which made it hedge —
  e.g. an `eatime` traffic spike was narrated as *"a significant peak around a
  specific date and time"* while the exact date and magnitude sat in the output
  in front of it. The prompt now asks the model to name the headline figure (the
  date/category and the magnitude of the peak/spike/anomaly) and only forbids
  reproducing the whole table; the `is_grid_continuation` filter still catches
  the one real failure mode (continuing the numeric grid). Pi-validated 3/3 at
  production temperature: a spike now narrates as *"peak activity occurred on
  2026-03-13T14:00:00 … 226× the baseline."* Applies to every rune narration.

## [3.16.3] — 2026-06-24

### Fixed

- **Palantír all-clear is now reported consistently.** v3.16.2 injected the
  affirmative all-clear, but the `<recent_observations>` block carried no
  instruction on how to use it, so under the Pi's minimal system prompt the small
  model relayed a rich incident line but intermittently ignored the terse
  all-clear (≈half the turns fell back to a generic "could you provide more
  context" at the web UI's temperature 0.4). The block is now framed with a short
  directive — *"answer from it: report any active incident, or state all is clear
  when none is active"* — so "what's going on?" / "is everything ok?" is answered
  from the watcher state every time. Pi-validated: 6/6 quiet turns → "All is
  clear", and an injected incident still reports the incident, not a false clear.

## [3.16.2] — 2026-06-24

### Fixed

- **Palantír now gives the chat an affirmative all-clear.** v3.16.1 told the chat
  about a live incident or a just-fired stand-down, but when an incident resolved
  any other way — a confirmed cascade aging out, or a prediction that never
  cascaded — `recent_observations` returned nothing, so the chat got no context
  and fell back to a generic reply even though the badge had gone green. It also
  couldn't distinguish "all clear" from "no watcher running". Now every *fresh*
  watcher contributes exactly one line: the incident or stand-down when there's
  news, otherwise `all clear — watching, no active incident`, so asking "is
  everything ok?" gets a definite answer. A *stale* daemon still contributes
  nothing (its state is unknown — a quiet line would be a false all-clear).

## [3.16.1] — 2026-06-24

### Fixed

- **Palantír stand-down now reaches the chat as an all-clear.** When a predicted
  cascade never arrived, the watcher stood down and the web-UI badge flipped
  green, but the chat model was never told — the `<recent_observations>` block
  dropped the resolved incident silently, so the assistant could warn "watching
  for a cascade" and then revert to a generic greeting without ever reporting it
  cleared. A recent `clear` now surfaces as a transient all-clear ("window clear
  — no cascade") for 90s, then ages out so a resolved incident doesn't linger as
  stale context. `active_incident()` and the badge are unchanged.
- **Incident lab good-deploy run ends cleanly.** `scenario.sh` aborted under
  `set -e` on a healthy deploy because its summary line referenced `db.log`,
  which only exists when the bug arms on a bad deploy; the missing-file `wc`
  killed the script before the summary printed. The `db.log` references are now
  guarded (0 db errors when absent). Also marked `deploy.sh` / `ramp.sh` /
  `scenario.sh` executable — they were committed `100644`, so `./scenario.sh`
  failed with "Permission denied" on a fresh checkout. Test infrastructure only.

## [3.16.0] — 2026-06-22

### Added

- **Incident lab (`benchmarks/incident-lab/`).** The Pi-only log generator used to
  exercise the runes and the palantír is now in the repo and closer to a real
  service: realistic diurnal/bursty baseline traffic, a `response_time` on every
  request, and five scenarios — including two controls (`quiet`, `good-deploy`)
  that assert the *absence* of a false alarm, which a flat-traffic generator
  can't test. A deterministic simulator (`simulate.py`, seed → byte-identical
  across CPython 3.11/3.12) freezes the scenarios as committed goldens; a CI gate
  (`tests/incident_lab_goldens.rs`) runs `eacorrelate` on them and asserts the
  bad-deploy cascade timeline plus no false incident on the controls. Test
  infrastructure only — nothing in the shipped binary changes.

## [3.15.0] — 2026-06-19

### Added

- **Start/stop palantír watchers from the web UI.** The 🔮 badge dropdown now has
  a "watch a file…" field and a per-watcher ✕ stop, backed by `POST
  /api/palantir/watch` and `/api/palantir/stop`. No CLI needed to launch a
  logwatch daemon. The watched path is validated by a palantír-specific guard
  that permits read-only `/var/log` on top of `$HOME`/`/tmp` (kept separate from
  the LLM-tool guard, so this doesn't widen what `read_file`/`grep`/`ls` can
  reach), and alert sinks (`--notify exec:`/`webhook:`) are refused from the web
  to keep RCE/SSRF off the exposed surface.

## [3.14.1] — 2026-06-19

### Fixed

- **Palantír badge stayed red after an incident cleared.** The web-UI badge
  (and the chat Pipe's `<recent_observations>`) treated *any* recent alert as
  active, so a `clear` stand-down — the all-clear that fires when a predicted
  cascade never arrives — kept the badge red for the full 10-minute freshness
  window instead of returning it to green. The badge now distinguishes "an alert
  is recent" from "an incident is active": `predicted`/`confirmed`/`anomaly`
  read alerting (red), `clear` reads watching (green).

## [3.14.0] — 2026-06-19

### Added

- **Palantír status badge in the web UI.** The top heartbeat bar now shows a 🔮
  indicator of how many palantír daemons are watching and how many are alerting;
  clicking it opens a dropdown listing each watcher, its status, and its last
  alert. It reads a new `palantir` field on the existing `/api/system` response
  (no new endpoint or stream), so it updates on the same heartbeat poll as the
  rest of the bar. With the v3.13.0 chat-awareness, this completes the
  Palantír web-UI integration.

## [3.13.0] — 2026-06-18

### Added

- **Olorin's chat senses live palantír incidents.** When a palantír daemon is
  reporting a fresh incident, the chat turn's system prompt gains a
  `<recent_observations>` block — so asking "anything happening with the logs?"
  works, and incidents surface in conversation without being asked about. It
  reads the daemon's status snapshot; only fresh watchers with a recent alert
  are surfaced (quiet/stale watchers contribute nothing, so the prompt is
  unchanged outside an incident). Per-turn only, sanitized; rune narration is
  unaffected.

## [3.12.0] — 2026-06-18

### Added

- **Palantír daemon mode + lifecycle.** The log watcher can now run detached:
  - `palantir --alert <file> --daemon [--name NAME]` backgrounds the watcher
    (no controlling terminal; survives the shell — including the web-UI terminal
    — that launched it), logging to `~/.olorin/palantir/<name>.log`.
  - `palantir --status [--name NAME]` lists watchers (running/stopped + last
    alert).
  - `palantir --stop [--name NAME]` stops a watcher and cleans up its state.
  State (pid, an atomically-written status snapshot, and the log) lives under
  `~/.olorin/palantir/`. Foreground mode is unchanged. Unix only.

## [3.11.0] — 2026-06-18

### Added

- **Palantír rate-anomaly detection.** The log watcher now catches incidents
  with **no trigger line** — memory leaks, gradual degradation, traffic spikes —
  by flagging an abnormal rise in the per-bucket error rate against a robust
  trailing baseline (median + MAD, the same statistic `eatime --bucket series`
  uses, run incrementally; no new kernel). It complements the v1 trigger-during-
  lag detection — both run together, and a rate anomaly is suppressed while a
  trigger incident is already being tracked so one event isn't reported twice.
  False-positive control is deliberate: a warm-up before alerting, a required
  streak of consecutive anomalous buckets, an absolute floor, a MAD floor, and a
  cooldown. Tunable via the existing `--sensitivity low|med|high`. Verified live
  on the Pi: a no-trigger error ramp fired a `RATE ANOMALY` alert from a quiet
  baseline.

## [3.10.0] — 2026-06-18

### Added

- **Palantír alert sinks (`--notify`).** The log watcher can now route alerts
  beyond stdout, repeatably:
  - `webhook:URL` — HTTP POST a JSON body (`{text, source, kind, severity}`)
    via curl. Works with Slack/PagerDuty/Discord directly, or a WhatsApp webhook
    service (CallMeBot/Twilio) for phone alerts.
  - `exec:CMD` — run a command per alert with the alert in the environment
    (`PALANTIR_ALERT` / `_KIND` / `_SEVERITY`) — wire it to the local wa-bridge,
    `ntfy`, `mail`, desktop notifications, anything.
  - `stdout` remains the default. `--notify` is repeatable, so alerts can fan
    out to several destinations at once. A sink failure logs to stderr and never
    blocks the watcher or the other sinks.

## [3.9.0] — 2026-06-18

### Added

- **Palantír — predictive log watching (`olorin palantir --alert <file>`).** A new
  subsystem, peer to the runes: an ambient log watcher that warns *before* an
  error cascade instead of reporting it after. It tails a log file and, when a
  deploy-class trigger line appears, predicts an incoming cascade — with an ETA
  learned from the file's own history — then confirms if the errors arrive in
  the window or stands down if they don't.

  v1 is a foreground, single-file, trigger-during-lag watcher with stdout
  alerts and tunable `--sensitivity low|med|high`. Error classification reuses
  the same per-format sub-stream kernels the runes use (ISO/CLF/syslog/Apache/
  HDFS/JSON), so "error" means exactly what `eacorrelate`/`ealog` call an error;
  it needs no new SIMD kernel. Verified live on the Pi (NEON): learned a 10s
  deploy→error lag and predicted the cascade at the deploy line, before the
  user-visible error storm.

  Honest scope: it predicts the trigger+lag class of incidents, not
  precursor-free instant crashes. Deferred to later versions: folder/multi-file
  correlation, depleting-gauge trends, streaming rate-anomaly detection,
  WhatsApp/webhook alert sinks, and a background daemon mode.

## [3.8.5] — 2026-06-18

### Fixed

- **`eacorrelate` no longer collapses a short incident into one coarse bucket
  when a few timestamps are far off.** The shared correlation grid took its
  bucket width from the raw min..max of all event times, so a minority of
  outlier timestamps — a 1970 epoch from a missing field, a clock-skew future,
  or a yearless format (BSD syslog) parsed to a fixed reference era while its
  dated peers sit in the real year — stretched the span across months and
  snapped the bucket to the coarsest rung (up to a 1-week bucket), manufacturing
  a trivial `r=1.00` with no lag resolution. The grid span is now clipped to a
  Tukey fence (Q1−3·IQR, Q3+3·IQR); the fence scales with the data's own spread,
  so genuine multi-day incidents keep a wide window and outlier-free data is
  unchanged. A real ~50s deploy→errors incident that bucketed at 604800s now
  buckets at 1s.

## [3.8.4] — 2026-06-17

### Added

- **Cloud fallback now narrates rune output.** When no local model is loaded
  (cloud-only configuration), dropping a file or running a rune previously
  showed the kernel output with no model narration — the followup paths were
  wired only to the local engine. Both the web-UI/file-drop and REPL paths now
  narrate via the Anthropic client when there's no local engine, reusing the
  same narration prompt and the same grid/dump/empty discard filters. A failed
  or absent cloud call degrades to the bare kernel output, as before.

## [3.8.3] — 2026-06-17

### Fixed

- **Cloud fallback now points at a current model.** The hardcoded default
  cloud model was `claude-3-5-haiku-latest`, which aliased a model Anthropic
  retired on 2026-02-19 — so once a valid API key was configured, every cloud
  request failed with `Anthropic API error: model: claude-3-5-haiku-latest`
  (now visible thanks to the v3.8.2 error-surfacing fix). The default is now
  `claude-haiku-4-5`, in both the client default and the config display. Users
  who set a custom cloud model are unaffected.

## [3.8.2] — 2026-06-17

### Fixed

- **Cloud (Anthropic) failures no longer masked as "no backend".** After
  pasting an API key into the web-UI config modal and pressing Apply, a failing
  cloud request used to fall through to the generic *"No LLM backend available.
  Load a model or set ANTHROPIC_API_KEY."* message — making a configured-but-
  failing backend indistinguishable from no backend at all, so the key looked
  like it never applied. Both dispatch paths now surface the real reason
  (`Cloud inference failed: <reason>`, e.g. an invalid `x-api-key`), pointing
  straight at the fix. The no-backend message now fires only when no client is
  configured, where it is accurate.

### Changed

- The Anthropic curl call is bounded with `--connect-timeout 15 --max-time 120`
  so a stuck cloud request reports an error instead of hanging the UI.

## [3.8.1] — 2026-06-17

### Changed

- **Clearer "unsupported file" and rune help, everywhere.** Dropping a file
  whose type isn't supported now names the file, says it isn't supported, and
  lists what *is* (CSV, JSON/JSON Lines, Parquet, SQL, logs & text) instead of a
  terse one-liner that had silently gone stale (it omitted `.sql`). Naming a rune
  that doesn't exist — via `olorin rune <name>` or the REPL `/rune` — now prints
  the full list of available runes. The REPL `/help` gained an accurate runes
  section and a supported-files block. The format list and the rune list are now
  derived from the same place routing and the registry live, with drift-guard
  tests so they can't fall out of sync again.

### Internal

- Split the two source files that had reached the 500-line cap
  (`interface/server.rs` → `server_config.rs`; `core/router_streaming.rs` →
  `router_streaming_analyze.rs`), and refreshed the README project-stats table to
  measured values. No behavior change.

## [3.8.0] — 2026-06-16

### Added

- **`eacorrelate` / `eatime` read Apache error logs and HDFS/Hadoop logs.** Two
  more formats are auto-detected and flow through correlation and the incident
  timeline like any other log:
  - **Apache error** — `[Sun Dec 04 04:47:44 2005] [error] ...` (full year; the
    `[error]`/`[crit]` severity drives the error sub-stream). Apache 2.4
    fractional seconds are tolerated.
  - **HDFS / Hadoop** — `081109 203615 148 INFO dfs.DataNode: ...` (2-digit year
    read as 20YY; the `ERROR`/`FATAL` level field drives the error sub-stream).
  Each adds a SIMD timestamp kernel reusing the existing cross-arch idiom (no new
  compiler intrinsics); both verified x86 ≡ Pi NEON bit-identical on the public
  Loghub samples.

### Fixed

- **Error sub-streams now count one event per line, not per keyword.** A log line
  whose message repeats a level word (common in Apache prose — `[error] ... in
  error state`) previously contributed several identical error events; it now
  contributes one. Matches the line count exactly (e.g. the Loghub Apache sample:
  595, was 1134); no change for logs that mention a level once per line.

## [3.7.0] — 2026-06-16

### Added

- **`eacorrelate` reads pino / bunyan numeric severity levels.** JSON logs that
  encode severity as a number (`"level":50`) rather than a word now get their
  errors separated into the `(errors)` sub-stream, like string-level logs
  (`"level":"error"`, zap / zerolog) already did. A new SIMD kernel
  (`json_level_scan`) matches `level` values ≥ 50 — pino's error (50) and fatal
  (60), plus any custom level ≥ 100 — while leaving warn (40) and below in the
  baseline, matching the ERROR/FATAL semantics of the keyword sub-stream. So a
  deploy→error cascade in a pino feed is now visible to the incident timeline.
  No new compiler intrinsics; verified x86 ≡ Pi NEON bit-identical.

## [3.6.0] — 2026-06-16

### Added

- **`eacorrelate` separates errors in classic BSD syslog logs.** A syslog app
  log (`Jun 14 15:16:01 host app[42]: ERROR ...`) now gets an `(errors)`
  sub-stream alongside its all-events stream — the ERROR/FATAL lines, keyed to
  each line's timestamp — exactly as ISO, CLF (HTTP 5xx) and JSON logs already
  did. Previously syslog was the one detected format whose errors stayed folded
  into the baseline traffic, so a deploy→error cascade in a syslog feed was
  invisible to the incident timeline. No new kernel: the format-agnostic
  `log_level_scan` is reused with the syslog timestamp decoder. Verified
  x86 ≡ Pi NEON bit-identical.

## [3.5.1] — 2026-06-16

### Fixed

- **The HTML report now renders the incident timeline.** `olorin report` showed
  the raw cross-file correlations but dropped the assembled incident — so the
  shareable artifact was missing the deploy-anchored "why did my service die"
  headline that the terminal and web outputs already lead with. The report now
  opens with it (*"Deployment at 06:25 → app.log errors rise 22 seconds later"*)
  above the correlation evidence.

## [3.5.0] — 2026-06-16

### Added

- **`eacorrelate` / `eatime` read structured JSON logs.** ndjson is auto-detected
  and flows through the correlation and incident timeline like any text log. JSON
  with an **ISO-string** timestamp (`"ts":"2026-..."` — winston, zerolog,
  structlog, bunyan, serilog) already worked, since the ISO kernel matches the
  string inside the value. New this release: **numeric Unix-epoch** timestamps
  (`"ts":1749600000`, `"time":1749600000000` — Go zap, pino), via a SIMD kernel
  `json_epoch_scan.ea` that anchors a timestamp key (`time`/`ts`/`timestamp`/
  `@timestamp`) followed by digits; the unit (s/ms/µs) is inferred from
  magnitude, so a zap-seconds log and a pino-millis log correlate correctly.
  String log levels (`"level":"error"`) form the error sub-stream as usual.
  Numeric levels (pino `"level":50`) are not yet recognized.

## [3.4.0] — 2026-06-15

### Added

- **`eacorrelate` reads nginx/Apache 5xx as errors.** CLF access logs now
  contribute an `(errors)` sub-stream built from HTTP 5xx responses, the way
  ISO/syslog logs already split on `ERROR`/`FATAL`. So a deploy → 500s → traffic
  cascade in a real web stack is now visible to the correlation and incident
  passes, not just keyword-logged app errors. New SIMD kernel
  `clf_status_scan.ea` (same cross-arch `.==`-anchor idiom as `clf_scan`, no new
  eacompute intrinsics); verified bit-identical x86 ↔ Pi NEON.

- **Incident timeline anchors on a single deploy line.** A real deploy is one log
  line — too sparse to be a correlation stream — so the timeline used to anchor
  on the first error stream. It now snaps the anchor onto a discrete trigger
  event (a deploy) sitting at the cascade's inferred root, and re-bases the step
  lags onto it, so the story names the cause:

  ```
  incident timeline (confidence 0.94):
    Deployment at 12:56
    -> access.log (errors) rise 34 seconds later (r=0.94)
  ```

### Fixed

- **`eacorrelate` no longer invents multi-hour "incidents" from long logs.** The
  lag cap scaled with the window (`span/4`), so a multi-day syslog could report a
  confident "errors → auth rises 16h later" built from unrelated periodic noise.
  An absolute ceiling (`MAX_LAG_SECONDS`, 1h) now bounds reported lags — an
  incident cascade is minutes, not hours.

## [3.3.0] — 2026-06-15

### Added

- **`eacorrelate` incident timeline — "why did my service die".** When the
  cross-file correlations form a cascade, eacorrelate now assembles them into a
  single ordered story instead of a correlation list: it finds the cascade root
  (a discrete trigger like a deploy when the stream is sparse, else the first
  break), anchors on it, and orders the fallout by lag.

  ```
  incident timeline (confidence 0.91):
    Deployment at 14:02
    -> app errors rise 4 minutes later (r=0.93)
    -> request traffic drops 12 minutes later (anomaly 0.91)
  ```

  Two kinds of follower step: a **correlated** rise (a co-spiking stream,
  carrying its Pearson `r`) and a signed **drop** (`anomaly`) — a stream that
  *falls* within the incident window, detected as a downward robust-median/MAD
  break (kept as an observation, never a negative correlation, so the
  disjoint-era guard is not reopened). `confidence` is the weakest link. Wording
  is strictly temporal ("errors rise N later", never "the deploy *caused* it");
  a zero-lag step reads "at the same time"; no cascade → no timeline. Surfaced
  as an additive `incident` block on `RuneOutput v1` (`--json`) and the headline
  of the text/narration output. Drop detection verified end-to-end on a real
  systemd journal.
- **Two new timestamp grammars**, auto-detected across `eatime`, `eacorrelate`,
  and the incident timeline:
  - **Space-separated ISO** `YYYY-MM-DD HH:MM:SS` (Postgres / MySQL / Python
    `logging` / OpenStack; fractional seconds ignored) — the `timestamp_scan`
    kernel now accepts `T` *or* space at the separator.
  - **Classic BSD syslog** `MMM DD HH:MM:SS` (`Jun 14 15:16:01` — Linux
    `/var/log`, sshd, cron, network gear; incl. the space-padded `Jun  4 …`) via
    a new `syslog_scan` SIMD kernel. Yearless, so a fixed reference year is
    assigned: bucketing and lag are exact, but the displayed year and the
    `weekday` bucket are placeholders, and two syslog files from different real
    years may falsely overlap. `--format syslog` added.

### Fixed

- **`eacorrelate` spurious `r=1.00` on long-span logs.** A near-boundary lag
  could leave a tiny (e.g. 4-bucket) overlap window where Pearson hits `1.00`
  trivially — a real 28-day NASA traffic/errors split scored errors "following"
  traffic by **+654 h, r=1.00**. The active-window gate counted events, not
  buckets; now an overlap-window floor (`MIN_OVERLAP_BUCKETS`) plus a lag cap at
  a quarter of the span exclude it. Only bind on long-span/short grids; existing
  goldens unchanged.
- **`olorin report` on large single-file uploads.** The web-UI "download report"
  button base64-wrapped the file into a JSON body, overflowing the 128 MB cap on
  ~100 MB+ logs; single-file reports now stream raw via `/api/report_raw`.

### Internal

- **Standing fuzzing campaign.** Every file-parsing rune, the BPE tokenizer, and
  the encrypted vault's crash recovery are now continuously fuzzed (mutation +
  randomized on-disk crash-state injection), on **both x86 and Raspberry Pi
  NEON**, each harness paired with a negative control that proves it can fail.
  Tens of thousands of adversarial inputs, zero findings. Cross-architecture
  bit-identical output remains a CI gate.

## [3.2.1] — 2026-06-14

### Fixed

- **`eacrunch` GROUP BY text output no longer double-prints `count`.** A
  requested `--agg count` re-rendered the group's row count as a noisy
  `count=N.00` next to the intrinsic integer `count=N`; the text table now
  prints the count once. `--json` output is unchanged (the `count` agg stays
  in `aggs[]` for machine consumers).

### Performance

- **`eajson` drops the dead comma scan from `jsonl_struct`.** The kernel
  emitted every comma position into a full-length `i32` array that eajson
  never read (the key/value walk uses only quote + colon + newline +
  backslash positions). Removing comma detection + emission cuts the scan
  from 5 position arrays to 4. On the 558 MB GH Archive file (`--json`),
  peak RSS drops ~41 MB (1307 → 1265 MB) with byte-identical output
  (differential vs Python `json`: 0 mismatches).

## [3.2.0] — 2026-06-13

### Added

- **`eajson --depth N` multi-level nesting.** Nested objects now flatten to
  dotted keys (`http.req.headers.ua`) up to N levels deep — default raised
  from 1 to **4**, `--depth 0` = top-level keys only. The flattener was
  already recursive; this lifts the one-level cap and makes it configurable.
  On the 558 MB GH Archive file, eajson now surfaces deep keys like
  `payload.review.user.site_admin` with exact counts (differential vs Python
  `json`: 0 mismatches; keys deeper than the cap and arrays-of-objects remain
  out of scope, reported as omitted). **Behavior change:** files with 2+
  levels of nesting now report more keys by default.

- **`eaparquet` decodes INT96 timestamps and DECIMAL columns.** INT96 (the
  deprecated 12-byte Spark/Hive/Impala timestamp: nanos-of-day + Julian day)
  now labels as a timestamp and decodes footer min/max to ISO instants;
  DECIMAL columns decode their unscaled footer stats (INT32/INT64 LE, or
  FIXED_LEN_BYTE_ARRAY big-endian two's-complement) to the scaled value
  `unscaled / 10^scale`, reading scale from the modern `LogicalType.DECIMAL`
  or the legacy `ConvertedType` + `SchemaElement.scale`. Verified vs pyarrow
  (DECIMAL min/max exact; INT96 decode proven against hand-crafted bytes,
  since pyarrow omits INT96 statistics).

- **`eacrunch --where <col><op><value>` row filter.** Filter rows before
  aggregating (with `--by`/`--agg`) or before whole-column stats (standalone)
  — `SELECT … GROUP BY … WHERE …`. One predicate; operators `=` `!=` (string)
  and `>` `>=` `<` `<=` (numeric, non-numeric cells never satisfy an ordered
  comparison). On the group path it needs no kernel change — the predicate
  column is just another projected column in `csv_groupby_scan`, evaluated in
  Rust. `totals.rows` reports the matched-row count. Differentially verified
  against pandas (`df[df.col > v]`) on the 3M-row NYC-taxi file (0 mismatches).

## [3.1.0] — 2026-06-13

### Added

- **`eacrunch` GROUP BY (`--by <col> [--agg <op:col,...>]`).** Aggregate CSV
  rows grouped by a column instead of summarizing the whole file. `--agg`
  takes `sum`/`mean`/`min`/`max` `op:col` pairs plus a bare `count`; with no
  `--agg` you get per-group row counts. Driven by a new fused SIMD kernel
  (`csv_groupby_scan`) that projects only the key + aggregation columns in a
  single pass — no `len`-sized delimiter arrays, no full field grid, so
  scratch is O(rows·cols) not O(bytes) (peak RSS ~5.8× lower and ~1.8× faster
  than a grid-based pass on the 3M-row NYC-taxi CSV). Value parsing stays in
  Rust with the same finite-skip rule as whole-column stats, so group
  aggregates agree with column stats by construction — differentially verified
  against pandas `groupby().agg()` (0 mismatches) and cross-arch bit-identical
  (x86↔Pi NEON). Output adds the additive `groups[]` / `group_by` fields to
  the v1 `RuneOutput` (serialized only when grouping ran, so every other
  rune's JSON is byte-identical). Reachable from the one-shot CLI, `/rune`,
  and the LLM tool-call surface; high-cardinality keys (>~1M groups) fail loud
  rather than exhausting memory.

## [3.0.0] — 2026-06-13

Robustness campaign waves two–four: the encrypted vault, the web/WhatsApp
server, and the inference path stress-tested under failure and adversarial
input. Major version because the vault on-disk format changes incompatibly.

### ⚠ BREAKING

- **Vault on-disk format → v4 (`storage::vault`).** Append-only record log plus
  a double-buffered, two-fsync header commit, so a crash mid-append recovers
  every committed block (at worst losing only the in-flight one) instead of
  leaving the vault unopenable. **There is no v3→v4 migration: existing vaults
  are rejected as an unsupported version.** Back up `~/.olorin/vault/` if you
  care about its contents, or expect a fresh vault on first run. (Findings
  F1/F2.)

### Security

- **Vault concurrent-open lock (F3).** `Vault::open*` now takes an exclusive
  advisory file lock for its lifetime — two processes opening the same vault
  could previously append at the same offset and reuse a per-block ChaCha20
  nonce (a two-time-pad). A concurrent open is now rejected; the lock
  auto-releases on process exit.
- **Server auth-parser hardening (S3).** The bearer/cookie token parsers no
  longer slice fixed byte offsets on attacker-controlled header lines (a
  multibyte char at the boundary could panic the connection thread, pre-auth on
  an exposed bind). Boundary-safe prefix checks now.

### Fixed

- **Inference context-window guard (W1).** `Engine::generate` now refuses a
  prompt that fills the context window and bounds the decode loop, so an
  over-window prompt returns a clean error instead of panicking the KV cache
  (was a process crash on a long paste).
- **Server request-body memory + connection bounds (S1/S2).** The request body
  is read incrementally up to the cap instead of eagerly allocating the declared
  Content-Length (no more multi-MB allocation from a lying header), and the
  accept loop caps concurrent connections (`OLORIN_MAX_CONN`, default 64).

## [2.13.0] — 2026-06-12

Robustness wave one: every rune diffed against independent ground-truth
tools on real public files (NYC-taxi parquet, GH Archive, Chinook, NASA
logs). Four findings, zero crashes / OOM / wrong counts — every emitted
count matched exactly; the findings were presentation, semantics, and
contract, the class synthetic fixtures structurally cannot surface. All
four are resolved here.

### Changed

- **eajson null contract now matches eaparquet.** Keys that were JSON
  `null` in some records were silently dropped — `count` undercounted the
  non-null values and `null_count` was never reported (a key null in every
  record vanished). `count` is now total presence (non-null + null),
  `null_count` is always populated, and statistics are computed over the
  non-null values. A key null in every record stays omitted (eajson is
  value-typed, with no schema to declare an all-null column). **Output
  change:** every eajson field now carries `null_count`, and `count`
  includes nulls.
- **eaparquet TIMESTAMP columns render as ISO instants.** They previously
  reported min/max as the raw int64 epoch count (e.g.
  `1230764502000000`). Modern Parquet (pyarrow) annotates timestamps only
  via `LogicalType` — the deprecated `ConvertedType` is absent — so the
  footer reader now parses the `LogicalType` union (time unit +
  `isAdjustedToUTC`). Columns become `kind=timestamp` with ISO min/max,
  unit-aware, with a trailing `Z` only when UTC-adjusted. Verified on the
  3,066,766-row NYC-taxi file.

### Fixed

- **SQL Server / SQLite `[bracket]`-quoted identifiers are stripped.**
  easql reported `[Album]` instead of `Album` on the real Chinook SQLite
  dump, which broke eadiff's cross-dialect name matching. Counts were
  always correct.
- **Structured `--json` output is no longer rejected by the 32 KB cap.**
  That cap bounds what reaches the LLM context, but `--json` output is
  never narrated — it goes to stdout or another rune's input. Oversized
  machine output (e.g. an eatime series over GH Archive) was being replaced
  by an `output exceeded` error, corrupting the
  `olorin rune … --json > out.json` contract. Structured JSON now passes
  through whole; only the human-readable path stays capped.
- **eatime series bucket count is hard-capped.** A pathological timestamp
  span (one outlier instant years or millennia from the rest, as in real
  GH Archive data) could explode the series to 100 k+ buckets. The count is
  now bounded regardless of span.

### Internal

- `storage::parquet` split: the Thrift footer structs and decoders move to
  a new `storage::parquet_meta` module so both files stay under the
  500-line cap.

## [2.12.1] — 2026-06-12

### Fixed

- **eacorrelate no longer reports spurious correlations between files from
  disjoint time periods.** Found in the wild on day one: two NASA-log slices
  covering different weeks scored r=1.00 at a +5-day lag. Two defects, one
  root cause (window statistics vs global statistics):
  - The score was the cosine of *globally* z-scored overlap windows — a
    window with **zero events** z-scores to a constant vector, and a constant
    against anything can reach ±1. Scores are now the **per-window Pearson r**
    (window means/variances via prefix sums, O(1) per lag, kernel unchanged);
    a constant window has zero variance and scores 0. Both overlap windows
    must also contain ≥ 3 actual events — a correlation claim requires both
    streams *active* in the compared window.
  - Findings are now **positive-only**: files recorded in different periods
    anti-correlate by presence alone (one is active where the other is
    silent — the same NASA pair scored r=−0.54 at +3 days), and negative
    rate-correlation across event files is that artifact, not behavior.
  All existing goldens unchanged (for healthy windows Pearson ≡ the old
  cosine within wire rounding — the definitions diverge only where the old
  one was wrong). Differential oracle updated to the same spec + 3
  disjoint-era trap scenarios (18/18); new disjoint-era E2E regression.

## [2.12.0] — 2026-06-12

### Added

- **HTML reports — the shareable artifact.** `olorin report <files…> [-o
  out.html]` runs the deterministic file-drop pipeline (rune per file +
  `eacorrelate` across ≥ 2) and writes **one self-contained HTML file**:
  inline CSS, inline SVG charts, zero external assets, zero JavaScript.
  Findings banner first, then per-file sections with stats, time-series
  charts (same `col_reduce` kernel envelope as the terminal block bars;
  anomaly buckets tinted red, dashed median baseline), and flagged spikes.
  No model, no wall-clock timestamp — same inputs, same report; every
  file-derived string HTML-escaped. The 3am incident renders as ~10 KB.
- **`POST /api/report`** — same payload as `/api/analyze`, response is the
  report as an attachment. Never touches the model or the vault.
- **"📄 download report"** link in the web chat after every file-drop
  analysis — re-posts the same files to `/api/report` and saves the result.

## [2.11.1] — 2026-06-11

### Fixed

- **Correlation narration now works on the Pi (NEON).** v2.11.0 fed the
  machine-shaped findings block straight into the narration prompt; Gemma 4 on
  NEON either continued the pattern verbatim (echoed data fragments) or emitted
  nothing — x86 narrated fine, so this only surfaced on the Pi install-gate.
  Three changes, all Pi-verified on the real model:
  - The narration prompt gets a **prose** rendering of the top finding
    (`findings_for_prompt`) instead of the patterned block — nothing to
    continue. The user-visible block is unchanged.
  - The prose leads the prompt (conclusion first), restoring the original
    design intent.
  - On aarch64 with ≥ 3 files the full multi-block prompt still degenerates
    (immediate EOS or junk that slips the dump filter), so the prompt shrinks
    to the prose conclusion alone — same playbook as the aarch64 minimal chat
    prompt — and narration is skipped entirely when there are no findings.
    Also ~4× faster narration on the Pi (smaller prefill).
  Verified: 2-file drops narrate 4/4 on NEON, 3-file drops 3/3, all voicing
  the deploy→error lag; x86 e2e narration unchanged-or-better.
- `OLORIN_DEBUG_NARRATION=1` prints discarded narrations and which filter
  fired — narration discards were previously invisible, which is why the
  NEON echo failure needed a debug build to diagnose.

## [2.11.0] — 2026-06-11

### Added

- **`eacorrelate` — cross-file lag correlation.** New rune + new portable
  `corr_sweep.ea` kernel: drop 2–8 timestamped files (ISO-8601 or CLF,
  auto-detected per file) and Olorin reports which event streams move
  together, at what lag, and where the alignment peaks — *"syslog (errors)
  follows deploys.csv by +240s (r=0.93, peak 2026-06-11T03:02:00)"*. Every
  file contributes its event stream (ISO logs also a second ERROR/FATAL
  sub-stream); all streams share one time grid (512 target buckets — lag
  resolution is bucket width), get z-scored, and every cross-file pair is
  swept over ±128 lags in SIMD. Scores are the cosine of the lag-aligned
  overlap windows, bounded [-1, 1]; findings are direction-normalized
  (`stream_a` always follows `stream_b`) and carry the peak instant plus the
  honest `width_seconds` resolution. Entirely deterministic — the model never
  sees raw data. Verified by an independent numpy re-implementation of the
  full pipeline (`benchmarks/eacorrelate_diff.py`, 15/15 randomized scenarios)
  and a cross-arch golden.
- **Multi-file drop runs the correlation automatically.** Dropping ≥ 2 files
  into the web UI now streams the eacorrelate findings after the per-file
  kernel outputs, and the findings *lead* the narration prompt, so the model
  opens with the conclusion ("errors follow deploys by 4 minutes") instead of
  hunting for one across summaries. Silence when nothing correlates — no
  findings, no block.
- **`correlations[]` in the RuneOutput v1 contract.** Additive optional block
  (exactly the `anomalies[]` pattern): serialized only when non-empty, so all
  existing rune JSON stays byte-identical. Scores ride the wire rounded to 4
  decimals for cross-arch golden stability.
- **The 3am incident fixture** (`tests/fixtures/incident/`): deploys.csv +
  syslog with ERROR bursts 4 minutes after each deploy + an nginx CLF access
  log with matching retry storms. Drives the integration tests, the
  model-gated narration e2e, and the Pi install-gate scenario — and doubles
  as the launch demo asset.

## [2.10.0] — 2026-06-11

### Added

- **`easql` file-drop auto-pick.** Dropping a `.sql` file into the web UI now
  routes to `easql` automatically (`src/runes/select.rs`), the same deterministic
  extension-first gesture that already handles `.csv`/`.jsonl`/`.parquet` — no
  model decision, bulletproof on the Pi. The rune was previously reachable only
  from the CLI/REPL.

### Changed

- **`easql` attribution is now exact at any scale.** The SIMD sweep is chunked
  over newline-aligned windows instead of a single pass with an 8 192-position
  cap, so per-table row counts stay exact on dumps with millions of statements
  (`pg_dump --inserts` / `mysqldump --skip-extended-insert`, one INSERT per row).
  A window denser than the per-call buffer resumes past its last marker; nibble
  functions read the full buffer, so a statement straddling a window tail is
  still counted whole. Verified by a 100 K-statement two-table test that forces
  both window tiling and the in-window resume; real Chinook dumps unchanged
  (11 tables, 15 607 rows, both dialects).

## [2.9.0] — 2026-06-11

### Added

- **`easql` rune — SQL-dump summarizer.** Drop a `pg_dump` or `mysqldump`
  `.sql` file → dialect, table count, per-table row + column counts, in one
  SIMD pass. A new `sql_scan` kernel sweeps `CREATE`/`INSERT`/`COPY`
  (case-insensitive, word-bounded) the same way `log_level_scan` does; the rune
  does the cheap per-marker scalar work (table name, newline-count rows in a
  `COPY … FROM stdin` block for Postgres, quote-aware value-tuple count for
  `INSERT … VALUES`). The tuple counter is single-quote-aware (a `),(` inside a
  string value is not a row) and skips the optional column list
  (`INSERT INTO t (c1, c2) VALUES …`, which would otherwise read as one extra
  row per statement). Dialect is detected structurally, not by trusting a header
  comment: `COPY` → postgres; a backtick → mysql; a `pg_catalog`/`\connect`/
  `standard_conforming` fingerprint → postgres (so `pg_dump --inserts` is
  labeled correctly); else `sql`. Whole-word keyword matching means a
  `-- Create Tables` comment no longer registers a phantom table. Output maps
  tables onto the v1 `categories` contract, so the block-bar chart, `--json`
  pipe, and `eadiff` work on a SQL dump for free. Verified per-table against a
  real SQLite engine on the Chinook mysqldump + pg_dump (0 mismatches, 15 607
  rows) on Pi 5 NEON, and guarded by a pinned-stack large-dump canary. Scope:
  pg_dump COPY + INSERT dumps; file-drop auto-pick and per-table row
  attribution past 8192 keyword markers are follow-ups.

## [2.8.5] — 2026-06-11

WARNING/CRITICAL log levels return, now stack-safe (requires eacompute ≥ 1.15.1).

### Added

- **ealog counts `WARNING` and `CRITICAL` again — now stack-safe.** The feature
  (Python `logging` / syslog spellings folding into WARN/FATAL) was reverted in
  v2.8.3 because the larger `log_level_scan` SIMD body overflowed the stack on
  logs ≥ ~1 MB. Root cause was in the compiler: eacompute `alloca`'d loop-body
  locals *inside* the loop, so stack grew per iteration. Fixed upstream in
  **eacompute 1.15.1** (`hoist loop-body allocas to the function entry block`).
  With that, the kernel is rebuilt unchanged and verified safe by the
  pinned-stack canary (`tests/ealog_large_log_canary.rs`) — which runs ealog on
  an 8 MiB log inside a 2 MiB stack and aborted on the old compiler, passes now.
  Verified on x86 and Raspberry Pi 5 (NEON). **Requires building with eacompute
  ≥ 1.15.1.**

## [2.8.4] — 2026-06-10

Two rune correctness fixes from the robustness pass (differential vs
pandas/pyarrow). No format or behavior changes.

### Fixed

- **eajson: a JSON number that overflows f64 no longer poisons a field's
  stats.** JSON can't write `inf`, but a value like `1e400` parses to
  `f64::INFINITY`, and a single such value otherwise propagated through
  `sum`/`mean` (serialized as null) while `min`/`max` survived — the same
  internally-inconsistent, silently-wrong summary fixed in eacrunch. Non-finite
  values are now excluded, so the field summarizes its finite values
  consistently. Found by differential testing during the runes robustness pass.
- **eaparquet: a `u64` column's max stat no longer saturates to `i64::MAX`.**
  The per-row-group stat reduction round-trips the value through `f64` and then
  back via `as i64`, whose saturating float→int cast pinned any value above
  `i64::MAX` to `2^63` (~9.22e18) — so a `u64` max of `2^64-1` reported as ~half
  its true magnitude. Out-of-`i64`-range reduced values now stay `f64`. Found by
  differential testing against pyarrow during the runes robustness pass.

## [2.8.3] — 2026-06-10

Hotfix: revert the ealog WARNING/CRITICAL kernel change to stop a crash.

### Fixed

- **ealog no longer crashes (stack overflow) on logs ≥ ~1 MB.** The
  `WARNING`/`CRITICAL` matching added to the `log_level_scan` SIMD kernel in
  v2.8.1 made the per-chunk SIMD body large enough that stack usage grew with
  the input, overflowing the main thread's stack on any log of roughly a
  megabyte or more — i.e. most real logs. The kernel and the rune are reverted
  to their pre-v2.8.1 form (the five base levels `DEBUG`/`INFO`/`WARN`/`ERROR`/
  `FATAL`). `WARNING`/`CRITICAL` detection is temporarily removed and will
  return with a stack-safe implementation.

## [2.8.2] — 2026-06-10

Web-UI chart rendering fix for non-Linux viewers.

### Fixed

- **Block-bar charts no longer shear in the web UI on non-Linux clients.** The
  chart `.chart` CSS led with `DejaVu Sans Mono`, a font present on Linux/the Pi
  but not on Windows/macOS; remote viewers fell back to a font whose Block
  Elements glyphs (`▁▂▃▄▅▆▇█`) render at a different advance width than the
  digits/spaces, so the columns drifted and the bars looked sheared. A ~9 KB
  subset of DejaVu Sans Mono (ASCII + Box Drawing + Block Elements) is now
  embedded in `chat.html` via `@font-face`, so the grid renders at uniform
  width on every client OS. No new runtime/build dependency; the binary grows
  ~9 KB. Verified vertical on a Windows browser viewing the Pi. (The plot text
  itself was always column-aligned — this was purely a client-font fallback.)

## [2.8.1] — 2026-06-10

Bug-fix release. Corrections surfaced by the runes robustness pass (rune
correctness vs ground-truth tools) and by live use of the v2.8.0 web auth gate.

### Added

- **ealog now counts the `WARNING` and `CRITICAL` spellings.** Python's
  `logging` and syslog emit `WARNING`/`CRITICAL` as their literal level names,
  which the scanner previously missed (it matched only `WARN`/.../`FATAL`),
  silently undercounting those logs. `WARNING` folds into the WARN bucket and
  `CRITICAL` into FATAL (same severity tiers). Added to the `log_level_scan`
  SIMD kernel — scalar and SIMD-body paths — with word boundaries preserved
  (`WARNINGS`/`CRITICALLY`/`UNCRITICAL` still don't match). Verified on x86 and
  Raspberry Pi 5 (NEON). Found by the runes robustness pass.

### Fixed

- **eacrunch: a `nan`/`inf` cell no longer poisons a numeric column's stats.**
  Rust's `f64::parse` accepts the literals `nan`/`inf`/`infinity`, so a single
  such cell in an otherwise-numeric column propagated `NaN` through `sum`/`mean`
  (serialized as `null`) while `min`/`max` survived — an internally
  inconsistent, silently-wrong summary. Non-finite parses are now excluded, so
  the column summarizes its finite values consistently (count/min/max/sum/mean
  all agree). Found by differential testing against pandas during the runes
  robustness pass.
- **Web auth gate: a stale `olorin_auth` cookie no longer shadows a valid
  `?token=`.** When the server was restarted with a new `OLORIN_AUTH_TOKEN`, a
  browser still holding the previous session's cookie could not re-authenticate
  by pasting the correct `?token=` URL — the gate checked only the first
  presented credential (cookie before query) and returned `401`. It now accepts
  if *any* of Bearer header / cookie / query token matches, so a fresh
  `?token=` always recovers a browser with a stale cookie (and re-sets the
  cookie on the way in). Workaround on 2.8.0: open in a private window or clear
  the `olorin_auth` cookie.

## [2.8.0] — 2026-06-10

Security hardening across the agent tool sandbox and the web server, from a
full-codebase audit. No format, schema, or rune-output changes; one operational
behavior change for networked deployments (see Changed).

### Security

- **Agent tool sandbox hardened against prompt-injection abuse.** Three gaps
  where the file/network tools could be steered past their guards are closed.
  The `http`/`fetch` tool now refuses any non-`http(s)` URL — previously
  `http file:///etc/shadow` read arbitrary local files (bypassing the path
  guard) and other schemes opened SSRF/exfil channels; curl is additionally
  pinned with `--proto`/`--proto-redir` so a redirect can't reach `file://`.
  The shell command classifier no longer misses destructive or exfil commands
  hidden behind newlines or `$(…)`/backtick substitution, nor sensitive paths
  split by shell quotes (`~/.s""sh/id_rsa`). The `read`/`write`/`grep`/`ls`
  path guard now resolves symlinks and re-checks the real target, so a link
  under `$HOME` pointing into `~/.ssh` or `~/.olorin` can no longer slip past
  the denylist.
- **Web server is fail-closed when exposed off-loopback.** Binding a
  non-loopback address now requires `OLORIN_AUTH_TOKEN`, and every request must
  present it (`Authorization: Bearer`, an `olorin_auth` cookie, or a `?token=`
  bootstrap that sets the cookie). Loopback binds — the default — are unchanged
  and need no token. The token check is constant-time and runs before any
  dispatch, covering every endpoint including the term WebSocket.

### Changed

- **`OLORIN_BIND` to a non-loopback address now requires `OLORIN_AUTH_TOKEN`.**
  Previously the web UI could bind `0.0.0.0` with no authentication, exposing
  the shell/http/write tools to the network. If you run Olorin on a LAN
  address, set `OLORIN_AUTH_TOKEN=<secret>` and open
  `http://<host>:<port>/?token=<secret>` once; otherwise the server now exits
  at startup rather than expose the tool-running endpoints unauthenticated.
  Local (loopback) use is unaffected.

### Removed

- Dead `platform/hwid.rs` — the machine-id vault-key path it backed was
  superseded by passphrase + Argon2id in v2.0.0.

## [2.7.0] — 2026-06-10

SIMD charts and a scriptable rune CLI. The file-drop analyst now *draws* the
event rate over time, and any rune can be run one-shot straight from the shell.

### Added

- **Block-bar charts for time-series runes.** Drop a timestamped log into the
  web UI — or run `eatime --bucket series` in the REPL — and Olorin renders the
  event rate over time as a block-bar chart with spikes flagged and the median
  baseline drawn. A new `col_reduce` SIMD kernel downsamples the series to the
  canvas width (peak-per-column, so a one-bucket spike still towers); a single
  renderer serves both surfaces — ANSI colour in the REPL, monospace in the web
  chat. The y-axis auto-zooms for high-floor rate series and the x-axis carries
  dates across multi-day spans. Verified on a Pi 5 (NEON), both surfaces.
- **`olorin rune <name> [args…]` one-shot CLI.** Runs a single rune
  non-interactively and writes only its answer to stdout — no banner, no model
  load, no REPL — so `olorin rune eatime --bucket series --json access.log >
  out.json` yields clean JSON for matplotlib / jq / pandas. Works for every
  rune; exit codes 0 success / 1 failure / 2 usage.

## [2.6.0] — 2026-06-09

A recall correctness fix: updating a fact mid-conversation now takes effect.

### Fixed

- **Recall surfaced a stale fact after an update.**  With recall on, telling
  Olorin "my name is now X" and then asking again still returned the *old*
  value. The recall context builder deduplicated results *before* filtering
  out the question's own echo, so the freshest answer was dropped as a
  near-duplicate of that echo — which was then itself filtered, leaving a
  stale entry. Self-matches are now filtered before dedup, so the most recent
  fact wins.

## [2.5.0] — 2026-06-09

Instant web terminal. The shell tile in the web UI was slow and didn't echo
typed input — every keystroke was a fresh HTTP POST. It now streams over a
single WebSocket, so typing is instant on the Pi kiosk, and the path is
hardened against leaks and a couple of sharp edges the rework surfaced.

### Changed

- **Web terminal streams over WebSocket.**  Replaces the per-keystroke HTTP
  POST + Server-Sent-Events transport with one persistent WebSocket
  connection. Typing is instant; the old per-keystroke round-trip and missing
  echo are gone. Verified on a Pi 5 kiosk.

### Fixed

- **Web-terminal connection and session leaks.**  Closing a terminal tab now
  reliably tears down the server-side streaming thread, frees the session slot
  (the 8-session cap no longer fills up over time), and SIGTERMs the shell
  child — previously all three leaked until the process exited. A periodic
  WebSocket keepalive also detects a half-open socket so an idle, silently-
  dropped connection can't spin forever.
- **Blocked-command feedback restored.**  A command rejected by the safety
  scanner flashes the terminal tile border red again (the signal was lost in
  the transport rework).

### Security

- **Passwords no longer linger in the web terminal.**  Optimistic client-side
  echo could paint a typed character at a no-echo prompt (`sudo`, `ssh`),
  where the shell never overwrites it — leaving plaintext on screen. The
  client can't distinguish a normal prompt from a no-echo read (readline keeps
  `ECHO` off at the prompt too), so local echo was removed; the terminal now
  shows only what the shell itself echoes.
- **WebSocket frame-size bound.**  Inbound frames are capped at 16 MiB before
  allocation, so a malformed or malicious declared length can't trigger an
  unbounded allocation and abort the process.

## [2.4.0] — 2026-06-08

Fast chat on the Pi. The third and final latency win in the series: with the
prefill tax gone (2.3.0), the remaining cost was the model's chain-of-thought,
so on aarch64 it's now off by default.

### Changed

- **Thinking off by default on aarch64.**  Once the minimal system prompt
  removed the prefill tax, Gemma 4's `<|think|>` reasoning was the entire
  remaining chat cost. On the Pi it buys nothing — tool calls don't fire and
  reasoning holds without it — so it now defaults off there (x86 keeps it on).
  Measured on a Pi 5: a factual question went from 17.1s to **1.6s**, and the
  reasoning answer stayed correct *and* started showing its work (the
  chain-of-thought now lands in the visible answer instead of a discarded
  hidden block). The full series arc for a factual chat: 32s → 17s → 1.6s.

### Added

- **`/think` toggle.**  `/think` flips the model's chain-of-thought for the
  session; `/think on` / `/think off` set it explicitly. The fast path is the
  default; turn thinking on for a genuinely hard reasoning question.

## [2.3.0] — 2026-06-08

Faster on the Pi. Two independent latency wins that together cut local
narration and chat response time several-fold — by spending **fewer model
tokens**, not chasing per-token speed (decode is already at the
memory-bandwidth floor, so token count is the only lever left).

### Changed

- **Narration skips chain-of-thought.**  Narration restates a rune's
  already-computed result, so Gemma 4's `<|think|>` reasoning is pure cost.
  Disabling it for narration cut a Pi 5 narration from ~82s to ~18s (4.5×) —
  and the summaries got *sharper*, since the model stops abstracting away
  from the concrete numbers the kernel handed it.

- **Minimal chat system prompt on aarch64.**  The ~2.6 KB tools-framing
  system prompt was re-prefilled every chat turn — ~30s of the Pi's "30-40s"
  chat latency (a trivial answer took 32s with the block, 2s without). On
  aarch64 the NEON forward pass can't emit the `<tool_call>` XML that block
  frames anyway, so the Pi now uses a one-line identity prompt: factual chat
  32s → 19.5s, and reasoning accuracy improved (the block had been priming
  the model to reconcile non-existent tool context). x86_64 keeps the full
  block for autonomous tool-calling. The working Pi tool paths — `/weather`,
  the intent classifier ("weather haparanda"), and file-drop analysis — are
  unaffected.

## [2.2.1] — 2026-06-05

### Fixed

- **Scan time keeps sub-millisecond resolution.**  `eatime` and `ealog`
  rounded the SIMD scan to whole milliseconds, so a sub-millisecond pass
  displayed as an uninformative `0 ms` — exactly when the kernel is
  fastest. They now print microseconds below a millisecond
  (`scan: 716 µs`) and milliseconds above, via a shared `format_scan_time`.

## [2.2.0] — 2026-06-05

The file-drop analyst: drop a file into the web UI and Olorin analyzes it
on-device — a **deterministic rune pick**, a **SIMD kernel scan**, and a
**local-model narration**, with no autonomous tool-call in the loop. The
user's drop gesture *is* the "analyze this" decision, which is what makes
the whole flow reliable on a Raspberry Pi.

- **Deterministic rune selection (`pick_rune`).**  Maps a dropped file to
  the right rune by extension, with an ISO-8601 / CLF timestamp sniff that
  splits logs between `eatime` (timing spikes) and `ealog` (severities).
  No model call — so it's arch-independent and reliable on NEON.
- **Web UI drop zone + 📎 attach button.**  Drag a file (or several) onto
  the chat; each file's kernel output streams in, followed by a one- or
  two-sentence explanation from the local model.
- **`/api/analyze` + `/api/analyze_raw`.**  Multi-file drops go through a
  base64 JSON endpoint (with one cross-file correlation narration);
  single files **stream raw to disk in 64 KB chunks** (no base64, no
  in-memory JSON), so multi-GB logs upload without exhausting RAM. Body
  cap is configurable via `OLORIN_MAX_UPLOAD` (default 128 MB; raw path up
  to 4 GB). Staged uploads are deleted after analysis.
- Verified on a Raspberry Pi 5: a **1 GB real access log** (9.46 M CLF
  timestamps) SIMD-scanned in **768 ms**, a genuine traffic spike flagged
  and narrated locally, zero cloud.

### Fixed

- **`ealog` percentages** are now a share of all lines, not of the
  matched-severity subset — a log dominated by an untracked level (e.g.
  Apache `[notice]`) no longer reports the lone present level at 100%.
- **`ealog` samples** are deduplicated by line, so a line with two matched
  keywords (`[error] … in error state`) is no longer shown twice.
- **Multi-file narration** drops its synthesis line when the small model
  reformats the inputs into a table or over-thinks to silence, instead of
  emitting a garbled summary; the per-file rune outputs stand clean.
- **Web UI**: `no-cache` headers on served HTML (embedded `chat.html`
  changes every deploy); **live cpu/temp heartbeat during analysis**
  (`/api/system` no longer blocks on the ctx mutex the dispatch thread
  holds for the duration of a run).

## [2.1.0] — 2026-06-04

Feature release for `eatime`: the rune graduates from *describing* a log
to *detecting when its rate broke*, and learns a second timestamp
grammar.

- **`--bucket series` — chronological spike detection.**  Where `hour`
  and `weekday` collapse the time axis, `series` keeps it: it bins the
  file's span into auto-width buckets (1s…1w, ~120 buckets) and flags the
  buckets where the event rate broke from a **robust median/MAD
  baseline**.  Median-not-mean by design, so a large spike cannot inflate
  its own threshold and hide; a flat (MAD = 0) series falls back to a
  ratio test.  Spikes surface in `--json` as an additive `anomalies[]`
  array (`bucket`, `count`, `baseline`, `ratio`, `score`), emitted only
  when non-empty so every existing `--json` consumer and `eadiff` are
  byte-for-byte unaffected.
- **Common Log Format support.**  `eatime` now auto-detects ISO-8601 and
  the Apache/nginx access-log format `[dd/MMM/yyyy:hh:mm:ss]` (new
  `clf_scan` SIMD kernel, same cross-arch anchor idiom as
  `timestamp_scan`, zero new compiler intrinsics), and dispatches the
  matching kernel.  Detection runs both kernels over a 64 KB head and
  picks the dominant grammar, so the sniff can never disagree with the
  scan; force it with `--format iso|clf|auto`.  All three bucket modes
  work on both grammars via a unified epoch path.
- **Tooling.**  `benchmarks/eatime_diff.py` validates `series` bucket
  counts bit-for-bit against an independent pandas/regex grouping (ISO and
  CLF); verified on a 313K-timestamp systemd journal and cross-arch
  byte-identical on Raspberry Pi 5 (aarch64 NEON) vs x86_64.

No output contract changes for existing modes — `anomalies[]` is purely
additive and the ISO hour/weekday/`--json` output is unchanged.

## [2.0.8] — 2026-06-03

Correctness-hardening release for the rune family.  A real-input
differential audit — each rune fed a realistic file, its output diffed
against the incumbent tool (pandas, grep, pyarrow, python `json`) —
surfaced a silent correctness failure in four of the six runes.  Each
had passed its unit tests, which used clean fixtures (uppercase
severities, unquoted CSV, signed ints, scalar JSON); none exercised the
messy real-world shape that breaks them.  All four are fixed here with a
real-input regression test and ground-truth verification.  `eatime`
(validated on real GitHub-event data) and `eadiff` (directional deltas,
timestamp shifts, malformed/empty inputs) were audited and confirmed
correct.  No output contract changes for already-valid inputs — only
previously-wrong outputs change.

### Fixed

- **`eacrunch` — RFC-4180 quoted CSV (PR #10).**  `csv_scan` counted
  every comma and newline, including those inside double-quoted fields,
  so a quoted value with an embedded delimiter (`"Smith, John"`)
  silently mis-aligned every column after it — confident, wrong top-N
  with no error.  `kernels/csv_scan.ea` now threads a quote-parity bit
  through its existing scalar emit pass: a `"` toggles in-quote state,
  in-quote delimiters are not emitted, and escaped `""` resolves by
  double-toggle.  No new eacompute intrinsics; bit-identical across x86
  SSE2 / ARM NEON.  `eacrunch` additionally strips surrounding quotes and
  unescapes `""` from text values (via `Cow`, so the numeric hot path
  stays allocation-free).  Now matches pandas on quoted input.
- **`ealog` — case-insensitive severity matching (PR #11).**
  `log_level_scan` compared candidate bytes against UPPERCASE-only codes,
  so lowercase/mixed-case severities (`error`, `Error`, `warn`) counted
  as zero — and the rune reported "no severity keywords found" on real
  logs from Go (zap/logrus), Python `logging`, nginx, and journald, which
  all emit lowercase.  Candidate letter bytes are now folded with
  `| 0x20` before comparison in both the scalar and SIMD paths; only
  upper/lower letter pairs share a folded value, so non-letters never
  alias a keyword byte, and delimiter/newline checks keep the original
  bytes (word-boundary rejection of `ERROR_HANDLER` unchanged).
  Cross-arch bit-identical; uppercase goldens unchanged.
- **`eaparquet` — unsigned integer columns; INT96 mislabel (PR #12).**
  `UINT_8/16/32/64` columns are stored in a signed `INT32/INT64` physical
  type, distinguished only by the schema's `ConvertedType`.  The footer
  decoder ignored `ConvertedType` and decoded every stat as two's-
  complement signed, so any unsigned value above the signed max wrapped
  to a large negative number, reported with `success:true`.
  `storage/parquet.rs` now reads `ConvertedType` (SchemaElement field 6)
  and decodes `UINT` columns unsigned (correct sign and magnitude; the
  f64 stat pipeline keeps its pre-existing >2^53 precision ceiling, the
  same one signed `i64` already had).  Also fixes a cosmetic bug where
  any statistics-less Number column was labeled `[INT96 timestamp;
  min/max not decoded]`; now the neutral `[min/max not available]`.
- **`eajson` — numeric arrays no longer byte-decoded (PR #13).**  eajson
  ran every JSON array value through `decode_byte_array`, which accepts
  any array of integers 0–255 — intended only for systemd's binary
  `MESSAGE` field.  So an ordinary numeric array
  (`{"latencies":[12,45,78]}`) was silently reinterpreted as a binary
  string and rendered as garbage control characters.  The byte-decode is
  now gated to keys whose leaf is `MESSAGE`; any other array is skipped,
  consistent with the existing array-of-objects / array-of-strings
  scope.  The structural scan itself was already correct — embedded
  `,`/`:`/`{`/escaped-quote inside string values never desynced key
  pairing (verified across 200 lines of real GitHub-event JSON).

### Documentation

- **README + `docs/runes.md` `eatime` example rewritten around real data
  (PR #9).**  Replaces the synthetic 1.2 GB sample with a reproducible
  run on public GitHub-event data from gharchive.org, measured on a
  Raspberry Pi 5, including input acquisition (`curl … | gunzip`) and the
  valid-input rule (RFC3339 `T` separator; space-separated syslog/Apache
  not matched).  Throughput claims are now split into the isolated
  `timestamp_scan` kernel figure (1.80 GB/s Pi 5 NEON, 6.34 GB/s Ryzen
  SSE2) and the density-dependent end-to-end rune figure (~1.4–3 GB/s),
  matching `benchmarks/results.md`.

### Verified

- Per-rune ground-truth diff: `eacrunch` vs pandas (quoted CSV), `ealog`
  vs grep on real `/var/log`, `eaparquet` vs pyarrow (UINT32/UINT64),
  `eajson` vs python `json`.
- New regression tests: `csv_scan_skips_quoted_delimiters`,
  `eacrunch_handles_quoted_csv`, `case_insensitive_lowercase` /
  `case_insensitive_mixed` / `case_fold_no_false_positive`,
  `ealog_counts_lowercase_and_mixed_case`,
  `eaparquet_decodes_unsigned_columns`,
  `eajson_does_not_byte_decode_numeric_arrays`.
- Cross-arch parity gate green; all existing goldens unchanged (the
  fixes only alter previously-wrong outputs).

## [2.0.7] — 2026-05-22

Windows x86_64 returns to the release matrix.  Previous attempts to
build natively on `windows-latest` were blocked by eacompute's
native-Windows linker passing `/NODEFAULTLIB` to `lld-link`, leaving
`expf` and other libc-math symbols unresolved.  v2.0.7 sidesteps the
issue by cross-compiling from Ubuntu via MinGW-w64, matching the
local dev flow — eacompute's MinGW path (`src/lib.rs:219`)
statically links MinGW's libc, so kernels calling `expf`/`fmaf`
resolve cleanly.  No behavioral changes for Linux users.

WhatsApp bridge (`wa-bridge.exe`) is still skipped on Windows
pending the `mattn/go-sqlite3` → `modernc.org/sqlite` swap.

### Changed

- **`.github/workflows/release.yml` — Windows entry restored.**
  Runs on `ubuntu-24.04` with `rust_target: x86_64-pc-windows-gnu`,
  installs `mingw-w64` from apt, builds Olorin via
  `cargo build --release --target x86_64-pc-windows-gnu`.  Staging
  step looks in `target/<triple>/release/` when a `rust_target` is
  set in the matrix.
- **`.github/workflows/ci.yml` — new `build-windows-cross` job.**
  Mirrors the release configuration so PRs catch Windows breakage
  before tag day.  ~1 min build, no MSVC infrastructure.
- **`build.rs::find_ea`** — replaced `Command::new("which")` with a
  manual cross-platform PATH walk.  Git for Windows' `which` returns
  MSYS-style paths that Windows `CreateProcess` rejects with OS
  error 3.  Adds an `EA` env var override for explicit cross-compile
  setups.

### Verified

- CI run 26273492385 green on `main`: Windows cross-build 1m15s,
  Linux build+test 4m52s.
- `olorin.exe` artifact produced at
  `target/x86_64-pc-windows-gnu/release/olorin.exe`.

## [2.0.6] — 2026-05-21

Argon2id `argon2_block_compress` x86 path rewritten from scalar
u64 to AVX2 SIMD u64x4.  The 16-u64 Argon2 P transform state is held
as four u64x4 vectors (rows a/b/c/d as lanes) so the four column G'
calls run in lockstep; the diagonal phase uses the canonical
Blake2b rot-left-by-row permutation to align lanes.  Zero new
eacompute intrinsics required — pure operator chain on the v1.12
u64x4 type plus a couple of compile-time-constant shuffles.

ARM (Pi 5 production) path stays scalar.  An attempt to write the
same kernel as u64x2 NEON SIMD regressed 1.7× because NEON has no
u64×u64 SIMD multiply instruction and LLVM falls back to scalar mul
via FMOV/MUL/INS lane round-trips.  Full memo in
`memory/project_argon2_u64x2_arm_null.md`.

### Changed

- **`kernels/argon2_block.ea` — x86 AVX2 SIMD path (u64x4) with
  `#[cfg(x86_64)]` gate.**  The masked BlaMka multiply
  `(a .& mask) .* (b .& mask)` lowers to a single `vpmuludq` per
  BlaMka site (LLVM `combineMul` detects provably-zero upper 32 bits
  and picks `vpmuludq` directly).  Rotate-by-32 lowers to `vpshufd
  $177` (single halves-swap uop on modern x86); rotate-by-24/16/63
  lower to `vpshufb` and shift|or pairs as LLVM sees fit.
  Diagonalization is three `vpermq` per `p_at` half.  Eight `store`
  calls per output tile.
- **`kernels/argon2_block_arm.ea` — NEW.**  Scalar path, identical
  source to the pre-v2.0.6 cross-arch `argon2_block.ea`.  build.rs
  strips the `_arm` suffix and produces `argon2_block.so` for the
  aarch64 target.

### Verified

- `argon2id_rfc9106_section_5_2_vector` KAT bit-exact on both
  x86_64 and aarch64.
- `vault_default_params_are_deterministic` KAT passes on both
  arches.

### Not done (parked for next session)

- ARM SIMD `argon2_block_compress` via explicit `wmul_u64_lo(u32x4,
  u32x4) -> u64x2` (eacompute v1.12.0) with bitcast + lane
  deinterleave to feed u32x4 inputs from masked u64x2.  Estimated
  2-3× win over scalar on Pi 5; the operator-chain attempt is null,
  the explicit-intrinsic path has not been tried.  Design sketch
  and risk list in `memory/project_argon2_u64x2_arm_null.md`.

### Requires

- eacompute ≥ v1.12 at build time (`u64x4` type + full operator
  suite).  `release.yml` uses `petlukk/eacompute@main`, currently
  at `v1.14.0-4` — covered.

### Performance

No perf claim in this release.  Per the discipline established in
the v2.0.5 addendum: rigorous A/B benches require co-built binaries
and interleaved run order.  The x86 SIMD asm is verified to emit
`vpmuludq` and `vpshufd $177` (the optimal patterns) — the
performance probe is a follow-up.

## [2.0.5] — 2026-05-21

`gemma4_gelu` kernel rewritten on top of the v1.14.0 `tanh_approx_f32`
intrinsic.  The prior implementation derived tanh from
`exp_poly_f32` via the identity `tanh(x) = 1 - 2 / (exp(2x) + 1)` plus
a manual `[-50, 50]` clamp for the exp domain.  That identity is
catastrophic-cancellation-prone near `x ≈ 0` — the numerator
`exp(2x) - 1` and the denominator `exp(2x) + 1` both approach 2 from
opposite directions, and small f32 rounding errors in `exp_poly_f32`
get amplified by the subsequent division.  The v1.14.0 changelog
named Olorin's `gemma4_gelu` as the motivating consumer for the
replacement; this lands the swap.

### Changed

- **`kernels/gemma4_gelu.ea` — SIMD path now calls `tanh_approx_f32`
  directly.**  Rational `P(x²) · x / Q(x²)` approximation, ~3e-7 max
  abs error, internal clamp to `[-9, 9]` (where tanh saturates to ±1
  within a few ulps).  Removes 3 splats (`v_two`, `v_clamp_hi`,
  `v_clamp_lo`), the manual clamp, the `exp_poly_f32` call, and the
  `1 - 2/(e+1)` reconstruction — one intrinsic call replaces the
  composition.  Scalar tail (≤3 elements per call) is unchanged.
- **`step14_gelu_vs_llama_ref` parity vs llama scalar reference:**
  max abs 4.8e-7, max rel 5e-5 — well under the 1e-3 / 1e-4
  test tolerances.

### Performance

- **Pi 5 (Cortex-A76, OLORIN_THREADS=3, performance governor pinned,
  q3kffnimpl production model):** decode median 137.8 ms/token →
  136.9 ms/token (-0.9 ms, +0.7 % t/s, 7.26 → 7.30 t/s).  The
  improvement is consistent across every percentile (min/p25/median/
  p75/p95 all -0.6 to -0.7 %), distributions translated rather than
  reshaped.  ~3200 decode-timing samples per binary, 3 runs × 120 s
  each.  GELU is 0.5 % of per-token decode time; the observed end-
  to-end win exceeds the pure-GELU share, plausibly because the
  smaller kernel reduces L1i pressure on the surrounding
  `gemv_gate+up` bandwidth-bound path.

### Requires

- eacompute ≥ v1.14.0 at build time (`tanh_approx_f32` intrinsic).
  `release.yml` checks out `petlukk/eacompute@main`, which is at
  `v1.14.0-4` as of this release — covered.

### Performance addendum — RETRACTING THE PERFORMANCE CLAIM (measured 2026-05-21, interleaved A/B)

**The `+0.7 % t/s decode` claim in the Performance section above is
withdrawn.**  A rigorous remeasurement on the same day this release
shipped — both v2.0.4 and v2.0.5 binaries co-built from source on
the same toolchain, runs interleaved (`A r1 → B r1 → A r2 → ...`),
OLORIN_THREADS=3, performance governor pinned, q3kffnimpl model,
same fixed 672-token prompt — shows v2.0.5 and v2.0.4 are
statistically indistinguishable on **both** prefill and decode.

#### Prefill (10 interleaved runs per binary, per-stage means ± σ in ms)

| stage | v2.0.4 | v2.0.5 | Δ |
|---|---:|---:|---:|
| total prefill | 27030.4 ± 47.2 | 27081.7 ± 68.5 | +0.19 % (+2.0σ, noise) |
| attention | 5112.4 ± 10.7 | 5112.0 ± 12.3 | -0.01 % (zero) |
| `gelu_mul` | 413.9 ± 3.2 | 410.4 ± 4.3 | **-0.84 %** (-2.0σ, the kernel changed) |
| all other stages | unchanged within noise | | |

`gelu_mul` is 1.5 % of prefill — propagated effect ≈ 0.01 %, within noise.

#### Decode (5 interleaved runs per binary, ~1500 per-token samples per binary)

| percentile | v2.0.4 (ms) | v2.0.5 (ms) | Δ |
|---|---:|---:|---:|
| min | 134.40 | 134.60 | +0.20 |
| p25 | 135.90 | 136.10 | +0.20 |
| **median** | **137.70** | **137.70** | **0 (identical)** |
| p75 | 139.70 | 139.80 | +0.10 |
| p95 | 141.20 | 141.20 | 0 |

σ ratio v204/v205 = 2.75 / 3.12, comparable — no environmental
disparity, interleaving worked.  Distributions sit on top of each
other rather than "translated" as the original claim implied.

#### What v2.0.5 actually delivers

The catastrophic-cancellation-elimination is real: `gemma4_gelu`
SIMD output is bit-close to llama's scalar reference (max abs
4.8e-7, max rel 5e-5, well under the 1e-3 / 1e-4 tolerances).  The
worry the original entry described — that the identity
`tanh(x) = 1 - 2/(exp(2x) + 1)` amplifies small `exp_poly_f32`
errors near `x ≈ 0` — was addressed by switching to v1.14.0's
`tanh_approx_f32`.  **v2.0.5 is a correctness improvement with no
measurable end-to-end performance delta on Pi 5.**

#### Why the original claim was wrong

Both the `+0.7 % decode` and the (also-retracted, then re-checked)
`-4 % attention` claims came from sequential A/B benches:
`v2.0.4 r1..rN` first, then `v2.0.5 r1..rN`.  On a Pi 5 sharing
the host with kiosk chrome and routine background load, the second
binary's run window often catches a more-settled state than the
first's.  σ on v2.0.4 stages was 5-15× larger than σ on v2.0.5
stages in the prefill repro — the smoking-gun for time-based drift,
not a code-level effect.  Interleaving the runs forces both
binaries to see the same time-distribution of Pi state, and the
delta becomes code-attributable.  Both prefill and decode go to
zero under this protocol.

Discipline going forward: A/B perf benches use co-built binaries
(both built from source the same day on the same toolchain) AND
interleaved run order.  Either alone is insufficient.

## [2.0.4] — 2026-05-20

Follow-up to v2.0.3's narration prompt-shape fix.  v2.0.3 stopped the
trailing-template-echo failure mode but the same long-output runes
(eatime, eajson) then started rambling / hallucinating fragments from
the data instead of summarising it — a different failure but still
garbage on top of clean kernel output.

### Changed

- **`build_narration_prompt` skips narration for answers over 600 bytes.**
  Empirically calibrated on the production Pi: working runes top out
  around 422 bytes (eaparquet), failing runes start around 857 bytes
  (eatime).  600 B splits the two regimes with ~180 B headroom on
  the lower side and ~250 B on the upper side.  Above the threshold
  the kernel output is shown unaccompanied — no LLM call, no decode
  wait, no garbage narration appended.  Bad narration was strictly
  worse than no narration; this restores the "kernel output is the
  product, model narration is a bonus" contract.
- `NARRATION_MAX_ANSWER_BYTES` is a `pub const`; tests assert the
  threshold rather than hard-coding bytes counts.

## [2.0.3] — 2026-05-20

Two production-data-discovered correctness fixes, bundled as one patch:
the eajson + eacrunch numeric-stats kernels lose precision above the
f32 lossless boundary (commit `9295897`), and the rune-narration prompt
makes Gemma 4 echo a trailing template fragment for repetitive-pattern
rune outputs.  Both were caught while running the install scripts on
the production Pi 5 against real public datasets on 2026-05-20.

### Fixed

- **eajson / eacrunch numeric stats — f32 → f64.**  The aggregator
  was parsing JSON numbers / CSV cells to f32 and feeding the
  `f32_stats` SIMD kernel.  Above the ~16.7M lossless f32 boundary,
  distinct integers collapse into a single f32 bucket.  Symptom:
  GitHub `payload.push_id` values (~3.4×10¹⁰, f32 spacing 4096)
  all reported as a single identical value above the actual data
  max.  Fixed by switching the aggregator to `Vec<f64>` and calling
  the (already-shipped) `f64_stats` kernel.  Schema unchanged
  (`NumericStats` was always `f64`); behaviour-only change.  Two new
  regression tests with the original 25-push-id sample bake the
  oracle in.
- **Rune-narration prompt — drop the trailing instruction.**  The
  user prompt used to repeat the system prompt's "respond in 1-2
  plain-English sentences" instruction at the end.  For runes with
  long repetitive output (eatime's 24 hour buckets, eajson's
  ~28 key lines), Gemma 4 generated next-most-likely tokens that
  *continued the trailing instruction* instead of summarising the
  data — sometimes hallucinating markdown bolding and repeating the
  echo 2-3 times.  The new user prompt is just
  `Output of \`<rune>\`:\n\n<answer>` with no trailing nag; the
  system prompt at `NARRATION_SYSTEM_PROMPT` carries the entire
  instruction load.  Five new tests in
  `tests/narration_prompt_shape.rs` lock the new shape in
  (no template fragments leak through, data is preserved verbatim,
  long outputs don't reintroduce a trailing instruction).
  Existing `runes_llm_wiring` tests updated to match.

## [2.0.2] — 2026-05-20

Release-pipeline patch for v2.0.1.  The v2.0.1 tag was created but
never produced an attached GitHub Release because the Windows matrix
entry failed at eacompute's `llvm-sys` step (chocolatey's `llvm`
package is binaries-only, not a dev SDK with `llvm-config.exe`).
Rather than block the public-launch release on a half-day Windows
LLVM setup, v2.0.2 ships **Linux x86_64 + Linux aarch64 only**;
Windows is deferred to a later release.

### Changed

- **`.github/workflows/release.yml`** — Windows dropped from the build
  matrix.  Reinstating it needs an LLVM 18 MSVC developer
  distribution on the runner (the recipe is in
  petlukk/Ea_showcase/`build-windows.bat`: `LLVM_SYS_*_PREFIX` pointing
  at a real dev install, plus the `libxml2s.lib` stub), not just the
  binaries-only chocolatey `llvm` package.
- **`README.md`** — Install section reflects the actual platform
  coverage (Linux x86_64 / aarch64).  The PowerShell installer
  remains in `scripts/install.ps1` for when Windows releases resume,
  but the README no longer advertises it as installable today.

### Known limitations

- **Windows is not in this release.**  No `olorin.exe` or
  `wa-bridge.exe` is published for v2.0.2.  Windows users who want to
  run Olorin can build from source per the contributor instructions
  in the README.  CI work to ship Windows binaries cleanly is
  tracked as the next platform-coverage item.

## [2.0.1] — 2026-05-20

Public-launch package.  No code-behaviour changes, no on-disk format
changes — this is the release that adds a `curl | sh` install path
plus per-platform release binaries, so external users can grab Olorin
without a Rust + Ea + Go toolchain on their machine.  Olorin also
gains a native env-file reader so the installer's API-key prompt
turns into something the binary actually picks up at startup.

### Added

- **`scripts/install.sh`** — pipe-safe bash installer (Linux x86_64 +
  aarch64).  Resolves the latest release tag from GitHub, downloads
  `olorin-<target>`, optionally prompts for `ANTHROPIC_API_KEY` and
  writes `~/.olorin/env` mode 0600, optionally downloads the WhatsApp
  bridge, optionally updates the user's `~/.bashrc` / `~/.zshrc`.
  Verifies SHA256 against the release's `SHA256SUMS` when published.
  All prompts route through `/dev/tty` so `curl | sh` works.
- **`scripts/install.ps1`** — PowerShell mirror for Windows x86_64.
  Installs to `%LOCALAPPDATA%\Olorin\bin`, updates User-scope PATH
  via `[Environment]::SetEnvironmentVariable`, same env-file +
  bridge + checksum flow as the bash script.
- **`src/config.rs` + `load_env_file()`** — native reader for
  `~/.olorin/env` (`KEY=VALUE`, `#` comments, leading `export `
  tolerated, paired single/double quotes stripped).  Called as the
  first line of `main()`; process env wins over file env, so any
  `ANTHROPIC_API_KEY` already exported by the shell still beats the
  installer's file write.
- **`tests/config_env_file.rs`** — 11 parser tests (quoting,
  comments, export-prefix, invalid keys, embedded equals, trailing
  whitespace, unmatched quotes).
- **`.github/workflows/release.yml`** — matrix build on `v*` tag
  push (or manual `workflow_dispatch`) across Linux x86_64,
  Linux aarch64 (`ubuntu-24.04-arm`), and Windows x86_64
  (`windows-2022`).  Each runner builds eacompute, Olorin, and the
  Go bridge (Go 1.25, CGo enabled, `-trimpath -ldflags='-s -w'`).
  A separate release job collects all artifacts, generates
  `SHA256SUMS`, and publishes a GitHub Release with auto-generated
  notes.  `workflow_dispatch` input is validated against `v[0-9]*`
  before being fed to `action-gh-release`.
- **`SECURITY.md`** — disclosure policy at the repo root: GitHub
  Security Advisories as the private channel, 7-day ack / 30-day
  fix best-effort, supported-versions table, explicit in-scope
  (vault crypto, SecureBuffer, path/shell guards, prompt injection,
  constant-time, memory safety) and out-of-scope (weak passphrase,
  sophisticated injection, host compromise, side channels, DoS,
  third-party models, cloud fallback) lists.

### Changed

- **`README.md`** — leads with `curl | sh` and `iwr | iex` install
  commands; the from-source build instructions move below as the
  contributor path.  Project-stats table updated: 44 → 49 logical
  kernels (matches reality).
- **`docs/architecture.md`** — refreshed counts: 64 → 71 kernel
  source files, 42 → 49 logical kernels, 60 → 97 test files,
  318 → 571 tests, 20 → 19 built-in tools.

## [2.0.0] — 2026-05-19

Arc #3 (passphrase + Argon2id KDF, the v2.0 vault story) — Blake2b
primitive, full Argon2id, the `derive_key` rewrite + salt
persistence + vault format bump to v3, the interactive REPL
passphrase prompt, and the Web UI + WhatsApp fail-fast guard.  v2
vaults are rejected; per [[feedback-no-migration-for-private-repo]]
there is no migration path.  Major bump because the on-disk vault
format is incompatible with v1.2.x — existing vaults must be
regenerated under the new passphrase flow.

### Added (this slice)

- **`src/platform/random.rs`** — cryptographically secure random
  bytes via the OS entropy source (Linux `getrandom(2)` syscall,
  fallback `/dev/urandom`; Windows `BCryptGenRandom`).  Used to
  generate the per-vault salt at first open.
- **Salt persistence** — `<vault_dir>/vault.salt` holds 16 random
  bytes generated at first vault create.  The salt is treated as
  public-but-stable: it makes Argon2id outputs unique across vaults
  but isn't itself a secret.  Losing the salt means losing the
  vault (no recovery path — exactly what the threat model says).
- **`Vault::open_with(dir, passphrase, kdf)`** — explicit KDF
  parameters for tests.  Production callers use `Vault::open()`,
  which pins `Params::VAULT_DEFAULT` (64 MiB, t=3, p=1).
- **`Params::TEST_FAST`** — minimum-cost Argon2id profile (8 KiB,
  t=1) for the vault-test suite; keeps test wall-clock under control
  without skipping the KDF entirely.
- **`tests/vault_passphrase.rs`** — 5 tests covering salt creation,
  salt reuse, wrong-passphrase rejection, v2 vault rejection, and
  `derive_key` determinism.

### Added (REPL prompt slice)

- **`src/platform/term.rs`** — `read_secret(prompt)` reads a line
  from `/dev/tty` (Unix) or `CONIN$` (Windows) with echo disabled,
  returning the bytes in a SecureBuffer.  Termios / ConsoleMode is
  restored even on error paths.  No third-party dep — direct
  `tcsetattr` + `BCryptGenRandom`-style FFI through `libc`.
- **`Router::prompt_for_passphrase(is_new)`** — wraps `read_secret`;
  prompts once for an existing vault, twice (with confirmation) for
  a fresh vault so a typo doesn't lock conversation history.

### Changed (REPL prompt slice)

- **`Router::open_vault` passphrase sourcing** — interactive tty
  prompt is now the primary source; `OLORIN_PASSPHRASE` env var
  stays available as a non-interactive fallback (CI / scripts).
  Persistence is disabled only when both are unavailable.

### Added (Web UI + WhatsApp auth slice)

- **`DispatchContext::has_vault()`** — true iff the encrypted vault
  opened.  Server entry points read this before binding a port; the
  REPL ignores it (interactive use can still ask one-off questions
  without persistence).
- **`tests/server_vault_required.rs`** — 2 end-to-end tests: both
  `--serve` and `--whatsapp` exit non-zero with an explanatory
  stderr message when no passphrase source resolves (no tty + no
  `OLORIN_PASSPHRASE`).  The WhatsApp test also asserts that the
  bridge subprocess is *not* spawned in the failure case.

### Changed (Web UI + WhatsApp auth slice)

- **`--serve` and `--whatsapp` refuse to start without a vault.** An
  unattended server with persistence silently disabled would lose
  every conversation on restart and never load the vault-stored API
  key — a footgun the operator never gets to see.  The REPL keeps
  the lenient "disabled, but you can still chat" path.
- **WhatsApp bridge spawn order** — `run_whatsapp` now builds the
  dispatch context (which prompts for the passphrase) *before*
  spawning the bridge subprocess, so a failed vault open no longer
  leaves an orphan bridge process behind.

### Changed (this slice)

- **`Vault::open` signature** — `(dir: &Path, passphrase: &[u8])`
  instead of `(dir: &Path)`.  Migration impact: every
  in-repo caller updated.  `OLORIN_PASSPHRASE` env var sources the
  production passphrase until task #4 adds an interactive prompt.
- **`key::derive_key` signature** — `(passphrase, salt, params) ->
  Result<[u8; 32]>`.  Hwid mixing removed entirely (arc #4
  collapsed, per [[next-security-arcs-2026-05-18]]).
- **Vault format version 2 → 3.**  Byte layout unchanged; only the
  version byte advanced.  v2 vaults return `"unsupported vault
  version"` on open.

Removed the v1 → v2 vault migration shipped in v1.1.0.  Olorin has
always been a private, single-user repo; the migration served a
userbase that doesn't exist, so it was dead-on-arrival per the
"no premature features" hard rule.  Clearing it now also shrinks
the baseline that arc #3 (passphrase + Argon2id) will land on.

v1 vaults are no longer recognised — opening one now returns
`unsupported vault version`.  No v1 vaults exist in the wild, so this
has no observable user impact; it just stops shipping the upgrade path
for a hypothetical migration.

### Added

- **`kernels/blake2b.ea`** — Blake2b compression function (RFC 7693
  §3.2) as a generic-arch Ea kernel.  IV constants are passed in via
  a `*u64 constants` parameter rather than baked into the kernel
  source: the IVs are data (fractional bits of √2..√19), not
  algorithm, and pushing them to the Rust side keeps the kernel free
  of magic constants — same data-vs-algorithm split that GGUF weights
  use.  Scalar u64 ops; same source compiles on x86_64 and aarch64.
- **`src/storage/blake2b.rs`** — variable-output (1..=64 byte) Rust
  wrapper around the kernel: one-shot `hash()` plus a streaming
  `Hasher` API that defers compression of the staged block until
  another byte is known to be coming (required for correct counter
  + final-flag accounting per RFC 7693 §3.3).
- **`tests/blake2b_kat.rs`** — 7 known-answer tests: RFC 7693
  Appendix A `Blake2b-512("abc")`, Blake2b-512/256 empty-input
  references, chunked-vs-one-shot parity, exact-block-boundary
  behaviour, multi-block input handling, and digest-length distinctness.
- **`kernels/argon2_block.ea`** — Argon2 G compression
  (RFC 9106 §3.4) as an Eä kernel.  Same `g_prime` / `p_at` pattern
  as Blake2b but with the Argon2-specific extra-multiply
  (`2 * trunc(a) * trunc(b)`) on every addition — the one thing that
  stops Blake2b from being usable as a drop-in inner step.  Column
  step uses a gather/scatter through a 16-u64 scratch buffer; same
  source compiles on x86_64 and aarch64.
- **`src/storage/argon2id.rs`** — full Argon2id (RFC 9106): H₀
  construction, H′ variable-output Blake2b with chained-mode for
  T > 64, segment fill with Argon2i / Argon2d mode switch at SL/2 in
  slice 0 of pass 0, and the XOR-into-existing-block path for
  passes > 0.  Public surface: `argon2id(password, salt, secret, ad,
  Params, out)` and `Params::VAULT_DEFAULT` (64 MiB, t=3, p=1).
- **`tests/argon2id_kat.rs`** — 5 tests including the canonical
  RFC 9106 §5.2 vector (byte-exact), determinism / wrong-passphrase
  regression guard for `Params::VAULT_DEFAULT`, and input-validation
  rejections (short salt, zero iterations, output-length mismatch).
- **`src/kernels/ffi_crypto.rs`** — split out from `ffi.rs` so that
  file stays under the 500-LOC hard rule with Blake2b + Argon2 added.
  Same re-export pattern as `ffi_data.rs`.

### Removed

- `Vault::migrate_v1_to_v2` and the v1 dispatch arm in
  `Vault::open_existing`; the function now opens v2 directly (the
  separate `open_v2` helper was folded back into `open_existing`).
- `vault_format::read_all_v1_plaintexts` and its `OpenOptions` /
  `Read` / `Seek` / `crypto` / `key` imports.
- `tests/vault_migration_v1_to_v2.rs` (191 lines) — the entire test
  file covering the removed migration path.

### Changed

- `src/storage/vault.rs` 465 → 417 lines.
- `src/storage/vault_format.rs` 261 → 188 lines.

## [1.2.2] — 2026-05-18

Internal refactor only: `src/storage/vault.rs` was 702 lines, breaking
the project's 500-line hard rule.  All on-disk-format types and helpers
(`VaultHeaderV2`, `IndexEntry`, nonce derivation, AAD building, v1
plaintext reader) move to a new sibling `src/storage/vault_format.rs`.
Public API is preserved: `olorin::storage::vault::{VaultHeaderV2,
HEADER_SIZE_V2}` still resolve via `pub use`.  v1.2.1 vaults open
unchanged; no on-disk format change.

Originally also queued for this release: arc #4 (hwid-mixing one-way
function).  Cancelled mid-session — the proposed swap from XOR-cascade
to xxhash64-keyed chains would have rotated `derive_key()`'s output,
silently breaking every existing v1.2.x vault.  Folded into the v2.0
arc #3 work (passphrase + Argon2id) where key derivation breaks by
design.

### Changed

- **`src/storage/vault.rs` 702 → 465 lines** by extracting all on-disk
  format code to `src/storage/vault_format.rs` (261 lines).  No
  behaviour change; all 318 tests pass unchanged.
- **`src/storage/mod.rs`** — registers `vault_format` as a private
  sibling module.  External callers use the `vault.rs` re-exports.

## [1.2.1] — 2026-05-18

Release-readiness patch: first public-ready Olorin. Adds MIT LICENSE,
GitHub Actions CI, status badges, a restructured README with the rune
catalog and architecture details extracted to `docs/`, and a full
Security & threat model section calling out what the vault protects
against and what is queued for v2.0. One small test-stability fix
(no production-code change).

### Added

- **`LICENSE`** — MIT.
- **`.github/workflows/ci.yml`** — Linux x86_64 build + test, with
  sibling `eacompute@main` checked out and cached by SHA. First green
  run: 9m7s; subsequent runs hit the cache.
- **`docs/runes.md`** — full rune catalog moved out of README. Per-rune
  samples, the `--json` chaining contract, and limits live here.
- **`docs/architecture.md`** — source layout, kernel inventory, runtime
  layout, hard-rules contract — moved out of README for contributors.
- **Security & threat model section** in README — three structured
  parts (protects-against / does-not-yet / designed-for), names the
  v2.0 passphrase + Argon2id work explicitly.
- **CI + License badges** at the top of README.

### Changed

- **README trimmed from 654 → 242 lines.** Pitch + Olorin Pipe + a
  runes section that pushes runes as the categorical differentiator
  now land above the fold. Performance tables already lived in
  `benchmarks/results.md`; README now mentions and links rather than
  reproducing.
- **`.gitignore`** — `/docs/` rule changed to `/docs/*` with explicit
  `!` negations for the two public docs files. Internal docs
  (`PI5_OPTIMIZATION_PLAN.md`, `handoffs/`, `superpowers/`) stay
  local-only.

### Fixed

- **`thread_detection` env-var race** — test serialized via Mutex.
  Previously could flake under parallel test execution when other
  tests mutated `OLORIN_THREADS`.



Inbound safety upgrade: score-based injection matching with multi-language
patterns + two-form normalization closes word-variant, punctuation/spacing,
and Swedish-language bypasses against v1.1.x's exact-keyword matcher.
Outbound scan, leak detection, and the fused SIMD candidate kernel are
unchanged.

### Added

- **Score-based injection matcher.** Patterns carry a score (1 = weak,
  2 = strong).  Input is blocked when the total >= `INJECT_THRESHOLD`
  (2).  Single weak signal alone does not block — natural language often
  contains words like "ignore", "system", "forget" — but two weak signals
  OR one strong signal does.
- **Two-form normalization.** Input is matched against both a *spaceful*
  form (Unicode lowercase, alphanumerics with non-alnum runs collapsed to
  single ASCII spaces) and a *spaceless* form (alnum-only).  Spaceful
  catches punctuation bypasses ("ignore.previous", "ignore   previous");
  spaceless catches letter-spacing ("i g n o r e p r e v i o u s" →
  "ignoreprevious").  Single-word patterns match only spaceful with
  word-boundary requirements to avoid cross-language false-positives
  (Swedish "ignorera" does NOT trigger English "ignore").
- **Swedish injection patterns.** Strong: `ignorera tidigare`,
  `ignorera alla tidigare`, `glöm allt`, `du är nu`, `låtsas vara`,
  `agera som`, `nya instruktioner`, `uppdaterade instruktioner`,
  `strunta i`, `föregående instruktioner`.  Weak: `ignorera`, `glöm`.
- **Adversarial corpus at `tests/safety_inbound_corpus.rs`** — 56 cases
  across 8 categories (naive_en, variant_en, obfusc, sv, chatml, fp_en,
  fp_sv, fp_overbroad_en).  Strict-asserts every case.  Adding a corpus
  case is how new bypasses (or new false-positives) get tracked into CI.

### Changed

- **`act as` is now a weak signal (1).**  v1.1.x treated it as a binary
  block, incorrectly catching legit "Can you act as a code reviewer for
  this PR?" requests.  The injection-shaped pattern "Act as a system
  administrator" still blocks (`act as` weak + `system` weak = 2).

### Fixed (bypasses against the v1.1.x exact-keyword matcher)

- Word variants — `ignored previous`, `ignoring previous`,
  `Forgetting all instructions`, `Acting as a system admin`,
  `Pretending to be the user`, `You were now a pirate`.
- Punctuation insertion — `ignore.previous.instructions`,
  `ignore,previous,instructions`, `Ignore-previous-instructions`,
  non-break-space variants.
- Multi-space and letter-spacing — `ignore   previous instructions`,
  `i g n o r e   p r e v i o u s`, `Y O U  A R E  N O W an admin`.
- Swedish-language injection — all `sv` corpus cases (10/10) now caught.
  v1.1.x caught only one (`system: du är nu en pirat`) because of the
  literal `system:` prefix.

### Out of scope (memo-tracked for future)

- Indirect framing ("what would you say if not bound by your rules") —
  the score-based matcher doesn't cover semantic injection without
  trigger words.  Full-ML option from `project_next_security_arcs.md`
  arc #2 stays a separate project.
- Other locales (Spanish, French, German) — pattern table extension
  only; same matcher design.

## [1.1.1] — 2026-05-15

Doc-only patch.  README architecture overview and "The Vault" diagram
brought up to date with the v1.1.0 AEAD format.  No code change; no
on-disk format change; v1.1.0 vaults open unchanged.

### Fixed

- README's "The Vault" diagram still showed `block --> xxHash64 verify
  --> ChaCha20 decrypt` for the read path.  Now reads `Poly1305 verify
  --> ChaCha20 decrypt` to match the v1.1.0 AEAD format.
- README intro to "The Vault" still said "encrypted at rest using
  ChaCha20" — updated to "ChaCha20-Poly1305 AEAD" so first-time readers
  see the authenticated half.
- README write-path diagram updated from "ChaCha20 encrypt" to
  "ChaCha20-Poly1305 seal" (matches `src/storage/aead.rs::seal`).
- README search-path diagram now shows the per-block Poly1305 verify
  gate that runs before each `FusedSearcher` candidate — surfacing the
  verify-then-search behavior added in v1.1.0.
- README architecture tree's `storage/key.rs` entry no longer advertises
  `xxHash64`.  The function survives as an internal helper used only by
  the v1→v2 migration verifier; it is no longer part of the live vault
  read or write flow.

## [1.1.0] — 2026-05-14

Vault crypto upgrade: per-block ChaCha20-Poly1305 AEAD with binding AAD,
a Poly1305 tag over the header + index, constant-time tag compare on
open, and verify-then-search in the fused decrypt path.  v1 vaults
auto-migrate to v2 on first open by a v1.1.0 binary, with the original
preserved as `vault.bin.v1.bak`.

### Added

- **Vault format v2 (ChaCha20-Poly1305 AEAD).** Per-block integrity now
  uses Poly1305 (RFC 8439) over `aad || ciphertext` instead of v1's
  xxhash check.  The AAD binds `key_id || version || nonce_counter ||
  timestamp || histogram`, so a swapped or copied block fails open even
  if the attacker controls both halves.
- **Header MAC.** The 64-byte v2 header carries a `header_tag` field
  that Poly1305-MACs `header[0..46] || serialized_index` using a
  domain-separated nonce (the high bit of the 4-byte counter slot is
  reserved for this domain — see the tightened wrap guard below).  Any
  byte flipped inside the MAC region — `block_count`, `index_offset`,
  `key_id`, `nonce_seed_8`, `header_rewrites`, or any index entry —
  fails open with `Error::Vault("vault header or index has been
  tampered")`.
- **Auto-migration of v1 vaults.** On first open by a v1.1.0 binary,
  v1 vaults are streamed, xxhash-verified, re-encrypted as v2 blocks,
  and atomically renamed into place.  The original is preserved as
  `vault.bin.v1.bak`.  A corrupt v1 fails before any backup or write,
  so users can't lose data to a silent migration error.
- **Constant-time tag verification.** `poly1305_verify` exported by
  both x86 and ARM kernels runs OR-reduce + branchless `is_zero` on
  the 16-byte XOR difference.  No early-exit; no byte-position
  timing leak.
- **Verify-then-search.** `Vault::search` MAC-verifies each candidate
  block before handing it to `FusedSearcher`.  A tampered block drops
  silently — no block index, no partial line, no signal reaches the
  result set.
- `src/storage/aead.rs` — `seal` / `open` / `verify` (verify-only,
  no decrypt) for ChaCha20-Poly1305 with AAD.
- `kernels/poly1305.ea` + `kernels/poly1305_arm.ea` — Poly1305 SIMD
  kernel, 5×26-bit radix using `wmul_u64_lo` / `wmul_u64_hi`.
- Cross-arch bit-identity golden at `tests/fixtures/aead_golden.bin`
  (1040 bytes = 1024 ct + 16 tag).  x86 ≡ ARM NEON.

### Changed

- **Vault block counter wrap tightened** from `u32::MAX` to
  `0x80000000`.  The high bit of the 4-byte counter slot is now
  reserved for the header-MAC nonce domain, so an honest counter must
  never set it.  At one append per second this still gives ~68 years
  before exhaustion.
- `IndexEntry.xxhash` is now `_reserved` — on-disk layout preserved
  (zeros written, ignored on read), so a v1→v2 migration can stream
  entries without shifting offsets.
- `FusedSearcher::search` now takes a `ctr_init: i32` parameter
  (v1 vaults passed 0, v2 passes 1 — counter 0 is the Poly1305 OTK).
- `Vault::read_encrypted_block` returns `(ct, tag, nonce)` instead of
  `(ct, nonce)`; only `Vault::search` uses it.

### Removed (breaking on-disk format)

- v1 vault format.  v1 vaults auto-migrate to v2 on first open;
  external tooling that parsed v1 directly must update.
- `Vault::last_block_hash()` — xxhash slot is gone.  Per-block AEAD
  tag is a stronger and constant-time integrity check.

### Security findings closed

- v1's xxhash-on-plaintext was non-cryptographic — an attacker who
  could write to `vault.bin` could substitute matching plaintext +
  precomputed xxhash and read it back without detection.  v2's
  Poly1305 tag is key-dependent so this attack now requires the key.
- v1 had no header integrity — `block_count`, `nonce_seed`, or index
  bytes could be tampered with at rest and the vault would open
  normally.  v2's `header_tag` closes this.

## [1.0.0] — 2026-05-12

Marks the rune family as production-ready. **No new features over
0.9.4** — this is a stability commitment, not a feature release.
Promotes the contracts that solidified over the v0.8.x and v0.9.x
lines to "stable" status.

### Stable contracts

The following surfaces are now stable and will not change without
a corresponding 2.0 bump:

- **`RuneOutput` v1 schema** and its compact JSONL wire format
  (`schema_version: 1`, `rune`, `source`, `totals`, `fields[]`,
  `categories[]`, `samples[]`, `error`). The v1 shape carried six
  runes and every `FieldKind` variant without a field addition;
  the design is settled.
- **`--json` mode behavior**: structured output is emitted
  verbatim (no `<rune_output>` wrap, no `[timing: …]` footer, no
  LLM narration). Refusal paths emit JSON too. Safety scan runs
  on the raw bytes regardless of format.
- **Six rune names and their core argument shape**: `eacrunch`,
  `eajson`, `eaparquet`, `ealog`, `eatime`, `eadiff`. Each accepts
  `[--json] <path>`; `eatime` also accepts `--bucket hour|weekday`;
  `eadiff` takes two paths.
- **Cross-rune chaining semantics**: match by exact name across
  `fields[]` and `categories[]`. Numeric deltas signed in
  `numeric.mean`. Bool fields split into paired
  `<col>.true_delta` / `<col>.false_delta`. Timestamp emits
  `<col>.unique_delta` + `<col>.min_shift_s` / `<col>.max_shift_s`.
  Text emits `<col>.unique_delta` plus per-value
  `<col>:<value>.count_delta` and `[appeared in top]` /
  `[disappeared from top]` markers. Asymmetric structural changes
  use `[appeared] <name>` / `[disappeared] <name>` Mixed markers.
  Categories use `+<name>` / `-<name>` for symmetric deltas.

### Out of scope for the 1.0 commitment

These exist in the codebase but are not promised stable:

- Internal kernel function signatures, the FFI loader, and the
  `KernelTable` shape. Private to the implementation.
- Specific throughput numbers (measured but workload-dependent).
- The Gemma 4 inference stack. Still evolving on a separate
  cadence; the 1.0 promise covers the rune family only.
- Web UI / WhatsApp gateway. Present and functional but not
  audited to the same depth as the rune dispatch path.

### Final state at 1.0

- 6 runes
- 1 cross-arch SIMD kernel family (csv_scan, jsonl_struct,
  log_level_scan, timestamp_scan, f32_stats, f64_stats), all
  validated on x86 SSE2 and ARM NEON
- 77 test suites, 423 passing tests, 0 failures
- 0 build warnings
- No file over the 500-LOC cap except the two documented
  chacha20 fused decrypt+search kernels (Ea has no module
  system; monolithic is required)

## [0.9.4] — 2026-05-12

### Fixed

- `--json` REPL dispatch contract: `/rune <name> --json …` now emits
  the structured JSONL answer verbatim. Previously the answer was
  wrapped in `<rune_output untrusted="true">` tags (broke
  parseability), suffixed with a `[timing: Nµs]` footer (broke
  JSONL line semantics), and routed through the LLM for narration
  (defeated the user's machine-output intent). The safety scan
  still runs on the raw JSON bytes — prompt-injection patterns
  inside file-derived string values stay blocked regardless of
  format.

### Changed

- `RuneResult` gains a `structured: bool` field set by each rune's
  `run()` when `--json` is in the args. Threads through
  `wrap_rune_result` (skips wrap), `build_narration_prompt` (returns
  `None`), and the REPL dispatch path (omits footer).

## [0.9.3] — 2026-05-12

### Added

- `eadiff` now emits Timestamp range-shift seconds: alongside the
  existing `<col>.unique_delta`, Timestamp diffs emit
  `<col>.min_shift_s` and `<col>.max_shift_s` carrying the signed
  second-deltas of the range endpoints. A range moving forward by
  one day reads as `+86400.00`. Garbage ISO strings silently omit
  the shift fields instead of crashing or emitting fake values.
- `eadiff` now emits Text top-N value comparison: per-value entries
  for each Text column. Values present in both top-N with different
  counts emit as `<col>:<value>.count_delta`; values appearing in
  only one side emit as `[appeared in top] <col>:<value>` /
  `[disappeared from top] <col>:<value>` Mixed markers.
- `benchmarks/timestamp_scan_bench.c` — C-driver throughput
  benchmark for the eatime kernel. Measured **6.34 GB/s on Ryzen
  7700X (SSE2)** and **1.80 GB/s on Pi 5 Cortex-A76 (NEON)** on a
  100 MB synthetic log.

### Changed

- `tests/runes_eadiff.rs` split at the FieldKind axis to keep both
  files under the 500-LOC cap: core dispatch + asymmetric + chain
  tests in the original file, per-Kind diff modes in
  `tests/runes_eadiff_kinds.rs`. Shared helpers extracted to
  `tests/common/eadiff_helpers.rs`.

## [0.9.2] — 2026-05-12

### Added

- `eadiff` handles every `FieldKind`. Number deltas unchanged. Bool
  fields split into paired `<col>.true_delta` / `<col>.false_delta`.
  Timestamp and Text emit `<col>.unique_delta`. Asymmetric keys (in
  one input but not the other) emit `[appeared] <name>` /
  `[disappeared] <name>` with kind=Mixed. Mismatched kinds emit
  `[kind-changed] <name>`. Same encoding works for `fields[]` and
  `categories[]`.
- `eatime --bucket weekday`: 7-bucket Mon..Sun histogram via
  Zeller's congruence on the year/month/day digits the kernel
  already extracted. Same `timestamp_scan` pass, different scalar
  post-processing. Default `--bucket hour` keeps the v0.9.1
  24-slot output bit-identical.

## [0.9.1] — 2026-05-12

### Added

- New rune `eatime`: ISO-8601 hour-of-day histogram. New SIMD
  kernel `timestamp_scan.ea` emits byte offsets where
  `YYYY-MM-DDT` occurs in a file; Rust extracts HH and buckets.
  All 24 hour-of-day slots always emitted (deterministic for
  downstream `eadiff`). Single `.ea` source; cross-arch via
  structural-anchor + scalar-validate idiom (no `movemask`, no
  `sat_sub`). Bit-exact bucket counts on x86 SSE2 and ARM NEON.
- New rune `eadiff`: structural delta between two prior `--json`
  rune outputs. Match-by-name across `fields[]` and `categories[]`.
  Generic — the same code handles `eatime × eatime`,
  `eacrunch × eacrunch`, `eaparquet × eaparquet`, etc. — never
  branches on the source rune.
- `src/kernels/ffi_data.rs` extracted from `ffi.rs` (data-plane
  kernel wrappers: csv_scan, jsonl_struct_scan, log_level_scan,
  timestamp_scan, f32_stats, f64_stats). Re-exported so existing
  call sites continue to compile.

## [0.9.0] — 2026-05-12

The structured-output release. Every rune now produces a stable
`RuneOutput` v1 contract that downstream runes (and shell pipes)
can consume mechanically.

### Added

- `RuneOutput` v1 schema in `src/runes/output.rs`. Cross-rune JSON
  contract with three orthogonal axes — `fields[]` (per-column),
  `categories[]` (bucketed counts), `samples[]` (exemplar
  records) — plus `source`, `totals`, and `error`. NaN/Inf
  serialize as `null`; round-trips through
  `to_json` / `from_json` via `storage::json` (zero new deps).
- `--json` flag on all four existing runes (eacrunch, eajson,
  ealog, eaparquet). Refusal paths also emit JSON so chained
  downstream runes get a parseable failure rather than free-form
  text.

### Changed

- All four existing runes migrated to use `RuneOutput` as the
  source of truth. Both the JSON form and the legacy text form are
  rendered from the same in-memory `RuneOutput` — they cannot
  drift. Zero schema fields added during the migration; the v1
  shape was sized correctly from the union of existing rune
  outputs.
- `src/runes/eajson.rs` split into `eajson.rs` + the new
  `eajson_aggregate.rs` to keep both files under the 500-LOC cap.

### Fixed

- `open_capped` `NotFound` error now reads `"file not found"`
  instead of the debug-format `"open failed: NotFound"`. Latent
  in all four migrated runes; surfaced by the structured-output
  failure-path tests.
- Deterministic top-N text ordering in `eacrunch` and `eajson`:
  same-count values now sort by count desc, value asc. Previously
  HashMap-seed-dependent across runs.

## [0.8.5] — 2026-05-11

### Fixed

- Windows rune allowlist: `std::fs::canonicalize` on Windows
  returns the verbatim form (`\\?\C:\…`), which
  `Path::starts_with` treats as a distinct prefix from the
  non-verbatim home. Now canonicalizes `home` too — no-op on Unix.
  Latent since v0.8.1 because no rune was invoked in the
  windows-port acceptance criteria.

## [0.8.4] — 2026-05-11

### Fixed

- Kernel extraction is now atomic: write-to-tmp + rename pattern
  with per-PID/per-thread tmp paths. `cargo test` parallel mode
  works again — previously concurrent processes raced on
  `~/.olorin/lib/{version}/lib<name>.so` writes.

## [0.8.3] — 2026-05-11

### Added

- `ealog` now records byte offsets of the first 5 high-severity
  (ERROR / FATAL) matches and surfaces sample lines in the rune
  output. Kernel extended with a `(out_positions, max_positions,
  out_n_positions, scratch)` signature pair using the same
  store-mask-to-scratch + scalar-walk pattern as `csv_scan` and
  `jsonl_struct`. `reduce_add` early-out keeps the no-match common
  case scratch-free; Pi 5 NEON perf unchanged at 0.43 GB/s.

## [0.8.2] — 2026-05-11

### Added

- New rune `ealog`: SIMD log severity scanner. Counts word-bounded
  `DEBUG / INFO / WARN / ERROR / FATAL` plus newline bytes in one
  pass. Cross-arch single `.ea` source; validated bit-exact on x86
  SSE2 (Ryzen 7700X) and ARM NEON (Pi 5 Cortex-A76). Measured 0.70
  GB/s on Ryzen, 0.43 GB/s on Pi 5. The
  `reduce_add`-on-u8x16-with-`select` idiom (no `movemask`) is the
  template future cross-arch byte-scanning kernels follow.

## [0.8.1] — 2026-05-11

### Added

- Windows parity release. Olorin now builds and runs natively on
  Windows: inference (mmap GGUF + 35-layer forward + 16-thread
  futex pool), embedded ConPTY terminal panel (input + output +
  resize), web chat + SSE, slash commands (`/cpu`, `/weather`,
  `/time`, `/sh`), vault key derivation machine-bound (HKLM
  MachineGuid). 22 commits from `windows-port` branch
  rebase-merged via the repository's first PR.

### Changed

- Cross-platform abstractions added under `src/platform/`: every
  libc-equivalent syscall behind a thin wrapper (mmap, futex,
  home_dir, hwid, sysinfo) or behind a trait + `_unix.rs` /
  `_windows.rs` backends (Spawner, PtyBackend). Result:
  `cargo check --target x86_64-pc-windows-gnu` produces a finite,
  mechanical TODO list of cfg-gated unimplemented backends.

## [0.8.0] — 2026-05-07

### Added

- Repositioned as "deterministic SIMD analyst with LLM narration".
  README rewritten to lead with the runes (kernel-first) story.
- `--strict` CLI mode: disables the LLM entirely. No model load,
  no narration. Starts in ~25 ms vs ~25 s with the model. Useful
  for fast one-shot CLI use and security-conscious deployments
  that need a categorical "this binary will never call an LLM"
  guarantee.
- `--audit <path>`: writes a JSON Lines log of every dispatch
  turn. Two events per turn (input received + dispatch result
  with phase + microsecond timing). Captures metadata only — no
  input text or rune output content.
- New rune `eaparquet`: Parquet footer summarizer. Reads
  metadata only (never decodes column data). Per-column min/max
  via `f64_stats` SIMD reduction across row groups. Milliseconds
  even on multi-GB files.
- `benchmarks/` driver: eacrunch vs pandas on 10K / 100K / 1M
  transaction CSVs. Numbers in
  [`benchmarks/results.md`](benchmarks/results.md).

### Changed

- 4 over-cap source files split (`router`, `router_tools`,
  `server`, `parquet`); helper modules extracted
  (`thrift_compact`, `router_streaming`).

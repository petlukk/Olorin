//! eacorrelate — cross-file lag correlation. Given 2..8 files carrying
//! timestamps, finds which event streams move together, at what lag,
//! and where the alignment peaks ("errors spiked 4 minutes after the
//! deploy").
//!
//! SIMD strategy: per file the existing position kernels extract the
//! event streams (`timestamp_scan`/`clf_scan` via `stream::scan_for`,
//! `log_level_scan` for the ERROR/FATAL sub-stream); all streams are
//! bucketed onto ONE shared time grid, z-scored, and every cross-file
//! pair is swept by the `corr_sweep` kernel over ±MAX_LAG_BUCKETS lags.
//! Rust does the small glue (bucketing, z-score, argmax) mirroring the
//! eatime/anomaly split: kernels for the bandwidth- and MAC-heavy work,
//! scalar orchestration around them.
//!
//! Grid width targets 512 buckets (vs eatime's 120) because lag
//! resolution IS bucket width; findings report `width_seconds` so the
//! precision is always visible. No model involvement — narration (PR 4)
//! only ever sees the compact findings.

use super::{Rune, RuneResult, OutputSafety};
use super::common::{resolve_path, open_capped, truncate_answer, PathError};
use super::correlation::Correlation;
use super::incident;
use super::output::{Category, RuneOutput, Totals};
use super::stream::{self, Format, MAX_POSITIONS};
use super::substream;
use super::timekey::seconds_to_iso;
use crate::kernels::ffi;
use std::path::PathBuf;
use std::time::Instant;

const RUNE_VERSION: i64 = 1;
/// Finer grid than eatime's 120: lag resolution equals bucket width.
const TARGET_BUCKETS: i64 = 512;
const MAX_LAG_BUCKETS: i64 = 128;
/// Absolute ceiling on a reported lag, in seconds. The `span/4` cap below is a
/// STATISTICAL bound (keep enough overlap for a credible Pearson r); this is a
/// PHYSICAL one. An incident cascade — deploy → errors → traffic drop — plays
/// out in seconds-to-minutes, never hours. On a multi-day log `span/4` alone
/// permits absurd lags (a real srv1174152 syslog/auth pair reported a
/// confidence-0.60 "errors → auth rises 16h later", lag 59400s, r=0.62, built
/// from unrelated fwupd noise + SSH bot bursts — found 2026-06-15). Cross-file
/// alignment beyond an hour on long logs is dominated by diurnal periodicity,
/// not causation. Trade-off: cascades slower than 1h on multi-day spans are not
/// claimed — feed a narrower window to resolve those.
const MAX_LAG_SECONDS: i64 = 3600;
/// A correlation window must span at least this many buckets — Pearson over a
/// handful of points is meaningless (hits ±1 trivially). Floor for short grids.
const MIN_OVERLAP_BUCKETS: usize = 8;
const SCORE_THRESHOLD: f64 = 0.5;
/// Streams with fewer events than this can align by luck; skip them.
const MIN_EVENTS: usize = 3;
const TOP_K: usize = 3;
const MAX_FILES: usize = 8;

pub struct Eacorrelate;
pub const RUNE: Eacorrelate = Eacorrelate;

impl Rune for Eacorrelate {
    fn name(&self) -> &'static str { "eacorrelate" }
    fn description(&self) -> &'static str {
        "Correlate event streams across 2-8 timestamped files via SIMD. \
         Buckets every file's events (ISO-8601, CLF, or syslog auto-detected; \
         ERROR/FATAL lines in ISO logs and HTTP 5xx in CLF access logs form a \
         second 'errors' stream per log) onto one time grid and sweeps all \
         cross-file pairs over ±128 lags with the corr_sweep kernel. Reports \
         the strongest lags as correlations[] — 'events in A follow events in \
         B by N seconds'. Args: [--json] <path> <path> [...]."
    }
    fn usage(&self) -> &'static str { "eacorrelate [--json] <path> <path> [...]" }
    fn output_safety(&self) -> OutputSafety { OutputSafety::UntrustedQuoted }

    fn run(&self, args: &str) -> RuneResult {
        let t0 = Instant::now();
        let mut json_mode = false;
        let mut paths: Vec<&str> = Vec::new();
        for tok in args.split_whitespace() {
            if tok == "--json" { json_mode = true; } else { paths.push(tok); }
        }
        let out = if paths.len() < 2 || paths.len() > MAX_FILES {
            error_output(&format!("usage: {} (got {} file(s))", self.usage(), paths.len()))
        } else {
            let pairs: Vec<(String, String)> = paths.iter()
                .map(|p| (basename(p), p.to_string()))
                .collect();
            correlate_files(&pairs)
        };
        let answer = if json_mode {
            out.to_json()
        } else if let Some(err) = &out.error {
            err.clone()
        } else {
            format_text(&out)
        };
        RuneResult {
            answer:     truncate_answer(&answer),
            details:    None,
            success:    out.success,
            timing_us:  t0.elapsed().as_micros() as u64,
            structured: json_mode,
        }
    }
}

fn error_output(msg: &str) -> RuneOutput {
    let mut out = RuneOutput::new("eacorrelate", RUNE_VERSION);
    out.success = false;
    out.error = Some(msg.to_string());
    out
}

/// One event stream: a file's timestamps, or its ERROR/FATAL subset.
struct EventStream {
    name:     String,
    file_idx: usize,
    epochs:   Vec<i64>,
}

/// Library entry shared by the rune (display = path basename) and the
/// file-drop analyst (display = the upload's friendly filename, path =
/// the staged tmp file). Stream names in the findings come from
/// `display`, so the narration speaks the user's filenames.
pub fn correlate_files(files: &[(String, String)]) -> RuneOutput {
    let home = crate::home_dir().unwrap_or_else(|| PathBuf::from("/tmp"));
    let mut streams: Vec<EventStream> = Vec::new();
    let mut triggers: Vec<incident::TriggerCandidate> = Vec::new();
    let mut scan_us_total: u64 = 0;

    for (idx, (display, path)) in files.iter().enumerate() {
        let bytes = match resolve_path(path, &home).and_then(|p| open_capped(&p, &home)) {
            Ok(b) => b,
            Err(PathError::NotFound) =>
                return error_output(&format!("file not found: {path}")),
            Err(PathError::TooLarge(n)) =>
                return error_output(&format!("file too large: {n} bytes ({path})")),
            Err(PathError::OutsideAllowlist) =>
                return error_output(&format!("path rejected: outside allowlist (~ or /tmp only): {path}")),
            Err(PathError::Io(e)) =>
                return error_output(&format!("io error: {e} ({path})")),
        };
        if bytes.len() > i32::MAX as usize {
            return error_output(&format!("file too large for scan: {} bytes ({path})", bytes.len()));
        }
        let format = stream::detect_format(&bytes);
        let scan = stream::scan_for(&bytes, format, MAX_POSITIONS);
        scan_us_total += scan.scan_us;
        let epochs = stream::positions_to_epochs(&bytes, &scan.positions, format);
        if epochs.len() >= MIN_EVENTS {
            // Error sub-stream first (it borrows scan.positions), so the
            // streams vector still lists the all-events stream first. ISO/syslog
            // logs split on ERROR/FATAL keywords; CLF access logs on HTTP 5xx.
            let errors = match format {
                Format::Iso => substream::iso_errors(&bytes, &scan.positions),
                Format::Clf => substream::clf_errors(&bytes, &scan.positions),
                Format::JsonEpoch => substream::json_errors(&bytes, &scan.positions),
                Format::Syslog => substream::syslog_errors(&bytes, &scan.positions),
                Format::Apache => substream::apache_errors(&bytes, &scan.positions),
            };
            streams.push(EventStream { name: display.clone(), file_idx: idx, epochs });
            if errors.len() >= MIN_EVENTS {
                streams.push(EventStream {
                    name: format!("{display} (errors)"),
                    file_idx: idx,
                    epochs: errors,
                });
            }
        } else if !epochs.is_empty() {
            // Too sparse to be a correlation stream (e.g. a one-line deploy log).
            // Keep its events as discrete trigger candidates the incident builder
            // can snap its anchor onto — how a real deploy actually appears.
            triggers.extend(epochs.into_iter().map(|t| incident::TriggerCandidate {
                time:   t,
                source: display.clone(),
            }));
        }
    }

    let mut out = RuneOutput::new("eacorrelate", RUNE_VERSION);
    out.totals = Totals {
        rows:    streams.iter().map(|s| s.epochs.len() as u64).sum(),
        scan_us: scan_us_total,
    };
    out.categories = streams.iter().map(|s| Category {
        name:  s.name.clone(),
        count: s.epochs.len() as u64,
    }).collect();

    let distinct_files: std::collections::BTreeSet<usize> =
        streams.iter().map(|s| s.file_idx).collect();
    if distinct_files.len() < 2 {
        // Not an error: the honest answer is "nothing to correlate".
        return out;
    }
    let (correlations, incident) = correlate(&streams, &triggers);
    out.correlations = correlations;
    out.incident = incident;
    out
}

/// Bucket all streams onto one grid, z-score, sweep every cross-file
/// pair with the corr_sweep kernel, keep the TOP_K strongest findings.
fn correlate(streams: &[EventStream], triggers: &[incident::TriggerCandidate]) -> (Vec<Correlation>, Option<incident::Incident>) {
    let gmin = streams.iter().flat_map(|s| s.epochs.iter()).min().copied().unwrap_or(0);
    let gmax = streams.iter().flat_map(|s| s.epochs.iter()).max().copied().unwrap_or(0);
    let span = gmax - gmin;
    if span <= 0 {
        return (Vec::new(), None); // everything in one instant — no lag structure
    }
    let width = stream::auto_width(span, TARGET_BUCKETS);
    let n = (span / width) as usize + 1;
    // Cap the lag at a quarter of the grid so the overlap window stays at least
    // 3/4 of the span — a lag that consumes most of the window leaves too few
    // buckets for a credible Pearson r (the NASA +654h r=1.00 artifact). On the
    // usual fine grids (n in the hundreds) n/4 exceeds MAX_LAG_BUCKETS, so this
    // only binds on short, long-span logs — exactly where the artifact lived.
    // Then clamp again to MAX_LAG_SECONDS of physical plausibility: on a coarse
    // multi-day grid n/4 is still tens of hours, so this absolute ceiling is the
    // bound that actually kills the diurnal "16h incident" artifact. At least one
    // bucket survives so co-occurring streams can still align.
    let max_lag_abs = (MAX_LAG_SECONDS / width).max(1);
    let max_lag = MAX_LAG_BUCKETS
        .min(n as i64 - 1)
        .min(n as i64 / 4)
        .min(max_lag_abs)
        .max(0) as i32;

    // Bucketed series per stream; None when flat (zero variance).
    let series: Vec<Option<StreamSeries>> = streams.iter()
        .map(|s| bucket_series(&s.epochs, gmin, width, n))
        .collect();

    let mut findings: Vec<Correlation> = Vec::new();
    let mut scores = vec![0.0f32; 2 * max_lag as usize + 1];
    for i in 0..streams.len() {
        for j in (i + 1)..streams.len() {
            if streams[i].file_idx == streams[j].file_idx {
                continue; // a file trivially correlates with its own subset
            }
            let (Some(sa), Some(sb)) = (&series[i], &series[j]) else { continue };
            unsafe {
                ffi::corr_sweep(sa.z.as_ptr(), sb.z.as_ptr(), n as i32, max_lag, scores.as_mut_ptr());
            }
            // The kernel returns dot/overlap per lag; turn each into the
            // PER-WINDOW Pearson r via prefix sums. Global-window cosine
            // (v2.11–2.12.0) was degenerate on disjoint-era inputs: a
            // zero-event overlap window z-scores to a constant and the
            // cosine of a constant against anything can reach ±1 — two
            // non-overlapping NASA-log slices scored r=1.00 at +5 days
            // (found in the wild, 2026-06-12). Pearson subtracts the
            // window means, so a constant window has zero variance and
            // scores 0; windows with fewer than MIN_EVENTS real events
            // are rejected outright — a correlation claim requires both
            // streams ACTIVE in the compared window.
            // POSITIVE correlations only. Negative rate-correlation
            // across two event files is almost always the other face of
            // the disjoint-era artifact — files recorded in different
            // periods anti-correlate by presence alone (where one is
            // active the other is silent; the same NASA pair scored
            // r=-0.54 at +3 days). The 3am story is co-occurrence.
            let (best_slot, best_score) = scores.iter().copied().enumerate()
                .map(|(slot, raw)| (slot, pearson_at_lag(raw, slot, max_lag, n, sa, sb)))
                .max_by(|x, y| x.1.partial_cmp(&y.1).expect("scores are finite"))
                .expect("scores is non-empty");
            if best_score < SCORE_THRESHOLD {
                continue;
            }
            let lag = best_slot as i64 - max_lag as i64;
            // Normalize direction: stream_a is the follower (lag >= 0).
            // corr_sweep pairs a[k+lag] with b[k], so a positive lag
            // already means "stream i follows stream j".
            let (fi, fj, lag) = if lag >= 0 { (i, j, lag) } else { (j, i, -lag) };
            let (a, b) = (
                &series[fi].as_ref().unwrap().z,
                &series[fj].as_ref().unwrap().z,
            );
            let peak = peak_bucket(a, b, lag as usize, gmin, width);
            findings.push(Correlation {
                stream_a:      streams[fi].name.clone(),
                stream_b:      streams[fj].name.clone(),
                lag_seconds:   lag * width,
                score:         best_score,
                peak_bucket:   peak,
                events_a:      streams[fi].epochs.len() as u64,
                events_b:      streams[fj].epochs.len() as u64,
                width_seconds: width,
            });
        }
    }
    // Strongest first; name pair breaks exact ties deterministically.
    findings.sort_by(|x, y| {
        y.score.partial_cmp(&x.score).expect("scores are finite")
            .then_with(|| x.stream_a.cmp(&y.stream_a))
            .then_with(|| x.stream_b.cmp(&y.stream_b))
    });
    findings.truncate(TOP_K);

    // Assemble the incident here, where the shared grid (gmin/width/n) is live —
    // Stage 2's drop pass needs it to bucket each stream within the window.
    let views: Vec<incident::StreamView> = streams.iter()
        .map(|s| incident::StreamView { name: &s.name, epochs: &s.epochs })
        .collect();
    let incident = incident::build_incident(&views, &findings, triggers, gmin, width, n);
    (findings, incident)
}

/// One stream on the shared grid: the z-scored series the kernel sweeps,
/// plus prefix sums that make any overlap window's mean, variance, and
/// raw event count an O(1) subtract.
struct StreamSeries {
    z:      Vec<f32>,
    sum:    Vec<f64>, // prefix of z
    sumsq:  Vec<f64>, // prefix of z²
    events: Vec<u64>, // prefix of raw counts
}

/// Turn the kernel's dot/overlap for one lag slot into the per-window
/// Pearson r — bounded [-1, 1], zero for a constant window (Pearson is
/// undefined there, and "silence correlates with nothing" is the honest
/// reading), zero when either window holds fewer than MIN_EVENTS events.
fn pearson_at_lag(
    raw: f32, slot: usize, max_lag: i32, n: usize,
    sa: &StreamSeries, sb: &StreamSeries,
) -> f64 {
    let lag = slot as i64 - max_lag as i64;
    let (off_a, off_b, m) = if lag >= 0 {
        (lag as usize, 0usize, n - lag as usize)
    } else {
        (0usize, (-lag) as usize, n - (-lag) as usize)
    };
    // A correlation over a handful of buckets is not a correlation: Pearson
    // over m points hits ±1 trivially as m shrinks, so a near-boundary lag
    // (overlap = a few buckets) manufactures a spurious r=1.00 even though
    // both windows hold thousands of events. The active-window gate below
    // counts EVENTS, not buckets, so it can't catch this — a real 28-day
    // NASA log scored errors "following" traffic by +654h, r=1.00, on a
    // 4-bucket overlap (found 2026-06-15 via real-data incident testing).
    // Require enough buckets for the statistic to mean anything.
    if m < MIN_OVERLAP_BUCKETS {
        return 0.0;
    }
    let ev_a = sa.events[off_a + m] - sa.events[off_a];
    let ev_b = sb.events[off_b + m] - sb.events[off_b];
    if (ev_a as usize) < MIN_EVENTS || (ev_b as usize) < MIN_EVENTS {
        return 0.0;
    }
    let mf = m as f64;
    let dot = raw as f64 * mf; // undo the kernel's /overlap
    let sum_a = sa.sum[off_a + m] - sa.sum[off_a];
    let sum_b = sb.sum[off_b + m] - sb.sum[off_b];
    let var_a = (sa.sumsq[off_a + m] - sa.sumsq[off_a]) - sum_a * sum_a / mf;
    let var_b = (sb.sumsq[off_b + m] - sb.sumsq[off_b]) - sum_b * sum_b / mf;
    if var_a <= 1e-9 || var_b <= 1e-9 {
        return 0.0; // (near-)constant window — Pearson undefined
    }
    let cov = dot - sum_a * sum_b / mf;
    (cov / (var_a * var_b).sqrt()).clamp(-1.0, 1.0)
}

/// Bucket epochs onto the shared grid, z-score the counts, and build the
/// window prefix sums. Returns None for a globally flat series (zero
/// variance) — correlation is undefined.
fn bucket_series(epochs: &[i64], gmin: i64, width: i64, n: usize) -> Option<StreamSeries> {
    let mut counts = vec![0.0f64; n];
    for &e in epochs {
        let idx = ((e - gmin) / width) as usize;
        if idx < n { counts[idx] += 1.0; }
    }
    let mean = counts.iter().sum::<f64>() / n as f64;
    let var = counts.iter().map(|c| (c - mean) * (c - mean)).sum::<f64>() / n as f64;
    if var == 0.0 {
        return None;
    }
    let inv_std = 1.0 / var.sqrt();
    let z: Vec<f32> = counts.iter().map(|c| ((c - mean) * inv_std) as f32).collect();

    let mut sum = Vec::with_capacity(n + 1);
    let mut sumsq = Vec::with_capacity(n + 1);
    let mut events = Vec::with_capacity(n + 1);
    let (mut s, mut s2, mut ev) = (0.0f64, 0.0f64, 0u64);
    sum.push(0.0);
    sumsq.push(0.0);
    events.push(0);
    for (zi, ci) in z.iter().zip(&counts) {
        s += *zi as f64;
        s2 += (*zi as f64) * (*zi as f64);
        ev += *ci as u64;
        sum.push(s);
        sumsq.push(s2);
        events.push(ev);
    }
    Some(StreamSeries { z, sum, sumsq, events })
}

/// Grid instant where the lag-aligned overlap product peaks — the
/// moment of strongest co-occurrence, reported in the follower's frame.
fn peak_bucket(a: &[f32], b: &[f32], lag: usize, gmin: i64, width: i64) -> String {
    let n = a.len();
    let m = n - lag;
    let mut best_i = 0usize;
    let mut best_p = f32::NEG_INFINITY;
    for i in 0..m {
        let p = a[i + lag] * b[i];
        if p > best_p {
            best_p = p;
            best_i = i;
        }
    }
    seconds_to_iso(gmin + ((best_i + lag) as i64) * width)
}

fn basename(path: &str) -> String {
    path.rsplit(['/', '\\']).next().unwrap_or(path).to_string()
}

/// Compact findings-only block (~100 B per finding) streamed to the
/// user in the multi-file drop. None when the run failed or found
/// nothing. Prefill is the Pi bottleneck, so this stays terse by
/// construction (TOP_K findings, one line each).
pub fn findings_block(out: &RuneOutput) -> Option<String> {
    if !out.success || out.correlations.is_empty() {
        return None;
    }
    let mut s = String::from("Cross-file correlations:\n");
    for c in &out.correlations {
        s.push_str(&format!(
            "{} follows {} by +{} seconds (correlation {:.2}, peak at {})\n",
            c.stream_a, c.stream_b, c.lag_seconds, c.score, c.peak_bucket,
        ));
    }
    Some(s)
}

/// Prose rendering of the top finding for the narration PROMPT. The
/// machine-shaped `findings_block` lines bait Gemma 4 on NEON into
/// continuing the pattern instead of summarizing (the same "content
/// already looks like a complete analysis" trap `build_narration_prompt`
/// documents), so the model gets one flowing sentence with nothing to
/// continue. Top finding only — the user-visible block carries the rest.
pub fn findings_for_prompt(out: &RuneOutput) -> Option<String> {
    if !out.success {
        return None;
    }
    // Lead with the assembled incident timeline when there is one — it IS the
    // conclusion ("why did my service die"); the raw correlations are evidence.
    if let Some(inc) = &out.incident {
        return Some(incident::incident_for_prompt(inc));
    }
    let c = out.correlations.first()?;
    Some(format!(
        "A correlation pass across the files found that events in {} \
         consistently happen about {} seconds after events in {}, most \
         strongly around {}.",
        c.stream_a, c.lag_seconds, c.stream_b, c.peak_bucket,
    ))
}

fn format_text(out: &RuneOutput) -> String {
    let mut buf = String::with_capacity(512);
    buf.push_str(&format!("events:      {}\n", out.totals.rows));
    buf.push_str(&format!("streams:     {}\n", out.categories.len()));
    buf.push_str(&format!("scan:        {}\n", super::common::format_scan_time(out.totals.scan_us)));
    buf.push('\n');
    if out.categories.is_empty() {
        buf.push_str("(no timestamped streams found — need >= 2 files with >= 3 events each)\n");
        return buf;
    }
    for c in &out.categories {
        buf.push_str(&format!("  {:<28} {:>10}\n", c.name, c.count));
    }
    buf.push('\n');
    if out.correlations.is_empty() {
        buf.push_str("correlations: none (no cross-file pair crossed the threshold)\n");
        return buf;
    }
    buf.push_str(&format!("correlations: {} finding(s)\n", out.correlations.len()));
    for c in &out.correlations {
        buf.push_str(&format!(
            "  {} follows {} by +{}s (r={:.2}, peak {}, bucket {}s)\n",
            c.stream_a, c.stream_b, c.lag_seconds, c.score, c.peak_bucket, c.width_seconds,
        ));
    }
    if let Some(inc) = &out.incident {
        buf.push('\n');
        buf.push_str(&incident::format_incident(inc));
    }
    buf
}

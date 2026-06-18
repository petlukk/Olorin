//! Trigger-during-lag detector for the logwatch palantír.
//!
//! The honest core of "alert before the error": most cascades have a trigger —
//! a deploy / restart / reload — and the user-visible errors follow some lag
//! later (the incident-lab's `db fails at deploy → 5xx LAG seconds later`). On a
//! live trigger line we alert immediately, with an ETA learned from the file's
//! own history, *before* the error storm. If errors then appear inside the
//! window we escalate; if the window passes quiet we stand down.
//!
//! Error classification reuses the exact per-format sub-stream kernels the runes
//! use (`substream::*_errors`), so "error" here means what `eacorrelate`/`ealog`
//! call an error. Triggers are rare, so a line-level keyword match suffices —
//! no kernel needed. The state machine takes `now` as an argument (arrival-time
//! epoch seconds) and is pure, so it is fully unit-testable without sleeping.

use crate::runes::stream::{self, Format};
use crate::runes::substream;
use crate::runes::timekey::seconds_to_iso;

/// Lines naming a deploy-class event. Case-insensitive substring match.
const TRIGGERS: [&str; 7] =
    ["deploy", "rollout", "redeploy", "rollback", "restart", "reload", "released"];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sensitivity { Low, Medium, High }

impl Sensitivity {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "low" => Some(Self::Low),
            "med" | "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            _ => None,
        }
    }
    /// (window multiplier on the learned lag, error count to confirm a cascade).
    fn knobs(self) -> (i64, usize) {
        match self {
            Self::High => (3, 1),
            Self::Medium => (2, 2),
            Self::Low => (1, 3),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum Alert {
    /// A trigger fired; a cascade is predicted (with an ETA when a lag is known).
    Predicted { at: i64, eta: Option<i64>, window: i64 },
    /// Errors crossed the confirm threshold inside the window.
    Confirmed { trigger_at: i64, at: i64, errors: usize },
    /// The window passed without a cascade.
    Clear { trigger_at: i64, window: i64 },
}

impl Alert {
    /// Machine-readable kind, for webhook payloads and exec env.
    pub fn kind(&self) -> &'static str {
        match self {
            Alert::Predicted { .. } => "predicted",
            Alert::Confirmed { .. } => "confirmed",
            Alert::Clear { .. } => "clear",
        }
    }

    /// Severity for alert routing: a prediction is a warning, a confirmed
    /// cascade is critical, a stand-down is informational.
    pub fn severity(&self) -> &'static str {
        match self {
            Alert::Predicted { .. } => "warning",
            Alert::Confirmed { .. } => "critical",
            Alert::Clear { .. } => "info",
        }
    }

    /// One human-facing line for stdout.
    pub fn render(&self) -> String {
        match self {
            Alert::Predicted { at, eta, window } => {
                let head = format!("⚠  PALANTÍR  {}  trigger detected", seconds_to_iso(*at));
                match eta {
                    Some(e) => format!(
                        "{head} — cascade predicted by {} (historical lag), watching {window}s…",
                        seconds_to_iso(*e)
                    ),
                    None => format!("{head} — no historical lag yet, watching {window}s for a cascade…"),
                }
            }
            Alert::Confirmed { trigger_at, at, errors } => format!(
                "🔴 PALANTÍR  {}  CASCADE CONFIRMED — {errors} error(s), {}s after the trigger",
                seconds_to_iso(*at), at - trigger_at
            ),
            Alert::Clear { trigger_at, window } => format!(
                "✓  PALANTÍR  {}  window clear — no cascade {window}s after the trigger",
                seconds_to_iso(trigger_at + window)
            ),
        }
    }
}

enum Phase {
    Idle,
    Armed { trigger_at: i64, until: i64, errors: usize },
    Cooldown { until: i64 },
}

pub struct Detector {
    window:   i64,
    confirm:  usize,
    cooldown: i64,
    lag:      Option<i64>,
    phase:    Phase,
}

impl Detector {
    /// Build from a learned `lag` (None when history had no trigger→error
    /// pattern) and a sensitivity. The watch window scales with the lag so a
    /// slow cascade gets a proportionally longer window.
    pub fn new(lag: Option<i64>, sensitivity: Sensitivity) -> Self {
        let (mult, confirm) = sensitivity.knobs();
        let base = lag.unwrap_or(60);
        let window = (base * mult).max(45);
        Self { window, confirm, cooldown: window, lag, phase: Phase::Idle }
    }

    pub fn window(&self) -> i64 { self.window }
    pub fn lag(&self) -> Option<i64> { self.lag }

    /// Advance the machine with one poll's worth of observations at arrival
    /// time `now` (epoch seconds): `triggers` trigger lines and `errors` error
    /// events seen this tick. Call every poll, with (0, 0) when nothing arrived.
    pub fn observe(&mut self, now: i64, triggers: usize, errors: usize) -> Vec<Alert> {
        let mut out = Vec::new();

        // 1. Expire the current phase if its deadline passed.
        match &self.phase {
            Phase::Armed { trigger_at, until, errors: seen } if now >= *until => {
                if *seen < self.confirm {
                    out.push(Alert::Clear { trigger_at: *trigger_at, window: self.window });
                }
                self.phase = Phase::Idle;
            }
            Phase::Cooldown { until } if now >= *until => self.phase = Phase::Idle,
            _ => {}
        }

        // 2. Fold in this tick's observations.
        match &mut self.phase {
            Phase::Cooldown { .. } => {} // suppress: one incident, one alert chain
            Phase::Armed { trigger_at, errors: seen, .. } => {
                *seen += errors;
                if *seen >= self.confirm {
                    out.push(Alert::Confirmed { trigger_at: *trigger_at, at: now, errors: *seen });
                    self.phase = Phase::Cooldown { until: now + self.cooldown };
                }
                // A fresh trigger mid-window just extends the watch; deploys
                // often log several lines, so we don't re-predict.
                else if triggers > 0 {
                    self.phase = Phase::Armed {
                        trigger_at: *trigger_at, until: now + self.window, errors: *seen,
                    };
                }
            }
            Phase::Idle if triggers > 0 => {
                out.push(Alert::Predicted {
                    at: now,
                    eta: self.lag.map(|l| now + l),
                    window: self.window,
                });
                self.phase = Phase::Armed { trigger_at: now, until: now + self.window, errors };
                if errors >= self.confirm {
                    out.push(Alert::Confirmed { trigger_at: now, at: now, errors });
                    self.phase = Phase::Cooldown { until: now + self.cooldown };
                }
            }
            Phase::Idle => {}
        }
        out
    }
}

/// Count trigger lines and error events in a chunk. Errors go through the same
/// per-format sub-stream kernel the runes use; triggers are a line-level keyword
/// match.
pub fn classify_chunk(bytes: &[u8], fmt: Format) -> (usize, usize) {
    let triggers = bytes.split(|&b| b == b'\n').filter(|l| line_is_trigger(l)).count();
    let scan = stream::scan_for(bytes, fmt, stream::MAX_POSITIONS);
    let errors = errors_for(fmt, bytes, &scan.positions).len();
    (triggers, errors)
}

/// Median trigger→next-error lag (seconds) over a file's history, or None when
/// no trigger is followed by an error within the hour. Drives the ETA.
pub fn learn_lag(bytes: &[u8], fmt: Format) -> Option<i64> {
    let scan = stream::scan_for(bytes, fmt, stream::MAX_POSITIONS);
    let mut errors = errors_for(fmt, bytes, &scan.positions);
    if errors.is_empty() {
        return None;
    }
    errors.sort_unstable();
    let mut gaps: Vec<i64> = Vec::new();
    for line in bytes.split(|&b| b == b'\n') {
        if !line_is_trigger(line) {
            continue;
        }
        let Some(t) = first_epoch(line, fmt) else { continue };
        if let Some(&e) = errors.iter().find(|&&e| e > t && e - t <= 3600) {
            gaps.push(e - t);
        }
    }
    if gaps.is_empty() {
        return None;
    }
    gaps.sort_unstable();
    Some(gaps[gaps.len() / 2])
}

fn errors_for(fmt: Format, bytes: &[u8], ts: &[i32]) -> Vec<i64> {
    match fmt {
        Format::Iso       => substream::iso_errors(bytes, ts),
        Format::Clf       => substream::clf_errors(bytes, ts),
        Format::Syslog    => substream::syslog_errors(bytes, ts),
        Format::JsonEpoch => substream::json_errors(bytes, ts),
        Format::Apache    => substream::apache_errors(bytes, ts),
        Format::Hdfs      => substream::hdfs_errors(bytes, ts),
    }
}

fn first_epoch(line: &[u8], fmt: Format) -> Option<i64> {
    let scan = stream::scan_for(line, fmt, 2);
    stream::positions_to_epochs(line, &scan.positions, fmt).into_iter().next()
}

fn line_is_trigger(line: &[u8]) -> bool {
    let lower: Vec<u8> = line.iter().map(u8::to_ascii_lowercase).collect();
    TRIGGERS.iter().any(|kw| window_contains(&lower, kw.as_bytes()))
}

fn window_contains(hay: &[u8], needle: &[u8]) -> bool {
    !needle.is_empty() && hay.len() >= needle.len() && hay.windows(needle.len()).any(|w| w == needle)
}

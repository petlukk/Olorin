//! Incident timeline — turns eacorrelate's pairwise lag correlations into a
//! single ordered story: *what happened, in order, and how tightly it holds
//! together*. The leap from "tool" to "answer" — nobody asks for a correlation
//! matrix, everybody asks "why did my service die?".
//!
//! This is an assembly + framing layer over the existing correlation engine,
//! NOT new math. eacorrelate already finds the directed edges ("errors follow
//! the deploy by +240s, r=0.93", with the disjoint-era false positive killed by
//! the active-window gate). Here we:
//!   1. find the ROOT of the cascade (a stream that leads but never follows),
//!   2. anchor on it — a discrete TRIGGER event ("Deployment at 14:02") when the
//!      root is sparse, else the root's own break ("error spike at 14:06"),
//!   3. order the followers by cumulative lag from the anchor into a timeline,
//!   4. report the WEAKEST-LINK confidence (min r across the chain).
//!
//! Honesty: the wording is strictly temporal — "errors increase 4 minutes
//! later", never "the deploy CAUSED it". It reads causal to a human; it only
//! ever claims correlation + ordering. `confidence` is min(r), the honest "how
//! solid is the weakest step", never a causal probability.
//!
//! Lives outside `output.rs` (which is at the 500-LOC cap) and is excluded from
//! the rune registry in `build.rs`, exactly like `correlation.rs`.

use super::correlation::Correlation;
use super::timekey::{iso_to_seconds, seconds_to_iso};
use crate::storage::json::{Object, Value};

/// A stream is "trigger-like" (a discrete event source, e.g. a deploy log) when
/// it has at most this many events AND is far sparser than the busiest stream.
const TRIGGER_MAX_EVENTS: u64 = 50;
/// ...sparser by at least this factor than the busiest stream in the cascade.
const TRIGGER_SPARSITY: u64 = 10;

/// Minimal per-stream metadata the incident builder needs from eacorrelate.
pub struct StreamMeta {
    pub name:   String,
    pub events: u64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Anchor {
    /// "trigger" (a discrete event began the cascade) or "spike" (the root is a
    /// rate stream whose own break began it).
    pub kind:   String,
    pub stream: String,
    pub time:   String, // ISO instant
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Step {
    pub stream:      String,
    pub lag_seconds: i64,    // cumulative, from the anchor
    /// "increase" for a correlated co-spike (stage 1). "decrease" is reserved
    /// for the signed drop-detection follow-up.
    pub direction:   String,
    pub score:       f64,    // weakest r along the path that reached this stream
    pub kind:        String, // "correlated"
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Incident {
    pub anchor:     Anchor,
    pub steps:      Vec<Step>,
    pub confidence: f64, // min(step.score) — the weakest link
}

/// Assemble an incident from the streams and the (already direction-normalized,
/// positive-only) correlations. `None` when there is no cascade to tell.
pub fn build_incident(streams: &[StreamMeta], correlations: &[Correlation]) -> Option<Incident> {
    if correlations.is_empty() {
        return None;
    }
    let events_of = |name: &str| streams.iter().find(|s| s.name == name).map(|s| s.events);

    // Edge a<-b means "a follows b": b leads, a follows.
    let mut leads = std::collections::BTreeMap::<&str, u32>::new();
    let mut follows = std::collections::BTreeMap::<&str, u32>::new();
    for c in correlations {
        *leads.entry(c.stream_b.as_str()).or_default() += 1;
        *follows.entry(c.stream_a.as_str()).or_default() += 1;
    }

    // Root = a leader that never follows (the source of the cascade). Among
    // those, prefer the sparsest (most trigger-like). Fall back to the strongest
    // edge's leader if the graph has no clean source (a cycle).
    let root: String = leads.keys()
        .filter(|n| !follows.contains_key(**n))
        .min_by_key(|n| events_of(n).unwrap_or(u64::MAX))
        .map(|n| n.to_string())
        .unwrap_or_else(|| correlations[0].stream_b.clone());

    // Anchor time: the strongest edge the root leads gives the follower's peak;
    // subtracting that edge's lag lands on the leader's moment — the deploy
    // instant for "errors peaked at 14:06, lag 240s -> deploy at 14:02".
    let anchor_edge = correlations.iter()
        .find(|c| c.stream_b == root)
        .or_else(|| correlations.first())?;
    let anchor_epoch = iso_to_seconds(&anchor_edge.peak_bucket)
        .map(|p| p - anchor_edge.lag_seconds)
        .unwrap_or(0);

    // Trigger vs spike: a sparse root that is a discrete event source is a
    // "trigger"; a busy rate stream that broke is a "spike".
    let root_events = events_of(&root).unwrap_or(0);
    let busiest = streams.iter().map(|s| s.events).max().unwrap_or(0);
    let is_trigger = root_events > 0
        && root_events <= TRIGGER_MAX_EVENTS
        && root_events.saturating_mul(TRIGGER_SPARSITY) <= busiest;

    let anchor = Anchor {
        kind:   if is_trigger { "trigger".into() } else { "spike".into() },
        stream: root.clone(),
        time:   seconds_to_iso(anchor_epoch),
    };

    let steps = walk_cascade(&root, correlations);
    if steps.is_empty() {
        return None;
    }
    let confidence = steps.iter().map(|s| s.score)
        .fold(f64::INFINITY, f64::min);

    Some(Incident { anchor, steps, confidence: round4(confidence) })
}

/// Shortest-lag traversal from the root over the correlation edges (leader ->
/// follower, weight = lag). Each reached stream carries its cumulative lag and
/// the weakest r on the path that reached it. Graphs are tiny (<= TOP_K edges),
/// so a Bellman-Ford-style relaxation is plenty.
fn walk_cascade(root: &str, correlations: &[Correlation]) -> Vec<Step> {
    let mut lag = std::collections::BTreeMap::<String, i64>::new();
    let mut path_min = std::collections::BTreeMap::<String, f64>::new();
    lag.insert(root.to_string(), 0);
    path_min.insert(root.to_string(), f64::INFINITY);

    // Relax |edges| times: any shortest path visits each edge at most once.
    for _ in 0..correlations.len() {
        for c in correlations {
            // edge: c.stream_b (leader) -> c.stream_a (follower), weight lag.
            if let Some(&leader_lag) = lag.get(&c.stream_b) {
                let cand = leader_lag + c.lag_seconds;
                let improved = lag.get(&c.stream_a).map(|&l| cand < l).unwrap_or(true);
                if improved {
                    lag.insert(c.stream_a.clone(), cand);
                    let leader_min = *path_min.get(&c.stream_b).unwrap_or(&f64::INFINITY);
                    path_min.insert(c.stream_a.clone(), leader_min.min(c.score));
                }
            }
        }
    }

    let mut steps: Vec<Step> = lag.iter()
        .filter(|(name, _)| name.as_str() != root)
        .map(|(name, &l)| Step {
            stream:      name.clone(),
            lag_seconds: l,
            direction:   "increase".into(),
            score:       round4(*path_min.get(name).unwrap_or(&0.0)),
            kind:        "correlated".into(),
        })
        .collect();
    // Timeline order: soonest after the anchor first, name breaks ties.
    steps.sort_by(|a, b| a.lag_seconds.cmp(&b.lag_seconds).then_with(|| a.stream.cmp(&b.stream)));
    steps
}

fn round4(v: f64) -> f64 {
    if !v.is_finite() { return 0.0; }
    (v * 10_000.0).round() / 10_000.0
}

/// Human label for the anchor — "Deployment" for a recognizable change source,
/// otherwise the stream's own name. Wording only; the structured `anchor.stream`
/// is always the raw name.
fn anchor_label(anchor: &Anchor) -> String {
    let low = anchor.stream.to_lowercase();
    if anchor.kind == "trigger" {
        if low.contains("deploy") { return "Deployment".into(); }
        if low.contains("release") || low.contains("rollout") || low.contains("ship") {
            return "Release".into();
        }
        return format!("{} event", anchor.stream);
    }
    format!("{} spike", anchor.stream)
}

/// "4 minutes" / "12 seconds" / "2 hours" — the largest whole unit.
fn humanize_lag(secs: i64) -> String {
    let s = secs.max(0);
    if s >= 3600 { format!("{} hour{}", s / 3600, if s / 3600 == 1 { "" } else { "s" }) }
    else if s >= 60 { format!("{} minute{}", s / 60, if s / 60 == 1 { "" } else { "s" }) }
    else { format!("{s} second{}", if s == 1 { "" } else { "s" }) }
}

fn step_verb(step: &Step) -> &'static str {
    if step.direction == "decrease" { "drops" } else { "rises" }
}

/// Multi-line incident timeline for the text view and the web UI.
pub fn format_incident(inc: &Incident) -> String {
    let mut s = format!("incident timeline (confidence {:.2}):\n", inc.confidence);
    let t = inc.anchor.time.get(11..16).unwrap_or(&inc.anchor.time); // HH:MM
    s.push_str(&format!("  {} at {}\n", anchor_label(&inc.anchor), t));
    for step in &inc.steps {
        // A zero lag is co-occurrence, not a cascade — say so rather than
        // "rises 0 seconds later" (honest: same-bucket, no lead/follow).
        if step.lag_seconds == 0 {
            s.push_str(&format!("  -> {} {} at the same time (r={:.2})\n",
                step.stream, step_verb(step), step.score));
        } else {
            s.push_str(&format!("  -> {} {} {} later (r={:.2})\n",
                step.stream, step_verb(step), humanize_lag(step.lag_seconds), step.score));
        }
    }
    s
}

/// One flowing sentence for the narration PROMPT — the model opens with the
/// conclusion. Prose (not the machine-shaped lines) so NEON Gemma summarizes
/// rather than pattern-continues, the same trap `findings_for_prompt` documents.
pub fn incident_for_prompt(inc: &Incident) -> String {
    let first = match inc.steps.first() {
        Some(s) => s,
        None => return String::new(),
    };
    let t = inc.anchor.time.get(11..16).unwrap_or(&inc.anchor.time);
    format!(
        "An incident timeline assembled from the files: after {} at {}, {} {} \
         about {} later, with the cascade holding together at confidence {:.2}.",
        anchor_label(&inc.anchor).to_lowercase(), t, first.stream, step_verb(first),
        humanize_lag(first.lag_seconds), inc.confidence,
    )
}

// ─── JSON codec (additive `incident` object on RuneOutput) ───────────────────

pub fn incident_to_obj(inc: &Incident) -> Object {
    let mut anchor = Object::new();
    anchor.set("kind",   Value::Str(inc.anchor.kind.clone()));
    anchor.set("stream", Value::Str(inc.anchor.stream.clone()));
    anchor.set("time",   Value::Str(inc.anchor.time.clone()));

    let steps: Vec<Value> = inc.steps.iter().map(|s| {
        let mut o = Object::new();
        o.set("stream",      Value::Str(s.stream.clone()));
        o.set("lag_seconds", Value::I64(s.lag_seconds));
        o.set("direction",   Value::Str(s.direction.clone()));
        o.set("score",       Value::F64(round4(s.score)));
        o.set("kind",        Value::Str(s.kind.clone()));
        Value::Object(Box::new(o))
    }).collect();

    let mut o = Object::new();
    o.set("anchor",     Value::Object(Box::new(anchor)));
    o.set("steps",      Value::Array(steps));
    o.set("confidence", Value::F64(round4(inc.confidence)));
    o
}

/// Read the optional `incident` object off a parsed RuneOutput object.
pub fn incident_from_runeoutput(o: &Object) -> Result<Option<Incident>, String> {
    let inc = match o.get_object("incident") {
        Some(i) => i,
        None => return Ok(None),
    };
    let a = inc.get_object("anchor").ok_or("incident.anchor missing")?;
    let anchor = Anchor {
        kind:   a.get_str("kind").ok_or("incident.anchor.kind missing")?.to_string(),
        stream: a.get_str("stream").ok_or("incident.anchor.stream missing")?.to_string(),
        time:   a.get_str("time").ok_or("incident.anchor.time missing")?.to_string(),
    };
    let mut steps = Vec::new();
    if let Some(Value::Array(arr)) = inc.get("steps") {
        for v in arr {
            if let Value::Object(so) = v {
                steps.push(Step {
                    stream:      so.get_str("stream").ok_or("incident.step.stream missing")?.to_string(),
                    lag_seconds: so.get_i64("lag_seconds").ok_or("incident.step.lag_seconds missing")?,
                    direction:   so.get_str("direction").unwrap_or("increase").to_string(),
                    score:       so.get_f64("score").unwrap_or(0.0),
                    kind:        so.get_str("kind").unwrap_or("correlated").to_string(),
                });
            }
        }
    }
    let confidence = inc.get_f64("confidence").unwrap_or(0.0);
    Ok(Some(Incident { anchor, steps, confidence }))
}

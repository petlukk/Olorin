//! Correlation block of the RuneOutput v1 contract — struct + JSON codec.
//!
//! Lives outside `output.rs` only to keep that file under the 500-LOC
//! cap; the field itself is on `RuneOutput` and serializes additively
//! (only when non-empty), exactly like `anomalies[]`.

use crate::storage::json::{Object, Value};

/// One cross-stream finding from `eacorrelate`. Direction is normalized
/// at build time so `lag_seconds >= 0` always: events in `stream_a`
/// happen `lag_seconds` AFTER events in `stream_b` ("a follows b").
/// `peak_bucket` is the grid instant where the lag-aligned overlap
/// peaks — the moment the narration should name.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Correlation {
    pub stream_a:      String,
    pub stream_b:      String,
    pub lag_seconds:   i64,
    /// Cosine of the lag-aligned overlap windows of the two z-scored
    /// series — bounded [-1, 1] by Cauchy-Schwarz. Rounded to 4 decimals
    /// at serialization so the cross-arch goldens stay byte-stable
    /// against last-ULP drift.
    pub score:         f64,
    pub peak_bucket:   String,
    pub events_a:      u64,
    pub events_b:      u64,
    pub width_seconds: i64,
}

/// Round to 4 decimals for the wire. f64 can't hold most decimal
/// fractions exactly, but the same input rounds to the same f64 on
/// every arch, which is all byte-stable goldens need.
fn round4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}

pub(super) fn correlation_to_obj(c: &Correlation) -> Object {
    let mut o = Object::new();
    o.set("stream_a",      Value::Str(c.stream_a.clone()));
    o.set("stream_b",      Value::Str(c.stream_b.clone()));
    o.set("lag_seconds",   Value::I64(c.lag_seconds));
    o.set("score",         Value::F64(round4(c.score)));
    o.set("peak_bucket",   Value::Str(c.peak_bucket.clone()));
    o.set("events_a",      Value::I64(c.events_a as i64));
    o.set("events_b",      Value::I64(c.events_b as i64));
    o.set("width_seconds", Value::I64(c.width_seconds));
    o
}

pub(super) fn correlation_from_obj(o: &Object) -> Result<Correlation, String> {
    Ok(Correlation {
        stream_a: o.get_str("stream_a").ok_or("correlation.stream_a missing")?.to_string(),
        stream_b: o.get_str("stream_b").ok_or("correlation.stream_b missing")?.to_string(),
        lag_seconds: o.get_i64("lag_seconds").ok_or("correlation.lag_seconds missing")?,
        score: o.get_f64("score").unwrap_or(0.0),
        peak_bucket: o.get_str("peak_bucket").ok_or("correlation.peak_bucket missing")?.to_string(),
        events_a: o.get_i64("events_a").ok_or("correlation.events_a missing")? as u64,
        events_b: o.get_i64("events_b").ok_or("correlation.events_b missing")? as u64,
        width_seconds: o.get_i64("width_seconds").ok_or("correlation.width_seconds missing")?,
    })
}

//! anomaly — deterministic spike detection over a chronological count
//! series. Used by `eatime --bucket series` to surface the buckets where
//! the event rate broke from baseline.
//!
//! Robust by construction: the baseline is the **median** bucket count
//! and the spread is the **MAD** (median absolute deviation), so a large
//! spike cannot inflate its own baseline the way a mean/σ pair would —
//! the very bucket we want to flag would otherwise hide itself.
//!
//! A perfectly flat series has MAD = 0 (the common synthetic / low-noise
//! case), which makes the robust z-score undefined. There we fall back to
//! a ratio test against the median. Both paths flag **upward** spikes
//! only; rate dips are out of scope for v1.

use super::output::Anomaly;
use super::timekey::seconds_to_iso;

/// Robust z-score threshold (≈ 4σ). Conservative — favors zero false
/// positives, matching the rune differential-audit release gate.
const Z_THRESHOLD: f64 = 4.0;
/// Ratio threshold for the MAD = 0 fallback: a bucket must exceed the
/// median by this factor to count as a spike.
const RATIO_THRESHOLD: f64 = 3.0;
/// Absolute floor for a spike. Guards against flagging trivial jitter on
/// tiny baselines (e.g. median 1 → a bucket of 4 is not an incident).
const MIN_ABS_SPIKE: f64 = 10.0;
/// Below this many buckets there isn't enough signal to define a
/// baseline; detection is skipped and an empty list is returned.
const MIN_BUCKETS: usize = 8;

/// Scan a chronological count series for upward spikes. `counts[i]` is
/// the number of events in the bucket starting at
/// `min_epoch + i * width_secs` seconds-since-2000.
pub fn detect(counts: &[u64], min_epoch: i64, width_secs: i64) -> Vec<Anomaly> {
    if counts.len() < MIN_BUCKETS {
        return Vec::new();
    }
    let median = median_u64(counts);
    let mad    = median_abs_dev(counts, median);
    let sigma  = 1.4826 * mad; // MAD → σ-equivalent for normal data.

    let mut out: Vec<Anomaly> = Vec::new();
    for (i, &c) in counts.iter().enumerate() {
        let cf = c as f64;
        if cf <= median {
            continue; // upward spikes only
        }
        let (flagged, score) = if sigma > 0.0 {
            let z = (cf - median) / sigma;
            (z >= Z_THRESHOLD, z)
        } else {
            // Flat baseline (MAD = 0): ratio test with an absolute floor.
            let ratio = if median > 0.0 { cf / median } else { cf };
            let big_enough = cf - median >= MIN_ABS_SPIKE;
            let spikes = if median > 0.0 { ratio >= RATIO_THRESHOLD } else { true };
            (big_enough && spikes, ratio)
        };
        if flagged {
            let ratio = if median > 0.0 { cf / median } else { f64::INFINITY };
            out.push(Anomaly {
                bucket:   seconds_to_iso(min_epoch + (i as i64) * width_secs),
                count:    c,
                baseline: median,
                ratio,
                score,
            });
        }
    }
    out
}

fn median_u64(xs: &[u64]) -> f64 {
    let mut v: Vec<u64> = xs.to_vec();
    v.sort_unstable();
    median_sorted_u64(&v)
}

fn median_sorted_u64(sorted: &[u64]) -> f64 {
    let n = sorted.len();
    if n == 0 { return 0.0; }
    if n % 2 == 1 {
        sorted[n / 2] as f64
    } else {
        (sorted[n / 2 - 1] as f64 + sorted[n / 2] as f64) / 2.0
    }
}

fn median_abs_dev(xs: &[u64], median: f64) -> f64 {
    let mut dev: Vec<f64> = xs.iter().map(|&c| (c as f64 - median).abs()).collect();
    dev.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = dev.len();
    if n == 0 { return 0.0; }
    if n % 2 == 1 {
        dev[n / 2]
    } else {
        (dev[n / 2 - 1] + dev[n / 2]) / 2.0
    }
}

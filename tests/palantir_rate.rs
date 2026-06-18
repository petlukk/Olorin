//! Palantír — streaming rate-anomaly detector. No trigger line needed: it flags
//! an abnormal rise in the per-bucket error rate against a robust trailing
//! baseline. These tests pin the false-positive controls, which are the point.

use olorin::palantir::watch::{Alert, RateDetector, Sensitivity};

/// Feed `n` quiet (zero-error) buckets to get past the warm-up.
fn warm(r: &mut RateDetector, t: &mut i64, n: usize) {
    for _ in 0..n {
        assert!(r.observe(*t, 0).is_none(), "no alert during/after a quiet warm-up");
        *t += 1;
    }
}

#[test]
fn fires_on_a_sustained_ramp_after_warmup() {
    let mut r = RateDetector::new(Sensitivity::High); // k=3, streak=2
    let mut t = 0;
    warm(&mut r, &mut t, 15);
    // First elevated bucket: anomalous but the streak isn't met yet.
    assert!(r.observe(t, 12).is_none()); t += 1;
    // Second consecutive elevated bucket → confirmed anomaly.
    let a = r.observe(t, 12);
    assert!(matches!(a, Some(Alert::Anomaly { rate: 12, .. })), "expected anomaly, got {a:?}");
}

#[test]
fn no_alert_before_warmup_even_with_a_spike() {
    let mut r = RateDetector::new(Sensitivity::High);
    // A big spike in the first few buckets must not fire — no baseline yet.
    for t in 0..10 {
        assert!(r.observe(t, 100).is_none(), "fired before warm-up at t={t}");
    }
}

#[test]
fn a_single_spike_does_not_fire() {
    let mut r = RateDetector::new(Sensitivity::High); // streak=2
    let mut t = 0;
    warm(&mut r, &mut t, 15);
    assert!(r.observe(t, 50).is_none()); t += 1; // streak = 1
    assert!(r.observe(t, 0).is_none());  t += 1; // back to quiet → streak resets
    assert!(r.observe(t, 0).is_none());          // still nothing
}

#[test]
fn counts_below_the_floor_never_fire_on_a_quiet_baseline() {
    let mut r = RateDetector::new(Sensitivity::High);
    let mut t = 0;
    warm(&mut r, &mut t, 15);
    // 2 errors/bucket, sustained, on an all-zero baseline — below the absolute
    // floor, so it must stay silent (a couple of stray errors aren't an incident).
    for _ in 0..6 {
        assert!(r.observe(t, 2).is_none(), "fired below the absolute floor");
        t += 1;
    }
}

#[test]
fn normal_variation_around_a_noisy_baseline_does_not_fire() {
    let mut r = RateDetector::new(Sensitivity::High); // k=3
    let mut t = 0;
    // Baseline alternating 8/12 → median 10, MAD 2, threshold = 10 + 3*2 = 16.
    for i in 0..30 {
        let _ = r.observe(t, if i % 2 == 0 { 8 } else { 12 });
        t += 1;
    }
    assert!(r.observe(t, 14).is_none(), "14 is within normal variation"); t += 1;
    assert!(r.observe(t, 15).is_none(), "15 is within normal variation"); t += 1;
    // A real jump well past the threshold fires after the streak.
    assert!(r.observe(t, 40).is_none()); t += 1; // streak 1
    let a = r.observe(t, 40);
    assert!(matches!(a, Some(Alert::Anomaly { .. })), "expected anomaly, got {a:?}");
}

#[test]
fn cooldown_suppresses_repeats() {
    let mut r = RateDetector::new(Sensitivity::High);
    let mut t = 0;
    warm(&mut r, &mut t, 15);
    assert!(r.observe(t, 30).is_none()); t += 1;          // streak 1
    let fired = r.observe(t, 30);                          // streak 2 → alert
    assert!(matches!(fired, Some(Alert::Anomaly { .. })));
    let at = t;
    // Within the cooldown window, even a bigger spike is suppressed.
    assert!(r.observe(at + 5, 200).is_none(), "should be in cooldown");
    assert!(r.observe(at + 40, 200).is_none(), "still in cooldown");
}

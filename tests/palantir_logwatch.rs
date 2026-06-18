//! Palantír v1 — logwatch trigger-during-lag watcher.
//!
//! The state machine takes `now` as an argument, so the prediction logic is
//! tested deterministically with no sleeping. Kernel-backed classification and
//! the history learn-pass are tested against crafted ISO logs, and the tailer
//! against a real growing temp file.

use olorin::palantir::tail::Tailer;
use olorin::palantir::watch::{classify_chunk, learn_lag, Alert, Detector, Sensitivity};
use olorin::runes::stream::{self, Format};
use std::io::Write;

// ── state machine (pure, no kernels) ────────────────────────────────────────

#[test]
fn predicts_on_trigger_then_confirms_on_errors() {
    // High: confirm=1, window=max(lag*3, 45)=45.
    let mut d = Detector::new(Some(10), Sensitivity::High);
    assert_eq!(
        d.observe(100, 1, 0),
        vec![Alert::Predicted { at: 100, eta: Some(110), window: 45 }],
        "a trigger must predict before any error, with an ETA from the learned lag",
    );
    assert_eq!(
        d.observe(108, 0, 1),
        vec![Alert::Confirmed { trigger_at: 100, at: 108, errors: 1 }],
        "an error inside the window confirms the cascade",
    );
}

#[test]
fn stands_down_when_window_passes_quiet() {
    let mut d = Detector::new(Some(10), Sensitivity::High); // window 45
    assert!(matches!(d.observe(100, 1, 0).as_slice(), [Alert::Predicted { .. }]));
    assert_eq!(
        d.observe(146, 0, 0),
        vec![Alert::Clear { trigger_at: 100, window: 45 }],
        "no errors past the window → stand down",
    );
}

#[test]
fn no_eta_without_a_learned_lag() {
    // Medium: confirm=2, window=max(60*2,45)=120.
    let mut d = Detector::new(None, Sensitivity::Medium);
    assert_eq!(
        d.observe(0, 1, 0),
        vec![Alert::Predicted { at: 0, eta: None, window: 120 }],
    );
    assert!(d.observe(10, 0, 1).is_empty(), "one error is below the Medium confirm threshold");
    assert_eq!(
        d.observe(20, 0, 1),
        vec![Alert::Confirmed { trigger_at: 0, at: 20, errors: 2 }],
    );
}

#[test]
fn cooldown_suppresses_repeat_alerts_then_recovers() {
    let mut d = Detector::new(Some(10), Sensitivity::High); // window/cooldown 45
    assert!(matches!(d.observe(100, 1, 0).as_slice(), [Alert::Predicted { .. }]));
    assert!(matches!(d.observe(105, 0, 1).as_slice(), [Alert::Confirmed { .. }]));
    assert!(d.observe(120, 1, 5).is_empty(), "cooldown suppresses a re-trigger mid-incident");
    assert!(
        matches!(d.observe(160, 1, 0).as_slice(), [Alert::Predicted { .. }]),
        "after cooldown a fresh trigger predicts again",
    );
}

// ── kernel-backed classification + learn pass ───────────────────────────────

fn iso_history() -> Vec<u8> {
    let mut s = String::new();
    for sec in 0..30 {
        s += &format!("2026-06-18T06:40:{sec:02} INFO heartbeat ok\n");
    }
    s += "2026-06-18T06:40:30 INFO deploy released v2\n";
    for k in 0..5 {
        s += &format!("2026-06-18T06:40:42 ERROR db pool exhausted #{k}\n");
    }
    s.into_bytes()
}

#[test]
fn learns_trigger_to_error_lag_from_history() {
    olorin::kernels::ffi::init().unwrap();
    let hist = iso_history();
    let fmt = stream::detect_format(&hist);
    assert_eq!(fmt, Format::Iso);
    assert_eq!(learn_lag(&hist, fmt), Some(12), "deploy 06:40:30 → first error 06:40:42 = 12s");
}

#[test]
fn classify_chunk_counts_triggers_and_errors() {
    olorin::kernels::ffi::init().unwrap();
    let chunk = b"2026-06-18T07:00:00 INFO deploy released v3\n\
                  2026-06-18T07:00:10 ERROR upstream timeout\n\
                  2026-06-18T07:00:10 ERROR upstream timeout\n\
                  2026-06-18T07:00:11 INFO request served\n";
    let (t, e) = classify_chunk(chunk, Format::Iso);
    assert_eq!(t, 1, "one deploy line");
    assert_eq!(e, 2, "two ERROR lines");
}

// ── tailer over a real growing file ─────────────────────────────────────────

#[test]
fn tailer_reports_appended_complete_lines() {
    let path = format!("/tmp/olorin_palantir_tail_{}.log", std::process::id());
    std::fs::write(&path, b"pre-existing\n").unwrap();
    let mut t = Tailer::at_end(&path);
    assert!(t.poll().is_empty(), "starts at EOF — existing content is not replayed");

    append(&path, b"new one\nnew two\n");
    assert_eq!(t.poll(), vec!["new one".to_string(), "new two".to_string()]);

    append(&path, b"partial"); // no newline yet
    assert!(t.poll().is_empty(), "a partial line is buffered until its newline");
    append(&path, b" done\n");
    assert_eq!(t.poll(), vec!["partial done".to_string()]);

    let _ = std::fs::remove_file(&path);
}

// ── end to end: predict at the trigger, before any error ────────────────────

#[test]
fn end_to_end_predicts_before_the_error_then_confirms() {
    olorin::kernels::ffi::init().unwrap();
    let mut d = Detector::new(Some(12), Sensitivity::High); // window 45, eta = now+12

    // Poll 1 (arrival t=1000): the deploy line — no error exists yet.
    let (t, e) = classify_chunk(b"2026-06-18T07:00:00 INFO deploy released\n", Format::Iso);
    assert!(
        matches!(d.observe(1000, t, e).as_slice(), [Alert::Predicted { eta: Some(1012), .. }]),
        "must alert at the trigger, ahead of any error",
    );

    // Poll 2 (arrival t=1012): the cascade arrives inside the window.
    let (t, e) = classify_chunk(b"2026-06-18T07:00:12 ERROR boom\n", Format::Iso);
    assert!(
        matches!(d.observe(1012, t, e).as_slice(), [Alert::Confirmed { .. }]),
        "errors inside the window confirm the predicted cascade",
    );
}

fn append(path: &str, bytes: &[u8]) {
    let mut f = std::fs::OpenOptions::new().append(true).open(path).unwrap();
    f.write_all(bytes).unwrap();
}

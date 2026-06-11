//! Step 4 of eacorrelate: the multi-file drop runs the cross-file lag
//! correlation BEFORE narration — findings stream like kernel output and
//! lead the narration prompt. Driven through `analyze_files_streaming`
//! with no engine (the deterministic path); the full 3am narration is
//! the model-gated #[ignore] test used as the Pi install-gate scenario.

use olorin::core::router::{DispatchContext, StreamEvent};
use std::path::{Path, PathBuf};
use std::sync::mpsc;

fn collect(rx: mpsc::Receiver<StreamEvent>) -> (String, bool, Option<String>) {
    let mut text = String::new();
    let mut done = false;
    let mut done_full = None;
    while let Ok(ev) = rx.try_recv() {
        match ev {
            StreamEvent::Token(t) => text.push_str(&t),
            StreamEvent::Done { full_text } => { done = true; done_full = Some(full_text); }
            _ => {}
        }
    }
    (text, done, done_full)
}

/// Stage a committed incident fixture into /tmp (path allowlist). `tag`
/// keeps each test's copies distinct — tests run in parallel and clean
/// up after themselves, so shared names would race.
fn stage(tag: &str, name: &str) -> String {
    let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/incident")
        .join(name);
    let dst = format!("/tmp/olorin_incident_{tag}_{name}");
    std::fs::copy(&src, &dst).unwrap_or_else(|e| panic!("stage {name}: {e}"));
    dst
}

#[test]
fn multi_file_drop_streams_correlation_findings() {
    let _ = olorin::kernels::ffi::init();
    let log = stage("two", "syslog.log");
    let csv = stage("two", "deploys.csv");

    let mut ctx = DispatchContext::new_no_engine(None);
    let (tx, rx) = mpsc::channel();
    ctx.analyze_files_streaming(
        &[("syslog.log".into(), log.clone()), ("deploys.csv".into(), csv.clone())],
        &tx,
    );
    drop(tx);
    let (text, done, full) = collect(rx);

    assert!(done);
    assert!(
        text.contains("ran `eacorrelate` across 2 files"),
        "correlation block missing: {text:?}"
    );
    assert!(
        text.contains("syslog.log (errors) follows deploys.csv by +240 seconds"),
        "planted lag not reported: {text:?}"
    );
    // Findings are part of the final full_text (vault/Done parity).
    assert!(full.unwrap_or_default().contains("ran `eacorrelate`"));

    let _ = std::fs::remove_file(&log);
    let _ = std::fs::remove_file(&csv);
}

#[test]
fn three_file_incident_correlates_clf_too() {
    let _ = olorin::kernels::ffi::init();
    let log = stage("three", "syslog.log");
    let csv = stage("three", "deploys.csv");
    let clf = stage("three", "access.log");

    let mut ctx = DispatchContext::new_no_engine(None);
    let (tx, rx) = mpsc::channel();
    ctx.analyze_files_streaming(
        &[
            ("syslog.log".into(), log.clone()),
            ("deploys.csv".into(), csv.clone()),
            ("access.log".into(), clf.clone()),
        ],
        &tx,
    );
    drop(tx);
    let (text, done, _) = collect(rx);

    assert!(done);
    assert!(text.contains("ran `eacorrelate` across 3 files"), "{text:?}");
    assert!(
        text.contains("follows deploys.csv by +240 seconds"),
        "deploy lag missing: {text:?}"
    );

    let _ = std::fs::remove_file(&log);
    let _ = std::fs::remove_file(&csv);
    let _ = std::fs::remove_file(&clf);
}

#[test]
fn uncorrelated_drop_stays_silent_about_correlations() {
    let _ = olorin::kernels::ffi::init();
    // Independent scatters: per-file runes still run, but no eacorrelate
    // block may appear (silence is the honest finding).
    let mut state: u64 = 0xfeed_beef_1234_5678;
    let mut scatter = |name: &str, salt: u64| -> String {
        let mut buf = String::new();
        let mut secs: Vec<i64> = (0..150)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(salt | 1);
                ((state >> 33) % (8 * 3600)) as i64
            })
            .collect();
        secs.sort_unstable();
        for s in secs {
            buf.push_str(&format!(
                "2026-06-11T{:02}:{:02}:{:02} INFO event\n",
                s / 3600, (s % 3600) / 60, s % 60
            ));
        }
        let path = format!("/tmp/{name}");
        std::fs::write(&path, buf).unwrap();
        path
    };
    let a = scatter("olorin_incident_rand_a.log", 7);
    let b = scatter("olorin_incident_rand_b.log", 99);

    let mut ctx = DispatchContext::new_no_engine(None);
    let (tx, rx) = mpsc::channel();
    ctx.analyze_files_streaming(&[("a.log".into(), a.clone()), ("b.log".into(), b.clone())], &tx);
    drop(tx);
    let (text, done, _) = collect(rx);

    assert!(done);
    assert!(
        !text.contains("eacorrelate"),
        "no findings -> no correlation block: {text:?}"
    );

    let _ = std::fs::remove_file(&a);
    let _ = std::fs::remove_file(&b);
}

#[test]
#[ignore = "model-gated e2e: the 3am incident -> narration leads with the deploy/error lag (Pi install-gate scenario)"]
fn e2e_3am_incident_narration() {
    if !Path::new(&olorin::home_dir().unwrap().join(".olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")).exists() {
        eprintln!("SKIP: no model");
        return;
    }
    let _ = olorin::kernels::ffi::init();
    let log = stage("e2e", "syslog.log");
    let csv = stage("e2e", "deploys.csv");
    let clf = stage("e2e", "access.log");

    let mut ctx = DispatchContext::new(None, None);
    let (tx, rx) = mpsc::channel();
    ctx.analyze_files_streaming(
        &[
            ("syslog.log".into(), log),
            ("deploys.csv".into(), csv),
            ("access.log".into(), clf),
        ],
        &tx,
    );
    drop(tx);
    let (text, done, full) = collect(rx);

    assert!(done);
    assert!(text.contains("ran `eacorrelate` across 3 files"));
    let full = full.unwrap_or_default();
    eprintln!("=== 3am incident output ===\n{full}");
}

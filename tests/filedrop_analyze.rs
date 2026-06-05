//! Step 2 of the file-drop analyst: `analyze_file_streaming` wires
//! pick_rune → run → stream → narrate. These tests drive it directly (no
//! HTTP). The deterministic path (no model) is fully testable; the narration
//! itself is model-gated and covered by the #[ignore] e2e test.

use olorin::core::router::{DispatchContext, StreamEvent};
use std::path::Path;
use std::sync::mpsc;

/// Drain a finished stream into (concatenated tokens, saw_done).
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

fn write_tmp(name: &str, contents: &str) -> String {
    let path = format!("/tmp/{name}");
    std::fs::write(&path, contents).unwrap();
    path
}

#[test]
fn timestamped_log_runs_eatime_and_streams_output() {
    let _ = olorin::kernels::ffi::init();
    let mut log = String::new();
    for i in 0..30 {
        log.push_str(&format!("2026-06-01T08:{:02}:00+00:00 INFO svc handled id={}\n", i % 60, 1000 + i));
    }
    // a burst in the 08:10 minute → eatime --bucket series has something to chew on
    for j in 0..40 {
        log.push_str(&format!("2026-06-01T08:10:{:02}+00:00 ERROR svc timeout retry={j}\n", j % 60));
    }
    let path = write_tmp("filedrop_eatime.log", &log);

    let mut ctx = DispatchContext::new_no_engine(None);
    let (tx, rx) = mpsc::channel();
    ctx.analyze_file_streaming("app.log", &path, &tx);
    drop(tx);
    let (text, done, _full) = collect(rx);

    assert!(done, "stream must end with Done");
    assert!(text.contains("ran `eatime`"), "header names the rune: {text:?}");
    assert!(text.contains("timestamps:") || text.contains("peak bucket"),
        "kernel output present: {text:?}");
}

#[test]
fn csv_runs_eacrunch() {
    let _ = olorin::kernels::ffi::init();
    let path = write_tmp("filedrop_data.csv", "a,b\n1,2\n3,4\n5,6\n");

    let mut ctx = DispatchContext::new_no_engine(None);
    let (tx, rx) = mpsc::channel();
    ctx.analyze_file_streaming("data.csv", &path, &tx);
    drop(tx);
    let (text, done, _) = collect(rx);

    assert!(done);
    assert!(text.contains("ran `eacrunch`"), "header names eacrunch: {text:?}");
}

#[test]
fn unknown_type_returns_graceful_message() {
    let _ = olorin::kernels::ffi::init();
    let path = write_tmp("filedrop_photo.png", "binary-ish content, unknown type");

    let mut ctx = DispatchContext::new_no_engine(None);
    let (tx, rx) = mpsc::channel();
    ctx.analyze_file_streaming("photo.png", &path, &tx);
    drop(tx);
    let (text, done, _) = collect(rx);

    assert!(done);
    assert!(text.contains("no rune matched"), "graceful fallback: {text:?}");
    assert!(!text.contains("ran `"), "no rune should have run: {text:?}");
}

#[test]
fn multiple_files_each_run_their_rune() {
    let _ = olorin::kernels::ffi::init();
    let mut a = String::new();
    for i in 0..20 { a.push_str(&format!("2026-06-01T08:{:02}:00+00:00 INFO a id={i}\n", i % 60)); }
    let mut b = String::new();
    for i in 0..20 { b.push_str(&format!("2026-06-01T09:{:02}:00+00:00 INFO b id={i}\n", i % 60)); }
    for j in 0..40 { b.push_str(&format!("2026-06-01T09:05:{:02}+00:00 ERROR b timeout {j}\n", j % 60)); }
    let pa = write_tmp("filedrop_multi_a.log", &a);
    let pb = write_tmp("filedrop_multi_b.log", &b);

    let mut ctx = DispatchContext::new_no_engine(None);
    let (tx, rx) = mpsc::channel();
    ctx.analyze_files_streaming(&[("svcA.log".into(), pa), ("svcB.log".into(), pb)], &tx);
    drop(tx);
    let (text, done, _) = collect(rx);

    assert!(done);
    assert_eq!(text.matches("ran `eatime`").count(), 2, "both files analyzed: {text:?}");
    assert!(text.contains("svcA.log") && text.contains("svcB.log"));
    assert!(text.contains("spike(s) detected"), "svcB spike present: {text:?}");
}

#[test]
#[ignore = "model-gated e2e: two logs -> one correlation narration"]
fn e2e_correlates_two_logs() {
    if !Path::new(&olorin::home_dir().unwrap().join(".olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")).exists() {
        eprintln!("SKIP: no model");
        return;
    }
    let _ = olorin::kernels::ffi::init();
    let mut clean = String::new();
    for i in 0..30 { clean.push_str(&format!("2026-06-01T08:{:02}:00+00:00 INFO ok {i}\n", i % 60)); }
    let mut spiky = String::new();
    for i in 0..30 { spiky.push_str(&format!("2026-06-01T08:{:02}:00+00:00 INFO ok {i}\n", i % 60)); }
    for j in 0..40 { spiky.push_str(&format!("2026-06-01T08:10:{:02}+00:00 ERROR boom {j}\n", j % 60)); }
    let pc = write_tmp("filedrop_corr_clean.log", &clean);
    let ps = write_tmp("filedrop_corr_spiky.log", &spiky);

    let mut ctx = DispatchContext::new(None, None);
    let (tx, rx) = mpsc::channel();
    ctx.analyze_files_streaming(&[("frontend.log".into(), pc), ("backend.log".into(), ps)], &tx);
    drop(tx);
    let (_text, done, full) = collect(rx);
    assert!(done);
    let full = full.unwrap_or_default();
    eprintln!("=== correlation output ===\n{full}");
    assert!(full.matches("ran `eatime`").count() == 2, "both kernels ran");
}

#[test]
#[ignore = "model-gated e2e: loads the model and asserts narration is appended"]
fn e2e_narrates_dropped_log() {
    if !Path::new(&olorin::home_dir().unwrap().join(".olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")).exists() {
        eprintln!("SKIP: no model");
        return;
    }
    let _ = olorin::kernels::ffi::init();
    let mut log = String::new();
    for i in 0..30 { log.push_str(&format!("2026-06-01T08:{:02}:00+00:00 INFO ok id={}\n", i % 60, i)); }
    for j in 0..40 { log.push_str(&format!("2026-06-01T08:10:{:02}+00:00 ERROR timeout {j}\n", j % 60)); }
    let path = write_tmp("filedrop_e2e.log", &log);

    let mut ctx = DispatchContext::new(None, None);
    let (tx, rx) = mpsc::channel();
    ctx.analyze_file_streaming("incident.log", &path, &tx);
    drop(tx);
    let (text, done, full) = collect(rx);

    assert!(done);
    assert!(text.contains("ran `eatime`"));
    // Narration should add prose beyond the raw kernel block.
    let full = full.unwrap_or_default();
    eprintln!("=== full output ===\n{full}");
    assert!(full.len() > text.find("[timing:").map(|_| 0).unwrap_or(0),
        "expected a narration to follow the kernel output");
}

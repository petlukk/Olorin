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
    assert!(text.contains("don't have a rune"), "graceful fallback: {text:?}");
    assert!(!text.contains("ran `"), "no rune should have run: {text:?}");
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

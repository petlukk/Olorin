//! CI canary for the ealog stack-overflow class (the v2.8.1 #49 regression,
//! reverted in v2.8.3).
//!
//! That crash grew the kernel's stack usage *per SIMD-loop iteration*, so it
//! overflowed the 8 MB main thread at ~1 MB of input. An adversarial 1 MB case
//! already existed and reproduced it locally — but it slipped through CI,
//! because the overflow threshold is environment-sensitive (runner stack
//! ulimit, eacompute codegen) and 1 MB sat right at the edge on the runner.
//!
//! This test removes that variable. It runs ealog on a multi-megabyte log
//! inside a thread with a PINNED small stack. A correct kernel uses a fixed
//! per-call frame and fits any input on this stack; a regression that grows
//! stack per iteration overflows it deterministically on every runner and
//! aborts the test process — exactly the CI failure we want. If you re-add
//! WARNING/CRITICAL (or any heavy kernel work) and this test starts aborting
//! the process, the kernel is growing stack per iteration again.
//!
//! Calibration: the correct kernel's frame is fixed regardless of input, so a
//! 2 MiB stack holds it with wide margin (it runs fine on the 8 MB main thread
//! for arbitrarily large logs); the 8 MiB input gives a growth-regression a
//! large overflow margin even if eacompute's per-iteration cost is small.

use olorin::runes::output::RuneOutput;
use olorin::runes::run_rune;

const PINNED_STACK: usize = 2 * 1024 * 1024; // 2 MiB — small, fixed, runner-independent
const LOG_BYTES: usize = 8 * 1024 * 1024; // 8 MiB of log

#[test]
fn ealog_large_log_does_not_grow_stack() {
    olorin::kernels::ffi::init().expect("kernel init");

    // ~8 MiB multi-line log (a realistic shape: timestamped INFO lines).
    let line = b"2026-06-11T09:00:00 INFO request handled ok\n";
    let mut data = Vec::with_capacity(LOG_BYTES + line.len());
    while data.len() < LOG_BYTES {
        data.extend_from_slice(line);
    }
    let expected_lines = data.iter().filter(|&&b| b == b'\n').count() as u64;
    let path = std::env::temp_dir()
        .join(format!("olorin_ealog_canary_{}.log", std::process::id()));
    std::fs::write(&path, &data).unwrap();
    let path_str = path.to_string_lossy().into_owned();

    let handle = std::thread::Builder::new()
        .stack_size(PINNED_STACK)
        .spawn(move || {
            let res = run_rune("ealog", &format!("--json {path_str}")).expect("ealog runs");
            RuneOutput::from_json(res.answer.as_bytes()).expect("parse RuneOutput")
        })
        .expect("spawn pinned-stack thread");

    // A per-iteration stack-growth regression overflows the pinned 2 MiB stack
    // inside this join and aborts the process (Rust can't unwind a stack
    // overflow) — which is the red CI signal. A correct kernel returns here.
    let out = handle
        .join()
        .expect("ealog overflowed a 2 MiB stack — kernel is growing stack per iteration");
    let _ = std::fs::remove_file(&path);

    assert!(out.success, "ealog should summarize a large log: {:?}", out.error);
    let info = out
        .categories
        .iter()
        .find(|c| c.name == "INFO")
        .map(|c| c.count)
        .unwrap_or(0);
    // Every line is INFO; allow slack but confirm it actually scanned the file.
    assert!(
        info >= expected_lines - 1,
        "expected ~{expected_lines} INFO lines, got {info}"
    );
}

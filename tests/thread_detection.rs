//! Sanity checks for hardware-detection helpers in
//! `inference::threadpool` — `detect_thread_count` and
//! `detect_prefill_ubatch`.
//!
//! All four tests mutate or read `OLORIN_THREADS` / `OLORIN_PREFILL_UBATCH`.
//! `cargo test` runs tests within one binary in parallel by default, and
//! env-var mutation is process-global, so the tests race without a lock.
//! `ENV_LOCK` serializes them.  Poisoning is treated as recoverable: a
//! prior panic leaves the env in whatever state the next test cares about
//! resetting, so we should not chain-panic on the lock itself.

use olorin::inference::threadpool::{detect_prefill_ubatch, detect_thread_count};
use std::sync::{Mutex, MutexGuard};

static ENV_LOCK: Mutex<()> = Mutex::new(());

fn env_guard() -> MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

#[test]
fn detect_thread_count_is_plausible() {
    let _g = env_guard();

    let n = detect_thread_count();
    assert!(n >= 1, "detect_thread_count returned {n}, must be >= 1");

    let logical = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(1);
    if std::env::var("OLORIN_THREADS").is_err() {
        assert!(
            n <= logical,
            "detect_thread_count {n} exceeds logical-CPU count {logical}"
        );
    }
}

#[test]
fn olorin_threads_env_var_overrides() {
    let _g = env_guard();

    unsafe { std::env::set_var("OLORIN_THREADS", "3"); }
    assert_eq!(detect_thread_count(), 3);
    unsafe { std::env::set_var("OLORIN_THREADS", "bogus"); }
    // Invalid value → falls through to sysfs / available_parallelism, must be >= 1.
    assert!(detect_thread_count() >= 1);
    unsafe { std::env::remove_var("OLORIN_THREADS"); }
}

#[test]
fn prefill_ubatch_env_var_overrides() {
    let _g = env_guard();

    // Explicit override — must be respected regardless of hardware.
    unsafe { std::env::set_var("OLORIN_PREFILL_UBATCH", "32"); }
    assert_eq!(detect_prefill_ubatch(), 32);

    unsafe { std::env::set_var("OLORIN_PREFILL_UBATCH", "128"); }
    assert_eq!(detect_prefill_ubatch(), 128);

    // "0" or invalid → explicitly disabled (returns sentinel).
    unsafe { std::env::set_var("OLORIN_PREFILL_UBATCH", "0"); }
    assert_eq!(detect_prefill_ubatch(), usize::MAX);

    unsafe { std::env::set_var("OLORIN_PREFILL_UBATCH", "nonsense"); }
    // Garbage text → falls through to sysfs detection, must be either
    // 64 (big L3) or usize::MAX (small/no L3). Anything else is a bug.
    let fallback = detect_prefill_ubatch();
    assert!(
        fallback == 64 || fallback == usize::MAX,
        "unexpected auto-detect value {fallback}"
    );

    unsafe { std::env::remove_var("OLORIN_PREFILL_UBATCH"); }
}

#[test]
fn prefill_ubatch_sysfs_detection_consistent() {
    let _g = env_guard();

    // With no override, detection result must be either 64 (big-L3 path)
    // or usize::MAX (small/no L3 path). Which one is hardware-specific.
    unsafe { std::env::remove_var("OLORIN_PREFILL_UBATCH"); }
    let n = detect_prefill_ubatch();
    assert!(n == 64 || n == usize::MAX, "unexpected default {n}");
}

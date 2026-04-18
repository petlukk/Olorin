//! Sanity checks for `inference::threadpool::detect_thread_count`.

use olorin::inference::threadpool::detect_thread_count;

#[test]
fn detect_thread_count_is_plausible() {
    // Test must leave OLORIN_THREADS clean. If an outer shell has it set,
    // the env-var branch is exercised; otherwise we fall into sysfs/fallback.
    // In either case, the answer should be a plausible worker count:
    //   - at least 1
    //   - at most the logical-CPU count reported by the stdlib
    //     (physical-core detection can't return MORE than logical-CPU count)
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
    // SAFETY: env-var mutation is process-global. Running this test under
    // --test-threads=1 (or in its own binary, which it is as a separate .rs
    // file) avoids interleaving with other tests in this file.
    unsafe { std::env::set_var("OLORIN_THREADS", "3"); }
    assert_eq!(detect_thread_count(), 3);
    unsafe { std::env::set_var("OLORIN_THREADS", "bogus"); }
    // Invalid value → falls through to sysfs / available_parallelism, must be >= 1.
    assert!(detect_thread_count() >= 1);
    unsafe { std::env::remove_var("OLORIN_THREADS"); }
}

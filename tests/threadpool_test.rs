//! Tests for SpinBarrier.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[test]
fn spin_barrier_4_threads_1000_rounds() {
    let n_threads = 4usize;
    let n_rounds = 1000;
    let barrier = Arc::new(olorin::inference::threadpool::SpinBarrier::new(n_threads));
    let counter = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..n_threads).map(|_tid| {
        let b = Arc::clone(&barrier);
        let c = Arc::clone(&counter);
        std::thread::spawn(move || {
            for _ in 0..n_rounds {
                c.fetch_add(1, Ordering::Relaxed);
                b.wait();
                // After barrier, all threads should have incremented
            }
        })
    }).collect();

    for h in handles { h.join().unwrap(); }

    // Each thread incremented n_rounds times
    assert_eq!(counter.load(Ordering::Relaxed), n_threads * n_rounds);
}

#[test]
fn spin_barrier_correctness() {
    // Verify that after each barrier, all threads have completed their work.
    // Use a shared array where thread i writes to slot i before barrier,
    // then all threads read all slots after barrier.
    let n_threads = 4usize;
    let barrier = Arc::new(olorin::inference::threadpool::SpinBarrier::new(n_threads));
    let slots: Arc<Vec<AtomicUsize>> = Arc::new((0..n_threads).map(|_| AtomicUsize::new(0)).collect());
    let errors = Arc::new(AtomicUsize::new(0));

    let handles: Vec<_> = (0..n_threads).map(|tid| {
        let b = Arc::clone(&barrier);
        let s = Arc::clone(&slots);
        let e = Arc::clone(&errors);
        std::thread::spawn(move || {
            for round in 1..=100usize {
                // Write my slot
                s[tid].store(round, Ordering::Relaxed);
                b.wait();
                // Read all slots — they should all be `round`
                for j in 0..n_threads {
                    if s[j].load(Ordering::Relaxed) != round {
                        e.fetch_add(1, Ordering::Relaxed);
                    }
                }
                b.wait(); // sync before next round overwrites
            }
        })
    }).collect();

    for h in handles { h.join().unwrap(); }
    assert_eq!(errors.load(Ordering::Relaxed), 0, "barrier failed to synchronize");
}

// GraphPool execute/OpList tests removed — the OpList/GraphCtx API was
// replaced by GraphPool::run_graph with an inline closure. GraphPool is
// exercised end-to-end by gemma4_parallel_regression.

/// When threads arrive at the barrier at very different times, early
/// arrivers must not busy-spin until the late thread arrives — they must
/// fall to the futex slow path so a preempted peer can get its core back.
///
/// Scenario: tid=0 runs ~300 ms of CPU work; tid 1..n arrive at the
/// barrier almost immediately. After the bounded spin (~30k iters, ~15 µs),
/// they must call `futex_wait`. Each of the n-1 fast threads should do
/// exactly one `futex_wait` call (maybe two if spurious wake).
///
/// We assert futex_wait was called — a direct signal the slow path was
/// taken.
///
/// Marked `#[ignore]`: the `FUTEX_WAIT_CALLS` counter is process-wide and
/// other barrier-using tests in this file bump it when they run
/// concurrently (the default `cargo test` behavior on a 4-core box spawns
/// 4-thread workloads that sometimes exhaust the barrier's spin budget
/// under the contention of other tests running). Run explicitly:
/// `cargo test --release --test threadpool_test -- --ignored`.
#[test]
#[ignore]
fn spin_barrier_falls_to_futex_on_asymmetric_arrival() {
    let n_threads = 4usize;
    let barrier = Arc::new(olorin::inference::threadpool::SpinBarrier::new(n_threads));
    let wait_ms = 300u64;

    let calls_before = olorin::inference::threadpool::futex_wait_call_count();
    let wall_start = Instant::now();

    let handles: Vec<_> = (0..n_threads).map(|tid| {
        let b = Arc::clone(&barrier);
        std::thread::spawn(move || {
            if tid == 0 {
                let deadline = Instant::now() + Duration::from_millis(wait_ms);
                let mut acc: u64 = 0;
                let mut i: u64 = 0;
                while Instant::now() < deadline {
                    for _ in 0..10_000 {
                        acc = acc.wrapping_add(i).wrapping_mul(0x9e3779b97f4a7c15);
                        i = i.wrapping_add(1);
                    }
                }
                std::hint::black_box(acc);
            }
            b.wait();
        })
    }).collect();

    for h in handles { h.join().unwrap(); }

    let wall_ms = wall_start.elapsed().as_millis() as u64;
    let futex_calls =
        olorin::inference::threadpool::futex_wait_call_count() - calls_before;

    assert!(
        wall_ms >= wait_ms / 2,
        "work finished too fast to measure ({} ms); calibrate wait_ms", wall_ms
    );
    // Expect a small number of futex_wait calls — each fast thread blocks
    // once and is woken by the last arrival. Allow modest slack for spurious
    // wake-ups (EINTR / perf-subsystem signals under `cargo test`). The real
    // failure mode is hundreds-of-calls hot-spin; anything under 30 proves
    // the slow path blocked.
    assert!(
        futex_calls >= n_threads - 1,
        "expected >= {} futex_wait calls, got {} — slow path never taken",
        n_threads - 1, futex_calls
    );
    assert!(
        futex_calls < 30,
        "futex_wait called {} times in {} ms — hot-spinning instead of blocking",
        futex_calls, wall_ms
    );
}

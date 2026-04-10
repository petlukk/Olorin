//! Tests for SpinBarrier and GraphPool.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use olorin::inference::graph::{GraphCtx, OpList};
use olorin::inference::threadpool::GraphPool;

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

#[test]
fn graph_pool_3_ops() {
    let pool = GraphPool::new();
    let nt = pool.thread_count();

    // 3 ops: each increments a counter. After execute(), counter should be 3 * n_threads.
    let counter: &'static AtomicUsize = Box::leak(Box::new(AtomicUsize::new(0)));

    let mut ops = OpList::new();
    ops.push(move |_ith, _nth, _ctx| { counter.fetch_add(1, Ordering::Relaxed); });
    ops.push(move |_ith, _nth, _ctx| { counter.fetch_add(1, Ordering::Relaxed); });
    ops.push(move |_ith, _nth, _ctx| { counter.fetch_add(1, Ordering::Relaxed); });

    pool.execute(&ops);
    assert_eq!(counter.load(Ordering::Relaxed), 3 * nt);
}

#[test]
fn graph_pool_barrier_ordering() {
    // Op1: thread 0 writes value. Op2: all threads read it.
    // Barrier between ops ensures op1 completes before op2 starts.
    let pool = GraphPool::new();
    let val: &'static AtomicUsize = Box::leak(Box::new(AtomicUsize::new(0)));
    let errors: &'static AtomicUsize = Box::leak(Box::new(AtomicUsize::new(0)));

    let mut ops = OpList::new();
    ops.push(move |ith, _nth, _ctx| {
        if ith == 0 { val.store(42, Ordering::Relaxed); }
    });
    ops.push(move |_ith, _nth, _ctx| {
        if val.load(Ordering::Relaxed) != 42 {
            errors.fetch_add(1, Ordering::Relaxed);
        }
    });

    pool.execute(&ops);
    assert_eq!(errors.load(Ordering::Relaxed), 0, "barrier didn't sync between ops");
}

#[test]
fn graph_pool_multiple_executions() {
    // Execute 10 times — tests that pool resets correctly between graphs.
    let pool = GraphPool::new();
    let nt = pool.thread_count();
    let counter: &'static AtomicUsize = Box::leak(Box::new(AtomicUsize::new(0)));

    for _ in 0..10 {
        let mut ops = OpList::new();
        ops.push(move |_ith, _nth, _ctx| { counter.fetch_add(1, Ordering::Relaxed); });
        pool.execute(&ops);
    }
    assert_eq!(counter.load(Ordering::Relaxed), 10 * nt);
}

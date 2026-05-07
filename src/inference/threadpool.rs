//! Thread pool for parallel compute dispatch.
//!
//! `GraphPool` + `SpinBarrier`: atomic kickoff via a versioned `n_graph`
//! counter, bounded spin then futex-block at barriers. Matches the behavior
//! of llama.cpp's `ggml_graph_compute` compiled with OpenMP (`GOMP_barrier`),
//! which is the form Pi/Debian distributes.

use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicUsize, Ordering, fence};
use std::sync::{Condvar, Mutex};
use std::thread::{self, JoinHandle};

use crate::platform::futex::{wait as futex_wait, wake_all as futex_wake_all};

// Thread-count / cache-size detection lives in its own module to keep this
// file focused on the threadpool primitives. Re-exported so external callers
// keep the `threadpool::detect_*` path.
pub use super::threadpool_detect::{detect_prefill_ubatch, detect_thread_count};

/// Diagnostic re-export: total `futex_wait` invocations since process start.
pub fn futex_wait_call_count() -> usize {
    crate::platform::futex::wait_call_count()
}


// ---------------------------------------------------------------------------
// SpinBarrier — bounded spin then futex block, matching GOMP_barrier
// ---------------------------------------------------------------------------

/// Iterations of `spin_loop()` before falling to futex block. ~15-60 µs at
/// 2.4 GHz. Sized to cover jitter between threads arriving at a barrier
/// during a tight graph dispatch, without burning cycles when a thread has
/// been preempted by the kernel. GOMP default for few-core systems is
/// comparable.
const BARRIER_SPIN_BUDGET: u32 = 30_000;

/// Atomic barrier. All n_threads call wait(); last arrival resets and wakes
/// everyone. Fast path is a short spin on `n_barrier_passed`; slow path is
/// a futex block so preempted threads can get the CPU back.
///
/// Ref: llama.cpp ggml-cpu.c ggml_barrier() with GGML_USE_OPENMP → GOMP_barrier.
#[repr(C, align(64))]
pub struct SpinBarrier {
    n_threads: i32,
    n_barrier: AtomicI32,
    n_barrier_passed: AtomicU32,
}

impl SpinBarrier {
    pub fn new(n_threads: usize) -> Self {
        Self {
            n_threads: n_threads as i32,
            n_barrier: AtomicI32::new(0),
            n_barrier_passed: AtomicU32::new(0),
        }
    }

    #[inline]
    pub fn wait(&self) {
        if self.n_threads <= 1 { return; }

        let old_passed = self.n_barrier_passed.load(Ordering::Relaxed);

        // Enter barrier (full seq-cst fence).
        let n = self.n_barrier.fetch_add(1, Ordering::SeqCst);

        if n == self.n_threads - 1 {
            // Last arrival — reset counter, bump pass counter, wake all waiters.
            self.n_barrier.store(0, Ordering::Relaxed);
            self.n_barrier_passed.fetch_add(1, Ordering::SeqCst);
            futex_wake_all(&self.n_barrier_passed);
            return;
        }

        // Fast path: bounded spin covers tight arrivals (µs scale).
        for _ in 0..BARRIER_SPIN_BUDGET {
            if self.n_barrier_passed.load(Ordering::Relaxed) != old_passed {
                fence(Ordering::SeqCst);
                return;
            }
            std::hint::spin_loop();
        }

        // Slow path: block on futex so a preempted peer can get its core back.
        while self.n_barrier_passed.load(Ordering::Relaxed) == old_passed {
            futex_wait(&self.n_barrier_passed, old_passed);
        }
        fence(Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------------------
// GraphPool — graph-loop threading matching llama.cpp exactly
// Ref: ggml-cpu.c ggml_graph_compute / ggml_graph_compute_kickoff /
//      ggml_graph_compute_secondary_thread / ggml_graph_compute_check_for_work
// ---------------------------------------------------------------------------

/// Packed n_graph: (graph_counter << 16) | n_active_threads.
/// Workers detect new work by comparing n_graph against their last value.
/// Ref: ggml-cpu.c GGML_THREADPOOL_N_THREADS_BITS
const N_THREADS_BITS: u32 = 16;
const N_THREADS_MASK: i32 = (1 << N_THREADS_BITS) - 1;

/// Spin-poll rounds before falling to condvar sleep.
/// Ref: ggml-cpu.c ggml_graph_compute_poll_for_work (1024*128*poll, poll=1)
const SPIN_POLL_ROUNDS: u64 = 1024 * 128;

type GraphWorkFn = unsafe fn(*const (), usize, usize, &SpinBarrier, &AtomicI32);

/// Graph pool matching llama.cpp's threadpool model exactly:
/// - N-1 worker threads spin-poll then condvar-sleep between dispatches
/// - Kickoff: store n_graph (SeqCst) + broadcast under lock
/// - Main thread (ith=0) participates via same work function
/// - Spin-barriers between ops, work-stealing for matmul
/// Ref: ggml-cpu.c:3237 ggml_graph_compute
pub struct GraphPool {
    shared: *mut GraphPoolShared,
    workers: Vec<JoinHandle<()>>,
    n_threads: usize,
}

#[repr(C, align(64))]
struct GraphPoolShared {
    // Condvar for sleep fallback (hybrid poll/wait like llama.cpp)
    mutex: Mutex<()>,
    cond: Condvar,
    // (graph_counter << 16) | n_active_threads
    // SeqCst store in kickoff, Relaxed poll in workers
    // Ref: ggml-cpu.c:475 atomic_int n_graph
    n_graph: AtomicI32,
    stop: AtomicBool,
    // Work function + data: set before n_graph SeqCst store,
    // read after worker's SeqCst fence — happens-before guaranteed.
    work_fn: AtomicUsize,
    work_data: AtomicUsize,
    // Spin-barrier between ops (all n_threads participate)
    pub barrier: SpinBarrier,
    // Work-stealing for matmul
    pub current_chunk: AtomicI32,
    n_threads: usize,
}

impl GraphPool {
    pub fn new() -> Self {
        let n_threads = detect_thread_count();

        let shared = Box::into_raw(Box::new(GraphPoolShared {
            mutex: Mutex::new(()),
            cond: Condvar::new(),
            n_graph: AtomicI32::new(0),
            stop: AtomicBool::new(false),
            work_fn: AtomicUsize::new(0),
            work_data: AtomicUsize::new(0),
            barrier: SpinBarrier::new(n_threads),
            current_chunk: AtomicI32::new(0),
            n_threads,
        }));

        // Spawn N-1 workers (ith=1..N-1). Main thread is ith=0.
        // 8 MB stack: batched_matmul_step's call chain needs more than the 2 MB default.
        // Ref: ggml-cpu.c:3212
        let mut workers = Vec::with_capacity(n_threads.saturating_sub(1));
        for tid in 1..n_threads {
            let s = shared as usize;
            let handle = thread::Builder::new()
                .stack_size(32 * 1024 * 1024)
                .spawn(move || {
                    let shared = unsafe { &*(s as *const GraphPoolShared) };
                    graph_worker_loop(shared, tid);
                })
                .expect("spawn graph worker");
            workers.push(handle);
        }

        GraphPool { shared, workers, n_threads }
    }

    pub fn thread_count(&self) -> usize { self.n_threads }
    pub fn barrier(&self) -> &SpinBarrier { unsafe { &(*self.shared).barrier } }
    pub fn chunk(&self) -> &AtomicI32 { unsafe { &(*self.shared).current_chunk } }

    /// Execute work on all threads. Main thread (ith=0) participates directly.
    /// Ref: ggml-cpu.c:3296-3299 kickoff then compute_thread(&workers[0])
    pub fn run_graph<F>(&self, f: &F)
    where F: Fn(usize, usize, &SpinBarrier, &AtomicI32) + Send + Sync
    {
        unsafe fn trampoline<F: Fn(usize, usize, &SpinBarrier, &AtomicI32)>(
            data: *const (), tid: usize, nth: usize,
            barrier: &SpinBarrier, chunk: &AtomicI32,
        ) {
            let f = &*(data as *const F);
            f(tid, nth, barrier, chunk);
        }

        let shared = unsafe { &*self.shared };

        // Set work fn/data BEFORE SeqCst store — relaxed stores become visible
        // to workers after their SeqCst fence (thread_sync).
        shared.work_fn.store(trampoline::<F> as usize, Ordering::Relaxed);
        shared.work_data.store(f as *const F as *const () as usize, Ordering::Relaxed);

        // Kickoff: update n_graph + broadcast UNDER lock.
        // Ref: ggml-cpu.c:3126 ggml_graph_compute_kickoff
        {
            let _guard = shared.mutex.lock().unwrap();
            let old = shared.n_graph.load(Ordering::Relaxed);
            let counter = (old >> N_THREADS_BITS) + 1;
            let new_val = (counter << N_THREADS_BITS)
                | (shared.n_threads as i32 & N_THREADS_MASK);
            shared.n_graph.store(new_val, Ordering::SeqCst);
            shared.cond.notify_all();
        }

        // Main thread participates as ith=0.
        // Ref: ggml-cpu.c:3299 ggml_graph_compute_thread(&workers[0])
        f(0, self.n_threads, &shared.barrier, &shared.current_chunk);
    }
}

/// Worker thread main loop matching llama.cpp secondary_thread.
/// Ref: ggml-cpu.c:3088 ggml_graph_compute_secondary_thread
fn graph_worker_loop(shared: &GraphPoolShared, tid: usize) {
    let mut last_graph: i32 = 0;

    loop {
        // Phase 1: Spin-poll n_graph for new work.
        // Ref: ggml-cpu.c:3054 ggml_graph_compute_poll_for_work
        let mut pending = false;
        for _ in 0..SPIN_POLL_ROUNDS {
            if shared.stop.load(Ordering::Relaxed) { return; }
            let ng = shared.n_graph.load(Ordering::Relaxed);
            if ng != last_graph {
                pending = (tid as i32) < (ng & N_THREADS_MASK);
                last_graph = ng;
                break;
            }
            std::hint::spin_loop();
        }

        // Phase 2: Condvar sleep if spin-poll found nothing.
        // Ref: ggml-cpu.c:3069 ggml_graph_compute_check_for_work (mutex path)
        if !pending {
            let mut guard = shared.mutex.lock().unwrap();
            loop {
                if shared.stop.load(Ordering::Relaxed) { return; }
                let ng = shared.n_graph.load(Ordering::Relaxed);
                if ng != last_graph {
                    pending = (tid as i32) < (ng & N_THREADS_MASK);
                    last_graph = ng;
                    break;
                }
                guard = shared.cond.wait(guard).unwrap();
            }
            drop(guard);
        }

        if shared.stop.load(Ordering::Relaxed) { return; }
        if !pending { continue; }

        // Thread sync: SeqCst fence ensures work_fn/work_data stores are visible.
        // Ref: ggml-cpu.c:3044 ggml_graph_compute_thread_sync
        fence(Ordering::SeqCst);

        // Execute work function.
        // Ref: ggml-cpu.c:3116-3118
        let work_fn = shared.work_fn.load(Ordering::Relaxed);
        let work_data = shared.work_data.load(Ordering::Relaxed);
        let f: GraphWorkFn = unsafe { std::mem::transmute(work_fn) };
        unsafe {
            f(work_data as *const (), tid, shared.n_threads,
              &shared.barrier, &shared.current_chunk);
        }
    }
}

impl Drop for GraphPool {
    fn drop(&mut self) {
        let shared = unsafe { &*self.shared };
        // Signal stop + bump n_graph to wake spinning workers
        shared.stop.store(true, Ordering::SeqCst);
        {
            let _guard = shared.mutex.lock().unwrap();
            let old = shared.n_graph.load(Ordering::Relaxed);
            shared.n_graph.store(old.wrapping_add(1), Ordering::SeqCst);
            shared.cond.notify_all();
        }
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
        unsafe { drop(Box::from_raw(self.shared)); }
    }
}

// Safety: shared is heap-allocated and outlives all workers
unsafe impl Send for GraphPool {}
unsafe impl Sync for GraphPool {}

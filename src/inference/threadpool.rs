//! Thread pools for parallel compute dispatch.
//!
//! Two implementations:
//! - `ThreadPool`: legacy mutex/condvar dispatch (used during migration)
//! - `SpinBarrier`: atomic barrier matching llama.cpp ggml_barrier()

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicUsize, Ordering, fence};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

// ---------------------------------------------------------------------------
// Thread-count detection
// ---------------------------------------------------------------------------

/// Count physical (non-SMT) CPU cores via Linux sysfs.
/// Returns None if the sysfs interface isn't available or unreadable
/// (e.g. non-Linux, sandboxed containers).
fn physical_core_count_sysfs() -> Option<usize> {
    let entries = std::fs::read_dir("/sys/devices/system/cpu").ok()?;
    let mut siblings_first: HashSet<u32> = HashSet::new();
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("cpu") || !name[3..].chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        let path = entry.path().join("topology/thread_siblings_list");
        let Ok(txt) = std::fs::read_to_string(&path) else { continue };
        // Format: "0,8" or "0-1" or "0". First sibling ID = physical core representative.
        let first = txt
            .trim()
            .split(|c: char| c == ',' || c == '-')
            .next()?
            .parse::<u32>()
            .ok()?;
        siblings_first.insert(first);
    }
    if siblings_first.is_empty() { None } else { Some(siblings_first.len()) }
}

/// Decide worker thread count. Priority:
/// 1. `OLORIN_THREADS` env var (positive integer).
/// 2. Physical-core count from sysfs — ignores SMT siblings on x86,
///    equals logical count on ARM (no SMT on Cortex-A76 / Pi 5).
/// 3. `std::thread::available_parallelism()` fallback.
/// 4. `1` last-resort.
pub fn detect_thread_count() -> usize {
    if let Ok(s) = std::env::var("OLORIN_THREADS") {
        if let Ok(n) = s.trim().parse::<usize>() {
            if n >= 1 { return n; }
        }
    }
    if let Some(n) = physical_core_count_sysfs() {
        return n;
    }
    thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

// ---------------------------------------------------------------------------
// SpinBarrier — matches llama.cpp ggml_barrier() exactly
// ---------------------------------------------------------------------------

/// Atomic spin-barrier. All n_threads call wait(); last arrival resets and
/// signals others. Spin-loop uses YIELD (ARM) / PAUSE (x86).
/// Ref: llama.cpp ggml-cpu.c:562
#[repr(C, align(64))]
pub struct SpinBarrier {
    n_threads: i32,
    n_barrier: AtomicI32,
    n_barrier_passed: AtomicI32,
}

impl SpinBarrier {
    pub fn new(n_threads: usize) -> Self {
        Self {
            n_threads: n_threads as i32,
            n_barrier: AtomicI32::new(0),
            n_barrier_passed: AtomicI32::new(0),
        }
    }

    #[inline]
    pub fn wait(&self) {
        if self.n_threads <= 1 { return; }

        let old_passed = self.n_barrier_passed.load(Ordering::Relaxed);

        // Enter barrier (full seq-cst fence)
        let n = self.n_barrier.fetch_add(1, Ordering::SeqCst);

        if n == self.n_threads - 1 {
            // Last thread — reset counter and signal
            self.n_barrier.store(0, Ordering::Relaxed);
            self.n_barrier_passed.fetch_add(1, Ordering::SeqCst);
            return;
        }

        // Spin until last thread signals
        while self.n_barrier_passed.load(Ordering::Relaxed) == old_passed {
            std::hint::spin_loop();
        }
        std::sync::atomic::fence(Ordering::SeqCst);
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

// ---------------------------------------------------------------------------
// Legacy ThreadPool (mutex/condvar dispatch)
// ---------------------------------------------------------------------------

static NOOP_FN: &(dyn Fn(usize, usize) + Send + Sync) = &|_, _| {};

struct WorkState {
    funcs: [*const dyn Fn(usize, usize); 3],
    bounds: [usize; 3],
    n_groups: usize,
    generation: u64,
    shutdown: bool,
}

unsafe impl Send for WorkState {}
unsafe impl Sync for WorkState {}

pub struct ThreadPool {
    shared: Arc<(Mutex<WorkState>, Condvar)>,
    done: Arc<AtomicUsize>,
    done_signal: Arc<(Mutex<bool>, Condvar)>,
    workers: Vec<JoinHandle<()>>,
    n_threads: usize,
}

impl ThreadPool {
    pub fn new() -> Self {
        let n_threads = detect_thread_count();

        let noop_ptr = NOOP_FN as *const dyn Fn(usize, usize);
        let shared = Arc::new((
            Mutex::new(WorkState {
                funcs: [noop_ptr, noop_ptr, noop_ptr],
                bounds: [0, 0, 0],
                n_groups: 0,
                generation: 0,
                shutdown: false,
            }),
            Condvar::new(),
        ));
        let done = Arc::new(AtomicUsize::new(0));
        let done_signal = Arc::new((Mutex::new(false), Condvar::new()));

        let mut workers = Vec::with_capacity(n_threads);
        for tid in 0..n_threads {
            let shared = Arc::clone(&shared);
            let done = Arc::clone(&done);
            let done_signal = Arc::clone(&done_signal);
            let handle = thread::Builder::new()
                .stack_size(32 * 1024 * 1024)
                .spawn(move || {
                let mut last_gen: u64 = 0;
                loop {
                    let (funcs, bounds, n_groups);
                    {
                        let (lock, cvar) = &*shared;
                        let mut state = lock.lock().unwrap();
                        while state.generation == last_gen && !state.shutdown {
                            state = cvar.wait(state).unwrap();
                        }
                        if state.shutdown {
                            return;
                        }
                        last_gen = state.generation;
                        funcs = state.funcs;
                        bounds = state.bounds;
                        n_groups = state.n_groups;
                    }
                    let n_active_total = bounds[n_groups - 1];
                    if tid < n_active_total {
                        if tid < bounds[0] {
                            let f = unsafe { &*funcs[0] };
                            f(tid, bounds[0]);
                        } else if n_groups >= 2 && tid < bounds[1] {
                            let f = unsafe { &*funcs[1] };
                            f(tid - bounds[0], bounds[1] - bounds[0]);
                        } else if n_groups >= 3 && tid < bounds[2] {
                            let f = unsafe { &*funcs[2] };
                            f(tid - bounds[1], bounds[2] - bounds[1]);
                        }
                    }
                    if done.fetch_sub(1, Ordering::AcqRel) == 1 {
                        let (lock, cvar) = &*done_signal;
                        let mut finished = lock.lock().unwrap();
                        *finished = true;
                        cvar.notify_one();
                    }
                }
            }).expect("spawn pool worker");
            workers.push(handle);
        }

        ThreadPool { shared, done, done_signal, workers, n_threads }
    }

    pub fn thread_count(&self) -> usize {
        self.n_threads
    }

    fn dispatch(
        &self,
        funcs: [*const dyn Fn(usize, usize); 3],
        bounds: [usize; 3],
        n_groups: usize,
    ) {
        self.done.store(self.n_threads, Ordering::Release);
        {
            let mut finished = self.done_signal.0.lock().unwrap();
            *finished = false;
        }

        {
            let (lock, cvar) = &*self.shared;
            let mut state = lock.lock().unwrap();
            state.funcs = funcs;
            state.bounds = bounds;
            state.n_groups = n_groups;
            state.generation += 1;
            cvar.notify_all();
        }

        {
            let (lock, cvar) = &*self.done_signal;
            let mut finished = lock.lock().unwrap();
            while !*finished {
                finished = cvar.wait(finished).unwrap();
            }
        }
    }

    pub fn run(&self, n: usize, f: impl Fn(usize, usize) + Send + Sync) {
        debug_assert!(n <= self.n_threads, "n ({n}) > pool size ({})", self.n_threads);
        if n == 0 { return; }
        let func_ref: &dyn Fn(usize, usize) = &f;
        let func_ref: &dyn Fn(usize, usize) = unsafe { std::mem::transmute(func_ref) };
        self.dispatch(
            [func_ref as *const _, NOOP_FN as *const _, NOOP_FN as *const _],
            [n, 0, 0],
            1,
        );
    }

    pub fn run_split3(
        &self,
        n1: usize, f1: impl Fn(usize, usize) + Send + Sync,
        n2: usize, f2: impl Fn(usize, usize) + Send + Sync,
        n3: usize, f3: impl Fn(usize, usize) + Send + Sync,
    ) {
        debug_assert!(
            n1 + n2 + n3 <= self.n_threads,
            "split3 {} + {} + {} > pool {}", n1, n2, n3, self.n_threads
        );
        if n1 + n2 + n3 == 0 { return; }
        let r1: &dyn Fn(usize, usize) = &f1;
        let r2: &dyn Fn(usize, usize) = &f2;
        let r3: &dyn Fn(usize, usize) = &f3;
        let r1: &dyn Fn(usize, usize) = unsafe { std::mem::transmute(r1) };
        let r2: &dyn Fn(usize, usize) = unsafe { std::mem::transmute(r2) };
        let r3: &dyn Fn(usize, usize) = unsafe { std::mem::transmute(r3) };
        self.dispatch(
            [r1 as *const _, r2 as *const _, r3 as *const _],
            [n1, n1 + n2, n1 + n2 + n3],
            3,
        );
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        {
            let (lock, cvar) = &*self.shared;
            let mut state = lock.lock().unwrap();
            state.shutdown = true;
            cvar.notify_all();
        }
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

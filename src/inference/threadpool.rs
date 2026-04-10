//! Thread pools for parallel compute dispatch.
//!
//! Two implementations:
//! - `ThreadPool`: legacy mutex/condvar dispatch (used during migration)
//! - `SpinBarrier`: atomic barrier matching llama.cpp ggml_barrier()

use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

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
// GraphPool — graph-loop threading matching llama.cpp
// ---------------------------------------------------------------------------

use super::graph::{GraphCtx, OpList};

/// Thread pool where all threads execute the entire op-list together.
/// Workers sleep (condvar) between tokens, spin-barrier between ops.
/// Matches llama.cpp's ggml_graph_compute_thread loop.
pub struct GraphPool {
    shared: Box<GraphPoolShared>,
    workers: Vec<JoinHandle<()>>,
    n_threads: usize,
}

struct GraphPoolShared {
    // Sleep/wake between tokens
    mutex: Mutex<GraphPoolState>,
    cond: Condvar,
    // Spin-barrier between ops
    barrier: SpinBarrier,
    // Work-stealing for matmul
    ctx: GraphCtx,
}

struct GraphPoolState {
    ops_ptr: usize,  // *const OpList, valid only during execute()
    n_ops: usize,
    generation: u64,
    n_threads_total: usize,
    done_count: AtomicUsize,
    shutdown: bool,
}

// Safety: ops_ptr is only read while execute() holds the OpList alive.
unsafe impl Send for GraphPoolState {}
unsafe impl Sync for GraphPoolState {}

impl GraphPool {
    pub fn new() -> Self {
        let n_threads = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let shared = Box::new(GraphPoolShared {
            mutex: Mutex::new(GraphPoolState {
                ops_ptr: 0,
                n_ops: 0,
                generation: 0,
                n_threads_total: n_threads,
                done_count: AtomicUsize::new(0),
                shutdown: false,
            }),
            cond: Condvar::new(),
            barrier: SpinBarrier::new(n_threads),
            ctx: GraphCtx::new(),
        });

        // Leak shared to get 'static reference for workers
        let shared_ref: &'static GraphPoolShared = unsafe {
            &*(Box::into_raw(shared) as *const GraphPoolShared)
        };

        let mut workers = Vec::with_capacity(n_threads);
        for tid in 0..n_threads {
            let s = shared_ref as *const GraphPoolShared as usize;
            let handle = thread::spawn(move || {
                let shared = unsafe { &*(s as *const GraphPoolShared) };
                let mut last_gen: u64 = 0;
                loop {
                    // Sleep until new graph or shutdown
                    let (ops_ptr, n_ops);
                    {
                        let mut state = shared.mutex.lock().unwrap();
                        while state.generation == last_gen && !state.shutdown {
                            state = shared.cond.wait(state).unwrap();
                        }
                        if state.shutdown { return; }
                        last_gen = state.generation;
                        ops_ptr = state.ops_ptr;
                        n_ops = state.n_ops;
                    }

                    // Execute all ops with spin-barriers (llama.cpp graph loop)
                    let ops = unsafe { &*(ops_ptr as *const OpList) };
                    for i in 0..n_ops {
                        ops.ops[i](tid, n_threads, &shared.ctx);
                        if i + 1 < n_ops {
                            shared.barrier.wait();
                        }
                    }

                    // Signal completion
                    if shared.mutex.lock().unwrap().done_count
                        .fetch_sub(1, Ordering::AcqRel) == 1
                    {
                        // Last thread — notify dispatcher
                        shared.cond.notify_all();
                    }
                }
            });
            workers.push(handle);
        }

        // Reconstruct Box from leaked pointer for Drop
        let shared = unsafe { Box::from_raw(shared_ref as *const _ as *mut GraphPoolShared) };

        GraphPool { shared, workers, n_threads }
    }

    pub fn thread_count(&self) -> usize {
        self.n_threads
    }

    /// Execute an op-list. All threads run all ops with spin-barriers between.
    /// Blocks until all threads complete.
    pub fn execute(&self, ops: &OpList) {
        if ops.len() == 0 { return; }

        {
            let mut state = self.shared.mutex.lock().unwrap();
            state.ops_ptr = ops as *const OpList as usize;
            state.n_ops = ops.len();
            state.done_count.store(self.n_threads, Ordering::Release);
            state.generation += 1;
        }
        self.shared.cond.notify_all();

        // Wait for all threads to finish
        {
            let mut state = self.shared.mutex.lock().unwrap();
            while state.done_count.load(Ordering::Acquire) != 0 {
                state = self.shared.cond.wait(state).unwrap();
            }
        }
    }

    /// Access the shared GraphCtx (for resetting current_chunk before matmul ops).
    pub fn ctx(&self) -> &GraphCtx {
        &self.shared.ctx
    }
}

impl Drop for GraphPool {
    fn drop(&mut self) {
        {
            let mut state = self.shared.mutex.lock().unwrap();
            state.shutdown = true;
        }
        self.shared.cond.notify_all();
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

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
        let n_threads = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

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
            let handle = thread::spawn(move || {
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
            });
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

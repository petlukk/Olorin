//! Legacy mutex/condvar thread pool used during the migration to
//! `GraphPool`. Still in use for a handful of call sites (e.g. the
//! `forward_one_graph` entry point used by the parallel-regression test).
//!
//! New code should use `GraphPool` from the parent module.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use super::threadpool::detect_thread_count;

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

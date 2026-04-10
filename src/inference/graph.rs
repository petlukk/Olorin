//! Op-list for graph-loop threading.
//!
//! Each op is a function executed by all threads with their (ith, nth).
//! Threads spin-barrier between ops, matching llama.cpp's graph compute loop.

use std::sync::atomic::AtomicI32;

/// Context passed to every graph op — shared mutable state + work-stealing counter.
pub struct GraphCtx {
    /// Work-stealing chunk counter for matmul ops (like llama.cpp current_chunk).
    /// Reset to n_threads before each matmul op; threads atomic_fetch_add to grab work.
    pub current_chunk: AtomicI32,
}

impl GraphCtx {
    pub fn new() -> Self {
        Self {
            current_chunk: AtomicI32::new(0),
        }
    }
}

/// A graph operation. All threads call this with (ith, nth, ctx).
/// The function splits work internally by ith/nth.
pub type GraphOp = Box<dyn Fn(usize, usize, &GraphCtx) + Send + Sync>;

/// Ordered list of ops for one forward pass.
pub struct OpList {
    pub ops: Vec<GraphOp>,
}

impl OpList {
    pub fn new() -> Self {
        Self { ops: Vec::with_capacity(512) }
    }

    pub fn push(&mut self, op: impl Fn(usize, usize, &GraphCtx) + Send + Sync + 'static) {
        self.ops.push(Box::new(op));
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }
}

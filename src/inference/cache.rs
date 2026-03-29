//! EakvCache stub — placeholder for Task 8.
//!
//! Provides the types and methods that forward passes call.
//! Real implementation will replace this in Task 8.

use crate::error::Result;

pub struct KernelTable;

impl KernelTable {
    pub fn init() -> Result<Self> {
        Ok(KernelTable)
    }
}

pub struct EakvCache {
    _priv: (),
}

impl EakvCache {
    pub fn new(
        _n_layers: i32,
        _n_kv_heads: i32,
        _head_dim: i32,
        _max_seq_len: i32,
        _kt: KernelTable,
    ) -> Result<Self> {
        Ok(EakvCache { _priv: () })
    }

    pub fn append(&mut self, _data: &[f32], _layer: i32, _kv: i32, _count: i32) -> Result<()> {
        Ok(())
    }

    pub fn advance(&mut self, _n: i32) -> Result<()> {
        Ok(())
    }
}

pub mod attention {
    use super::EakvCache;

    pub fn attention_scores(
        _cache: &EakvCache,
        _q: &[f32],
        _layer: i32,
        _n_heads: i32,
        _n_kv_heads: i32,
        _scores: &mut [f32],
    ) {
        // Stub: zero scores
        for s in _scores.iter_mut() { *s = 0.0; }
    }

    pub fn attention_output(
        _cache: &EakvCache,
        _scores: &[f32],
        _layer: i32,
        _n_heads: i32,
        _n_kv_heads: i32,
        _out: &mut [f32],
    ) {
        // Stub: zero output
        for v in _out.iter_mut() { *v = 0.0; }
    }
}

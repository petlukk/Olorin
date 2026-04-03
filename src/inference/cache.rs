//! KV cache — f16 storage, identical to llama.cpp.
//!
//! Layout: [layer][kv_idx][head][seq * head_dim] as f16 (u16).
//! No quantization, no rotation. Direct f32->f16 conversion via Ea kernels.

use crate::error::{Error, Result};
use crate::kernels::ffi_inference as ffi;

pub struct F16KvCache {
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    seq_len: usize,
    data: Vec<u16>,
    slot_elems: usize,
}

unsafe impl Send for F16KvCache {}

impl F16KvCache {
    pub fn new(
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        if n_layers == 0 || n_kv_heads == 0 || head_dim == 0 || max_seq_len == 0 {
            return Err(Error::Inference("F16KvCache: invalid dimensions".into()));
        }
        // slot = one KV buffer for one layer (K or V)
        // [head][max_seq_len * head_dim]
        let slot_elems = n_kv_heads * max_seq_len * head_dim;
        let total = n_layers * 2 * slot_elems;
        let data = vec![0u16; total];

        Ok(Self {
            n_layers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            seq_len: 0,
            data,
            slot_elems,
        })
    }

    // ── Accessors ──

    pub fn seq_len(&self) -> usize { self.seq_len }
    pub fn len(&self) -> usize { self.seq_len }
    pub fn n_layers(&self) -> usize { self.n_layers }
    pub fn n_kv_heads(&self) -> usize { self.n_kv_heads }
    pub fn head_dim(&self) -> usize { self.head_dim }
    pub fn max_seq_len(&self) -> usize { self.max_seq_len }

    pub fn checkpoint(&self) -> usize { self.seq_len }

    pub fn restore(&mut self, seq_len: usize) -> Result<()> {
        if seq_len > self.seq_len {
            return Err(Error::Inference(format!(
                "restore: seq_len {seq_len} out of range [0, {}]", self.seq_len
            )));
        }
        self.seq_len = seq_len;
        Ok(())
    }

    pub fn advance(&mut self, n: usize) -> Result<()> {
        if n == 0 {
            return Err(Error::Inference("advance: n must be > 0".into()));
        }
        if self.seq_len + n > self.max_seq_len {
            return Err(Error::Inference(format!(
                "advance: {} + {} > max {}", self.seq_len, n, self.max_seq_len
            )));
        }
        self.seq_len += n;
        Ok(())
    }

    pub fn clear(&mut self) { self.seq_len = 0; }

    // ── Slot offset ──

    /// Returns the starting index in `data` for slot (layer, kv_idx).
    /// kv_idx: 0 = K, 1 = V.
    #[inline]
    fn slot_offset(&self, layer: usize, kv_idx: usize) -> usize {
        (layer * 2 + kv_idx) * self.slot_elems
    }

    // ── Store ──

    /// Store f32 data into the cache as f16.
    ///
    /// `kv_idx`: 0 = K, 1 = V.
    /// Input layout: token-major `[n_tokens][n_kv_heads * head_dim]`.
    /// Stored as: `[head][seq * head_dim]` within each slot.
    pub fn store(&mut self, layer: usize, kv_idx: usize, data_f32: &[f32], n_tokens: usize) -> Result<()> {
        if n_tokens == 0 {
            return Err(Error::Inference("store: n_tokens must be > 0".into()));
        }
        if layer >= self.n_layers {
            return Err(Error::Inference(format!(
                "store: layer {layer} out of range [0, {})", self.n_layers
            )));
        }
        if kv_idx > 1 {
            return Err(Error::Inference("store: kv_idx must be 0 or 1".into()));
        }
        let token_elems = self.n_kv_heads * self.head_dim;
        if data_f32.len() < n_tokens * token_elems {
            return Err(Error::Inference("store: data too short".into()));
        }

        let base = self.slot_offset(layer, kv_idx);
        let hd = self.head_dim;
        let msl = self.max_seq_len;

        for t in 0..n_tokens {
            for h in 0..self.n_kv_heads {
                let src_off = t * token_elems + h * hd;
                let dst_off = base + h * msl * hd + (self.seq_len + t) * hd;
                unsafe {
                    ffi::f32_to_f16(
                        data_f32[src_off..].as_ptr(),
                        self.data[dst_off..].as_mut_ptr(),
                        hd as i32,
                    );
                }
            }
        }
        Ok(())
    }

    // ── Head pointers ──

    /// Pointer to cached K data for (layer, head).
    /// Points to `[max_seq_len * head_dim]` f16 values.
    pub fn k_head_ptr(&self, layer: usize, head: usize) -> *const u16 {
        let base = self.slot_offset(layer, 0);
        unsafe { self.data.as_ptr().add(base + head * self.max_seq_len * self.head_dim) }
    }

    /// Pointer to cached V data for (layer, head).
    /// Points to `[max_seq_len * head_dim]` f16 values.
    pub fn v_head_ptr(&self, layer: usize, head: usize) -> *const u16 {
        let base = self.slot_offset(layer, 1);
        unsafe { self.data.as_ptr().add(base + head * self.max_seq_len * self.head_dim) }
    }
}

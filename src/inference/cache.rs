//! KV cache with sliding window + shared layers.

use crate::inference::engine::AttnType;

// ---------------------------------------------------------------------------
// f32 → f16 bit conversion (inline helper, replaced by kernel in forward.rs)
// ---------------------------------------------------------------------------

#[inline]
fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    if exp == 255 {
        // Inf / NaN
        return (sign | 0x7C00 | if mantissa != 0 { 0x0200 } else { 0 }) as u16;
    }

    let new_exp = exp - 127 + 15;

    if new_exp >= 31 {
        // Overflow → Inf
        return (sign | 0x7C00) as u16;
    }

    if new_exp <= 0 {
        // Underflow → subnormal or zero
        if new_exp < -10 {
            return sign as u16;
        }
        let m = mantissa | 0x0080_0000;
        let shift = 1 - new_exp;
        let half = (m >> (shift + 13 - 1)) & 1;
        let result = m >> (shift + 13);
        return (sign | (result + half)) as u16;
    }

    let half = (mantissa >> 12) & 1;
    let result = ((new_exp as u32) << 10) | (mantissa >> 13);
    (sign | result + half) as u16
}

// ---------------------------------------------------------------------------
// KvCache
// ---------------------------------------------------------------------------

pub struct KvCache {
    k: Vec<Vec<u16>>,
    v: Vec<Vec<u16>>,
    shared_source: Vec<Option<usize>>,
    attn_types: Vec<AttnType>,
    window_size: usize,
    seq_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl KvCache {
    /// Create a new KV cache.
    ///
    /// - Shared layers (shared_source\[l\] = Some(src)) get empty vecs.
    /// - Sliding window layers allocate `n_kv_heads * window_size * head_dim`.
    /// - Global layers allocate `n_kv_heads * max_seq_len * head_dim`.
    pub fn new(
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        window_size: usize,
        max_seq_len: usize,
        attn_types: Vec<AttnType>,
        shared_source: Vec<Option<usize>>,
    ) -> Self {
        assert_eq!(attn_types.len(), n_layers);
        assert_eq!(shared_source.len(), n_layers);

        let mut k = Vec::with_capacity(n_layers);
        let mut v = Vec::with_capacity(n_layers);

        for l in 0..n_layers {
            if shared_source[l].is_some() {
                // Shared layer — no own storage
                k.push(Vec::new());
                v.push(Vec::new());
            } else {
                let cap = match attn_types[l] {
                    AttnType::SlidingWindow => n_kv_heads * window_size * head_dim,
                    AttnType::Global => n_kv_heads * max_seq_len * head_dim,
                };
                k.push(vec![0u16; cap]);
                v.push(vec![0u16; cap]);
            }
        }

        Self {
            k,
            v,
            shared_source,
            attn_types,
            window_size,
            seq_len: 0,
            n_kv_heads,
            head_dim,
        }
    }

    /// Store one token's K and V for a layer.
    ///
    /// `k_f32` and `v_f32` have length `n_kv_heads * head_dim`.
    /// Converts f32→f16 and writes into the ring buffer (sliding) or
    /// sequential position (global).
    pub fn store(&mut self, layer: usize, k_f32: &[f32], v_f32: &[f32]) {
        // Skip shared layers — they use another layer's storage
        if self.shared_source[layer].is_some() {
            return;
        }

        let stride = self.n_kv_heads * self.head_dim;
        debug_assert_eq!(k_f32.len(), stride);
        debug_assert_eq!(v_f32.len(), stride);

        let pos = match self.attn_types[layer] {
            AttnType::SlidingWindow => self.seq_len % self.window_size,
            AttnType::Global => self.seq_len,
        };

        let offset = pos * stride;

        let kb = &mut self.k[layer];
        let vb = &mut self.v[layer];

        for i in 0..stride {
            kb[offset + i] = f32_to_f16(k_f32[i]);
            vb[offset + i] = f32_to_f16(v_f32[i]);
        }
    }

    /// Pointer to K buffer for a layer, resolving shared source.
    #[inline]
    pub fn k_ptr(&self, layer: usize) -> *const u16 {
        let src = self.shared_source[layer].unwrap_or(layer);
        self.k[src].as_ptr()
    }

    /// Pointer to V buffer for a layer, resolving shared source.
    #[inline]
    pub fn v_ptr(&self, layer: usize) -> *const u16 {
        let src = self.shared_source[layer].unwrap_or(layer);
        self.v[src].as_ptr()
    }

    /// Number of KV positions available for attention at this layer.
    ///
    /// Sliding: `min(seq_len + 1, window_size)`
    /// Global:  `seq_len + 1`
    #[inline]
    pub fn attn_len(&self, layer: usize) -> usize {
        let src = self.shared_source[layer].unwrap_or(layer);
        match self.attn_types[src] {
            AttnType::SlidingWindow => (self.seq_len + 1).min(self.window_size),
            AttnType::Global => self.seq_len + 1,
        }
    }

    /// Advance sequence position by one token.
    #[inline]
    pub fn advance(&mut self) {
        self.seq_len += 1;
    }

    /// Current sequence position.
    #[inline]
    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    /// Reset cache — zero all buffers, seq_len back to 0.
    pub fn reset(&mut self) {
        self.seq_len = 0;
        for buf in self.k.iter_mut().chain(self.v.iter_mut()) {
            for b in buf.iter_mut() {
                *b = 0;
            }
        }
    }
}

//! KV cache — TurboQuant Q4 storage with fused attention kernels.
//!
//! Single flat allocation, per-group Q4 packing.
//! Kernels accessed via crate::kernels::ffi and ffi_inference.

use crate::error::{Error, Result};

/// Offsets into `data_buf` for one KV slot (K or V for one layer).
#[derive(Clone, Copy, Debug)]
struct KvSlice {
    weights_offset: usize,
    scales_offset:  usize,
    biases_offset:  usize,
}

/// Dummy handle — kernels are accessed globally via ffi/ffi_inference.
pub struct KernelTable;

impl KernelTable {
    pub fn init() -> Result<Self> {
        Ok(KernelTable)
    }
}

pub struct EakvCache {
    n_layers:         i32,
    n_kv_heads:       i32,
    head_dim:         i32,
    max_seq_len:      i32,
    seq_len:          i32,
    groups_per_token: i32,
    max_groups:       i32,
    data_buf:         Vec<u8>,
    kv:               Vec<KvSlice>,
    jl_signs:         [f32; 64],
}

unsafe impl Send for EakvCache {}

fn gen_jl_signs() -> [f32; 64] {
    let mut signs = [0.0f32; 64];
    let mut rng: u64 = 0x4F6C6F72696E4A4C;
    for s in signs.iter_mut() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        *s = if rng % 2 == 0 { 1.0 } else { -1.0 };
    }
    signs
}

impl EakvCache {
    pub fn new(
        n_layers:    i32,
        n_kv_heads:  i32,
        head_dim:    i32,
        max_seq_len: i32,
        _kt:         KernelTable,
    ) -> Result<Self> {
        if n_layers <= 0 || n_kv_heads <= 0 || head_dim <= 0 || max_seq_len <= 0 {
            return Err(Error::Inference("EakvCache: invalid dimensions".into()));
        }
        if (n_kv_heads * head_dim) % 64 != 0 {
            return Err(Error::Inference(
                "EakvCache: n_kv_heads * head_dim must be divisible by 64".into(),
            ));
        }

        let groups_per_token = (n_kv_heads * head_dim) / 64;
        let max_groups = groups_per_token * max_seq_len;

        // Per-slot: weights(32 bytes/group) + scales(4 bytes/group) + biases(4 bytes/group)
        let blk = max_groups as usize * 40;
        let n_slots = n_layers as usize * 2;
        let total = blk * n_slots;
        let data_buf = vec![0u8; total];

        let mut kv = Vec::with_capacity(n_slots);
        let mut off = 0usize;
        for _ in 0..n_layers {
            for _ in 0..2 {
                let w = off;  off += max_groups as usize * 32;
                let s = off;  off += max_groups as usize * 4;
                let b = off;  off += max_groups as usize * 4;
                kv.push(KvSlice { weights_offset: w, scales_offset: s, biases_offset: b });
            }
        }
        debug_assert_eq!(off, total);

        Ok(Self {
            n_layers, n_kv_heads, head_dim, max_seq_len,
            seq_len: 0, groups_per_token, max_groups,
            data_buf, kv, jl_signs: gen_jl_signs(),
        })
    }

    // ── Accessors ──

    pub fn seq_len(&self) -> i32           { self.seq_len }
    pub fn len(&self) -> i32               { self.seq_len }
    pub fn n_layers(&self) -> i32          { self.n_layers }
    pub fn n_kv_heads(&self) -> i32        { self.n_kv_heads }
    pub fn head_dim(&self) -> i32          { self.head_dim }
    pub fn max_seq_len(&self) -> i32       { self.max_seq_len }
    pub fn groups_per_token(&self) -> i32  { self.groups_per_token }
    pub fn jl_signs(&self) -> &[f32; 64]   { &self.jl_signs }

    pub fn groups_per_head(&self) -> i32 {
        self.max_seq_len * (self.head_dim / 64)
    }

    pub fn k_ptrs(&self, layer: i32) -> (*const u8, *const f32, *const f32) {
        let s = self.kv[layer as usize * 2];
        unsafe {(
            self.data_buf.as_ptr().add(s.weights_offset),
            self.data_buf.as_ptr().add(s.scales_offset) as *const f32,
            self.data_buf.as_ptr().add(s.biases_offset) as *const f32,
        )}
    }

    pub fn v_ptrs(&self, layer: i32) -> (*const u8, *const f32, *const f32) {
        let s = self.kv[layer as usize * 2 + 1];
        unsafe {(
            self.data_buf.as_ptr().add(s.weights_offset),
            self.data_buf.as_ptr().add(s.scales_offset) as *const f32,
            self.data_buf.as_ptr().add(s.biases_offset) as *const f32,
        )}
    }

    pub fn checkpoint(&self) -> i32 { self.seq_len }

    pub fn restore(&mut self, seq_len: i32) -> Result<()> {
        if seq_len < 0 || seq_len > self.seq_len {
            return Err(Error::Inference(format!(
                "restore: seq_len {seq_len} out of range [0, {}]", self.seq_len
            )));
        }
        self.seq_len = seq_len;
        Ok(())
    }

    pub fn advance(&mut self, n: i32) -> Result<()> {
        if n <= 0 {
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

    // ── Rotation helper ──

    fn rotate_groups(&self, buf: &mut [f32], n_groups: i32) {
        for g in 0..n_groups as usize {
            unsafe {
                crate::kernels::ffi::turbo_rotate(
                    buf.as_mut_ptr().add(g * 64),
                    self.jl_signs.as_ptr(),
                    64,
                );
            }
        }
    }

    // ── Bulk load ──

    /// Load raw f32 KV data for all layers.
    /// Layout: [layer][kv_idx][head][seq][dim].
    pub fn load_raw(&mut self, data: &[f32], seq_len: i32) -> Result<()> {
        if seq_len <= 0 || seq_len > self.max_seq_len {
            return Err(Error::Inference(
                format!("load_raw: seq_len {seq_len} out of range")
            ));
        }
        let hd = self.head_dim as usize;
        let nh = self.n_kv_heads as usize;
        let gpd = hd / 64;
        let gph = self.max_seq_len as usize * gpd;
        let n_gphead = seq_len as usize * gpd;
        let head_elems = seq_len as usize * hd;
        let elems_per_lkv = nh * head_elems;

        if data.len() < self.n_layers as usize * 2 * elems_per_lkv {
            return Err(Error::Inference("load_raw: data too short".into()));
        }

        let mut tmp = vec![0i32; n_gphead * 32];
        let mut rot_buf = vec![0.0f32; head_elems];

        for l in 0..self.n_layers as usize {
            for kv_idx in 0..2usize {
                let src_base = &data[(l * 2 + kv_idx) * elems_per_lkv..];
                let slot = self.kv[l * 2 + kv_idx];
                for h in 0..nh {
                    let src = &src_base[h * head_elems..h * head_elems + head_elems];
                    let group_base = h * gph;
                    rot_buf[..head_elems].copy_from_slice(src);
                    self.rotate_groups(&mut rot_buf[..head_elems], n_gphead as i32);
                    let scales_ptr = unsafe {
                        (self.data_buf.as_mut_ptr().add(slot.scales_offset) as *mut f32)
                            .add(group_base)
                    };
                    let biases_ptr = unsafe {
                        (self.data_buf.as_mut_ptr().add(slot.biases_offset) as *mut f32)
                            .add(group_base)
                    };
                    unsafe {
                        crate::kernels::ffi_inference::quantize_simd(
                            rot_buf.as_ptr(), tmp.as_mut_ptr(),
                            scales_ptr, biases_ptr, n_gphead as i32,
                        );
                    }
                    let w_start = slot.weights_offset + group_base * 32;
                    for i in 0..n_gphead * 32 {
                        self.data_buf[w_start + i] = tmp[i] as u8;
                    }
                }
            }
        }
        self.seq_len = seq_len;
        Ok(())
    }

    // ── Incremental append ──

    /// Append tokens to one (layer, kv_idx) slot.
    /// Input layout: [head][token][dim]. Does NOT advance seq_len.
    pub fn append(&mut self, data: &[f32], layer: i32, kv_idx: i32, n_tokens: i32) -> Result<()> {
        if n_tokens <= 0 {
            return Err(Error::Inference("append: n_tokens must be > 0".into()));
        }
        if layer < 0 || layer >= self.n_layers {
            return Err(Error::Inference(format!(
                "append: layer {layer} out of range [0, {})", self.n_layers
            )));
        }
        if kv_idx < 0 || kv_idx > 1 {
            return Err(Error::Inference("append: kv_idx must be 0 or 1".into()));
        }
        if self.seq_len + n_tokens > self.max_seq_len {
            return Err(Error::Inference(format!(
                "append: {} + {} > max {}", self.seq_len, n_tokens, self.max_seq_len
            )));
        }

        let hd = self.head_dim as usize;
        let nh = self.n_kv_heads as usize;
        let gpd = hd / 64;
        let gph = self.max_seq_len as usize * gpd;
        let n_gphead = n_tokens as usize * gpd;
        let head_elems = n_tokens as usize * hd;

        if data.len() < nh * head_elems {
            return Err(Error::Inference("append: data too short".into()));
        }

        let mut tmp = vec![0i32; n_gphead * 32];
        let mut rot_buf = vec![0.0f32; head_elems];
        let slot = self.kv[layer as usize * 2 + kv_idx as usize];

        for h in 0..nh {
            let src = &data[h * head_elems..h * head_elems + head_elems];
            let group_base = h * gph + self.seq_len as usize * gpd;
            rot_buf[..head_elems].copy_from_slice(src);
            self.rotate_groups(&mut rot_buf[..head_elems], n_gphead as i32);
            let scales_ptr = unsafe {
                (self.data_buf.as_mut_ptr().add(slot.scales_offset) as *mut f32)
                    .add(group_base)
            };
            let biases_ptr = unsafe {
                (self.data_buf.as_mut_ptr().add(slot.biases_offset) as *mut f32)
                    .add(group_base)
            };
            unsafe {
                crate::kernels::ffi_inference::quantize_simd(
                    rot_buf.as_ptr(), tmp.as_mut_ptr(),
                    scales_ptr, biases_ptr, n_gphead as i32,
                );
            }
            let w_start = slot.weights_offset + group_base * 32;
            for i in 0..n_gphead * 32 {
                self.data_buf[w_start + i] = tmp[i] as u8;
            }
        }
        Ok(())
    }

    // ── pub(crate) accessors for attention ──

    pub(crate) fn weights(&self, layer: i32, kv_idx: i32) -> &[u8] {
        let s = self.kv[layer as usize * 2 + kv_idx as usize];
        &self.data_buf[s.weights_offset..s.weights_offset + self.max_groups as usize * 32]
    }

    pub(crate) fn scales(&self, layer: i32, kv_idx: i32) -> &[f32] {
        let s = self.kv[layer as usize * 2 + kv_idx as usize];
        unsafe {
            std::slice::from_raw_parts(
                self.data_buf.as_ptr().add(s.scales_offset) as *const f32,
                self.max_groups as usize,
            )
        }
    }

    pub(crate) fn biases(&self, layer: i32, kv_idx: i32) -> &[f32] {
        let s = self.kv[layer as usize * 2 + kv_idx as usize];
        unsafe {
            std::slice::from_raw_parts(
                self.data_buf.as_ptr().add(s.biases_offset) as *const f32,
                self.max_groups as usize,
            )
        }
    }
}

// ── Attention ─────────────────────────────────────────────────────────────────

pub mod attention {
    use super::EakvCache;
    use crate::kernels::ffi_inference as ki;

    /// Compute attention scores for all query heads against cached K vectors.
    /// `queries`: `[n_q_heads * head_dim]` f32.
    /// `seq_len`: number of KV positions to attend over (including current token).
    /// `scores_out`: `[n_q_heads * seq_len]` f32.
    pub fn attention_scores(
        cache:      &EakvCache,
        queries:    &[f32],
        layer:      i32,
        n_q_heads:  i32,
        n_kv_heads: i32,
        seq_len:    i32,
        scores_out: &mut [f32],
    ) {
        let hd = cache.head_dim();
        let q_elems = (n_q_heads * hd) as usize;
        let n_q_groups = q_elems / 64;
        let groups_per_head = cache.max_seq_len() * (hd / 64);

        let mut rot_q = queries[..q_elems].to_vec();
        let signs = cache.jl_signs();
        for g in 0..n_q_groups {
            unsafe {
                crate::kernels::ffi::turbo_rotate(
                    rot_q.as_mut_ptr().add(g * 64),
                    signs.as_ptr(), 64,
                );
            }
        }

        let k_w = cache.weights(layer, 0);
        let k_s = cache.scales(layer, 0);
        let k_b = cache.biases(layer, 0);

        unsafe {
            if hd == 64 {
                if n_q_heads == n_kv_heads {
                    ki::fused_k_score_64(
                        rot_q.as_ptr(), k_w.as_ptr(), k_s.as_ptr(), k_b.as_ptr(),
                        scores_out.as_mut_ptr(), seq_len, n_q_heads, groups_per_head,
                    );
                } else {
                    ki::fused_k_score_gqa_64(
                        rot_q.as_ptr(), k_w.as_ptr(), k_s.as_ptr(), k_b.as_ptr(),
                        scores_out.as_mut_ptr(), seq_len,
                        n_q_heads, n_kv_heads, groups_per_head,
                    );
                }
            } else if n_q_heads == n_kv_heads {
                ki::fused_k_score(
                    rot_q.as_ptr(), k_w.as_ptr(), k_s.as_ptr(), k_b.as_ptr(),
                    scores_out.as_mut_ptr(), seq_len, n_q_heads, groups_per_head,
                );
            } else {
                ki::fused_k_score_gqa(
                    rot_q.as_ptr(), k_w.as_ptr(), k_s.as_ptr(), k_b.as_ptr(),
                    scores_out.as_mut_ptr(), seq_len,
                    n_q_heads, n_kv_heads, groups_per_head,
                );
            }
        }

    }

    /// Compute attention output by summing V vectors weighted by softmax scores.
    /// `weights_in`: `[n_q_heads * seq_len]` f32 (softmax'd).
    /// `seq_len`: number of KV positions (must match scores dimension).
    /// `output_out`: `[n_q_heads * head_dim]` f32.
    pub fn attention_output(
        cache:      &EakvCache,
        weights_in: &[f32],
        layer:      i32,
        n_q_heads:  i32,
        n_kv_heads: i32,
        seq_len:    i32,
        output_out: &mut [f32],
    ) {
        let hd = cache.head_dim();
        let groups_per_head = cache.max_seq_len() * (hd / 64);
        let v_w = cache.weights(layer, 1);
        let v_s = cache.scales(layer, 1);
        let v_b = cache.biases(layer, 1);

        unsafe {
            if hd == 64 {
                if n_q_heads == n_kv_heads {
                    ki::fused_v_sum_64(
                        weights_in.as_ptr(), v_w.as_ptr(), v_s.as_ptr(), v_b.as_ptr(),
                        output_out.as_mut_ptr(), seq_len, n_q_heads, groups_per_head,
                    );
                } else {
                    ki::fused_v_sum_gqa_64(
                        weights_in.as_ptr(), v_w.as_ptr(), v_s.as_ptr(), v_b.as_ptr(),
                        output_out.as_mut_ptr(), seq_len,
                        n_q_heads, n_kv_heads, groups_per_head,
                    );
                }
            } else if n_q_heads == n_kv_heads {
                ki::fused_v_sum(
                    weights_in.as_ptr(), v_w.as_ptr(), v_s.as_ptr(), v_b.as_ptr(),
                    output_out.as_mut_ptr(), seq_len, n_q_heads, groups_per_head,
                );
            } else {
                ki::fused_v_sum_gqa(
                    weights_in.as_ptr(), v_w.as_ptr(), v_s.as_ptr(), v_b.as_ptr(),
                    output_out.as_mut_ptr(), seq_len,
                    n_q_heads, n_kv_heads, groups_per_head,
                );
            }
        }

        // Inverse-rotate output to undo V pre-rotation.
        let out_elems = (n_q_heads * hd) as usize;
        let n_out_groups = out_elems / 64;
        let signs = cache.jl_signs();
        for g in 0..n_out_groups {
            unsafe {
                let ptr = output_out.as_mut_ptr().add(g * 64);
                crate::kernels::ffi::fwht_inplace(ptr, 64);
                crate::kernels::ffi::sign_flip(ptr, signs.as_ptr(), 64);
            }
        }
    }
}

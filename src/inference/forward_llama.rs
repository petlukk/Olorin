//! Transformer forward pass for Llama models with Q4_K_M quantization.

use crate::kernels::ffi_inference as ffi;
use crate::inference::forward::{apply_rope, build_rope_freqs, sample_into};
use crate::inference::math::{wipe_f32, wipe_i8, wipe_i16};
use crate::inference::matmul::embed_f16_lookup;
use crate::inference::matmul_q4k::{Q4K_BLOCK_BYTES, q4k_matmul_mt, q4k_matmul_work, q4k_matmul_residual_work, q4k_fused_gate_up_silu_work};
use crate::inference::matmul_q6k::{Q6K_BLOCK_BYTES, q6k_matmul_mt, q6k_matmul_work, q6k_matmul_residual_work};
use crate::inference::matmul_q4k::q4k_embed_lookup;
use crate::inference::matmul_q6k::q6k_embed_lookup;
use crate::inference::engine::BitNetModel;
use crate::inference::cache::F16KvCache;
use crate::inference::ptr::{SendPtr, SendMutPtr};
use crate::inference::threadpool::ThreadPool;

/// Return the index of the largest element in `logits`.
pub fn argmax(logits: &[f32]) -> u32 {
    logits.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Add bias vector to output buffer. No-op if bias is null (Llama models).
#[inline]
pub(crate) fn add_bias(buf: &mut [f32], bias: *const f32, n: usize) {
    if bias.is_null() { return; }
    for i in 0..n { buf[i] += unsafe { *bias.add(i) }; }
}

pub struct LlamaState {
    pub(crate) pool: ThreadPool,
    pub(crate) x: Vec<f32>,
    x_norm: Vec<f32>,
    x_q8_qs: Vec<i8>,
    x_q8_d: Vec<f32>,
    x_q8_bsums: Vec<i16>,
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) attn_out: Vec<f32>,
    attn_q8_qs: Vec<i8>,
    attn_q8_d: Vec<f32>,
    attn_q8_bsums: Vec<i16>,
    pub(crate) hidden: Vec<f32>,
    hidden_q8_qs: Vec<i8>,
    hidden_q8_d: Vec<f32>,
    hidden_q8_bsums: Vec<i16>,
    pub(crate) logits: Vec<f32>,
    pub(crate) tmp: Vec<f32>,
    pub(crate) kv_cache: F16KvCache,
    pub(crate) attn_scores: Vec<f32>,
    pub(crate) rope_freqs: Vec<f32>,
    pub(crate) sample_logits_buf: Vec<f32>,
    pub(crate) sample_probs: Vec<f32>,
    pub(crate) sample_indices: Vec<usize>,
    #[allow(dead_code)]
    pub(crate) max_seq_len: usize,
}

pub(crate) fn q8k_blocks(dim: usize) -> usize { dim / 256 }

pub(crate) fn embed_token(model: &BitNetModel, token: u32, out: &mut [f32]) {
    match model.embed_dtype {
        12 => q4k_embed_lookup(model.embed_weight_f16, token, out, model.hidden_dim),
        14 => q6k_embed_lookup(model.embed_weight_f16, token, out, model.hidden_dim),
        _ => embed_f16_lookup(model.embed_weight_f16, token, out, model.hidden_dim),
    }
}

impl LlamaState {
    pub fn new(model: &BitNetModel, max_seq_len: usize) -> Self {
        let h = model.hidden_dim;
        let f = model.ffn_dim;
        let v = model.vocab_size;
        let hb = q8k_blocks(h);
        let fb = q8k_blocks(f);
        let nh = model.n_heads;
        let kv_cache = F16KvCache::new(
            model.n_layers, model.n_kv_heads,
            model.head_dim, max_seq_len,
        ).expect("failed to create F16KvCache");
        LlamaState {
            pool: ThreadPool::new(),
            x: vec![0.0; h],
            x_norm: vec![0.0; h],
            x_q8_qs: vec![0; h + 16],
            x_q8_d: vec![0.0; hb],
            x_q8_bsums: vec![0; hb * 16],
            q: vec![0.0; h],
            k: vec![0.0; model.kv_dim],
            v: vec![0.0; model.kv_dim],
            attn_out: vec![0.0; h],
            attn_q8_qs: vec![0; h + 16],
            attn_q8_d: vec![0.0; hb],
            attn_q8_bsums: vec![0; hb * 16],
            hidden: vec![0.0; f],
            hidden_q8_qs: vec![0; f + 16],
            hidden_q8_d: vec![0.0; fb],
            hidden_q8_bsums: vec![0; fb * 16],
            logits: vec![0.0; v],
            tmp: vec![0.0; h.max(f)],
            kv_cache,
            attn_scores: vec![0.0; nh * max_seq_len],
            rope_freqs: vec![0.0; model.head_dim],
            sample_logits_buf: vec![0.0; v],
            sample_probs: vec![0.0; v],
            sample_indices: vec![0; v],
            max_seq_len,
        }
    }

    /// Process one token through one layer.
    pub(crate) fn process_layer(&mut self, model: &BitNetModel, layer: usize, x: &mut [f32], pos: usize) {
        let h = model.hidden_dim;
        let hd = model.head_dim;
        let nh = model.n_heads;
        let nkv = model.n_kv_heads;
        let kv = model.kv_dim;
        let f = model.ffn_dim;
        let h_nb = q8k_blocks(h);
        let f_nb = q8k_blocks(f);
        let h_row_stride = h_nb * Q4K_BLOCK_BYTES;
        let lw = &model.q4k_layers[layer];

        unsafe {
            ffi::rmsnorm_f32(x.as_ptr(), lw.attn_norm, self.x_norm.as_mut_ptr(), h as i32, model.rms_eps);
            ffi::quant_f32_q8k(self.x_norm.as_ptr(), self.x_q8_qs.as_mut_ptr(),
                self.x_q8_d.as_mut_ptr(), self.x_q8_bsums.as_mut_ptr(), h as i32);
        }


        // QKV — concurrent dispatch via ThreadPool
        let total = self.pool.thread_count();
        let wv_bb = lw.wv_block_bytes;
        if total >= 3 {
            let q_t = (total / 2).max(1);
            let rem = total - q_t;
            let k_t = rem / 2;
            let v_t = rem - k_t;
            let q8p = SendPtr(self.x_q8_qs.as_ptr());
            let q8d = SendPtr(self.x_q8_d.as_ptr());
            let q8b = SendPtr(self.x_q8_bsums.as_ptr());
            let q_out = SendMutPtr(self.q.as_mut_ptr());
            let k_out = SendMutPtr(self.k.as_mut_ptr());
            let v_out = SendMutPtr(self.v.as_mut_ptr());
            let (wq, wk, wv) = (SendPtr(lw.wq), SendPtr(lw.wk), SendPtr(lw.wv));
            let h_q6k_stride = h_nb * Q6K_BLOCK_BYTES;
            self.pool.run_split3(
                q_t, move |tid, _n| unsafe { q4k_matmul_work(wq.ptr(), h_row_stride, h_nb, q8p.ptr(), q8d.ptr(), q8b.ptr(), q_out.ptr(), h, tid, q_t); },
                k_t, move |tid, _n| unsafe { q4k_matmul_work(wk.ptr(), h_row_stride, h_nb, q8p.ptr(), q8d.ptr(), q8b.ptr(), k_out.ptr(), kv, tid, k_t); },
                v_t, move |tid, _n| unsafe {
                    if wv_bb == Q6K_BLOCK_BYTES { q6k_matmul_work(wv.ptr(), h_q6k_stride, h_nb, q8p.ptr(), q8d.ptr(), q8b.ptr(), v_out.ptr(), kv, tid, v_t); }
                    else { q4k_matmul_work(wv.ptr(), h_row_stride, h_nb, q8p.ptr(), q8d.ptr(), q8b.ptr(), v_out.ptr(), kv, tid, v_t); }
                },
            );
        } else {
            q4k_matmul_mt(lw.wq, h_row_stride, h_nb, self.x_q8_qs.as_ptr(), self.x_q8_d.as_ptr(), self.x_q8_bsums.as_ptr(), &mut self.q, h, &self.pool);
            q4k_matmul_mt(lw.wk, h_row_stride, h_nb, self.x_q8_qs.as_ptr(), self.x_q8_d.as_ptr(), self.x_q8_bsums.as_ptr(), &mut self.k, kv, &self.pool);
            if wv_bb == Q6K_BLOCK_BYTES {
                q6k_matmul_mt(lw.wv, h_nb * Q6K_BLOCK_BYTES, h_nb, self.x_q8_qs.as_ptr(), self.x_q8_d.as_ptr(), self.x_q8_bsums.as_ptr(), &mut self.v, kv, &self.pool);
            } else {
                q4k_matmul_mt(lw.wv, h_row_stride, h_nb, self.x_q8_qs.as_ptr(), self.x_q8_d.as_ptr(), self.x_q8_bsums.as_ptr(), &mut self.v, kv, &self.pool);
            }
        }


        // Bias — all applied BEFORE cache (llama.cpp style)
        add_bias(&mut self.q, lw.q_bias, h);
        add_bias(&mut self.k, lw.k_bias, kv);
        add_bias(&mut self.v, lw.v_bias, kv);

        build_rope_freqs(&mut self.rope_freqs, hd, pos, model.rope_theta);
        apply_rope(&mut self.q, &self.rope_freqs, hd, nh);
        apply_rope(&mut self.k, &self.rope_freqs, hd, nkv);

        // KV store (f32 -> f16)
        self.kv_cache.store(layer, 0, &self.k[..kv], 1).unwrap();
        self.kv_cache.store(layer, 1, &self.v[..kv], 1).unwrap();
        if layer == model.n_layers - 1 { self.kv_cache.advance(1).unwrap(); }

        // Attention: per-head Q*K -> softmax -> V*weights (f16 cache)
        let seq_len = pos + 1;
        let q_per_kv = nh / nkv;
        let rsqrt_hd = 1.0 / (hd as f32).sqrt();
        for kv_h in 0..nkv {
            let k_ptr = self.kv_cache.k_head_ptr(layer, kv_h);
            let v_ptr = self.kv_cache.v_head_ptr(layer, kv_h);
            for q_off in 0..q_per_kv {
                let q_h = kv_h * q_per_kv + q_off;
                let scores = &mut self.attn_scores[q_h * self.max_seq_len..q_h * self.max_seq_len + seq_len];
                unsafe {
                    ffi::attn_dot_f16(
                        self.q[q_h * hd..].as_ptr(), k_ptr,
                        scores.as_mut_ptr(), seq_len as i32, hd as i32,
                    );
                    ffi::softmax_f32(scores.as_mut_ptr(), seq_len as i32, rsqrt_hd);
                    ffi::attn_vsum_f16(
                        scores.as_ptr(), v_ptr,
                        self.attn_out[q_h * hd..].as_mut_ptr(),
                        seq_len as i32, hd as i32,
                    );
                }
            }
        }

        unsafe {
            ffi::quant_f32_q8k(self.attn_out.as_ptr(), self.attn_q8_qs.as_mut_ptr(),
                self.attn_q8_d.as_mut_ptr(), self.attn_q8_bsums.as_mut_ptr(), h as i32);
        }
        // Wo projection + residual (fused per-thread: x[r] += matmul[r])
        {
            let n_thr = self.pool.thread_count().min(h / 4).max(1);
            let w = SendPtr(lw.wo);
            let qs = SendPtr(self.attn_q8_qs.as_ptr());
            let qd = SendPtr(self.attn_q8_d.as_ptr());
            let qb = SendPtr(self.attn_q8_bsums.as_ptr());
            let o = SendMutPtr(x.as_mut_ptr());
            self.pool.run(n_thr, move |tid, _n| unsafe {
                q4k_matmul_residual_work(
                    w.ptr(), h_row_stride, h_nb,
                    qs.ptr(), qd.ptr(), qb.ptr(),
                    o.ptr(), h, tid, n_thr,
                );
            });
        }


        unsafe {
            ffi::rmsnorm_f32(x.as_ptr(), lw.ffn_norm, self.x_norm.as_mut_ptr(), h as i32, model.rms_eps);
            ffi::quant_f32_q8k(self.x_norm.as_ptr(), self.x_q8_qs.as_mut_ptr(),
                self.x_q8_d.as_mut_ptr(), self.x_q8_bsums.as_mut_ptr(), h as i32);
        }

        // Fused gate+up+SiLU
        {
            let q8p = SendPtr(self.x_q8_qs.as_ptr());
            let q8d = SendPtr(self.x_q8_d.as_ptr());
            let q8b = SendPtr(self.x_q8_bsums.as_ptr());
            let h_out = SendMutPtr(self.hidden.as_mut_ptr());
            let (wg, wu) = (SendPtr(lw.w_gate), SendPtr(lw.w_up));
            self.pool.run(total, move |tid, _n| unsafe {
                q4k_fused_gate_up_silu_work(wg.ptr(), wu.ptr(), h_row_stride, h_nb,
                    q8p.ptr(), q8d.ptr(), q8b.ptr(), h_out.ptr(), f, tid, total);
            });
        }

        unsafe {
            ffi::quant_f32_q8k(self.hidden.as_ptr(), self.hidden_q8_qs.as_mut_ptr(),
                self.hidden_q8_d.as_mut_ptr(), self.hidden_q8_bsums.as_mut_ptr(), f as i32);
        }
        // Down projection + residual (fused per-thread: x[r] += matmul[r])
        {
            let n_thr = self.pool.thread_count().min(h / 4).max(1);
            let w = SendPtr(lw.w_down);
            let qs = SendPtr(self.hidden_q8_qs.as_ptr());
            let qd = SendPtr(self.hidden_q8_d.as_ptr());
            let qb = SendPtr(self.hidden_q8_bsums.as_ptr());
            let o = SendMutPtr(x.as_mut_ptr());
            let down_q6k = lw.w_down_block_bytes == Q6K_BLOCK_BYTES;
            let f_row_stride_q6 = f_nb * Q6K_BLOCK_BYTES;
            let f_row_stride_q4 = f_nb * Q4K_BLOCK_BYTES;
            self.pool.run(n_thr, move |tid, _n| unsafe {
                if down_q6k {
                    q6k_matmul_residual_work(
                        w.ptr(), f_row_stride_q6, f_nb,
                        qs.ptr(), qd.ptr(), qb.ptr(),
                        o.ptr(), h, tid, n_thr,
                    );
                } else {
                    q4k_matmul_residual_work(
                        w.ptr(), f_row_stride_q4, f_nb,
                        qs.ptr(), qd.ptr(), qb.ptr(),
                        o.ptr(), h, tid, n_thr,
                    );
                }
            });
        }
    }

    /// Output projection: RMSNorm + quantize + matmul -> logits.
    pub(crate) fn output_proj(&mut self, model: &BitNetModel) {
        let h = model.hidden_dim;
        let h_nb = q8k_blocks(h);
        let h_row_stride = h_nb * Q4K_BLOCK_BYTES;
        unsafe {
            ffi::rmsnorm_f32(self.x.as_ptr(), model.norm_weight, self.x_norm.as_mut_ptr(), h as i32, model.rms_eps);
            ffi::quant_f32_q8k(self.x_norm.as_ptr(), self.x_q8_qs.as_mut_ptr(),
                self.x_q8_d.as_mut_ptr(), self.x_q8_bsums.as_mut_ptr(), h as i32);
        }
        if model.output_block_bytes == Q6K_BLOCK_BYTES {
            q6k_matmul_mt(model.output_weight, h_nb * Q6K_BLOCK_BYTES, h_nb, self.x_q8_qs.as_ptr(),
                self.x_q8_d.as_ptr(), self.x_q8_bsums.as_ptr(), &mut self.logits, model.vocab_size, &self.pool);
        } else {
            q4k_matmul_mt(model.output_weight, h_row_stride, h_nb, self.x_q8_qs.as_ptr(),
                self.x_q8_d.as_ptr(), self.x_q8_bsums.as_ptr(), &mut self.logits, model.vocab_size, &self.pool);
        }
    }

    /// Single-token forward pass.
    pub fn forward(&mut self, model: &BitNetModel, token: u32, pos: usize) {
        embed_token(model, token, &mut self.x);
        let mut x = std::mem::take(&mut self.x);
        for layer in 0..model.n_layers {
            self.process_layer(model, layer, &mut x, pos);
        }
        self.x = x;
        self.output_proj(model);
    }

    /// Single-token forward pass with per-step profiling.
    pub fn forward_profiled(&mut self, model: &BitNetModel, token: u32, pos: usize) {
        use std::time::Instant;
        let mut t_rmsnorm1 = 0u64;
        let mut t_quant1 = 0u64;
        let mut t_qkv = 0u64;
        let mut t_bias_rope = 0u64;
        let mut t_kv_store = 0u64;
        let mut t_attn = 0u64;
        let mut t_quant_wo = 0u64;
        let mut t_wo = 0u64;
        let mut t_rmsnorm2 = 0u64;
        let mut t_quant2 = 0u64;
        let mut t_ffn = 0u64;
        let mut t_quant_down = 0u64;
        let mut t_down = 0u64;

        embed_token(model, token, &mut self.x);
        let mut x = std::mem::take(&mut self.x);
        let h = model.hidden_dim;
        let hd = model.head_dim;
        let nh = model.n_heads;
        let nkv = model.n_kv_heads;
        let kv = model.kv_dim;
        let f = model.ffn_dim;
        let h_nb = q8k_blocks(h);
        let f_nb = q8k_blocks(f);
        let h_row_stride = h_nb * Q4K_BLOCK_BYTES;
        let total = self.pool.thread_count();

        for layer in 0..model.n_layers {
            let lw = &model.q4k_layers[layer];
            let wv_bb = lw.wv_block_bytes;

            let t0 = Instant::now();
            unsafe {
                ffi::rmsnorm_f32(x.as_ptr(), lw.attn_norm, self.x_norm.as_mut_ptr(), h as i32, model.rms_eps);
            }
            t_rmsnorm1 += t0.elapsed().as_micros() as u64;

            let t0 = Instant::now();
            unsafe {
                ffi::quant_f32_q8k(self.x_norm.as_ptr(), self.x_q8_qs.as_mut_ptr(),
                    self.x_q8_d.as_mut_ptr(), self.x_q8_bsums.as_mut_ptr(), h as i32);
            }
            t_quant1 += t0.elapsed().as_micros() as u64;

            let t0 = Instant::now();
            if total >= 3 {
                let q_t = (total / 2).max(1);
                let rem = total - q_t;
                let k_t = rem / 2;
                let v_t = rem - k_t;
                let q8p = SendPtr(self.x_q8_qs.as_ptr());
                let q8d = SendPtr(self.x_q8_d.as_ptr());
                let q8b = SendPtr(self.x_q8_bsums.as_ptr());
                let q_out = SendMutPtr(self.q.as_mut_ptr());
                let k_out = SendMutPtr(self.k.as_mut_ptr());
                let v_out = SendMutPtr(self.v.as_mut_ptr());
                let (wq, wk, wv) = (SendPtr(lw.wq), SendPtr(lw.wk), SendPtr(lw.wv));
                let h_q6k_stride = h_nb * Q6K_BLOCK_BYTES;
                self.pool.run_split3(
                    q_t, move |tid, _n| unsafe { q4k_matmul_work(wq.ptr(), h_row_stride, h_nb, q8p.ptr(), q8d.ptr(), q8b.ptr(), q_out.ptr(), h, tid, q_t); },
                    k_t, move |tid, _n| unsafe { q4k_matmul_work(wk.ptr(), h_row_stride, h_nb, q8p.ptr(), q8d.ptr(), q8b.ptr(), k_out.ptr(), kv, tid, k_t); },
                    v_t, move |tid, _n| unsafe {
                        if wv_bb == Q6K_BLOCK_BYTES { q6k_matmul_work(wv.ptr(), h_q6k_stride, h_nb, q8p.ptr(), q8d.ptr(), q8b.ptr(), v_out.ptr(), kv, tid, v_t); }
                        else { q4k_matmul_work(wv.ptr(), h_row_stride, h_nb, q8p.ptr(), q8d.ptr(), q8b.ptr(), v_out.ptr(), kv, tid, v_t); }
                    },
                );
            } else {
                q4k_matmul_mt(lw.wq, h_row_stride, h_nb, self.x_q8_qs.as_ptr(), self.x_q8_d.as_ptr(), self.x_q8_bsums.as_ptr(), &mut self.q, h, &self.pool);
                q4k_matmul_mt(lw.wk, h_row_stride, h_nb, self.x_q8_qs.as_ptr(), self.x_q8_d.as_ptr(), self.x_q8_bsums.as_ptr(), &mut self.k, kv, &self.pool);
                if wv_bb == Q6K_BLOCK_BYTES {
                    q6k_matmul_mt(lw.wv, h_nb * Q6K_BLOCK_BYTES, h_nb, self.x_q8_qs.as_ptr(), self.x_q8_d.as_ptr(), self.x_q8_bsums.as_ptr(), &mut self.v, kv, &self.pool);
                } else {
                    q4k_matmul_mt(lw.wv, h_row_stride, h_nb, self.x_q8_qs.as_ptr(), self.x_q8_d.as_ptr(), self.x_q8_bsums.as_ptr(), &mut self.v, kv, &self.pool);
                }
            }
            t_qkv += t0.elapsed().as_micros() as u64;

            let t0 = Instant::now();
            add_bias(&mut self.q, lw.q_bias, h);
            add_bias(&mut self.k, lw.k_bias, kv);
            add_bias(&mut self.v, lw.v_bias, kv);
            build_rope_freqs(&mut self.rope_freqs, hd, pos, model.rope_theta);
            apply_rope(&mut self.q, &self.rope_freqs, hd, nh);
            apply_rope(&mut self.k, &self.rope_freqs, hd, nkv);
            t_bias_rope += t0.elapsed().as_micros() as u64;

            let t0 = Instant::now();
            self.kv_cache.store(layer, 0, &self.k[..kv], 1).unwrap();
            self.kv_cache.store(layer, 1, &self.v[..kv], 1).unwrap();
            if layer == model.n_layers - 1 { self.kv_cache.advance(1).unwrap(); }
            t_kv_store += t0.elapsed().as_micros() as u64;

            let t0 = Instant::now();
            let seq_len = pos + 1;
            let q_per_kv = nh / nkv;
            let rsqrt_hd = 1.0 / (hd as f32).sqrt();
            for kv_h in 0..nkv {
                let k_ptr = self.kv_cache.k_head_ptr(layer, kv_h);
                let v_ptr = self.kv_cache.v_head_ptr(layer, kv_h);
                for q_off in 0..q_per_kv {
                    let q_h = kv_h * q_per_kv + q_off;
                    let scores = &mut self.attn_scores[q_h * self.max_seq_len..q_h * self.max_seq_len + seq_len];
                    unsafe {
                        ffi::attn_dot_f16(self.q[q_h * hd..].as_ptr(), k_ptr, scores.as_mut_ptr(), seq_len as i32, hd as i32);
                        ffi::softmax_f32(scores.as_mut_ptr(), seq_len as i32, rsqrt_hd);
                        ffi::attn_vsum_f16(scores.as_ptr(), v_ptr, self.attn_out[q_h * hd..].as_mut_ptr(), seq_len as i32, hd as i32);
                    }
                }
            }
            t_attn += t0.elapsed().as_micros() as u64;

            let t0 = Instant::now();
            unsafe {
                ffi::quant_f32_q8k(self.attn_out.as_ptr(), self.attn_q8_qs.as_mut_ptr(),
                    self.attn_q8_d.as_mut_ptr(), self.attn_q8_bsums.as_mut_ptr(), h as i32);
            }
            t_quant_wo += t0.elapsed().as_micros() as u64;

            let t0 = Instant::now();
            {
                let n_thr = self.pool.thread_count().min(h / 4).max(1);
                let w = SendPtr(lw.wo);
                let qs = SendPtr(self.attn_q8_qs.as_ptr());
                let qd = SendPtr(self.attn_q8_d.as_ptr());
                let qb = SendPtr(self.attn_q8_bsums.as_ptr());
                let o = SendMutPtr(x.as_mut_ptr());
                self.pool.run(n_thr, move |tid, _n| unsafe {
                    q4k_matmul_residual_work(w.ptr(), h_row_stride, h_nb, qs.ptr(), qd.ptr(), qb.ptr(), o.ptr(), h, tid, n_thr);
                });
            }
            t_wo += t0.elapsed().as_micros() as u64;

            let t0 = Instant::now();
            unsafe {
                ffi::rmsnorm_f32(x.as_ptr(), lw.ffn_norm, self.x_norm.as_mut_ptr(), h as i32, model.rms_eps);
            }
            t_rmsnorm2 += t0.elapsed().as_micros() as u64;

            let t0 = Instant::now();
            unsafe {
                ffi::quant_f32_q8k(self.x_norm.as_ptr(), self.x_q8_qs.as_mut_ptr(),
                    self.x_q8_d.as_mut_ptr(), self.x_q8_bsums.as_mut_ptr(), h as i32);
            }
            t_quant2 += t0.elapsed().as_micros() as u64;

            let t0 = Instant::now();
            {
                let q8p = SendPtr(self.x_q8_qs.as_ptr());
                let q8d = SendPtr(self.x_q8_d.as_ptr());
                let q8b = SendPtr(self.x_q8_bsums.as_ptr());
                let h_out = SendMutPtr(self.hidden.as_mut_ptr());
                let (wg, wu) = (SendPtr(lw.w_gate), SendPtr(lw.w_up));
                self.pool.run(total, move |tid, _n| unsafe {
                    q4k_fused_gate_up_silu_work(wg.ptr(), wu.ptr(), h_row_stride, h_nb,
                        q8p.ptr(), q8d.ptr(), q8b.ptr(), h_out.ptr(), f, tid, total);
                });
            }
            t_ffn += t0.elapsed().as_micros() as u64;

            let t0 = Instant::now();
            unsafe {
                ffi::quant_f32_q8k(self.hidden.as_ptr(), self.hidden_q8_qs.as_mut_ptr(),
                    self.hidden_q8_d.as_mut_ptr(), self.hidden_q8_bsums.as_mut_ptr(), f as i32);
            }
            t_quant_down += t0.elapsed().as_micros() as u64;

            let t0 = Instant::now();
            {
                let n_thr = self.pool.thread_count().min(h / 4).max(1);
                let w = SendPtr(lw.w_down);
                let qs = SendPtr(self.hidden_q8_qs.as_ptr());
                let qd = SendPtr(self.hidden_q8_d.as_ptr());
                let qb = SendPtr(self.hidden_q8_bsums.as_ptr());
                let o = SendMutPtr(x.as_mut_ptr());
                let down_q6k = lw.w_down_block_bytes == Q6K_BLOCK_BYTES;
                let f_row_stride_q6 = f_nb * Q6K_BLOCK_BYTES;
                let f_row_stride_q4 = f_nb * Q4K_BLOCK_BYTES;
                self.pool.run(n_thr, move |tid, _n| unsafe {
                    if down_q6k {
                        q6k_matmul_residual_work(w.ptr(), f_row_stride_q6, f_nb, qs.ptr(), qd.ptr(), qb.ptr(), o.ptr(), h, tid, n_thr);
                    } else {
                        q4k_matmul_residual_work(w.ptr(), f_row_stride_q4, f_nb, qs.ptr(), qd.ptr(), qb.ptr(), o.ptr(), h, tid, n_thr);
                    }
                });
            }
            t_down += t0.elapsed().as_micros() as u64;
        }
        self.x = x;
        self.output_proj(model);

        let total_us = t_rmsnorm1 + t_quant1 + t_qkv + t_bias_rope + t_kv_store + t_attn
            + t_quant_wo + t_wo + t_rmsnorm2 + t_quant2 + t_ffn + t_quant_down + t_down;
        let ms = |us: u64| us as f64 / 1000.0;
        let pct = |us: u64| if total_us > 0 { us as f64 / total_us as f64 * 100.0 } else { 0.0 };
        let nl = model.n_layers;
        eprintln!("\n--- decode profile (pos={pos}, {nl} layers, {:.1}ms) ---", ms(total_us));
        eprintln!("┌─────────────────────┬──────────┬───────┐");
        eprintln!("│ Step                │   ms     │   %   │");
        eprintln!("├─────────────────────┼──────────┼───────┤");
        eprintln!("│ rmsnorm (attn)      │ {:7.1} │ {:4.1}% │", ms(t_rmsnorm1), pct(t_rmsnorm1));
        eprintln!("│ quant_q8k (attn)    │ {:7.1} │ {:4.1}% │", ms(t_quant1), pct(t_quant1));
        eprintln!("│ QKV matmul          │ {:7.1} │ {:4.1}% │", ms(t_qkv), pct(t_qkv));
        eprintln!("│ bias+rope           │ {:7.1} │ {:4.1}% │", ms(t_bias_rope), pct(t_bias_rope));
        eprintln!("│ kv_store (f16)      │ {:7.1} │ {:4.1}% │", ms(t_kv_store), pct(t_kv_store));
        eprintln!("│ attention (f16)     │ {:7.1} │ {:4.1}% │", ms(t_attn), pct(t_attn));
        eprintln!("│ quant_q8k (Wo)      │ {:7.1} │ {:4.1}% │", ms(t_quant_wo), pct(t_quant_wo));
        eprintln!("│ Wo matmul+resid     │ {:7.1} │ {:4.1}% │", ms(t_wo), pct(t_wo));
        eprintln!("│ rmsnorm (FFN)       │ {:7.1} │ {:4.1}% │", ms(t_rmsnorm2), pct(t_rmsnorm2));
        eprintln!("│ quant_q8k (FFN)     │ {:7.1} │ {:4.1}% │", ms(t_quant2), pct(t_quant2));
        eprintln!("│ FFN gate+up+SiLU    │ {:7.1} │ {:4.1}% │", ms(t_ffn), pct(t_ffn));
        eprintln!("│ quant_q8k (down)    │ {:7.1} │ {:4.1}% │", ms(t_quant_down), pct(t_quant_down));
        eprintln!("│ down matmul+resid   │ {:7.1} │ {:4.1}% │", ms(t_down), pct(t_down));
        eprintln!("├─────────────────────┼──────────┼───────┤");
        eprintln!("│ TOTAL               │ {:7.1} │ 100%  │", ms(total_us));
        eprintln!("└─────────────────────┴──────────┴───────┘");
    }

    pub fn apply_repetition_penalty(&mut self, generated: &[u32], penalty: f32) {
        if penalty == 1.0 { return; }
        for &tok in generated {
            let idx = tok as usize;
            if idx < self.logits.len() {
                if self.logits[idx] > 0.0 { self.logits[idx] /= penalty; }
                else { self.logits[idx] *= penalty; }
            }
        }
    }

    pub fn logits(&self) -> &[f32] {
        &self.logits
    }

    pub fn sample_logits(&mut self, temperature: f32, top_k: usize, top_p: f32, min_p: f32) -> u32 {
        sample_into(
            &self.logits,
            &mut self.sample_logits_buf,
            &mut self.sample_probs,
            &mut self.sample_indices,
            temperature, top_k, top_p, min_p,
        )
    }
}

pub fn generate(
    model: &BitNetModel,
    prompt_tokens: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    repetition_penalty: f32,
    stop_ids: &[u32],
    max_seq_len: usize,
    mut on_token: impl FnMut(u32),
) -> (Vec<u32>, f64, f64) {
    use std::time::Instant;
    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let mut state = LlamaState::new(model, max_seq_len);
    let mut output = Vec::with_capacity(prompt_tokens.len() + max_tokens);

    let prefill_start = Instant::now();
    state.prefill(model, prompt_tokens);
    output.extend_from_slice(prompt_tokens);
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    let first_tok_start = Instant::now();
    let mut pos = prompt_tokens.len();
    let mut n_gen = 0u32;
    let mut first_tok_ms = 0.0;

    let decode_start = Instant::now();
    for step in 0..max_tokens {
        if pos >= max_seq_len { break; }
        state.apply_repetition_penalty(&output, repetition_penalty);
        let next = state.sample_logits(temperature, top_k, top_p, min_p);
        if stop_ids.contains(&next) { break; }
        output.push(next);
        on_token(next);
        if step == 0 { first_tok_ms = first_tok_start.elapsed().as_secs_f64() * 1000.0; }
        if step == 4 { state.forward_profiled(model, next, pos); }
        else { state.forward(model, next, pos); }
        pos += 1;
        n_gen += 1;
    }
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

    let ptps = prompt_tokens.len() as f64 / (prefill_ms / 1000.0);
    let dtps = if n_gen > 0 { n_gen as f64 / (decode_ms / 1000.0) } else { 0.0 };
    let avg = if n_gen > 0 { decode_ms / n_gen as f64 } else { 0.0 };
    eprintln!("\n--- perf ({n_threads} threads) ---");
    eprintln!("prefill:    {} tokens in {prefill_ms:.0}ms ({ptps:.1} tok/s)", prompt_tokens.len());
    eprintln!("first tok:  {first_tok_ms:.0}ms");
    eprintln!("decode:     {n_gen} tokens in {decode_ms:.0}ms ({dtps:.1} tok/s, {avg:.1}ms/tok)");
    (output, prefill_ms, decode_ms)
}

impl Drop for LlamaState {
    fn drop(&mut self) {
        wipe_f32(&mut self.x);
        wipe_f32(&mut self.x_norm);
        wipe_i8(&mut self.x_q8_qs);
        wipe_f32(&mut self.x_q8_d);
        wipe_i16(&mut self.x_q8_bsums);
        wipe_f32(&mut self.q);
        wipe_f32(&mut self.k);
        wipe_f32(&mut self.v);
        wipe_f32(&mut self.attn_out);
        wipe_i8(&mut self.attn_q8_qs);
        wipe_f32(&mut self.attn_q8_d);
        wipe_i16(&mut self.attn_q8_bsums);
        wipe_f32(&mut self.hidden);
        wipe_i8(&mut self.hidden_q8_qs);
        wipe_f32(&mut self.hidden_q8_d);
        wipe_i16(&mut self.hidden_q8_bsums);
        wipe_f32(&mut self.logits);
        wipe_f32(&mut self.tmp);
        wipe_f32(&mut self.sample_logits_buf);
        wipe_f32(&mut self.sample_probs);
    }
}

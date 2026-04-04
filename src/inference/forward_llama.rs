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

        // (attn_out trace moved after attention loop below)
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
        if layer == 0 {
            // Dump bias magnitudes
            if !lw.k_bias.is_null() {
                let kb = unsafe { std::slice::from_raw_parts(lw.k_bias, kv) };
                let k_mean_abs = kb.iter().map(|v| v.abs()).sum::<f32>() / kv as f32;
                let k_max = kb.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let k_out_l2: f32 = self.k[..kv].iter().map(|v| v*v).sum::<f32>().sqrt();
                let k_mean_out = self.k[..kv].iter().map(|v| v.abs()).sum::<f32>() / kv as f32;
                eprintln!("[L0 BIAS] K_bias mean_abs={k_mean_abs:.4} max={k_max:.4} | K_matmul_L2={k_out_l2:.4} K_matmul_mean_abs={k_mean_out:.4}");
            }
            if !lw.q_bias.is_null() {
                let qb = unsafe { std::slice::from_raw_parts(lw.q_bias, h) };
                let q_mean_abs = qb.iter().map(|v| v.abs()).sum::<f32>() / h as f32;
                let q_out_l2: f32 = self.q[..h].iter().map(|v| v*v).sum::<f32>().sqrt();
                eprintln!("[L0 BIAS] Q_bias mean_abs={q_mean_abs:.4} | Q_matmul_L2={q_out_l2:.4}");
            }
        }
        add_bias(&mut self.q, lw.q_bias, h);
        add_bias(&mut self.k, lw.k_bias, kv);
        add_bias(&mut self.v, lw.v_bias, kv);

        build_rope_freqs(&mut self.rope_freqs, hd, pos, model.rope_theta);
        apply_rope(&mut self.q, &self.rope_freqs, hd, nh);
        apply_rope(&mut self.k, &self.rope_freqs, hd, nkv);

        // pos=0 layer 0: dump Q, K, V after bias+rope (pre-cache)
        if pos == 0 && layer == 0 {
            let ql2: f32 = self.q.iter().take(h).map(|v| v*v).sum::<f32>().sqrt();
            let kl2: f32 = self.k.iter().take(kv).map(|v| v*v).sum::<f32>().sqrt();
            let vl2: f32 = self.v.iter().take(kv).map(|v| v*v).sum::<f32>().sqrt();
            eprintln!("[pos=0 L0] after bias+rope: Q_L2={ql2:.4} K_L2={kl2:.4} V_L2={vl2:.4}");
            eprintln!("[pos=0 L0] Q[0..4]={:?}", &self.q[..4]);
            eprintln!("[pos=0 L0] K[0..4]={:?}", &self.k[..4]);
            eprintln!("[pos=0 L0] V[0..4]={:?}", &self.v[..4]);
        }

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
                    // DEBUG: pos=0 layer 0 — verify attention is trivial (score=1.0, out=V)
                    if pos == 0 && layer == 0 && q_h == 0 {
                        eprintln!("[pos=0 L0 H0] attn score = {:?}", &scores[..seq_len]);
                    }
                    ffi::attn_vsum_f16(
                        scores.as_ptr(), v_ptr,
                        self.attn_out[q_h * hd..].as_mut_ptr(),
                        seq_len as i32, hd as i32,
                    );
                }
            }
        }

        // pos=0 L0: check Q8K quantization details
        if pos == 0 && layer == 0 {
            // Dump first Q8K block's scale and first few qs values
            eprintln!("[pos=0 L0] Q8K block 0: d={:.6} qs[0..8]={:?} bsums[0]={}",
                self.x_q8_d[0],
                &self.x_q8_qs[..8].iter().map(|&v| v as i32).collect::<Vec<_>>(),
                self.x_q8_bsums[0]);
            // Also check the input to quantization
            eprintln!("[pos=0 L0] x_norm[0..8]={:?}", &self.x_norm[..8]);
            // Check: is d always positive in our code?
            let neg_d_count = self.x_q8_d.iter().take(h_nb).filter(|&&d| d < 0.0).count();
            eprintln!("[pos=0 L0] negative q8_d count: {neg_d_count} / {h_nb}");
        }

        // First decode step, L0: verify Q4K dot against scalar reference
        if false && layer == 0 && std::env::var("OLORIN_VERIFY").is_ok() {
            let wq = lw.wq;
            let q8_qs = self.x_q8_qs.as_ptr();
            let q8_d = self.x_q8_d.as_ptr();
            let q8_bsums = self.x_q8_bsums.as_ptr();
            let pow2 = crate::inference::matmul_q4k::F16_POW2.as_ptr();
            unsafe {
                let kernel_val = ffi::q4k_dot_q8k(wq, q8_qs, q8_bsums, h_nb as i32, q8_d, pow2);
                // Pure scalar reference — no SIMD
                let mut ref_val = 0.0f32;
                for blk in 0..h_nb {
                    let w = wq.add(blk * 144);
                    let d_raw = *w as u16 | ((*w.add(1) as u16) << 8);
                    let dm_raw = *w.add(2) as u16 | ((*w.add(3) as u16) << 8);
                    let d = crate::inference::matmul::f16_to_f32(d_raw);
                    let dm = crate::inference::matmul::f16_to_f32(dm_raw);
                    let qd = *q8_d.add(blk);
                    let scales = std::slice::from_raw_parts(w.add(4), 12);
                    let mut sc = [0u8; 8];
                    let mut mn = [0u8; 8];
                    crate::inference::matmul_q4k::unpack_q4k_scales(scales, &mut sc, &mut mn);
                    let qs = w.add(16);
                    let mut sumi = 0i32;
                    for j in 0..4usize {
                        let (sl, sh) = (sc[2*j] as i32, sc[2*j+1] as i32);
                        for k in 0..32usize {
                            let byte = *qs.add(j * 32 + k);
                            let lo = (byte & 0xF) as i32;
                            let hi = (byte >> 4) as i32;
                            let q8l = *q8_qs.add(blk * 256 + j * 64 + k) as i32;
                            let q8h = *q8_qs.add(blk * 256 + j * 64 + 32 + k) as i32;
                            sumi += lo * q8l * sl + hi * q8h * sh;
                        }
                    }
                    let mut mins = 0i32;
                    for j in 0..8usize {
                        let bs_idx = blk * 16 + j * 2;
                        let bp = *q8_bsums.add(bs_idx) as i32 + *q8_bsums.add(bs_idx + 1) as i32;
                        mins += mn[j] as i32 * bp;
                    }
                    ref_val += d * qd * sumi as f32 - dm * qd * mins as f32;
                }
                eprintln!("[pos=0 L0] Q4K dot row0: kernel={kernel_val:.4} scalar_ref={ref_val:.4} diff={:.6}",
                    (kernel_val - ref_val).abs());
                eprintln!("[pos=0 L0] Q[0] (from matmul) = {:.4}", self.q[0]);

                // Verify ALL rows: compare matmul output vs scalar reference
                let mut max_diff = 0.0f32;
                let mut max_diff_row = 0usize;
                for row in 0..h {
                    let row_w = wq.add(row * h_nb * 144);
                    let mut ref_r = 0.0f32;
                    for blk in 0..h_nb {
                        let w = row_w.add(blk * 144);
                        let d_raw = *w as u16 | ((*w.add(1) as u16) << 8);
                        let dm_raw = *w.add(2) as u16 | ((*w.add(3) as u16) << 8);
                        let d = crate::inference::matmul::f16_to_f32(d_raw);
                        let dm = crate::inference::matmul::f16_to_f32(dm_raw);
                        let qd = *q8_d.add(blk);
                        let scales = std::slice::from_raw_parts(w.add(4), 12);
                        let mut sc = [0u8; 8];
                        let mut mn = [0u8; 8];
                        crate::inference::matmul_q4k::unpack_q4k_scales(scales, &mut sc, &mut mn);
                        let qs = w.add(16);
                        let mut sumi = 0i32;
                        for j in 0..4usize {
                            let (sl, sh) = (sc[2*j] as i32, sc[2*j+1] as i32);
                            for k in 0..32usize {
                                let byte = *qs.add(j * 32 + k);
                                let lo = (byte & 0xF) as i32;
                                let hi = (byte >> 4) as i32;
                                let q8l = *q8_qs.add(blk * 256 + j * 64 + k) as i32;
                                let q8h = *q8_qs.add(blk * 256 + j * 64 + 32 + k) as i32;
                                sumi += lo * q8l * sl + hi * q8h * sh;
                            }
                        }
                        let mut mins = 0i32;
                        for j in 0..8usize {
                            let bs_idx = blk * 16 + j * 2;
                            let bp = *q8_bsums.add(bs_idx) as i32 + *q8_bsums.add(bs_idx + 1) as i32;
                            mins += mn[j] as i32 * bp;
                        }
                        ref_r += d * qd * sumi as f32 - dm * qd * mins as f32;
                    }
                    let diff = (self.q[row] - ref_r).abs();
                    if diff > max_diff { max_diff = diff; max_diff_row = row; }
                }
                eprintln!("[pos=0 L0] ALL ROWS: max_diff={max_diff:.6} at row={max_diff_row} kernel={:.4} ref={:.4}",
                    self.q[max_diff_row], {
                        // recompute ref for that row
                        let row_w = wq.add(max_diff_row * h_nb * 144);
                        let mut ref_r = 0.0f32;
                        for blk in 0..h_nb {
                            let w = row_w.add(blk * 144);
                            let d_raw = *w as u16 | ((*w.add(1) as u16) << 8);
                            let dm_raw = *w.add(2) as u16 | ((*w.add(3) as u16) << 8);
                            let d = crate::inference::matmul::f16_to_f32(d_raw);
                            let dm = crate::inference::matmul::f16_to_f32(dm_raw);
                            let qd = *q8_d.add(blk);
                            let scales = std::slice::from_raw_parts(w.add(4), 12);
                            let mut sc = [0u8; 8];
                            let mut mn = [0u8; 8];
                            crate::inference::matmul_q4k::unpack_q4k_scales(scales, &mut sc, &mut mn);
                            let qs = w.add(16);
                            let mut sumi = 0i32;
                            for j in 0..4usize {
                                let (sl, sh) = (sc[2*j] as i32, sc[2*j+1] as i32);
                                for k in 0..32usize {
                                    let byte = *qs.add(j * 32 + k);
                                    let lo = (byte & 0xF) as i32;
                                    let hi = (byte >> 4) as i32;
                                    let q8l = *q8_qs.add(blk * 256 + j * 64 + k) as i32;
                                    let q8h = *q8_qs.add(blk * 256 + j * 64 + 32 + k) as i32;
                                    sumi += lo * q8l * sl + hi * q8h * sh;
                                }
                            }
                            let mut mins = 0i32;
                            for j in 0..8usize {
                                let bs_idx = blk * 16 + j * 2;
                                let bp = *q8_bsums.add(bs_idx) as i32 + *q8_bsums.add(bs_idx + 1) as i32;
                                mins += mn[j] as i32 * bp;
                            }
                            ref_r += d * qd * sumi as f32 - dm * qd * mins as f32;
                        }
                        ref_r
                    });
            }
        }

        if pos == 0 && layer == 0 {
            let al2: f32 = self.attn_out.iter().take(h).map(|v| v*v).sum::<f32>().sqrt();
            eprintln!("[pos=0 L0] attn_out_L2={al2:.4}");
            // Head 0: attn_out should equal the f16-roundtripped V
            let a_h0 = &self.attn_out[..hd];
            eprintln!("[pos=0 L0] attn_h0[0..4]={:?}", &a_h0[..4]);
            eprintln!("[pos=0 L0] V_orig[0..4]={:?}", &self.v[..4]);
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

        // DEBUG: trace per-layer stats on pos==28 (first decode after prefill of 24+4 tokens)
        if pos == 24 && layer == 2 {
            let x_l2: f32 = x.iter().map(|v| v*v).sum::<f32>().sqrt();
            let ao_l2: f32 = self.attn_out.iter().take(h).map(|v| v*v).sum::<f32>().sqrt();
            eprintln!("[L2] after Wo+resid: x_L2={x_l2:.2} attn_out_L2={ao_l2:.2} x[0..8]={:?}", &x[..8]);
        } else if pos == 24 {
            eprintln!("[L{layer}] x_L2={:.2}", x.iter().map(|v| v*v).sum::<f32>().sqrt());
        }

        // DEBUG: zero out dominant dimension to test dimensional drift hypothesis
        if std::env::var("OLORIN_ZERO_DIM2").is_ok() {
            x[2] = 0.0;
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

        if pos == 24 {
            eprintln!("[L{layer}] after FFN+resid: x[2]={:.4}", x[2]);
        }
    }

    /// Output projection: RMSNorm + quantize + matmul -> logits.
    pub(crate) fn output_proj(&mut self, model: &BitNetModel) {
        let h = model.hidden_dim;
        let h_nb = q8k_blocks(h);
        let h_row_stride = h_nb * Q4K_BLOCK_BYTES;
        unsafe {
            ffi::rmsnorm_f32(self.x.as_ptr(), model.norm_weight, self.x_norm.as_mut_ptr(), h as i32, model.rms_eps);
            // DEBUG: check x and x_norm before output projection
            let x_l2: f32 = self.x.iter().map(|v| v*v).sum::<f32>().sqrt();
            let xn_l2: f32 = self.x_norm.iter().take(h).map(|v| v*v).sum::<f32>().sqrt();
            eprintln!("[output_proj] x_L2={x_l2:.2} x_norm_L2={xn_l2:.2} x[0..4]={:?} xn[0..4]={:?}",
                &self.x[..4], &self.x_norm[..4]);
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
        if pos == 0 {
            let l2: f32 = self.x.iter().map(|v| v*v).sum::<f32>().sqrt();
            eprintln!("[embed] token={token} L2={l2:.6} x[0..8]={:?}", &self.x[..8]);
        }
        let mut x = std::mem::take(&mut self.x);
        for layer in 0..model.n_layers {
            self.process_layer(model, layer, &mut x, pos);
            // Trace per-layer x_L2 for first 2 tokens
            if pos <= 1 {
                let l2: f32 = x.iter().map(|v| v*v).sum::<f32>().sqrt();
                eprint!(" L{layer}={l2:.1}");
                if layer == model.n_layers - 1 { eprintln!(); }
            }
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
    // DEBUG: test f16 roundtrip
    if std::env::var("OLORIN_F16_TEST").is_ok() {
        let test_vals: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.1).collect();
        let mut f16_buf = vec![0u16; 128];
        let mut back = vec![0.0f32; 128];
        unsafe {
            ffi::f32_to_f16(test_vals.as_ptr(), f16_buf.as_mut_ptr(), 128);
            ffi::f16_to_f32(f16_buf.as_ptr(), back.as_mut_ptr(), 128);
        }
        let max_err = test_vals.iter().zip(back.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("[F16 roundtrip] max_err={max_err:.6} test[0]={:.4}→{:.4} test[64]={:.4}→{:.4}",
            test_vals[0], back[0], test_vals[64], back[64]);
    }
    let use_decode_only = std::env::var("OLORIN_NO_PREFILL").is_ok();
    if use_decode_only {
        eprintln!("[DEBUG] decode-only mode — processing prompt tokens one at a time");
        for (i, &tok) in prompt_tokens.iter().enumerate() {
            if i == 0 {
                embed_token(model, tok, &mut state.x);
            } else {
                state.forward(model, tok, i - 1);
            }
        }
        state.output_proj(model);
    } else {
        state.prefill(model, prompt_tokens);
    }
    output.extend_from_slice(prompt_tokens);
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    let first_tok_start = Instant::now();
    let mut pos = prompt_tokens.len();
    let mut n_gen = 0u32;
    let mut first_tok_ms = 0.0;

    let decode_start = Instant::now();
    for step in 0..max_tokens {
        if pos >= max_seq_len { break; }
        if step == 0 {
            // Check vocab size — are logits the right length?
            eprintln!("[step 0] logits.len()={} vocab_size={}", state.logits.len(), model.vocab_size);
            // Top-5 BEFORE penalty
            let mut indexed: Vec<(usize, f32)> = state.logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            eprintln!("--- top-10 logits BEFORE rep penalty ---");
            for &(i, v) in indexed.iter().take(10) {
                eprintln!("  token {i}: {v:.4}");
            }
            // Check if logits look "uniform" — entropy test
            let max_l = state.logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f64 = state.logits.iter().map(|&v| ((v - max_l) as f64).exp()).sum();
            let entropy: f64 = -(1.0 / sum_exp).ln();
            eprintln!("  range: [{:.4}, {:.4}] softmax_entropy={entropy:.2}",
                state.logits.iter().cloned().fold(f32::INFINITY, f32::min), max_l);
            eprintln!("  logits[0..8]={:?}", &state.logits[..8]);
            // Compare: llama.cpp gives logits[0..8]=[2.6111, 11.6444, 5.8988, 8.1306, 5.2622, 4.5004, 7.9542, 6.4317]

            // Verify output projection: scalar Q6K dot for row 0
            let h_nb = crate::inference::forward_llama::q8k_blocks(model.hidden_dim);
            let q6k_bb = crate::inference::matmul_q6k::Q6K_BLOCK_BYTES;
            unsafe {
                let ow = model.output_weight;
                let q8_qs = state.x_q8_qs.as_ptr();
                let q8_d = state.x_q8_d.as_ptr();
                // Scalar Q6K dot for row 0
                let mut ref_val = 0.0f64;
                for blk in 0..h_nb {
                    let bp = ow.add(blk * q6k_bb);
                    // Q6K layout: ql[128], qh[64], sc[16], d[2]
                    let d_raw = *(bp.add(208) as *const u16);
                    let d = crate::inference::matmul::f16_to_f32(d_raw) as f64;
                    let qd = *q8_d.add(blk) as f64;
                    let ql = bp;
                    let qh = bp.add(128);
                    let sc = bp.add(192) as *const i8;
                    let mut sumi = 0i32;
                    for half in 0..2usize {
                        for l in 0..32usize {
                            let is = l / 16;
                            let ql0 = *ql.add(half * 64 + l);
                            let ql32 = *ql.add(half * 64 + l + 32);
                            let qh_byte = *qh.add(half * 32 + l);
                            let q1 = ((ql0 & 0xF) | (((qh_byte >> 0) & 3) << 4)) as i8 as i32 - 32;
                            let q2 = ((ql32 & 0xF) | (((qh_byte >> 2) & 3) << 4)) as i8 as i32 - 32;
                            let q3 = ((ql0 >> 4) | (((qh_byte >> 4) & 3) << 4)) as i8 as i32 - 32;
                            let q4 = ((ql32 >> 4) | (((qh_byte >> 6) & 3) << 4)) as i8 as i32 - 32;
                            let s0 = *sc.add(half * 8 + is) as i32;
                            let s2 = *sc.add(half * 8 + is + 2) as i32;
                            let s4 = *sc.add(half * 8 + is + 4) as i32;
                            let s6 = *sc.add(half * 8 + is + 6) as i32;
                            let base = blk * 256 + half * 128;
                            sumi += q1 * *q8_qs.add(base + l) as i32 * s0;
                            sumi += q2 * *q8_qs.add(base + l + 32) as i32 * s2;
                            sumi += q3 * *q8_qs.add(base + l + 64) as i32 * s4;
                            sumi += q4 * *q8_qs.add(base + l + 96) as i32 * s6;
                        }
                    }
                    ref_val += d * qd * sumi as f64;
                }
                eprintln!("  [OUTPUT VERIFY] kernel logits[0]={:.6} scalar_ref={:.6} diff={:.6}",
                    state.logits[0], ref_val, (state.logits[0] as f64 - ref_val).abs());

                // Per-block comparison: run kernel for 1 block at a time vs scalar ref
                eprintln!("  [PER-BLOCK Q6K] (block, kernel_contrib, ref_contrib, diff):");
                for blk in 0..h_nb.min(4) {
                    let bp = ow.add(blk * q6k_bb);
                    let d_raw = *(bp.add(208) as *const u16);
                    let d = crate::inference::matmul::f16_to_f32(d_raw);
                    let qd = *q8_d.add(blk);
                    let d_arr_val = d * qd;

                    // Kernel: 1-block Q6K dot (weight ptr offset to this block)
                    let kernel_1blk = ffi::q6k_dot_q8k(
                        ow.add(blk * q6k_bb),
                        q8_qs.add(blk * 256),
                        state.x_q8_bsums.as_ptr().add(blk * 16),
                        1, // n_blocks = 1
                        &d_arr_val as *const f32,
                    );

                    // Scalar ref for this block
                    let ql = bp;
                    let qh = bp.add(128);
                    let sc = bp.add(192) as *const i8;
                    let mut sumi = 0i32;
                    for half in 0..2usize {
                        for l in 0..32usize {
                            let is = l / 16;
                            let ql0 = *ql.add(half * 64 + l);
                            let ql32 = *ql.add(half * 64 + l + 32);
                            let qh_byte = *qh.add(half * 32 + l);
                            let q1 = ((ql0 & 0xF) | (((qh_byte >> 0) & 3) << 4)) as i8 as i32 - 32;
                            let q2 = ((ql32 & 0xF) | (((qh_byte >> 2) & 3) << 4)) as i8 as i32 - 32;
                            let q3 = ((ql0 >> 4) | (((qh_byte >> 4) & 3) << 4)) as i8 as i32 - 32;
                            let q4 = ((ql32 >> 4) | (((qh_byte >> 6) & 3) << 4)) as i8 as i32 - 32;
                            let s0 = *sc.add(half * 8 + is) as i32;
                            let s2 = *sc.add(half * 8 + is + 2) as i32;
                            let s4 = *sc.add(half * 8 + is + 4) as i32;
                            let s6 = *sc.add(half * 8 + is + 6) as i32;
                            let base = blk * 256 + half * 128;
                            sumi += q1 * *q8_qs.add(base + l) as i32 * s0;
                            sumi += q2 * *q8_qs.add(base + l + 32) as i32 * s2;
                            sumi += q3 * *q8_qs.add(base + l + 64) as i32 * s4;
                            sumi += q4 * *q8_qs.add(base + l + 96) as i32 * s6;
                        }
                    }
                    let ref_1blk = d as f64 * qd as f64 * sumi as f64;

                    // Also compute maddubs-style (unsigned q6 * signed q8) sum and bsums correction
                    let mut maddubs_sum = 0i32;
                    let mut bsums_corr = 0i32;
                    for half in 0..2usize {
                        for l in 0..32usize {
                            let is = l / 16;
                            let ql0 = *ql.add(half * 64 + l);
                            let ql32 = *ql.add(half * 64 + l + 32);
                            let qh_byte = *qh.add(half * 32 + l);
                            // Unsigned q6 values (0..63)
                            let q1u = ((ql0 & 0xF) | (((qh_byte >> 0) & 3) << 4)) as i32;
                            let q2u = ((ql32 & 0xF) | (((qh_byte >> 2) & 3) << 4)) as i32;
                            let q3u = ((ql0 >> 4) | (((qh_byte >> 4) & 3) << 4)) as i32;
                            let q4u = ((ql32 >> 4) | (((qh_byte >> 6) & 3) << 4)) as i32;
                            let s0 = *sc.add(half * 8 + is) as i32;
                            let s2 = *sc.add(half * 8 + is + 2) as i32;
                            let s4 = *sc.add(half * 8 + is + 4) as i32;
                            let s6 = *sc.add(half * 8 + is + 6) as i32;
                            let base = blk * 256 + half * 128;
                            maddubs_sum += q1u * *q8_qs.add(base + l) as i32 * s0;
                            maddubs_sum += q2u * *q8_qs.add(base + l + 32) as i32 * s2;
                            maddubs_sum += q3u * *q8_qs.add(base + l + 64) as i32 * s4;
                            maddubs_sum += q4u * *q8_qs.add(base + l + 96) as i32 * s6;
                        }
                        // bsums correction for this half
                        for g in 0..8usize {
                            let bs = *state.x_q8_bsums.as_ptr().add(blk * 16 + half * 8 + g) as i32;
                            let s_val = *sc.add(half * 8 + g) as i32;
                            bsums_corr += 32 * s_val * bs;
                        }
                    }
                    if blk < 2 {
                        eprintln!("    blk={blk}: kernel={kernel_1blk:.6} ref={:.6} diff={:.6} maddubs={maddubs_sum} bsums_corr={bsums_corr} sumi_ref={sumi} check={}",
                            ref_1blk, (kernel_1blk as f64 - ref_1blk).abs(), maddubs_sum - bsums_corr);
                    } else {
                        eprintln!("    blk={blk}: kernel={kernel_1blk:.6} ref={:.6} diff={:.6}",
                            ref_1blk, (kernel_1blk as f64 - ref_1blk).abs());
                    }
                }
            }
        }
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

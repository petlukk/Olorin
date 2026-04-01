//! GEMM-style batched prefill for Llama/Q4_K models.

use crate::kernels::ffi_inference as ffi;
use crate::inference::forward::{apply_rope, build_rope_freqs};
use crate::inference::forward_llama::{LlamaState, embed_token, add_bias, q8k_blocks};
use crate::inference::math::softmax_rows;
use crate::inference::gemm_q4k::{BatchQ8K, q4k_fused_gemm_f32_mt, q4k_fused_silu_gemm_f32_mt};
use crate::inference::gemm_q6k::q6k_gemm_mt;
use crate::inference::matmul_q4k::Q4K_BLOCK_BYTES;
use crate::inference::matmul_q6k::Q6K_BLOCK_BYTES;
use crate::inference::engine::BitNetModel;
use crate::inference::cache;
/// Opaque pointer wrapper for vecadd thread dispatch.
#[derive(Clone, Copy)]
struct W {
    xs_mut: usize,  // *const *mut f32
    buf: usize,     // *const f32
    h: usize,
    n: usize,
}

impl LlamaState {
    /// GEMM-style batched prefill: load weight once, multiply all tokens.
    /// Uses fused quant+matmul kernels where possible (Q4K).
    /// Falls back to BatchQ8K for Q6K layers.
    pub fn prefill(&mut self, model: &BitNetModel, tokens: &[u32]) {
        let n = tokens.len();
        let (h, hd, nh, nkv, kv, f) = (
            model.hidden_dim, model.head_dim, model.n_heads,
            model.n_kv_heads, model.kv_dim, model.ffn_dim,
        );
        let (h_nb, f_nb) = (q8k_blocks(h), q8k_blocks(f));
        let h_rs = h_nb * Q4K_BLOCK_BYTES;
        let mut xs: Vec<Vec<f32>> = tokens.iter().map(|&tok| {
            let mut x = vec![0.0f32; h]; embed_token(model, tok, &mut x); x
        }).collect();

        // Flat f32 buffer for normalized activations (replaces BatchQ8K for Q4K path)
        let mut norm_all = vec![0.0f32; n * h];

        // BatchQ8K only needed for Q6K fallback layers
        let has_q6k_v = model.q4k_layers.iter().any(|lw| lw.wv_block_bytes == Q6K_BLOCK_BYTES);
        let has_q6k_down = model.q4k_layers.iter().any(|lw| lw.w_down_block_bytes == Q6K_BLOCK_BYTES);
        let mut bq_h = if has_q6k_v { Some(BatchQ8K::new(n, h)) } else { None };
        let mut bq_f = if has_q6k_down { Some(BatchQ8K::new(n, f)) } else { None };

        let (mut qs_all, mut ks_all, mut vs_all) = (
            vec![0.0f32; n*h], vec![0.0f32; n*kv], vec![0.0f32; n*kv],
        );
        let (mut attn_all, mut tmp_all, mut hidden_all) = (
            vec![0.0f32; n*h], vec![0.0f32; n*h], vec![0.0f32; n*f],
        );
        let nt = self.pool.thread_count();

        for layer in 0..model.n_layers {
            let lw = &model.q4k_layers[layer];

            // ── Parallel rmsnorm (attn input) → norm_all ──
            {
                let xs_raw: Vec<*const f32> = xs.iter().map(|x| x.as_ptr()).collect();
                let norm_ptr = norm_all.as_mut_ptr() as usize;
                let xs_usize = xs_raw.as_ptr() as usize;
                let norm_w = lw.attn_norm as usize;
                let eps = model.rms_eps;
                let h_dim = h;
                let n_tok = n;
                self.pool.run(nt.min(n), move |tid, nt_used| unsafe {
                    let xs = xs_usize as *const *const f32;
                    let out = norm_ptr as *mut f32;
                    let mut t = tid;
                    while t < n_tok {
                        ffi::rmsnorm_f32(*xs.add(t), norm_w as *const f32, out.add(t * h_dim), h_dim as i32, eps);
                        t += nt_used;
                    }
                });
            }

            // ── QKV matmul: fused f32→kernel (no Q8K buffer) ──
            q4k_fused_gemm_f32_mt(lw.wq, h_rs, h_nb, &norm_all, h, n, &mut qs_all, h, &self.pool);
            q4k_fused_gemm_f32_mt(lw.wk, h_rs, h_nb, &norm_all, h, n, &mut ks_all, kv, &self.pool);
            if lw.wv_block_bytes == Q6K_BLOCK_BYTES {
                // Q6K fallback: need BatchQ8K
                let bq = bq_h.as_mut().unwrap();
                for t in 0..n { bq.quantize(t, &norm_all[t*h..(t+1)*h]); }
                q6k_gemm_mt(lw.wv, h_nb * Q6K_BLOCK_BYTES, h_nb, bq, &mut vs_all, kv, &self.pool);
            } else {
                q4k_fused_gemm_f32_mt(lw.wv, h_rs, h_nb, &norm_all, h, n, &mut vs_all, kv, &self.pool);
            }

            // ── Attention: batch bias+RoPE, bulk KV append, sequential scoring ──

            // Apply biases and RoPE to all tokens first
            for t in 0..n {
                let q = &mut qs_all[t*h..(t+1)*h];
                let k = &mut ks_all[t*kv..(t+1)*kv];
                let v = &mut vs_all[t*kv..(t+1)*kv];
                add_bias(q, lw.q_bias, h);
                add_bias(v, lw.v_bias, kv);
                build_rope_freqs(&mut self.rope_freqs, hd, t, model.rope_theta);
                apply_rope(q, &self.rope_freqs, hd, nh);
                apply_rope(k, &self.rope_freqs, hd, nkv);
            }

            // Transpose K and V from [token][head*dim] to [head][token][dim] for bulk append
            let mut k_transposed = vec![0.0f32; nkv * n * hd];
            let mut v_transposed = vec![0.0f32; nkv * n * hd];
            for t in 0..n {
                for head in 0..nkv {
                    let src_off = t * kv + head * hd;
                    let dst_off = head * n * hd + t * hd;
                    k_transposed[dst_off..dst_off + hd]
                        .copy_from_slice(&ks_all[src_off..src_off + hd]);
                    v_transposed[dst_off..dst_off + hd]
                        .copy_from_slice(&vs_all[src_off..src_off + hd]);
                }
            }

            // Bulk append all N tokens to KV cache at once
            self.kv_cache.restore(0).unwrap();
            self.kv_cache.append(&k_transposed, layer as i32, 0, n as i32).unwrap();
            self.kv_cache.append(&v_transposed, layer as i32, 1, n as i32).unwrap();
            self.kv_cache.advance(n as i32).unwrap();

            // Pre-compute K-bias correction for all positions (if model has K bias)
            let has_k_bias = !lw.k_bias.is_null();
            let mut k_bias_dots: Vec<f32> = Vec::new();
            if has_k_bias {
                let q_per_kv = nh / nkv;
                let rsqrt_hd = 1.0 / (hd as f32).sqrt();
                let mut bias_freqs = vec![0.0f32; hd];
                let mut rotated_biases = vec![0.0f32; n * kv];
                for s in 0..n {
                    for i in 0..kv { rotated_biases[s * kv + i] = unsafe { *lw.k_bias.add(i) }; }
                    build_rope_freqs(&mut bias_freqs, hd, s, model.rope_theta);
                    apply_rope(&mut rotated_biases[s * kv..(s + 1) * kv], &bias_freqs, hd, nkv);
                }
                k_bias_dots.resize(n * nh * n, 0.0);
                for t in 0..n {
                    let q = &qs_all[t * h..(t + 1) * h];
                    let seq_len = t + 1;
                    for s in 0..seq_len {
                        let rb = &rotated_biases[s * kv..];
                        for kv_h in 0..nkv {
                            let kb_off = kv_h * hd;
                            for q_off in 0..q_per_kv {
                                let q_h = kv_h * q_per_kv + q_off;
                                let qb = q_h * hd;
                                let mut dot = 0.0f32;
                                for d in 0..hd { dot += q[qb + d] * rb[kb_off + d]; }
                                k_bias_dots[t * nh * n + q_h * n + s] = dot * rsqrt_hd;
                            }
                        }
                    }
                }
            }

            // Attention: fused causal kernel if available, else parallel per-token
            if !has_k_bias && ffi::has_fused_causal_attn() {
                let signs = self.kv_cache.jl_signs();
                let n_q_groups = (nh * hd) / 64;
                let mut rot_qs = qs_all[..n * h].to_vec();
                for qi in 0..n {
                    for g in 0..n_q_groups {
                        unsafe {
                            let ptr = rot_qs.as_mut_ptr().add(qi * h + g * 64);
                            crate::kernels::ffi::turbo_rotate(ptr, signs.as_ptr(), 64);
                        }
                    }
                }
                let gph = self.kv_cache.max_seq_len() * (hd as i32 / 64);
                let k_w = self.kv_cache.weights(layer as i32, 0);
                let k_s = self.kv_cache.scales(layer as i32, 0);
                let k_b = self.kv_cache.biases(layer as i32, 0);
                let v_w = self.kv_cache.weights(layer as i32, 1);
                let v_s = self.kv_cache.scales(layer as i32, 1);
                let v_b = self.kv_cache.biases(layer as i32, 1);
                let mut state = vec![0.0f32; n * nh * 130];
                unsafe {
                    ffi::fused_causal_attn_gqa(
                        rot_qs.as_ptr(), k_w.as_ptr(), k_s.as_ptr(), k_b.as_ptr(),
                        v_w.as_ptr(), v_s.as_ptr(), v_b.as_ptr(),
                        state.as_mut_ptr(), attn_all.as_mut_ptr(),
                        n as i32, n as i32, nh as i32, nkv as i32, gph,
                    );
                }
                let n_out_groups = (nh * hd) / 64;
                for qi in 0..n {
                    for g in 0..n_out_groups {
                        unsafe {
                            let ptr = attn_all.as_mut_ptr().add(qi * h + g * 64);
                            crate::kernels::ffi::fwht_inplace(ptr, 64);
                            crate::kernels::ffi::sign_flip(ptr, signs.as_ptr(), 64);
                        }
                    }
                }
            } else {
                let cache_ptr = &self.kv_cache as *const cache::EakvCache as usize;
                let qs_ptr = qs_all.as_ptr() as usize;
                let attn_ptr = attn_all.as_mut_ptr() as usize;
                let bias_ptr = if has_k_bias { k_bias_dots.as_ptr() as usize } else { 0 };
                let layer_i32 = layer as i32;
                let nh_i32 = nh as i32;
                let nkv_i32 = nkv as i32;
                let n_tokens = n;
                self.pool.run(nt.min(n), move |tid, nt_used| {
                    let mut t = tid;
                    while t < n_tokens {
                        let seq_len = (t + 1) as i32;
                        let mut scores = vec![0.0f32; nh * (t + 1)];
                        let q = unsafe { std::slice::from_raw_parts((qs_ptr as *const f32).add(t * h), h) };
                        let cache_ref = unsafe { &*(cache_ptr as *const cache::EakvCache) };
                        cache::attention::attention_scores(
                            cache_ref, q, layer_i32, nh_i32, nkv_i32, seq_len, &mut scores,
                        );
                        if bias_ptr != 0 {
                            let bias = unsafe { std::slice::from_raw_parts(
                                (bias_ptr as *const f32).add(t * nh * n_tokens), nh * n_tokens,
                            )};
                            for q_h in 0..nh {
                                for s in 0..(t + 1) {
                                    scores[q_h * (t + 1) + s] += bias[q_h * n_tokens + s];
                                }
                            }
                        }
                        softmax_rows(&mut scores, nh, t + 1);
                        let attn_out = unsafe { std::slice::from_raw_parts_mut(
                            (attn_ptr as *mut f32).add(t * h), h,
                        )};
                        cache::attention::attention_output(
                            cache_ref, &scores, layer_i32, nh_i32, nkv_i32, seq_len, attn_out,
                        );
                        t += nt_used;
                    }
                });
            }

            // ── Wo matmul: fused f32→kernel (attn_all already f32) ──
            q4k_fused_gemm_f32_mt(lw.wo, h_rs, h_nb, &attn_all, h, n, &mut tmp_all, h, &self.pool);

            // ── Parallel vecadd residual (attn) ──
            {
                let xs_raw: Vec<*mut f32> = xs.iter_mut().map(|x| x.as_mut_ptr()).collect();
                let w = W { xs_mut: xs_raw.as_ptr() as usize, buf: tmp_all.as_ptr() as usize, h, n };
                self.pool.run(nt.min(n), move |tid, nt_used| unsafe {
                    let xm = w.xs_mut as *const *mut f32;
                    let buf = w.buf as *const f32;
                    let mut t = tid;
                    while t < w.n {
                        let x = *xm.add(t);
                        ffi::vecadd_f32(x, buf.add(t * w.h), x, w.h as i32);
                        t += nt_used;
                    }
                });
            }

            // ── Parallel rmsnorm (ffn input) → norm_all ──
            {
                let xs_raw: Vec<*const f32> = xs.iter().map(|x| x.as_ptr()).collect();
                let norm_ptr = norm_all.as_mut_ptr() as usize;
                let xs_usize = xs_raw.as_ptr() as usize;
                let norm_w = lw.ffn_norm as usize;
                let eps = model.rms_eps;
                let h_dim = h;
                let n_tok = n;
                self.pool.run(nt.min(n), move |tid, nt_used| unsafe {
                    let xs = xs_usize as *const *const f32;
                    let out = norm_ptr as *mut f32;
                    let mut t = tid;
                    while t < n_tok {
                        ffi::rmsnorm_f32(*xs.add(t), norm_w as *const f32, out.add(t * h_dim), h_dim as i32, eps);
                        t += nt_used;
                    }
                });
            }

            // ── FFN gate+up+SiLU: fused f32→kernel ──
            q4k_fused_silu_gemm_f32_mt(lw.w_gate, lw.w_up, h_rs, h_nb, &norm_all, h, n, &mut hidden_all, f, &self.pool);

            // ── Down projection: fused or Q6K fallback ──
            if lw.w_down_block_bytes == Q6K_BLOCK_BYTES {
                let bq = bq_f.as_mut().unwrap();
                for t in 0..n { bq.quantize(t, &hidden_all[t*f..(t+1)*f]); }
                q6k_gemm_mt(lw.w_down, f_nb * Q6K_BLOCK_BYTES, f_nb, bq, &mut tmp_all, h, &self.pool);
            } else {
                q4k_fused_gemm_f32_mt(lw.w_down, f_nb * Q4K_BLOCK_BYTES, f_nb, &hidden_all, f, n, &mut tmp_all, h, &self.pool);
            }

            // ── Parallel vecadd residual (ffn) ──
            {
                let xs_raw: Vec<*mut f32> = xs.iter_mut().map(|x| x.as_mut_ptr()).collect();
                let w = W { xs_mut: xs_raw.as_ptr() as usize, buf: tmp_all.as_ptr() as usize, h, n };
                self.pool.run(nt.min(n), move |tid, nt_used| unsafe {
                    let xm = w.xs_mut as *const *mut f32;
                    let buf = w.buf as *const f32;
                    let mut t = tid;
                    while t < w.n {
                        let x = *xm.add(t);
                        ffi::vecadd_f32(x, buf.add(t * w.h), x, w.h as i32);
                        t += nt_used;
                    }
                });
            }
        }
        self.x[..h].copy_from_slice(&xs[n - 1]);
        self.output_proj(model);
    }
}

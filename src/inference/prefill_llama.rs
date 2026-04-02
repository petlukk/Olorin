//! GEMM-style batched prefill for Llama/Q4_K models.
//! Quantize all tokens to Q8K once, then batch-matmul against Q4K weights.

use crate::kernels::ffi_inference as ffi;
use crate::inference::forward::{apply_rope, build_rope_freqs};
use crate::inference::forward_llama::{LlamaState, embed_token, add_bias, q8k_blocks};
use crate::inference::math::softmax_rows;
use crate::inference::gemm_q4k::{BatchQ8K, q4k_gemm_mt, q4k_fused_silu_gemm_mt};
use crate::inference::gemm_q6k::q6k_gemm_mt;
use crate::inference::matmul_q4k::Q4K_BLOCK_BYTES;
use crate::inference::matmul_q6k::Q6K_BLOCK_BYTES;
use crate::inference::engine::BitNetModel;
use crate::inference::cache;

/// Opaque pointer wrapper for vecadd thread dispatch.
#[derive(Clone, Copy)]
struct W {
    xs_mut: usize,
    buf: usize,
    h: usize,
    n: usize,
}

impl LlamaState {
    /// Run all transformer layers for N tokens, return per-token hidden states.
    fn prefill_layers(&mut self, model: &BitNetModel, tokens: &[u32]) -> Vec<Vec<f32>> {
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

        let mut norm_all = vec![0.0f32; n * h];
        let mut bq_h = BatchQ8K::new(n, h);
        let mut bq_f = BatchQ8K::new(n, f);

        let (mut qs_all, mut ks_all, mut vs_all) = (
            vec![0.0f32; n*h], vec![0.0f32; n*kv], vec![0.0f32; n*kv],
        );
        let (mut attn_all, mut tmp_all, mut hidden_all) = (
            vec![0.0f32; n*h], vec![0.0f32; n*h], vec![0.0f32; n*f],
        );
        let nt = self.pool.thread_count();

        for layer in 0..model.n_layers {
            let lw = &model.q4k_layers[layer];

            // ── Parallel rmsnorm → norm_all, then batch quantize to Q8K ──
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
            // Parallel Q8K quantization of all N tokens
            {
                let qs_ptr = bq_h.qs.as_mut_ptr() as usize;
                let d_ptr = bq_h.d.as_mut_ptr() as usize;
                let bs_ptr = bq_h.bsums.as_mut_ptr() as usize;
                let norm_ptr = norm_all.as_ptr() as usize;
                let qs_stride = bq_h.qs_stride;
                let nb = bq_h.n_blocks;
                let h_dim = h;
                let n_tok = n;
                self.pool.run(nt.min(n), move |tid, nt_used| unsafe {
                    let mut t = tid;
                    while t < n_tok {
                        ffi::quant_f32_q8k(
                            (norm_ptr as *const f32).add(t * h_dim),
                            (qs_ptr as *mut i8).add(t * qs_stride),
                            (d_ptr as *mut f32).add(t * nb),
                            (bs_ptr as *mut i32).add(t * nb * 16),
                            h_dim as i32,
                        );
                        t += nt_used;
                    }
                });
            }

            // ── QKV matmul: Q4K × Q8K (pre-quantized) ──
            q4k_gemm_mt(lw.wq, h_rs, h_nb, &bq_h, &mut qs_all, h, &self.pool);
            q4k_gemm_mt(lw.wk, h_rs, h_nb, &bq_h, &mut ks_all, kv, &self.pool);
            if lw.wv_block_bytes == Q6K_BLOCK_BYTES {
                q6k_gemm_mt(lw.wv, h_nb * Q6K_BLOCK_BYTES, h_nb, &bq_h, &mut vs_all, kv, &self.pool);
            } else {
                q4k_gemm_mt(lw.wv, h_rs, h_nb, &bq_h, &mut vs_all, kv, &self.pool);
            }

            // ── Attention: batch bias+RoPE, bulk KV append, sequential scoring ──
            for t in 0..n {
                add_bias(&mut qs_all[t*h..(t+1)*h], lw.q_bias, h);
                add_bias(&mut vs_all[t*kv..(t+1)*kv], lw.v_bias, kv);
            }
            let mut rope_freqs = vec![0.0f32; hd];
            let pos_base = self.kv_cache.seq_len() as usize;
            for t in 0..n {
                build_rope_freqs(&mut rope_freqs, hd, pos_base + t, model.rope_theta);
                apply_rope(&mut qs_all[t*h..(t+1)*h], &rope_freqs, hd, nh);
                apply_rope(&mut ks_all[t*kv..(t+1)*kv], &rope_freqs, hd, nkv);
            }
            // Bulk KV append: all N tokens at once (head-major within each token)
            // append() uses self.seq_len as write position, n_tokens controls how many.
            // Data must be head-major: [head][n_tokens * head_dim]
            // But ks_all is token-major: [token][kv_heads * head_dim]
            // Transpose to head-major for bulk append:
            {
                let mut k_hm = vec![0.0f32; n * kv];  // head-major
                let mut v_hm = vec![0.0f32; n * kv];
                for h_idx in 0..nkv {
                    for t in 0..n {
                        let src_off = t * kv + h_idx * hd;
                        let dst_off = h_idx * n * hd + t * hd;
                        k_hm[dst_off..dst_off+hd].copy_from_slice(&ks_all[src_off..src_off+hd]);
                        v_hm[dst_off..dst_off+hd].copy_from_slice(&vs_all[src_off..src_off+hd]);
                    }
                }
                self.kv_cache.append(&k_hm, layer as i32, 0, n as i32).unwrap();
                self.kv_cache.append(&v_hm, layer as i32, 1, n as i32).unwrap();
                if layer == model.n_layers - 1 { self.kv_cache.advance(n as i32).unwrap(); }
            }

            let has_k_bias = !lw.k_bias.is_null();
            let seq_len = pos_base + n;

            if !has_k_bias && hd == 128 && ffi::has_fused_causal_attn() {
                // Flash causal attention: single kernel for all N queries
                let signs = self.kv_cache.jl_signs();
                let n_groups = (nh * hd) / 64;
                for t in 0..n {
                    for g in 0..n_groups {
                        let off = t * h + g * 64;
                        unsafe {
                            crate::kernels::ffi::turbo_rotate(
                                qs_all.as_mut_ptr().add(off), signs.as_ptr(), 64,
                            );
                        }
                    }
                }
                let (k_w, k_s, k_b) = self.kv_cache.k_ptrs(layer as i32);
                let (v_w, v_s, v_b) = self.kv_cache.v_ptrs(layer as i32);
                let gph = self.kv_cache.groups_per_head();
                let mut state = vec![0.0f32; n * nh * 130];
                unsafe {
                    ffi::fused_causal_attn_gqa(
                        qs_all.as_ptr(), k_w, k_s, k_b, v_w, v_s, v_b,
                        state.as_mut_ptr(), attn_all.as_mut_ptr(),
                        seq_len as i32, n as i32, nh as i32, nkv as i32, gph,
                    );
                }
                let n_out_groups = (nh * hd) / 64;
                for t in 0..n {
                    for g in 0..n_out_groups {
                        let off = t * h + g * 64;
                        unsafe {
                            let ptr = attn_all.as_mut_ptr().add(off);
                            crate::kernels::ffi::fwht_inplace(ptr, 64);
                            crate::kernels::ffi::sign_flip(ptr, signs.as_ptr(), 64);
                        }
                    }
                }
            } else {
                // Per-token 3-pass attention (with K-bias correction)
                let max_scores = nh * seq_len;
                let mut scores = vec![0.0f32; max_scores];
                for t in 0..n {
                    let qt = &qs_all[t*h..(t+1)*h];
                    cache::attention::attention_scores(&self.kv_cache, qt, layer as i32, nh as i32, nkv as i32, (pos_base + t + 1) as i32, &mut scores[..nh*(pos_base+t+1)]);
                    if has_k_bias {
                        let q_per_kv = nh / nkv;
                        let rsqrt_hd = 1.0 / (hd as f32).sqrt();
                        let sl = pos_base + t + 1;
                        let mut rotated_bias = vec![0.0f32; kv];
                        let mut bias_freqs = vec![0.0f32; hd];
                        for s in 0..sl {
                            for i in 0..kv { rotated_bias[i] = unsafe { *lw.k_bias.add(i) }; }
                            build_rope_freqs(&mut bias_freqs, hd, s, model.rope_theta);
                            apply_rope(&mut rotated_bias, &bias_freqs, hd, nkv);
                            for kv_h in 0..nkv {
                                let kb_off = kv_h * hd;
                                for q_off in 0..q_per_kv {
                                    let q_h = kv_h * q_per_kv + q_off;
                                    let mut dot = 0.0f32;
                                    for d in 0..hd { dot += qt[q_h*hd+d] * rotated_bias[kb_off+d]; }
                                    scores[q_h * sl + s] += dot * rsqrt_hd;
                                }
                            }
                        }
                    }
                    softmax_rows(&mut scores[..nh*(pos_base+t+1)], nh, pos_base + t + 1);
                    cache::attention::attention_output(&self.kv_cache, &scores[..nh*(pos_base+t+1)], layer as i32, nh as i32, nkv as i32, (pos_base + t + 1) as i32, &mut attn_all[t*h..(t+1)*h]);
                }
            }

            // ── Wo matmul: Q4K × Q8K ──
            {
                let qs_ptr = bq_h.qs.as_mut_ptr() as usize;
                let d_ptr = bq_h.d.as_mut_ptr() as usize;
                let bs_ptr = bq_h.bsums.as_mut_ptr() as usize;
                let src_ptr = attn_all.as_ptr() as usize;
                let qs_stride = bq_h.qs_stride;
                let nb = bq_h.n_blocks;
                let h_dim = h;
                let n_tok = n;
                self.pool.run(nt.min(n), move |tid, nt_used| unsafe {
                    let mut t = tid;
                    while t < n_tok {
                        ffi::quant_f32_q8k(
                            (src_ptr as *const f32).add(t * h_dim),
                            (qs_ptr as *mut i8).add(t * qs_stride),
                            (d_ptr as *mut f32).add(t * nb),
                            (bs_ptr as *mut i32).add(t * nb * 16),
                            h_dim as i32,
                        );
                        t += nt_used;
                    }
                });
            }
            q4k_gemm_mt(lw.wo, h_rs, h_nb, &bq_h, &mut tmp_all, h, &self.pool);

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

            // ── Parallel rmsnorm (FFN) → norm_all, then batch quantize ──
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
            {
                let qs_ptr = bq_h.qs.as_mut_ptr() as usize;
                let d_ptr = bq_h.d.as_mut_ptr() as usize;
                let bs_ptr = bq_h.bsums.as_mut_ptr() as usize;
                let src_ptr = norm_all.as_ptr() as usize;
                let qs_stride = bq_h.qs_stride;
                let nb = bq_h.n_blocks;
                let h_dim = h;
                let n_tok = n;
                self.pool.run(nt.min(n), move |tid, nt_used| unsafe {
                    let mut t = tid;
                    while t < n_tok {
                        ffi::quant_f32_q8k(
                            (src_ptr as *const f32).add(t * h_dim),
                            (qs_ptr as *mut i8).add(t * qs_stride),
                            (d_ptr as *mut f32).add(t * nb),
                            (bs_ptr as *mut i32).add(t * nb * 16),
                            h_dim as i32,
                        );
                        t += nt_used;
                    }
                });
            }

            // ── FFN gate+up+SiLU: Q4K × Q8K ──
            q4k_fused_silu_gemm_mt(lw.w_gate, lw.w_up, h_rs, h_nb, &bq_h, &mut hidden_all, f, &self.pool);

            // ── Down projection: Q4K or Q6K ──
            {
                let qs_ptr = bq_f.qs.as_mut_ptr() as usize;
                let d_ptr = bq_f.d.as_mut_ptr() as usize;
                let bs_ptr = bq_f.bsums.as_mut_ptr() as usize;
                let src_ptr = hidden_all.as_ptr() as usize;
                let qs_stride = bq_f.qs_stride;
                let nb = bq_f.n_blocks;
                let f_dim = f;
                let n_tok = n;
                self.pool.run(nt.min(n), move |tid, nt_used| unsafe {
                    let mut t = tid;
                    while t < n_tok {
                        ffi::quant_f32_q8k(
                            (src_ptr as *const f32).add(t * f_dim),
                            (qs_ptr as *mut i8).add(t * qs_stride),
                            (d_ptr as *mut f32).add(t * nb),
                            (bs_ptr as *mut i32).add(t * nb * 16),
                            f_dim as i32,
                        );
                        t += nt_used;
                    }
                });
            }
            if lw.w_down_block_bytes == Q6K_BLOCK_BYTES {
                q6k_gemm_mt(lw.w_down, f_nb * Q6K_BLOCK_BYTES, f_nb, &bq_f, &mut tmp_all, h, &self.pool);
            } else {
                q4k_gemm_mt(lw.w_down, f_nb * Q4K_BLOCK_BYTES, f_nb, &bq_f, &mut tmp_all, h, &self.pool);
            }

            // ── Parallel vecadd residual (FFN) ──
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
        xs
    }

    /// GEMM-style batched prefill: quantize all tokens to Q8K once,
    /// then load each weight matrix once and multiply all tokens.
    pub fn prefill(&mut self, model: &BitNetModel, tokens: &[u32]) {
        let xs = self.prefill_layers(model, tokens);
        let h = model.hidden_dim;
        let n = tokens.len();
        self.x[..h].copy_from_slice(&xs[n - 1]);
        self.output_proj(model);
    }

    /// Batched prefill that returns per-token argmax token IDs.
    /// Used by speculative decoding to verify draft tokens.
    pub fn prefill_verify(&mut self, model: &BitNetModel, tokens: &[u32]) -> Vec<u32> {
        let xs = self.prefill_layers(model, tokens);
        let h = model.hidden_dim;
        let mut result = Vec::with_capacity(xs.len());
        for hidden in &xs {
            self.x[..h].copy_from_slice(hidden);
            self.output_proj(model);
            result.push(super::forward_llama::argmax(&self.logits));
        }
        result
    }
}

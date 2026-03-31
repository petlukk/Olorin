//! GEMM-style batched prefill for Llama/Q4_K models.

use crate::kernels::ffi_inference as ffi;
use crate::inference::forward::{apply_rope, build_rope_freqs};
use crate::inference::forward_llama::{LlamaState, embed_token, add_bias, q8k_blocks, softmax_rows};
use crate::inference::gemm_q4k::{BatchQ8K, q4k_gemm_mt, q4k_fused_silu_gemm_mt};
use crate::inference::gemm_q6k::q6k_gemm_mt;
use crate::inference::matmul_q4k::Q4K_BLOCK_BYTES;
use crate::inference::matmul_q6k::Q6K_BLOCK_BYTES;
use crate::inference::engine::BitNetModel;
use crate::inference::cache;
/// Opaque pointer wrapper for thread dispatch — all raw pointers cast to usize.
/// pool.run is synchronous so pointed-to data outlives the dispatch.
#[derive(Clone, Copy)]
struct W {
    xs: usize,      // *const *const f32
    xs_mut: usize,  // *const *mut f32
    norm: usize,    // *const f32
    buf: usize,     // *const f32
    bq: usize,      // *mut BatchQ8K
    h: usize,
    f: usize,
    n: usize,
}

impl LlamaState {
    /// GEMM-style batched prefill: load weight once, multiply all tokens.
    /// Per-token rmsnorm/quantize/vecadd parallelized via ThreadPool.
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

            // ── Parallel rmsnorm + quantize (attn input) ──
            {
                let xs_raw: Vec<*const f32> = xs.iter().map(|x| x.as_ptr()).collect();
                let w = W { xs: xs_raw.as_ptr() as usize, xs_mut: 0, norm: lw.attn_norm as usize, buf: 0, bq: &mut bq_h as *mut BatchQ8K as usize, h, f, n };
                let eps = model.rms_eps;
                self.pool.run(nt.min(n), move |tid, nt_used| unsafe {
                    let mut nbuf = vec![0.0f32; w.h];
                    let xs = w.xs as *const *const f32;
                    let mut t = tid;
                    while t < w.n {
                        ffi::rmsnorm_f32(*xs.add(t), w.norm as *const f32, nbuf.as_mut_ptr(), w.h as i32, eps);
                        (*(w.bq as *mut BatchQ8K)).quantize(t, &nbuf);
                        t += nt_used;
                    }
                });
            }

            q4k_gemm_mt(lw.wq, h_rs, h_nb, &bq_h, &mut qs_all, h, &self.pool);
            q4k_gemm_mt(lw.wk, h_rs, h_nb, &bq_h, &mut ks_all, kv, &self.pool);
            if lw.wv_block_bytes == Q6K_BLOCK_BYTES {
                q6k_gemm_mt(lw.wv, h_nb * Q6K_BLOCK_BYTES, h_nb, &bq_h, &mut vs_all, kv, &self.pool);
            } else {
                q4k_gemm_mt(lw.wv, h_rs, h_nb, &bq_h, &mut vs_all, kv, &self.pool);
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
                // Pre-rotate bias for each position
                let mut rotated_biases = vec![0.0f32; n * kv];
                for s in 0..n {
                    for i in 0..kv { rotated_biases[s * kv + i] = unsafe { *lw.k_bias.add(i) }; }
                    build_rope_freqs(&mut bias_freqs, hd, s, model.rope_theta);
                    apply_rope(&mut rotated_biases[s * kv..(s + 1) * kv], &bias_freqs, hd, nkv);
                }
                // For each query token, compute dot with all bias positions
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

            // Sequential attention scoring (KV already in cache, seq_len = n)
            // attention_scores/output use explicit seq_len param, not cache.seq_len
            for t in 0..n {
                let q = &qs_all[t*h..(t+1)*h];
                let seq_len = t + 1;
                let scores = &mut self.attn_scores[..nh * seq_len];
                cache::attention::attention_scores(&self.kv_cache, q, layer as i32, nh as i32, nkv as i32, seq_len as i32, scores);
                if has_k_bias {
                    for q_h in 0..nh {
                        for s in 0..seq_len {
                            scores[q_h * seq_len + s] += k_bias_dots[t * nh * n + q_h * n + s];
                        }
                    }
                }
                softmax_rows(scores, nh, seq_len);
                let attn = &mut attn_all[t*h..(t+1)*h];
                cache::attention::attention_output(&self.kv_cache, scores, layer as i32, nh as i32, nkv as i32, seq_len as i32, attn);
            }

            // ── Parallel quantize attn output ──
            {
                let w = W { xs: 0, xs_mut: 0, norm: 0, buf: attn_all.as_ptr() as usize, bq: &mut bq_h as *mut BatchQ8K as usize, h, f, n };
                self.pool.run(nt.min(n), move |tid, nt_used| unsafe {
                    let buf = w.buf as *const f32;
                    let mut t = tid;
                    while t < w.n {
                        (*(w.bq as *mut BatchQ8K)).quantize(t, std::slice::from_raw_parts(buf.add(t * w.h), w.h));
                        t += nt_used;
                    }
                });
            }

            q4k_gemm_mt(lw.wo, h_rs, h_nb, &bq_h, &mut tmp_all, h, &self.pool);

            // ── Parallel vecadd residual (attn) ──
            {
                let xs_raw: Vec<*mut f32> = xs.iter_mut().map(|x| x.as_mut_ptr()).collect();
                let w = W { xs: 0, xs_mut: xs_raw.as_ptr() as usize, norm: 0, buf: tmp_all.as_ptr() as usize, bq: 0, h, f, n };
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

            // ── Parallel rmsnorm + quantize (ffn input) ──
            {
                let xs_raw: Vec<*const f32> = xs.iter().map(|x| x.as_ptr()).collect();
                let w = W { xs: xs_raw.as_ptr() as usize, xs_mut: 0, norm: lw.ffn_norm as usize, buf: 0, bq: &mut bq_h as *mut BatchQ8K as usize, h, f, n };
                let eps = model.rms_eps;
                self.pool.run(nt.min(n), move |tid, nt_used| unsafe {
                    let mut nbuf = vec![0.0f32; w.h];
                    let xs = w.xs as *const *const f32;
                    let mut t = tid;
                    while t < w.n {
                        ffi::rmsnorm_f32(*xs.add(t), w.norm as *const f32, nbuf.as_mut_ptr(), w.h as i32, eps);
                        (*(w.bq as *mut BatchQ8K)).quantize(t, &nbuf);
                        t += nt_used;
                    }
                });
            }

            q4k_fused_silu_gemm_mt(lw.w_gate, lw.w_up, h_rs, h_nb, &bq_h, &mut hidden_all, f, &self.pool);

            // ── Parallel quantize hidden ──
            {
                let w = W { xs: 0, xs_mut: 0, norm: 0, buf: hidden_all.as_ptr() as usize, bq: &mut bq_f as *mut BatchQ8K as usize, h, f, n };
                self.pool.run(nt.min(n), move |tid, nt_used| unsafe {
                    let buf = w.buf as *const f32;
                    let mut t = tid;
                    while t < w.n {
                        (*(w.bq as *mut BatchQ8K)).quantize(t, std::slice::from_raw_parts(buf.add(t * w.f), w.f));
                        t += nt_used;
                    }
                });
            }

            if lw.w_down_block_bytes == Q6K_BLOCK_BYTES {
                q6k_gemm_mt(lw.w_down, f_nb * Q6K_BLOCK_BYTES, f_nb, &bq_f, &mut tmp_all, h, &self.pool);
            } else {
                q4k_gemm_mt(lw.w_down, f_nb * Q4K_BLOCK_BYTES, f_nb, &bq_f, &mut tmp_all, h, &self.pool);
            }

            // ── Parallel vecadd residual (ffn) ──
            {
                let xs_raw: Vec<*mut f32> = xs.iter_mut().map(|x| x.as_mut_ptr()).collect();
                let w = W { xs: 0, xs_mut: xs_raw.as_ptr() as usize, norm: 0, buf: tmp_all.as_ptr() as usize, bq: 0, h, f, n };
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

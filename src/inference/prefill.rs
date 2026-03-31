//! Batched prefill: load each layer's weights once, multiply all prompt tokens.

use crate::kernels::ffi_inference as ffi;
use crate::inference::forward::{apply_rope, build_rope_freqs, softmax_rows, InferenceState};
use crate::inference::gemm_i2s::{BatchI8, i2s_gemm_mt, i2s_fused_sqrelu_gemm_mt};
use crate::inference::matmul::{embed_f16_lookup, i8_output_matmul_mt};
#[cfg(target_arch = "aarch64")]
use crate::inference::matmul::i8_output_matmul_speculative;
use crate::inference::engine::BitNetModel;
use crate::inference::cache;

impl InferenceState {
    pub fn prefill(&mut self, model: &BitNetModel, tokens: &[u32]) {
        let n = tokens.len();
        let (h, hd, nh, nkv, kv, f) = (
            model.hidden_dim, model.head_dim, model.n_heads,
            model.n_kv_heads, model.kv_dim, model.ffn_dim,
        );
        let mut xs: Vec<Vec<f32>> = tokens.iter().map(|&tok| {
            let mut x = vec![0.0f32; h];
            embed_f16_lookup(model.embed_weight_f16, tok, &mut x, h);
            x
        }).collect();

        let mut batch_h = BatchI8::new(n, h);
        let mut batch_f = BatchI8::new(n, f);
        let (mut qs_all, mut ks_all, mut vs_all) = (
            vec![0.0f32; n * h], vec![0.0f32; n * kv], vec![0.0f32; n * kv],
        );
        let (mut attn_all, mut tmp_all, mut hidden_all) = (
            vec![0.0f32; n * h], vec![0.0f32; n * h], vec![0.0f32; n * f],
        );

        for layer in 0..model.n_layers {
            let lw = &model.layers[layer];

            for t in 0..n {
                unsafe {
                    ffi::rmsnorm_f32(
                        xs[t].as_ptr(), lw.attn_norm,
                        self.x_norm.as_mut_ptr(), h as i32, model.rms_eps,
                    );
                }
                batch_h.quantize(t, &self.x_norm);
            }

            i2s_gemm_mt(lw.wq, lw.wq_scale, &batch_h, &mut qs_all, h, h, &self.pool);
            i2s_gemm_mt(lw.wk, lw.wk_scale, &batch_h, &mut ks_all, kv, h, &self.pool);
            i2s_gemm_mt(lw.wv, lw.wv_scale, &batch_h, &mut vs_all, kv, h, &self.pool);

            // Apply RoPE to all tokens, then bulk KV append
            for t in 0..n {
                let q = &mut qs_all[t * h..(t + 1) * h];
                let k = &mut ks_all[t * kv..(t + 1) * kv];
                build_rope_freqs(&mut self.rope_freqs, hd, t, model.rope_theta);
                apply_rope(q, &self.rope_freqs, hd, nh);
                apply_rope(k, &self.rope_freqs, hd, nkv);
            }
            let mut k_tr = vec![0.0f32; nkv * n * hd];
            let mut v_tr = vec![0.0f32; nkv * n * hd];
            for t in 0..n {
                for head in 0..nkv {
                    let src = t * kv + head * hd;
                    let dst = head * n * hd + t * hd;
                    k_tr[dst..dst + hd].copy_from_slice(&ks_all[src..src + hd]);
                    v_tr[dst..dst + hd].copy_from_slice(&vs_all[src..src + hd]);
                }
            }
            self.kv_cache.restore(0).unwrap();
            self.kv_cache.append(&k_tr, layer as i32, 0, n as i32).unwrap();
            self.kv_cache.append(&v_tr, layer as i32, 1, n as i32).unwrap();
            self.kv_cache.advance(n as i32).unwrap();

            // Parallel attention scoring
            {
                let pool_n = self.pool.thread_count().min(n);
                let cache_ptr = &self.kv_cache as *const cache::EakvCache as usize;
                let qs_ptr = qs_all.as_ptr() as usize;
                let attn_ptr = attn_all.as_mut_ptr() as usize;
                let layer_i32 = layer as i32;
                let nh_i32 = nh as i32;
                let nkv_i32 = nkv as i32;
                let n_tokens = n;
                self.pool.run(pool_n, move |tid, nt_used| {
                    let mut t = tid;
                    while t < n_tokens {
                        let seq_len = (t + 1) as i32;
                        let mut scores = vec![0.0f32; nh * (t + 1)];
                        let q = unsafe { std::slice::from_raw_parts((qs_ptr as *const f32).add(t * h), h) };
                        let cache_ref = unsafe { &*(cache_ptr as *const cache::EakvCache) };
                        cache::attention::attention_scores(
                            cache_ref, q, layer_i32, nh_i32, nkv_i32, seq_len, &mut scores,
                        );
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

            for t in 0..n {
                unsafe {
                    ffi::rmsnorm_f32(
                        attn_all[t * h..].as_ptr(), lw.attn_sub_norm,
                        attn_all[t * h..].as_mut_ptr(), h as i32, model.rms_eps,
                    );
                }
                batch_h.quantize(t, &attn_all[t * h..(t + 1) * h]);
            }
            i2s_gemm_mt(lw.wo, lw.wo_scale, &batch_h, &mut tmp_all, h, h, &self.pool);

            for t in 0..n {
                unsafe {
                    ffi::vecadd_f32(
                        xs[t].as_ptr(), tmp_all[t * h..].as_ptr(),
                        xs[t].as_mut_ptr(), h as i32,
                    );
                }
            }

            for t in 0..n {
                unsafe {
                    ffi::rmsnorm_f32(
                        xs[t].as_ptr(), lw.ffn_norm,
                        self.x_norm.as_mut_ptr(), h as i32, model.rms_eps,
                    );
                }
                batch_h.quantize(t, &self.x_norm);
            }
            i2s_fused_sqrelu_gemm_mt(
                lw.w_gate, lw.w_gate_scale,
                lw.w_up, lw.w_up_scale,
                &batch_h, &mut hidden_all, f, h,
                &self.pool,
            );

            for t in 0..n {
                unsafe {
                    ffi::rmsnorm_f32(
                        hidden_all[t * f..].as_ptr(), lw.ffn_sub_norm,
                        hidden_all[t * f..].as_mut_ptr(), f as i32, model.rms_eps,
                    );
                }
                batch_f.quantize(t, &hidden_all[t * f..(t + 1) * f]);
            }
            i2s_gemm_mt(
                lw.w_down, lw.w_down_scale, &batch_f,
                &mut tmp_all, h, f,
                &self.pool,
            );

            for t in 0..n {
                unsafe {
                    ffi::vecadd_f32(
                        xs[t].as_ptr(), tmp_all[t * h..].as_ptr(),
                        xs[t].as_mut_ptr(), h as i32,
                    );
                }
            }
        }

        self.x[..h].copy_from_slice(&xs[n - 1]);
        unsafe {
            ffi::rmsnorm_f32(
                self.x.as_ptr(), model.norm_weight, self.x_norm.as_mut_ptr(),
                h as i32, model.rms_eps,
            );
        }
        #[cfg(target_arch = "aarch64")]
        {
            if !model.embed_sketch.is_empty() {
                i8_output_matmul_speculative(
                    &model.embed_weight_i8, &model.embed_row_scales,
                    &model.embed_sketch, model.embed_sketch_dim,
                    &self.x_norm, &mut self.logits,
                    model.vocab_size, h, &self.pool, &mut self.spec_work,
                );
            } else {
                i8_output_matmul_mt(
                    &model.embed_weight_i8, &model.embed_row_scales,
                    &self.x_norm, &mut self.logits,
                    model.vocab_size, h,
                    &self.pool,
                );
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        i8_output_matmul_mt(
            &model.embed_weight_i8, &model.embed_row_scales,
            &self.x_norm, &mut self.logits,
            model.vocab_size, h,
            &self.pool,
        );
    }
}

//! Batched prefill: load each layer's weights once, multiply all prompt tokens.

use crate::kernels::ffi_inference as ffi;
use crate::inference::forward::{apply_rope, build_rope_freqs, InferenceState};
use crate::inference::gemm_i2s::{BatchI8, i2s_gemm_mt, i2s_fused_sqrelu_gemm_mt};
use crate::inference::matmul::{embed_f16_lookup, i8_output_matmul_mt};
#[cfg(target_arch = "aarch64")]
use crate::inference::matmul::i8_output_matmul_speculative;
use crate::inference::engine::BitNetModel;
use crate::kernels::ffi_inference as ffi_inf;

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
            // KV store: token-major, F16KvCache handles scatter
            self.kv_cache.restore(0).unwrap();
            self.kv_cache.store(layer, 0, &ks_all[..n*kv], n).unwrap();
            self.kv_cache.store(layer, 1, &vs_all[..n*kv], n).unwrap();
            self.kv_cache.advance(n).unwrap();

            // Per-token f16 attention
            let q_per_kv = nh / nkv;
            let rsqrt_hd = 1.0 / (hd as f32).sqrt();
            for t in 0..n {
                let causal_len = t + 1;
                let qt = &qs_all[t*h..(t+1)*h];
                for kv_h in 0..nkv {
                    let k_ptr = self.kv_cache.k_head_ptr(layer, kv_h);
                    let v_ptr = self.kv_cache.v_head_ptr(layer, kv_h);
                    for q_off in 0..q_per_kv {
                        let q_h = kv_h * q_per_kv + q_off;
                        let mut scores = vec![0.0f32; causal_len];
                        unsafe {
                            ffi_inf::attn_dot_f16(
                                qt[q_h * hd..].as_ptr(), k_ptr,
                                scores.as_mut_ptr(), causal_len as i32, hd as i32,
                            );
                            ffi_inf::softmax_f32(scores.as_mut_ptr(), causal_len as i32, rsqrt_hd);
                            ffi_inf::attn_vsum_f16(
                                scores.as_ptr(), v_ptr,
                                attn_all[t*h + q_h*hd..].as_mut_ptr(),
                                causal_len as i32, hd as i32,
                            );
                        }
                    }
                }
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

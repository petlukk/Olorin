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

impl LlamaState {
    /// GEMM-style batched prefill: load weight once, multiply all tokens.
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

        for layer in 0..model.n_layers {
            let lw = &model.q4k_layers[layer];
            for t in 0..n {
                unsafe { ffi::rmsnorm_f32(xs[t].as_ptr(), lw.attn_norm, self.x_norm_pub.as_mut_ptr(), h as i32, model.rms_eps); }
                bq_h.quantize(t, &self.x_norm_pub);
            }
            q4k_gemm_mt(lw.wq, h_rs, h_nb, &bq_h, &mut qs_all, h);
            q4k_gemm_mt(lw.wk, h_rs, h_nb, &bq_h, &mut ks_all, kv);
            if lw.wv_block_bytes == Q6K_BLOCK_BYTES {
                q6k_gemm_mt(lw.wv, h_nb * Q6K_BLOCK_BYTES, h_nb, &bq_h, &mut vs_all, kv);
            } else {
                q4k_gemm_mt(lw.wv, h_rs, h_nb, &bq_h, &mut vs_all, kv);
            }
            for t in 0..n {
                let (q, k, v) = (
                    &mut qs_all[t*h..(t+1)*h],
                    &mut ks_all[t*kv..(t+1)*kv],
                    &mut vs_all[t*kv..(t+1)*kv],
                );
                add_bias(q, lw.q_bias, h);
                add_bias(k, lw.k_bias, kv);
                add_bias(v, lw.v_bias, kv);
                build_rope_freqs(&mut self.rope_freqs, hd, t, model.rope_theta);
                apply_rope(q, &self.rope_freqs, hd, nh);
                apply_rope(k, &self.rope_freqs, hd, nkv);
                self.kv_cache.append(k, layer as i32, 0, 1).unwrap();
                self.kv_cache.append(v, layer as i32, 1, 1).unwrap();
                if layer == model.n_layers - 1 { self.kv_cache.advance(1).unwrap(); }
                let seq_len = t + 1;
                let scores = &mut self.attn_scores[..nh * seq_len];
                cache::attention::attention_scores(&self.kv_cache, q, layer as i32, nh as i32, nkv as i32, scores);
                softmax_rows(scores, nh, seq_len);
                let attn = &mut attn_all[t*h..(t+1)*h];
                cache::attention::attention_output(&self.kv_cache, scores, layer as i32, nh as i32, nkv as i32, attn);
            }
            for t in 0..n { bq_h.quantize(t, &attn_all[t*h..(t+1)*h]); }
            q4k_gemm_mt(lw.wo, h_rs, h_nb, &bq_h, &mut tmp_all, h);
            for t in 0..n {
                unsafe { ffi::vecadd_f32(xs[t].as_ptr(), tmp_all[t*h..].as_ptr(), xs[t].as_mut_ptr(), h as i32); }
            }
            for t in 0..n {
                unsafe { ffi::rmsnorm_f32(xs[t].as_ptr(), lw.ffn_norm, self.x_norm_pub.as_mut_ptr(), h as i32, model.rms_eps); }
                bq_h.quantize(t, &self.x_norm_pub);
            }
            q4k_fused_silu_gemm_mt(lw.w_gate, lw.w_up, h_rs, h_nb, &bq_h, &mut hidden_all, f);
            for t in 0..n { bq_f.quantize(t, &hidden_all[t*f..(t+1)*f]); }
            if lw.w_down_block_bytes == Q6K_BLOCK_BYTES {
                q6k_gemm_mt(lw.w_down, f_nb * Q6K_BLOCK_BYTES, f_nb, &bq_f, &mut tmp_all, h);
            } else {
                q4k_gemm_mt(lw.w_down, f_nb * Q4K_BLOCK_BYTES, f_nb, &bq_f, &mut tmp_all, h);
            }
            for t in 0..n {
                unsafe { ffi::vecadd_f32(xs[t].as_ptr(), tmp_all[t*h..].as_ptr(), xs[t].as_mut_ptr(), h as i32); }
            }
        }
        self.x[..h].copy_from_slice(&xs[n - 1]);
        self.output_proj(model);
    }
}

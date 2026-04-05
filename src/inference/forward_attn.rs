//! Gemma 4 per-layer forward: attention + FFN, split from forward.rs.
//!
//! Matches llama.cpp gemma4-iswa.cpp pipeline exactly:
//!   attn_norm -> Q(+K+V if has_kv) -> QK_norm(weighted) -> V_bare_norm
//!   -> rope(with freq_factors for global) -> cache store -> attn(scale=1.0)
//!   -> wo -> post_attn_norm -> +inpL -> ffn_norm -> gelu*up -> down
//!   -> post_ffn_norm -> +attn_out_res -> out_scale

use crate::inference::engine::Gemma4Model;
use crate::inference::matmul;
use crate::kernels::ffi_inference;

use super::forward::{bare_rmsnorm, compute_rope_tables, l2_norm, Gemma4State};

impl Gemma4State {
    /// Full layer forward pass matching llama.cpp gemma4-iswa.cpp.
    pub fn layer_forward(
        &mut self,
        model: &Gemma4Model,
        il: usize,
        pos: usize,
        diag: bool,
    ) {
        let hd = model.hidden_dim;
        let n_heads = model.n_heads;
        let n_kv_heads = model.n_kv_heads;
        let gqa_ratio = n_heads / n_kv_heads;
        let lw = &model.layers[il];
        let head_dim = model.head_dim_k[il];
        let head_dim_v = model.head_dim_v[il];
        let has_kv = model.kv_shared_source[il].is_none();

        // RoPE params per layer type
        let n_rot = if model.is_swa[il] {
            model.rope_dim_swa
        } else {
            model.rope_dim_global
        };
        let rope_theta = if model.is_swa[il] {
            model.rope_theta_swa
        } else {
            model.rope_theta_global
        };
        // Global layers use freq_factors for proportional RoPE
        let freq_factors: Option<&[f32]> = if !model.is_swa[il] {
            model.rope_freqs.as_deref()
        } else {
            None
        };

        compute_rope_tables(
            &mut self.cos_table,
            &mut self.sin_table,
            pos,
            n_rot,
            rope_theta,
            freq_factors,
        );

        // ── 1. Pre-attention RMSNorm (weight+1) ─────────────────────
        ffi_inference::gemma4_rmsnorm(
            self.x.as_ptr(),
            lw.attn_norm,
            self.x_norm.as_mut_ptr(),
            hd as i32,
            model.rms_eps,
        );

        // Quantize normed input for matmul
        matmul::quant_input(
            &self.x_norm,
            &mut self.q8_qs,
            &mut self.q8_d,
            &mut self.q8_bsums,
        );

        if diag && il == 0 {
            eprintln!("[gemma4] L0 attn_norm L2={:.4} (llama.cpp: 452.89)", l2_norm(&self.x_norm[..hd]));
            eprintln!("[gemma4] L0 attn_norm first4=[{:.4},{:.4},{:.4},{:.4}] (llama.cpp: [-10.64,-8.44,1.21,-12.26])",
                self.x_norm[0], self.x_norm[1], self.x_norm[2], self.x_norm[3]);
        }

        // ── 2. Q projection ─────────────────────────────────────────
        matmul::matvec(
            lw.wq_dtype, lw.wq,
            &self.q8_qs, &self.q8_d, &self.q8_bsums,
            &mut self.q, &mut self.q6k_d_scratch,
            n_heads * head_dim, hd,
        );

        if diag && il == 0 {
            eprintln!("[gemma4] L0 Q_proj L2={:.4} head_dim={} n_heads={}", l2_norm(&self.q[..n_heads * head_dim]), head_dim, n_heads);
        }

        // ── 3. Q norm (per-head, weight+1) + RoPE ───────────────────
        if !lw.q_norm.is_null() {
            for h in 0..n_heads {
                let off = h * head_dim;
                ffi_inference::gemma4_rmsnorm(
                    self.q.as_ptr().wrapping_add(off),
                    lw.q_norm,
                    self.kv_f32_scratch.as_mut_ptr(),
                    head_dim as i32,
                    model.rms_eps,
                );
                self.q[off..off + head_dim]
                    .copy_from_slice(&self.kv_f32_scratch[..head_dim]);
            }
        }

        // RoPE on Q (uses n_rot, not full head_dim)
        ffi_inference::gemma4_rope(
            self.q.as_mut_ptr(),
            self.cos_table.as_ptr(),
            self.sin_table.as_ptr(),
            head_dim as i32,
            n_heads as i32,
        );

        // ── 4. K/V projection + norm + RoPE (only if layer has KV) ──
        if has_kv {
            let kv_dim = n_kv_heads * head_dim;
            let kv_dim_v = n_kv_heads * head_dim_v;

            matmul::matvec(
                lw.wk_dtype, lw.wk,
                &self.q8_qs, &self.q8_d, &self.q8_bsums,
                &mut self.k, &mut self.q6k_d_scratch,
                kv_dim, hd,
            );
            matmul::matvec(
                lw.wv_dtype, lw.wv,
                &self.q8_qs, &self.q8_d, &self.q8_bsums,
                &mut self.v, &mut self.q6k_d_scratch,
                kv_dim_v, hd,
            );

            // K norm (per-head, weight+1)
            if !lw.k_norm.is_null() {
                for h in 0..n_kv_heads {
                    let off = h * head_dim;
                    ffi_inference::gemma4_rmsnorm(
                        self.k.as_ptr().wrapping_add(off),
                        lw.k_norm,
                        self.kv_f32_scratch.as_mut_ptr(),
                        head_dim as i32,
                        model.rms_eps,
                    );
                    self.k[off..off + head_dim]
                        .copy_from_slice(&self.kv_f32_scratch[..head_dim]);
                }
            }

            // V norm: BARE RMSNorm (no weight! just normalize)
            for h in 0..n_kv_heads {
                let off = h * head_dim_v;
                bare_rmsnorm(
                    &mut self.v[off..off + head_dim_v],
                    model.rms_eps,
                );
            }

            // RoPE on K
            ffi_inference::gemma4_rope(
                self.k.as_mut_ptr(),
                self.cos_table.as_ptr(),
                self.sin_table.as_ptr(),
                head_dim as i32,
                n_kv_heads as i32,
            );

            // Store K, V in cache
            let kv_dim = n_kv_heads * head_dim;
            let kv_dim_v = n_kv_heads * head_dim_v;
            self.cache.store(il, &self.k[..kv_dim], &self.v[..kv_dim_v]);
        }

        if diag && il == 0 {
            eprintln!("[gemma4] L0 attn_norm L2={:.4} (llama.cpp: 452.89)", l2_norm(&self.x_norm[..hd]));
            eprintln!("[gemma4] L0 Q pre-norm L2={:.4}", l2_norm(&self.q[..n_heads * head_dim]));
            // After QK norm:
            eprintln!("[gemma4] L0 Qnorm L2={:.4} (llama.cpp: 44.55)", l2_norm(&self.q[..n_heads * head_dim]));
            if has_kv {
                eprintln!("[gemma4] L0 Knorm L2={:.4} (llama.cpp: 2.03)", l2_norm(&self.k[..n_kv_heads * head_dim]));
                eprintln!("[gemma4] L0 Vnorm L2={:.4} (llama.cpp: 16.00)", l2_norm(&self.v[..n_kv_heads * head_dim_v]));
            }
        }

        // ── 5. Attention (GQA, scale=1.0) ────────────────────────────
        // llama.cpp: f_attention_scale = 1.0 for Gemma4
        let attn_scale = 1.0f32;
        let attn_len = self.cache.attn_len(il);
        let k_ptr = self.cache.k_ptr(il);
        let v_ptr = self.cache.v_ptr(il);
        let kv_dim = n_kv_heads * head_dim;

        self.attention_decode(
            n_heads, n_kv_heads, gqa_ratio, head_dim,
            kv_dim, attn_len, attn_scale, k_ptr, v_ptr,
        );

        if diag && il == 0 {
            eprintln!("[gemma4] L0 kqv_out L2={:.4} first4=[{:.4},{:.4},{:.4},{:.4}]",
                l2_norm(&self.attn_out[..n_heads * head_dim]),
                self.attn_out[0], self.attn_out[1], self.attn_out[2], self.attn_out[3]);
        }

        // ── 6. Wo + post-attention norm + residual ───────────────────
        let attn_out_dim = n_heads * head_dim;
        matmul::quant_input(
            &self.attn_out[..attn_out_dim],
            &mut self.q8_qs,
            &mut self.q8_d,
            &mut self.q8_bsums,
        );
        matmul::matvec(
            lw.wo_dtype, lw.wo,
            &self.q8_qs, &self.q8_d, &self.q8_bsums,
            &mut self.wo_out, &mut self.q6k_d_scratch,
            hd, attn_out_dim,
        );

        // post_attn_norm(wo_out) + inpL -> attn_res
        if !lw.post_attn_norm.is_null() {
            ffi_inference::gemma4_rmsnorm(
                self.wo_out.as_ptr(),
                lw.post_attn_norm,
                self.x_norm.as_mut_ptr(),
                hd as i32,
                model.rms_eps,
            );
            for i in 0..hd {
                self.attn_res[i] = self.x_norm[i] + self.x[i];
            }
        } else {
            for i in 0..hd {
                self.attn_res[i] = self.wo_out[i] + self.x[i];
            }
        }

        if diag && il == 0 {
            eprintln!("[gemma4] L0 post-attn L2={:.4} first4=[{:.4},{:.4},{:.4},{:.4}]",
                l2_norm(&self.attn_res[..hd]),
                self.attn_res[0], self.attn_res[1], self.attn_res[2], self.attn_res[3]);
        }

        // ── 7. FFN ───────────────────────────────────────────────────
        // Pre-FFN RMSNorm on attn_res
        ffi_inference::gemma4_rmsnorm(
            self.attn_res.as_ptr(),
            lw.ffn_norm,
            self.x_norm.as_mut_ptr(),
            hd as i32,
            model.rms_eps,
        );

        self.ffn(model, il);

        if diag && il == 0 {
            eprintln!("[gemma4] L0 ffn_out L2={:.4} first4=[{:.4},{:.4},{:.4},{:.4}]",
                l2_norm(&self.down[..hd]),
                self.down[0], self.down[1], self.down[2], self.down[3]);
        }

        // ── 8. Post-FFN norm + residual ──────────────────────────────
        // Residual is with attn_res (NOT inpL!)
        if !lw.post_ffn_norm.is_null() {
            ffi_inference::gemma4_rmsnorm(
                self.down.as_ptr(),
                lw.post_ffn_norm,
                self.x_norm.as_mut_ptr(),
                hd as i32,
                model.rms_eps,
            );
            for i in 0..hd {
                self.x[i] = self.x_norm[i] + self.attn_res[i];
            }
        } else {
            for i in 0..hd {
                self.x[i] = self.down[i] + self.attn_res[i];
            }
        }

        // ── 9. PLE ──────────────────────────────────────────────────
        if model.ple_dim > 0 && !lw.inp_gate.is_null() && !lw.proj.is_null() {
            let ple_dim = model.ple_dim;
            let ple_off = il * ple_dim;

            // Down-project: inp_gate @ x → ple_gate[ple_dim]
            matmul::quant_input(
                &self.x[..hd],
                &mut self.q8_qs,
                &mut self.q8_d,
                &mut self.q8_bsums,
            );
            matmul::matvec(
                lw.inp_gate_dtype, lw.inp_gate,
                &self.q8_qs, &self.q8_d, &self.q8_bsums,
                &mut self.ple_gate, &mut self.q6k_d_scratch,
                ple_dim, hd,
            );

            // GELU(gate) * ple_signal_slice
            ffi_inference::gelu_mul(
                self.ple_gate.as_ptr(),
                self.ple_signal[ple_off..].as_ptr(),
                self.ple_gate.as_mut_ptr(),
                ple_dim as i32,
            );

            // Up-project: proj @ gated → ple_out[hidden_dim]
            matmul::quant_input(
                &self.ple_gate[..ple_dim],
                &mut self.ple_q8_qs,
                &mut self.ple_q8_d,
                &mut self.ple_q8_bsums,
            );
            matmul::matvec(
                lw.proj_dtype, lw.proj,
                &self.ple_q8_qs, &self.ple_q8_d, &self.ple_q8_bsums,
                &mut self.ple_out, &mut self.q6k_d_scratch,
                hd, ple_dim,
            );

            // RMSNorm + residual add
            if !lw.post_norm.is_null() {
                ffi_inference::gemma4_rmsnorm(
                    self.ple_out.as_ptr(),
                    lw.post_norm,
                    self.ple_out.as_mut_ptr(),
                    hd as i32,
                    model.rms_eps,
                );
            }
            for i in 0..hd {
                self.x[i] += self.ple_out[i];
            }
        }

        // ── 10. Layer output scale ───────────────────────────────────
        let out_scale = lw.layer_output_scale;
        if out_scale != 1.0 {
            for v in self.x[..hd].iter_mut() {
                *v *= out_scale;
            }
        }

        if diag {
            eprintln!(
                "[gemma4] L{il} out L2={:.4} scale={:.6}",
                l2_norm(&self.x[..hd]), out_scale
            );
        }
    }

    /// GQA attention decode: Q dot K -> softmax(scale) -> weighted V sum.
    pub(crate) fn attention_decode(
        &mut self,
        n_heads: usize,
        _n_kv_heads: usize,
        gqa_ratio: usize,
        head_dim: usize,
        kv_dim: usize,
        attn_len: usize,
        scale: f32,
        k_ptr: *const u16,
        v_ptr: *const u16,
    ) {
        let stride = kv_dim;

        for h in 0..n_heads {
            let kv_h = h / gqa_ratio;
            let q_off = h * head_dim;
            let q_slice = &self.q[q_off..q_off + head_dim];

            // Q dot K for each cached position
            for p in 0..attn_len {
                let k_offset = p * stride + kv_h * head_dim;
                let k_src = unsafe { k_ptr.add(k_offset) };
                unsafe {
                    ffi_inference::f16_to_f32(
                        k_src,
                        self.kv_f32_scratch.as_mut_ptr(),
                        head_dim as i32,
                    );
                }
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_slice[d] * self.kv_f32_scratch[d];
                }
                self.attn_scores[p] = dot;
            }

            // Softmax with scale (1.0 for Gemma4)
            unsafe {
                ffi_inference::softmax_f32(
                    self.attn_scores.as_mut_ptr(),
                    attn_len as i32,
                    scale,
                );
            }

            // Weighted V sum
            let out_off = q_off;
            for d in 0..head_dim {
                self.attn_out[out_off + d] = 0.0;
            }
            for p in 0..attn_len {
                let v_offset = p * stride + kv_h * head_dim;
                let v_src = unsafe { v_ptr.add(v_offset) };
                unsafe {
                    ffi_inference::f16_to_f32(
                        v_src,
                        self.kv_f32_scratch.as_mut_ptr(),
                        head_dim as i32,
                    );
                }
                let s = self.attn_scores[p];
                for d in 0..head_dim {
                    self.attn_out[out_off + d] += s * self.kv_f32_scratch[d];
                }
            }
        }
    }

    /// FFN: GeGLU — gate/up dual matmul, GELU(gate)*up, down projection.
    pub(crate) fn ffn(&mut self, model: &Gemma4Model, layer: usize) {
        let hd = model.hidden_dim;
        let ffn_dim = model.ffn_dim[layer];
        let lw = &model.layers[layer];

        // Quantize x_norm for gate/up matmul
        matmul::quant_input(
            &self.x_norm,
            &mut self.q8_qs,
            &mut self.q8_d,
            &mut self.q8_bsums,
        );

        // Gate + up projection (fused when both Q4K, separate otherwise)
        if lw.w_gate_dtype == matmul::GGML_TYPE_Q4_K && lw.w_up_dtype == matmul::GGML_TYPE_Q4_K {
            matmul::q4k_matvec_dual(
                lw.w_gate,
                lw.w_up,
                &self.q8_qs, &self.q8_d, &self.q8_bsums,
                &mut self.gate, &mut self.up,
                ffn_dim, hd,
            );
        } else {
            matmul::matvec(
                lw.w_gate_dtype, lw.w_gate,
                &self.q8_qs, &self.q8_d, &self.q8_bsums,
                &mut self.gate, &mut self.q6k_d_scratch,
                ffn_dim, hd,
            );
            matmul::matvec(
                lw.w_up_dtype, lw.w_up,
                &self.q8_qs, &self.q8_d, &self.q8_bsums,
                &mut self.up, &mut self.q6k_d_scratch,
                ffn_dim, hd,
            );
        }

        // GELU(gate) * up
        ffi_inference::gelu_mul(
            self.gate.as_ptr(),
            self.up.as_ptr(),
            self.gate.as_mut_ptr(),
            ffn_dim as i32,
        );

        // Quantize gate (ffn_dim) for down projection
        matmul::quant_input(
            &self.gate[..ffn_dim],
            &mut self.ffn_q8_qs,
            &mut self.ffn_q8_d,
            &mut self.ffn_q8_bsums,
        );

        // Down projection: ffn_dim -> hidden_dim
        matmul::matvec(
            lw.w_down_dtype, lw.w_down,
            &self.ffn_q8_qs, &self.ffn_q8_d, &self.ffn_q8_bsums,
            &mut self.down, &mut self.q6k_d_scratch,
            hd, ffn_dim,
        );
    }
}

//! Gemma 4 batched per-layer forward (prompt eval).
//!
//! Mirrors `forward_attn.rs::layer_forward` stage-for-stage but operates on
//! `n_batch` prompt tokens in column-major layout. Per-column matmuls use
//! the existing `par_matvec` against unrepacked weights; the batched kernels
//! from Tasks 13–18 (rmsnorm, rope, gelu_mul, q8k_quant, attention,
//! KvCache::store_batch) are invoked wherever a single call can replace an
//! N-fold loop.
//!
//! Correctness contract: for `n_batch == 1` this must be bit-close to
//! `layer_forward` on the same inputs; for `n_batch > 1` the last column's
//! output must be bit-close to N independent `layer_forward` calls.
//!
//! Known limitations (intentional for Task 19 option-α):
//!   1. Weights are still in their mmap layout — gemm via repacked 8x8
//!      kernels is deferred to Task 20.
//!   2. Sliding-window causal cap assumes `pos_base + n_batch <= window`.
//!      Long-prompt chunking handles this at the forward_batch layer.

use crate::inference::engine::Gemma4Model;
use crate::inference::matmul;
use crate::kernels::ffi_inference;

use super::forward::{compute_rope_tables, Gemma4State};

impl Gemma4State {
    /// Full batched layer forward pass. `pos_base` is the sequence position
    /// of column 0; column k is at position `pos_base + k`.
    pub fn layer_forward_batch(
        &mut self,
        model: &Gemma4Model,
        il: usize,
        pos_base: usize,
        n_batch: usize,
        pool: &crate::inference::threadpool::ThreadPool,
    ) {
        let hd = model.hidden_dim;
        let n_heads = model.n_heads;
        let n_kv_heads = model.n_kv_heads;
        let gqa_ratio = n_heads / n_kv_heads;
        let lw = &model.layers[il];
        let head_dim = model.head_dim_k[il];
        let head_dim_v = model.head_dim_v[il];
        let has_kv = model.kv_shared_source[il].is_none();
        let q_stride = n_heads * head_dim;
        let kv_stride = n_kv_heads * head_dim;
        let kv_stride_v = n_kv_heads * head_dim_v;

        // RoPE params (layer-type dependent)
        let n_rot = if model.is_swa[il] { model.rope_dim_swa } else { model.rope_dim_global };
        let rope_theta = if model.is_swa[il] { model.rope_theta_swa } else { model.rope_theta_global };
        let freq_factors: Option<&[f32]> = if !model.is_swa[il] {
            model.rope_freqs.as_deref()
        } else {
            None
        };
        let half = n_rot / 2;

        // ── Build per-token cos/sin tables ───────────────────────────
        for k in 0..n_batch {
            let off = k * half;
            let (cos_slot, sin_slot) = (&mut self.batch_cos[off..off + half], &mut self.batch_sin[off..off + half]);
            // Borrow-checker: split via split_at_mut is unnecessary since
            // the two fields are disjoint. compute_rope_tables takes both
            // as &mut [f32] from the same struct but different fields —
            // OK via disjoint-field borrows when accessed directly.
            let _ = (cos_slot, sin_slot);
            // Redo with direct field access to satisfy NLL:
            compute_rope_tables(
                &mut self.batch_cos[off..off + half],
                &mut self.batch_sin[off..off + half],
                pos_base + k,
                n_rot,
                rope_theta,
                freq_factors,
            );
        }

        // ── 1. Batched pre-attention RMSNorm ─────────────────────────
        unsafe {
            ffi_inference::gemma4_rmsnorm_batched(
                self.batch_x.as_ptr(),
                lw.attn_norm,
                self.batch_x_norm.as_mut_ptr(),
                hd as i32,
                model.rms_eps,
                n_batch as i32,
            );
        }

        // ── 2. Per-column Q (+K/V) projection + per-head norms ──────
        for k in 0..n_batch {
            let x_off = k * hd;
            // Quantize this column's normed input once.
            matmul::quant_input(
                &self.batch_x_norm[x_off..x_off + hd],
                &mut self.q8_qs,
                &mut self.q8_d,
                &mut self.q8_bsums,
            );

            // Q projection into state.q (per-column scratch)
            matmul::par_matvec(
                pool,
                lw.wq_dtype,
                lw.wq,
                &self.q8_qs,
                &self.q8_d,
                &self.q8_bsums,
                &mut self.q,
                &mut self.q6k_d_scratch,
                q_stride,
                hd,
            );
            if !lw.q_norm.is_null() {
                super::forward_attn_heads::q_norm_per_head(
                    self, lw.q_norm, n_heads, head_dim, model.rms_eps, pool,
                );
            }
            // Copy Q column into batch_q
            let q_off = k * q_stride;
            self.batch_q[q_off..q_off + q_stride]
                .copy_from_slice(&self.q[..q_stride]);

            if has_kv {
                matmul::par_matvec(
                    pool,
                    lw.wk_dtype,
                    lw.wk,
                    &self.q8_qs,
                    &self.q8_d,
                    &self.q8_bsums,
                    &mut self.k,
                    &mut self.q6k_d_scratch,
                    kv_stride,
                    hd,
                );
                matmul::par_matvec(
                    pool,
                    lw.wv_dtype,
                    lw.wv,
                    &self.q8_qs,
                    &self.q8_d,
                    &self.q8_bsums,
                    &mut self.v,
                    &mut self.q6k_d_scratch,
                    kv_stride_v,
                    hd,
                );
                if !lw.k_norm.is_null() {
                    super::forward_attn_heads::k_norm_per_head(
                        self, lw.k_norm, n_kv_heads, head_dim, model.rms_eps, pool,
                    );
                }
                super::forward_attn_heads::v_bare_norm_per_head(
                    self, n_kv_heads, head_dim_v, model.rms_eps, pool,
                );

                let k_off = k * kv_stride;
                let v_off = k * kv_stride_v;
                self.batch_k[k_off..k_off + kv_stride]
                    .copy_from_slice(&self.k[..kv_stride]);
                self.batch_v[v_off..v_off + kv_stride_v]
                    .copy_from_slice(&self.v[..kv_stride_v]);
            }
        }

        // ── 3. Batched RoPE on Q (and K if has_kv) ───────────────────
        unsafe {
            ffi_inference::gemma4_rope_batched(
                self.batch_q.as_mut_ptr(),
                self.batch_cos.as_ptr(),
                self.batch_sin.as_ptr(),
                head_dim as i32,
                n_heads as i32,
                n_batch as i32,
            );
            if has_kv {
                ffi_inference::gemma4_rope_batched(
                    self.batch_k.as_mut_ptr(),
                    self.batch_cos.as_ptr(),
                    self.batch_sin.as_ptr(),
                    head_dim as i32,
                    n_kv_heads as i32,
                    n_batch as i32,
                );
            }
        }

        // ── 4. KV cache store (batched) ──────────────────────────────
        if has_kv {
            self.cache.store_batch(
                il,
                &self.batch_k[..kv_stride * n_batch],
                &self.batch_v[..kv_stride_v * n_batch],
                n_batch,
            );
        }

        // ── 5. Batched attention (GQA, scale=1.0) ────────────────────
        let attn_scale = 1.0f32;
        let k_ptr = self.cache.k_ptr(il);
        let v_ptr = self.cache.v_ptr(il);
        super::forward_attn_heads::attention_decode_batch(
            self,
            n_heads,
            n_kv_heads,
            gqa_ratio,
            head_dim,
            kv_stride,
            pos_base,
            n_batch,
            attn_scale,
            k_ptr,
            v_ptr,
            pool,
        );

        // ── 6. Per-column Wo projection ──────────────────────────────
        for k in 0..n_batch {
            let attn_off = k * q_stride;
            matmul::quant_input(
                &self.batch_attn_out[attn_off..attn_off + q_stride],
                &mut self.q8_qs,
                &mut self.q8_d,
                &mut self.q8_bsums,
            );
            matmul::par_matvec(
                pool,
                lw.wo_dtype,
                lw.wo,
                &self.q8_qs,
                &self.q8_d,
                &self.q8_bsums,
                &mut self.wo_out,
                &mut self.q6k_d_scratch,
                hd,
                q_stride,
            );
            let wo_off = k * hd;
            self.batch_wo_out[wo_off..wo_off + hd]
                .copy_from_slice(&self.wo_out[..hd]);
        }

        // ── post_attn_norm (batched) + residual with batch_x ─────────
        if !lw.post_attn_norm.is_null() {
            unsafe {
                ffi_inference::gemma4_rmsnorm_batched(
                    self.batch_wo_out.as_ptr(),
                    lw.post_attn_norm,
                    self.batch_x_norm.as_mut_ptr(),
                    hd as i32,
                    model.rms_eps,
                    n_batch as i32,
                );
            }
            for k in 0..n_batch {
                let off = k * hd;
                unsafe {
                    let a = self.batch_x_norm.as_ptr().add(off);
                    let b = self.batch_x.as_ptr().add(off);
                    let out = self.batch_attn_res.as_mut_ptr().add(off);
                    (ffi_inference::vec_add_f32)(a, b, out, hd as i32);
                }
            }
        } else {
            for k in 0..n_batch {
                let off = k * hd;
                unsafe {
                    let a = self.batch_wo_out.as_ptr().add(off);
                    let b = self.batch_x.as_ptr().add(off);
                    let out = self.batch_attn_res.as_mut_ptr().add(off);
                    (ffi_inference::vec_add_f32)(a, b, out, hd as i32);
                }
            }
        }

        // ── 7. FFN: ffn_norm → gate/up → gelu_mul_batched → down ─────
        unsafe {
            ffi_inference::gemma4_rmsnorm_batched(
                self.batch_attn_res.as_ptr(),
                lw.ffn_norm,
                self.batch_x_norm.as_mut_ptr(),
                hd as i32,
                model.rms_eps,
                n_batch as i32,
            );
        }
        let ffn_dim = model.ffn_dim[il];
        for k in 0..n_batch {
            let x_off = k * hd;
            matmul::quant_input(
                &self.batch_x_norm[x_off..x_off + hd],
                &mut self.q8_qs,
                &mut self.q8_d,
                &mut self.q8_bsums,
            );
            if lw.w_gate_dtype == matmul::GGML_TYPE_Q4_K
                && lw.w_up_dtype == matmul::GGML_TYPE_Q4_K
            {
                matmul::par_q4k_matvec_dual(
                    pool,
                    lw.w_gate,
                    lw.w_up,
                    &self.q8_qs,
                    &self.q8_d,
                    &self.q8_bsums,
                    &mut self.gate,
                    &mut self.up,
                    ffn_dim,
                    hd,
                );
            } else {
                matmul::par_matvec(
                    pool,
                    lw.w_gate_dtype,
                    lw.w_gate,
                    &self.q8_qs,
                    &self.q8_d,
                    &self.q8_bsums,
                    &mut self.gate,
                    &mut self.q6k_d_scratch,
                    ffn_dim,
                    hd,
                );
                matmul::par_matvec(
                    pool,
                    lw.w_up_dtype,
                    lw.w_up,
                    &self.q8_qs,
                    &self.q8_d,
                    &self.q8_bsums,
                    &mut self.up,
                    &mut self.q6k_d_scratch,
                    ffn_dim,
                    hd,
                );
            }
            let g_off = k * ffn_dim;
            self.batch_gate[g_off..g_off + ffn_dim]
                .copy_from_slice(&self.gate[..ffn_dim]);
            self.batch_up[g_off..g_off + ffn_dim]
                .copy_from_slice(&self.up[..ffn_dim]);
        }

        unsafe {
            ffi_inference::gelu_mul_batched(
                self.batch_gate.as_ptr(),
                self.batch_up.as_ptr(),
                self.batch_gate.as_mut_ptr(),
                ffn_dim as i32,
                n_batch as i32,
            );
        }

        for k in 0..n_batch {
            let g_off = k * ffn_dim;
            matmul::quant_input(
                &self.batch_gate[g_off..g_off + ffn_dim],
                &mut self.ffn_q8_qs,
                &mut self.ffn_q8_d,
                &mut self.ffn_q8_bsums,
            );
            matmul::par_matvec(
                pool,
                lw.w_down_dtype,
                lw.w_down,
                &self.ffn_q8_qs,
                &self.ffn_q8_d,
                &self.ffn_q8_bsums,
                &mut self.down,
                &mut self.q6k_d_scratch,
                hd,
                ffn_dim,
            );
            let d_off = k * hd;
            self.batch_down[d_off..d_off + hd]
                .copy_from_slice(&self.down[..hd]);
        }

        // ── 8. post_ffn_norm + residual with attn_res → batch_x ──────
        if !lw.post_ffn_norm.is_null() {
            unsafe {
                ffi_inference::gemma4_rmsnorm_batched(
                    self.batch_down.as_ptr(),
                    lw.post_ffn_norm,
                    self.batch_x_norm.as_mut_ptr(),
                    hd as i32,
                    model.rms_eps,
                    n_batch as i32,
                );
            }
            for k in 0..n_batch {
                let off = k * hd;
                unsafe {
                    let a = self.batch_x_norm.as_ptr().add(off);
                    let b = self.batch_attn_res.as_ptr().add(off);
                    let out = self.batch_x.as_mut_ptr().add(off);
                    (ffi_inference::vec_add_f32)(a, b, out, hd as i32);
                }
            }
        } else {
            for k in 0..n_batch {
                let off = k * hd;
                unsafe {
                    let a = self.batch_down.as_ptr().add(off);
                    let b = self.batch_attn_res.as_ptr().add(off);
                    let out = self.batch_x.as_mut_ptr().add(off);
                    (ffi_inference::vec_add_f32)(a, b, out, hd as i32);
                }
            }
        }

        // ── 9. PLE (per-column) ──────────────────────────────────────
        if model.ple_dim > 0 && !lw.inp_gate.is_null() && !lw.proj.is_null() {
            let ple_dim = model.ple_dim;
            let n_layers = model.n_layers;
            let ple_off_layer = il * ple_dim;

            for k in 0..n_batch {
                let x_off = k * hd;
                matmul::quant_input(
                    &self.batch_x[x_off..x_off + hd],
                    &mut self.q8_qs,
                    &mut self.q8_d,
                    &mut self.q8_bsums,
                );
                matmul::par_matvec(
                    pool,
                    lw.inp_gate_dtype,
                    lw.inp_gate,
                    &self.q8_qs,
                    &self.q8_d,
                    &self.q8_bsums,
                    &mut self.ple_gate,
                    &mut self.q6k_d_scratch,
                    ple_dim,
                    hd,
                );

                // GELU(gate) * ple_signal_slice for this token/layer
                let ple_src_off = k * (ple_dim * n_layers) + ple_off_layer;
                unsafe {
                    let g_ptr = self.ple_gate.as_mut_ptr();
                    let sig_ptr = self.batch_ple_signal.as_ptr().add(ple_src_off);
                    ffi_inference::gelu_mul(g_ptr, sig_ptr, g_ptr, ple_dim as i32);
                }

                matmul::quant_input(
                    &self.ple_gate[..ple_dim],
                    &mut self.ple_q8_qs,
                    &mut self.ple_q8_d,
                    &mut self.ple_q8_bsums,
                );
                matmul::par_matvec(
                    pool,
                    lw.proj_dtype,
                    lw.proj,
                    &self.ple_q8_qs,
                    &self.ple_q8_d,
                    &self.ple_q8_bsums,
                    &mut self.ple_out,
                    &mut self.q6k_d_scratch,
                    hd,
                    ple_dim,
                );

                if !lw.post_norm.is_null() {
                    ffi_inference::gemma4_rmsnorm(
                        self.ple_out.as_ptr(),
                        lw.post_norm,
                        self.ple_out.as_mut_ptr(),
                        hd as i32,
                        model.rms_eps,
                    );
                }
                unsafe {
                    let x_ptr = self.batch_x.as_mut_ptr().add(x_off);
                    let ple_ptr = self.ple_out.as_ptr();
                    (ffi_inference::vec_add_f32)(x_ptr, ple_ptr, x_ptr, hd as i32);
                }
            }
        }

        // ── 10. Layer output scale ───────────────────────────────────
        let out_scale = lw.layer_output_scale;
        if out_scale != 1.0 {
            ffi_inference::vec_scale_f32(
                self.batch_x.as_ptr(),
                self.batch_x.as_mut_ptr(),
                out_scale,
                (hd * n_batch) as i32,
            );
        }
    }
}

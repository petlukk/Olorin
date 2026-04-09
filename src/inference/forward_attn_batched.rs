//! Batched per-layer forward: repacked Q4K gemm + batched kernels.
//! Mirrors `forward_attn.rs::layer_forward` stage-for-stage.
//! Non-Q4K weights fall back to per-column `par_matvec`.

use crate::inference::engine::Gemma4Model;
use crate::inference::matmul;
use crate::inference::threadpool::ThreadPool;
use crate::kernels::ffi_inference;
use super::forward::{compute_rope_tables, Gemma4State};

/// Batched matmul: Q4K gemm when repacked, else per-column par_matvec.
#[allow(clippy::too_many_arguments)]
fn matmul_batched(
    pool: &ThreadPool,
    dtype: u32,
    weight_unpacked: *const u8,
    weight_packed: &[u8],
    q8_qs: &[i8],
    q8_d: &[f32],
    q8_bsums: &[i16],
    gemm_scratch: &mut [u8],
    gemm_acc_scratch: &mut [f32],
    q6k_d_scratch: &mut [f32],
    output: &mut [f32],
    n_rows: usize,
    n_cols: usize,
    n_batch: usize,
) {
    debug_assert_eq!(output.len(), n_rows * n_batch);
    if dtype == matmul::GGML_TYPE_Q4_K && !weight_packed.is_empty() {
        let pow2 = matmul::pow2_table();
        unsafe {
            ffi_inference::q4k_8x8_q8k_gemm(
                weight_packed.as_ptr(),
                q8_qs.as_ptr(),
                q8_d.as_ptr(),
                q8_bsums.as_ptr(),
                pow2.as_ptr(),
                gemm_scratch.as_mut_ptr(),
                gemm_acc_scratch.as_mut_ptr(),
                output.as_mut_ptr(),
                n_rows as i32,
                n_cols as i32,
                n_batch as i32,
            );
        }
    } else {
        // Non-Q4K fallback: slice the batched Q8K input per-column and
        // dispatch N independent par_matvec calls. Bit-exact with the
        // Task-19 path that drove this module before Task 20.
        let n_blocks = n_cols / 256;
        let qs_stride = n_cols + 12;
        for k in 0..n_batch {
            let qs = &q8_qs[k * qs_stride..(k + 1) * qs_stride];
            let d = &q8_d[k * n_blocks..(k + 1) * n_blocks];
            let bsums = &q8_bsums[k * n_blocks * 16..(k + 1) * n_blocks * 16];
            let out_col = &mut output[k * n_rows..(k + 1) * n_rows];
            matmul::par_matvec(
                pool,
                dtype,
                weight_unpacked,
                qs,
                d,
                bsums,
                out_col,
                q6k_d_scratch,
                n_rows,
                n_cols,
            );
        }
    }
}

impl Gemma4State {
    /// Full batched layer forward pass.
    pub fn layer_forward_batch(
        &mut self,
        model: &Gemma4Model,
        il: usize,
        pos_base: usize,
        n_batch: usize,
        pool: &ThreadPool,
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
        let ffn_dim = model.ffn_dim[il];

        // ── 0. Repack Q4K weights into per-layer scratch ─────────────
        let (rp_off, rp_sz) = super::engine_helpers::repack_layer(
            &mut self.layer_repack, lw,
            n_heads, n_kv_heads, head_dim, head_dim_v, hd, ffn_dim,
        );
        // Helper: get packed slice for weight index (0=wq, 1=wk, etc.)
        macro_rules! packed {
            ($idx:expr) => {
                &self.layer_repack[rp_off[$idx]..rp_off[$idx] + rp_sz[$idx]]
            };
        }

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

        // ── 2. Quantize batch_x_norm once (shared by Q/K/V) ──────────
        unsafe {
            ffi_inference::q8k_quant_batched(
                self.batch_x_norm.as_ptr(),
                self.batch_q8_qs.as_mut_ptr(),
                self.batch_q8_d.as_mut_ptr(),
                self.batch_q8_bsums.as_mut_ptr(),
                hd as i32,
                n_batch as i32,
            );
        }

        // ── 3. Q projection (gemm or fallback) ───────────────────────
        matmul_batched(
            pool, lw.wq_dtype, lw.wq, packed!(0),
            &self.batch_q8_qs, &self.batch_q8_d, &self.batch_q8_bsums,
            &mut self.gemm_scratch, &mut self.gemm_acc_scratch, &mut self.q6k_d_scratch,
            &mut self.batch_q[..q_stride * n_batch],
            q_stride, hd, n_batch,
        );

        // Q norm per head per column (helper operates on self.q)
        if !lw.q_norm.is_null() {
            for k in 0..n_batch {
                let q_off = k * q_stride;
                self.q[..q_stride]
                    .copy_from_slice(&self.batch_q[q_off..q_off + q_stride]);
                super::forward_attn_heads::q_norm_per_head(
                    self, lw.q_norm, n_heads, head_dim, model.rms_eps, pool,
                );
                self.batch_q[q_off..q_off + q_stride]
                    .copy_from_slice(&self.q[..q_stride]);
            }
        }

        if has_kv {
            // K projection
            matmul_batched(
                pool, lw.wk_dtype, lw.wk, packed!(1),
                &self.batch_q8_qs, &self.batch_q8_d, &self.batch_q8_bsums,
                &mut self.gemm_scratch, &mut self.gemm_acc_scratch, &mut self.q6k_d_scratch,
                &mut self.batch_k[..kv_stride * n_batch],
                kv_stride, hd, n_batch,
            );
            // V projection
            matmul_batched(
                pool, lw.wv_dtype, lw.wv, packed!(2),
                &self.batch_q8_qs, &self.batch_q8_d, &self.batch_q8_bsums,
                &mut self.gemm_scratch, &mut self.gemm_acc_scratch, &mut self.q6k_d_scratch,
                &mut self.batch_v[..kv_stride_v * n_batch],
                kv_stride_v, hd, n_batch,
            );

            // K norm + V bare norm per column (helpers operate on self.k/self.v)
            if !lw.k_norm.is_null() {
                for k in 0..n_batch {
                    let off = k * kv_stride;
                    self.k[..kv_stride]
                        .copy_from_slice(&self.batch_k[off..off + kv_stride]);
                    super::forward_attn_heads::k_norm_per_head(
                        self, lw.k_norm, n_kv_heads, head_dim, model.rms_eps, pool,
                    );
                    self.batch_k[off..off + kv_stride]
                        .copy_from_slice(&self.k[..kv_stride]);
                }
            }
            for k in 0..n_batch {
                let off = k * kv_stride_v;
                self.v[..kv_stride_v]
                    .copy_from_slice(&self.batch_v[off..off + kv_stride_v]);
                super::forward_attn_heads::v_bare_norm_per_head(
                    self, n_kv_heads, head_dim_v, model.rms_eps, pool,
                );
                self.batch_v[off..off + kv_stride_v]
                    .copy_from_slice(&self.v[..kv_stride_v]);
            }
        }

        // ── 4. Batched RoPE on Q (and K if has_kv) ───────────────────
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

        // ── 5. KV cache store (batched) ──────────────────────────────
        if has_kv {
            self.cache.store_batch(
                il,
                &self.batch_k[..kv_stride * n_batch],
                &self.batch_v[..kv_stride_v * n_batch],
                n_batch,
            );
        }

        // ── 6. Batched attention (GQA, scale=1.0) ────────────────────
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

        // ── 7. Wo projection: quant attn_out then gemm/fallback ──────
        unsafe {
            ffi_inference::q8k_quant_batched(
                self.batch_attn_out.as_ptr(),
                self.batch_q8_qs.as_mut_ptr(),
                self.batch_q8_d.as_mut_ptr(),
                self.batch_q8_bsums.as_mut_ptr(),
                q_stride as i32,
                n_batch as i32,
            );
        }
        matmul_batched(
            pool, lw.wo_dtype, lw.wo, packed!(3),
            &self.batch_q8_qs, &self.batch_q8_d, &self.batch_q8_bsums,
            &mut self.gemm_scratch, &mut self.gemm_acc_scratch, &mut self.q6k_d_scratch,
            &mut self.batch_wo_out[..hd * n_batch],
            hd, q_stride, n_batch,
        );

        // post_attn_norm (batched) + residual with batch_x
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
                    ffi_inference::vec_add_f32(a, b, out, hd as i32);
                }
            }
        } else {
            for k in 0..n_batch {
                let off = k * hd;
                unsafe {
                    let a = self.batch_wo_out.as_ptr().add(off);
                    let b = self.batch_x.as_ptr().add(off);
                    let out = self.batch_attn_res.as_mut_ptr().add(off);
                    ffi_inference::vec_add_f32(a, b, out, hd as i32);
                }
            }
        }

        // ── 8. FFN: ffn_norm → gate/up → gelu_mul → down ─────────────
        unsafe {
            ffi_inference::gemma4_rmsnorm_batched(
                self.batch_attn_res.as_ptr(),
                lw.ffn_norm,
                self.batch_x_norm.as_mut_ptr(),
                hd as i32,
                model.rms_eps,
                n_batch as i32,
            );
            ffi_inference::q8k_quant_batched(
                self.batch_x_norm.as_ptr(),
                self.batch_q8_qs.as_mut_ptr(),
                self.batch_q8_d.as_mut_ptr(),
                self.batch_q8_bsums.as_mut_ptr(),
                hd as i32,
                n_batch as i32,
            );
        }
        // gate + up separately (dual-matvec optimization is single-column only)
        matmul_batched(
            pool, lw.w_gate_dtype, lw.w_gate, packed!(4),
            &self.batch_q8_qs, &self.batch_q8_d, &self.batch_q8_bsums,
            &mut self.gemm_scratch, &mut self.gemm_acc_scratch, &mut self.q6k_d_scratch,
            &mut self.batch_gate[..ffn_dim * n_batch],
            ffn_dim, hd, n_batch,
        );
        matmul_batched(
            pool, lw.w_up_dtype, lw.w_up, packed!(5),
            &self.batch_q8_qs, &self.batch_q8_d, &self.batch_q8_bsums,
            &mut self.gemm_scratch, &mut self.gemm_acc_scratch, &mut self.q6k_d_scratch,
            &mut self.batch_up[..ffn_dim * n_batch],
            ffn_dim, hd, n_batch,
        );

        unsafe {
            ffi_inference::gelu_mul_batched(
                self.batch_gate.as_ptr(),
                self.batch_up.as_ptr(),
                self.batch_gate.as_mut_ptr(),
                ffn_dim as i32,
                n_batch as i32,
            );
            // Quantize gated intermediate for down proj
            ffi_inference::q8k_quant_batched(
                self.batch_gate.as_ptr(),
                self.batch_q8_qs.as_mut_ptr(),
                self.batch_q8_d.as_mut_ptr(),
                self.batch_q8_bsums.as_mut_ptr(),
                ffn_dim as i32,
                n_batch as i32,
            );
        }
        matmul_batched(
            pool, lw.w_down_dtype, lw.w_down, packed!(6),
            &self.batch_q8_qs, &self.batch_q8_d, &self.batch_q8_bsums,
            &mut self.gemm_scratch, &mut self.gemm_acc_scratch, &mut self.q6k_d_scratch,
            &mut self.batch_down[..hd * n_batch],
            hd, ffn_dim, n_batch,
        );

        // ── 9. post_ffn_norm + residual with attn_res → batch_x ──────
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
                    ffi_inference::vec_add_f32(a, b, out, hd as i32);
                }
            }
        } else {
            for k in 0..n_batch {
                let off = k * hd;
                unsafe {
                    let a = self.batch_down.as_ptr().add(off);
                    let b = self.batch_attn_res.as_ptr().add(off);
                    let out = self.batch_x.as_mut_ptr().add(off);
                    ffi_inference::vec_add_f32(a, b, out, hd as i32);
                }
            }
        }

        // ── 10. PLE (per-column, uses its own Q8K scratch) ───────────
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
                    ffi_inference::vec_add_f32(x_ptr, ple_ptr, x_ptr, hd as i32);
                }
            }
        }

        // ── 11. Layer output scale ───────────────────────────────────
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

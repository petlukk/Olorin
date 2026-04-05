//! Gemma 4 forward pass — decode (single token).
//!
//! Pipeline: embed -> (rmsnorm -> qkv -> qk_norm -> rope -> kv -> attn -> wo ->
//!   post_attn_norm -> residual -> rmsnorm -> ffn -> post_ffn_norm -> residual) x N
//!   -> final rmsnorm -> output matmul -> logit_softcap -> logits

use crate::inference::cache::KvCache;
use crate::inference::engine::{AttnType, Gemma4Model};
use crate::inference::matmul;
use crate::kernels::ffi_inference;

// ---------------------------------------------------------------------------
// Diagnostic helper
// ---------------------------------------------------------------------------

fn diag_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("GEMMA4_DIAG").is_ok())
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ---------------------------------------------------------------------------
// RoPE table computation
// ---------------------------------------------------------------------------

fn compute_rope_tables(
    cos: &mut [f32],
    sin: &mut [f32],
    pos: usize,
    head_dim: usize,
    theta: f32,
) {
    let half = head_dim / 2;
    for d in 0..half {
        let freq = 1.0 / theta.powf(2.0 * d as f32 / head_dim as f32);
        let angle = pos as f32 * freq;
        cos[d] = angle.cos();
        sin[d] = angle.sin();
    }
}

// ---------------------------------------------------------------------------
// Gemma4State
// ---------------------------------------------------------------------------

pub struct Gemma4State {
    // Activation buffers
    pub(crate) x: Vec<f32>,
    pub(crate) x_norm: Vec<f32>,

    // Q8K quantized input (for matmul)
    pub(crate) q8_qs: Vec<i8>,
    pub(crate) q8_d: Vec<f32>,
    pub(crate) q8_bsums: Vec<i16>,

    // QKV buffers
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,

    // Attention
    pub(crate) attn_out: Vec<f32>,
    pub(crate) attn_scores: Vec<f32>,
    // Scratch for f16->f32 conversion of one KV head row
    pub(crate) kv_f32_scratch: Vec<f32>,

    // FFN
    pub(crate) gate: Vec<f32>,
    pub(crate) up: Vec<f32>,
    pub(crate) down: Vec<f32>,

    // FFN Q8K (for quantizing FFN intermediate)
    pub(crate) ffn_q8_qs: Vec<i8>,
    pub(crate) ffn_q8_d: Vec<f32>,
    pub(crate) ffn_q8_bsums: Vec<i16>,

    // Output
    pub(crate) logits: Vec<f32>,

    // Q6K d_scratch for output matmul (if output is Q6K)
    pub(crate) q6k_d_scratch: Vec<f32>,

    // RoPE tables
    pub(crate) cos_table: Vec<f32>,
    pub(crate) sin_table: Vec<f32>,

    // KV cache
    pub cache: KvCache,
}

impl Gemma4State {
    pub fn new(model: &Gemma4Model, max_seq_len: usize) -> Self {
        let hd = model.hidden_dim;
        // Use max dimensions across all layers for buffer allocation
        let max_head_k = *model.head_dim_k.iter().max().unwrap_or(&512);
        let max_head_v = *model.head_dim_v.iter().max().unwrap_or(&512);
        let max_head = max_head_k.max(max_head_v);
        let max_qkv = model.n_heads * max_head_k;
        let max_kv = model.n_kv_heads * max_head;
        let n_blocks_hd = hd / 256;
        let max_ffn = *model.ffn_dim.iter().max().unwrap_or(&12288);
        let n_blocks_ffn = max_ffn / 256;
        let n_blocks_out = hd / 256;

        // Build AttnType vec from is_swa for KvCache
        let attn_types: Vec<AttnType> = model.is_swa.iter().map(|&swa| {
            if swa { AttnType::SlidingWindow } else { AttnType::Global }
        }).collect();

        let cache = KvCache::new(
            model.n_layers,
            model.n_kv_heads,
            model.head_dim_v.clone(),
            model.sliding_window,
            max_seq_len,
            attn_types,
            model.kv_shared_source.clone(),
        );

        Self {
            x: vec![0.0; hd],
            x_norm: vec![0.0; hd],

            q8_qs: vec![0; hd + 12],
            q8_d: vec![0.0; n_blocks_hd],
            q8_bsums: vec![0; n_blocks_hd * 16],

            q: vec![0.0; max_qkv],
            k: vec![0.0; max_kv],
            v: vec![0.0; max_kv],

            attn_out: vec![0.0; max_qkv],
            attn_scores: vec![0.0; max_seq_len],
            kv_f32_scratch: vec![0.0; max_head],

            gate: vec![0.0; max_ffn],
            up: vec![0.0; max_ffn],
            down: vec![0.0; hd],

            ffn_q8_qs: vec![0; max_ffn + 12],
            ffn_q8_d: vec![0.0; n_blocks_ffn],
            ffn_q8_bsums: vec![0; n_blocks_ffn * 16],

            logits: vec![0.0; model.vocab_size],
            q6k_d_scratch: vec![0.0; n_blocks_out * 4],

            cos_table: vec![0.0; max_head / 2],
            sin_table: vec![0.0; max_head / 2],

            cache,
        }
    }

    /// Run one decode step. Returns logits slice.
    pub fn forward_one(&mut self, model: &Gemma4Model, token_id: u32) -> &[f32] {
        let hd = model.hidden_dim;
        let n_heads = model.n_heads;
        let n_kv_heads = model.n_kv_heads;
        let gqa_ratio = n_heads / n_kv_heads;
        let pos = self.cache.seq_len();
        let diag = diag_enabled();

        // 1. Embedding lookup (Q6K dequant)
        matmul::q6k_embed_lookup(model.embed_weight, token_id as usize, &mut self.x, hd);

        // Gemma scaling: multiply embedding by sqrt(hidden_dim)
        let embed_scale = (hd as f32).sqrt();
        for v in self.x.iter_mut() {
            *v *= embed_scale;
        }

        if diag {
            eprintln!("[gemma4] pos={pos} embed L2={:.4}", l2_norm(&self.x));
        }

        // 2. Per-layer transformer blocks
        for layer in 0..model.n_layers {
            let lw = &model.layers[layer];
            let head_dim = model.head_dim_k[layer];
            let head_dim_v = model.head_dim_v[layer];
            let kv_dim = n_kv_heads * head_dim;
            let kv_dim_v = n_kv_heads * head_dim_v;
            let scale = 1.0 / (head_dim as f32).sqrt();

            // RoPE: use layer-appropriate theta and dim
            let rope_theta = if model.is_swa[layer] {
                model.rope_theta_swa
            } else {
                model.rope_theta_global
            };
            compute_rope_tables(
                &mut self.cos_table,
                &mut self.sin_table,
                pos,
                head_dim,
                rope_theta,
            );

            // a. Pre-attention RMSNorm
            ffi_inference::gemma4_rmsnorm(
                self.x.as_ptr(),
                lw.attn_norm,
                self.x_norm.as_mut_ptr(),
                hd as i32,
                model.rms_eps,
            );

            // Quantize x_norm to Q8K for matmul
            matmul::quant_input(
                &self.x_norm,
                &mut self.q8_qs,
                &mut self.q8_d,
                &mut self.q8_bsums,
            );

            // b-d. QKV projections (dims vary per layer)
            matmul::q4k_matvec(
                lw.wq,
                &self.q8_qs, &self.q8_d, &self.q8_bsums,
                &mut self.q, n_heads * head_dim, hd,
            );
            matmul::q4k_matvec(
                lw.wk,
                &self.q8_qs, &self.q8_d, &self.q8_bsums,
                &mut self.k, kv_dim, hd,
            );
            matmul::q4k_matvec(
                lw.wv,
                &self.q8_qs, &self.q8_d, &self.q8_bsums,
                &mut self.v, kv_dim_v, hd,
            );

            // e. QK norm — per-head RMSNorm before RoPE
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

            if diag && layer == 0 {
                eprintln!(
                    "[gemma4] L0 Qnorm L2={:.4} Knorm L2={:.4}",
                    l2_norm(&self.q[..n_heads * head_dim]),
                    l2_norm(&self.k[..kv_dim]),
                );
            }

            // f. RoPE on Q and K
            ffi_inference::gemma4_rope(
                self.q.as_mut_ptr(),
                self.cos_table.as_ptr(),
                self.sin_table.as_ptr(),
                head_dim as i32,
                n_heads as i32,
            );
            ffi_inference::gemma4_rope(
                self.k.as_mut_ptr(),
                self.cos_table.as_ptr(),
                self.sin_table.as_ptr(),
                head_dim as i32,
                n_kv_heads as i32,
            );

            // g. KV cache store
            self.cache.store(layer, &self.k[..kv_dim], &self.v[..kv_dim_v]);

            // h. Attention (GQA, single-token decode)
            let attn_len = self.cache.attn_len(layer);
            let k_ptr = self.cache.k_ptr(layer);
            let v_ptr = self.cache.v_ptr(layer);

            self.attention_decode(
                n_heads, n_kv_heads, gqa_ratio, head_dim,
                kv_dim, attn_len, scale, k_ptr, v_ptr,
            );

            // i. Output projection
            let attn_out_dim = n_heads * head_dim;
            matmul::quant_input(
                &self.attn_out[..attn_out_dim],
                &mut self.q8_qs,
                &mut self.q8_d,
                &mut self.q8_bsums,
            );
            matmul::q4k_matvec(
                lw.wo,
                &self.q8_qs, &self.q8_d, &self.q8_bsums,
                &mut self.down, hd, attn_out_dim,
            );

            // j. Post-attention RMSNorm + residual add
            if !lw.post_attn_norm.is_null() {
                ffi_inference::gemma4_rmsnorm(
                    self.down.as_ptr(),
                    lw.post_attn_norm,
                    self.x_norm.as_mut_ptr(),
                    hd as i32,
                    model.rms_eps,
                );
                for i in 0..hd {
                    self.x[i] += self.x_norm[i];
                }
            } else {
                for i in 0..hd {
                    self.x[i] += self.down[i];
                }
            }

            if diag {
                eprintln!(
                    "[gemma4] L{layer} post-attn L2={:.4}",
                    l2_norm(&self.x[..hd])
                );
            }

            // k. Pre-FFN RMSNorm
            ffi_inference::gemma4_rmsnorm(
                self.x.as_ptr(),
                lw.ffn_norm,
                self.x_norm.as_mut_ptr(),
                hd as i32,
                model.rms_eps,
            );

            // l. FFN (GeGLU) — per-layer ffn_dim
            self.ffn(model, layer);

            // m. Post-FFN RMSNorm + residual add
            if !lw.post_ffn_norm.is_null() {
                ffi_inference::gemma4_rmsnorm(
                    self.down.as_ptr(),
                    lw.post_ffn_norm,
                    self.x_norm.as_mut_ptr(),
                    hd as i32,
                    model.rms_eps,
                );
                for i in 0..hd {
                    self.x[i] += self.x_norm[i];
                }
            } else {
                for i in 0..hd {
                    self.x[i] += self.down[i];
                }
            }

            if diag {
                eprintln!(
                    "[gemma4] L{layer} post-ffn L2={:.4}",
                    l2_norm(&self.x[..hd])
                );
            }
        }

        // 3. Final RMSNorm
        ffi_inference::gemma4_rmsnorm(
            self.x.as_ptr(),
            model.norm_weight,
            self.x_norm.as_mut_ptr(),
            hd as i32,
            model.rms_eps,
        );

        // 4. Output logits (tied to embedding, always Q6K for Gemma 4)
        matmul::quant_input(
            &self.x_norm,
            &mut self.q8_qs,
            &mut self.q8_d,
            &mut self.q8_bsums,
        );

        if model.embed_dtype == matmul::GGML_TYPE_Q6_K {
            matmul::q6k_matvec(
                model.embed_weight,
                &self.q8_qs, &self.q8_d, &self.q8_bsums,
                &mut self.logits,
                &mut self.q6k_d_scratch,
                model.vocab_size, hd,
            );
        } else {
            matmul::q4k_matvec(
                model.embed_weight,
                &self.q8_qs, &self.q8_d, &self.q8_bsums,
                &mut self.logits, model.vocab_size, hd,
            );
        }

        // Apply logit soft-capping if configured
        if model.logit_softcap > 0.0 {
            let cap = model.logit_softcap;
            let inv_cap = 1.0 / cap;
            for l in self.logits.iter_mut() {
                *l = cap * (*l * inv_cap).tanh();
            }
        }

        // Advance cache position
        self.cache.advance();

        &self.logits
    }

    /// Reset state for a new sequence.
    pub fn reset(&mut self) {
        self.cache.reset();
    }
}

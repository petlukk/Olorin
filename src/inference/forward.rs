//! Gemma 4 forward pass — decode (single token).
//!
//! Pipeline matches llama.cpp gemma4-iswa.cpp EXACTLY:
//!   embed * sqrt(n_embd) -> per-layer(attn_norm -> Q/K/V -> QK_norm -> V_bare_norm
//!   -> rope -> cache -> attn(scale=1.0) -> wo -> post_attn_norm -> +inpL
//!   -> ffn_norm -> gelu_gate*up -> down -> post_ffn_norm -> +attn_out
//!   -> out_scale) -> final_norm -> output_matmul -> softcap

use crate::inference::cache::KvCache;
use crate::inference::engine::{AttnType, Gemma4Model};
use crate::inference::matmul;
use crate::kernels::ffi_inference;

// ---------------------------------------------------------------------------
// Diagnostic helper
// ---------------------------------------------------------------------------

pub(crate) fn diag_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("GEMMA4_DIAG").is_ok())
}

pub(crate) fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ---------------------------------------------------------------------------
// RoPE table computation
// ---------------------------------------------------------------------------

/// Compute cos/sin tables for RoPE. If freq_factors is Some, each frequency
/// is divided by freq_factors[d] (matching llama.cpp's rope_ext behaviour).
pub(crate) fn compute_rope_tables(
    cos: &mut [f32],
    sin: &mut [f32],
    pos: usize,
    n_rot: usize,
    theta: f32,
    freq_factors: Option<&[f32]>,
) {
    let half = n_rot / 2;
    for d in 0..half {
        let base_freq = 1.0 / theta.powf(2.0 * d as f32 / n_rot as f32);
        let freq = match freq_factors {
            Some(ff) => base_freq / ff[d],
            None => base_freq,
        };
        let angle = pos as f32 * freq;
        cos[d] = angle.cos();
        sin[d] = angle.sin();
    }
}

// ---------------------------------------------------------------------------
// Bare RMSNorm (no weight multiplication)
// ---------------------------------------------------------------------------

/// RMSNorm without weight: x = x * rsqrt(mean(x^2) + eps).
/// Used for V normalization in Gemma4 (matches ggml_rms_norm).
pub(crate) fn bare_rmsnorm(x: &mut [f32], eps: f32) {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>();
    let scale = 1.0 / ((ss / n as f32) + eps).sqrt();
    for v in x.iter_mut() {
        *v *= scale;
    }
}

// ---------------------------------------------------------------------------
// Gemma4State
// ---------------------------------------------------------------------------

pub struct Gemma4State {
    // Activation buffers
    pub(crate) x: Vec<f32>,       // current layer input (inpL)
    pub(crate) x_norm: Vec<f32>,  // scratch for norm output

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
    pub(crate) kv_f32_scratch: Vec<f32>,

    // Post-attention projection scratch (wo @ attn_out)
    pub(crate) wo_out: Vec<f32>,

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

    // Q6K d_scratch for output matmul
    pub(crate) q6k_d_scratch: Vec<f32>,

    // RoPE tables
    pub(crate) cos_table: Vec<f32>,
    pub(crate) sin_table: Vec<f32>,

    // Post-attention residual (attn_out_res in the pipeline)
    pub(crate) attn_res: Vec<f32>,

    // KV cache
    pub cache: KvCache,
}

impl Gemma4State {
    pub fn new(model: &Gemma4Model, max_seq_len: usize) -> Self {
        let hd = model.hidden_dim;
        let max_head_k = *model.head_dim_k.iter().max().unwrap_or(&512);
        let max_head_v = *model.head_dim_v.iter().max().unwrap_or(&512);
        let max_head = max_head_k.max(max_head_v);
        let max_qkv = model.n_heads * max_head_k;
        let max_kv = model.n_kv_heads * max_head;
        let n_blocks_hd = hd / 256;
        let max_ffn = *model.ffn_dim.iter().max().unwrap_or(&12288);
        let n_blocks_ffn = max_ffn / 256;
        let n_blocks_out = hd / 256;

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

            wo_out: vec![0.0; hd],

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

            attn_res: vec![0.0; hd],

            cache,
        }
    }

    /// Run one decode step. Returns logits slice.
    pub fn forward_one(&mut self, model: &Gemma4Model, token_id: u32) -> &[f32] {
        let hd = model.hidden_dim;
        let pos = self.cache.seq_len();
        let diag = diag_enabled();

        // ── Pre-loop: embed + scale ──────────────────────────────────
        matmul::q6k_embed_lookup(model.embed_weight, token_id as usize, &mut self.x, hd);
        let embed_scale = (hd as f32).sqrt();
        for v in self.x.iter_mut() {
            *v *= embed_scale;
        }

        if diag {
            eprintln!("[gemma4] pos={pos} embed L2={:.4}", l2_norm(&self.x));
        }

        // ── Per-layer transformer blocks ─────────────────────────────
        for il in 0..model.n_layers {
            self.layer_forward(model, il, pos, diag);
        }

        // ── Post-loop: final norm + output matmul + softcap ─────────
        ffi_inference::gemma4_rmsnorm(
            self.x.as_ptr(),
            model.norm_weight,
            self.x_norm.as_mut_ptr(),
            hd as i32,
            model.rms_eps,
        );

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

        if model.logit_softcap > 0.0 {
            let cap = model.logit_softcap;
            let inv_cap = 1.0 / cap;
            for l in self.logits.iter_mut() {
                *l = cap * (*l * inv_cap).tanh();
            }
        }

        self.cache.advance();
        &self.logits
    }

    /// Reset state for a new sequence.
    pub fn reset(&mut self) {
        self.cache.reset();
    }
}

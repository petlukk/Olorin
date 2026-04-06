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
use crate::inference::dequant;
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
    debug_assert!(half <= cos.len(), "rope: half={half} > cos={}", cos.len());
    debug_assert!(half <= sin.len(), "rope: half={half} > sin={}", sin.len());
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
    ffi_inference::bare_rmsnorm_f32(x.as_mut_ptr(), x.len() as i32, eps);
}

// ---------------------------------------------------------------------------
// Gemma4State
// ---------------------------------------------------------------------------

pub struct Gemma4State {
    // Activation buffers
    pub x: Vec<f32>,              // current layer input (inpL)
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
    pub attn_out: Vec<f32>,
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

    // PLE buffers
    pub ple_signal: Vec<f32>,
    pub(crate) ple_gate: Vec<f32>,
    pub(crate) ple_out: Vec<f32>,
    pub(crate) ple_q8_qs: Vec<i8>,
    pub(crate) ple_q8_d: Vec<f32>,
    pub(crate) ple_q8_bsums: Vec<i16>,

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

            // Sized for max(hd, max_qkv) since Wo quantizes attn_out (n_heads*head_dim)
            q8_qs: vec![0; max_qkv + 12],
            q8_d: vec![0.0; max_qkv / 256],
            q8_bsums: vec![0; (max_qkv / 256) * 16],

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
            q6k_d_scratch: vec![0.0; std::cmp::max(n_blocks_out, n_blocks_ffn) * 4],

            cos_table: vec![0.0; max_head / 2],
            sin_table: vec![0.0; max_head / 2],

            attn_res: vec![0.0; hd],

            ple_signal: vec![0.0; model.ple_dim * model.n_layers],
            ple_gate: vec![0.0; model.ple_dim.max(1)],
            ple_out: vec![0.0; hd],
            ple_q8_qs: vec![0; model.ple_dim + 12],
            ple_q8_d: vec![0.0; (model.ple_dim / 256).max(1)],
            ple_q8_bsums: vec![0; ((model.ple_dim / 256).max(1)) * 16],

            cache,
        }
    }

    /// Phase A: compute per-layer PLE signal for this token.
    /// Called once per token before the layer loop.
    pub fn prepare_ple(
        &mut self,
        model: &Gemma4Model,
        token_id: u32,
    ) {
        let ple_dim = model.ple_dim;
        if ple_dim == 0 || model.ple_token_embd.is_null() {
            return;
        }
        let n_layers = model.n_layers;
        let hd = model.hidden_dim;
        let total = ple_dim * n_layers;

        // 1. Q6K dequant: ple_token_embd[token_id] → raw signal, scale × √ple_dim
        dequant::q6k_dequant_row(
            model.ple_token_embd,
            token_id as usize,
            &mut self.ple_signal,
            total,
        );
        let embd_scale = (ple_dim as f32).sqrt();
        ffi_inference::vec_scale_f32(
            self.ple_signal.as_ptr(), self.ple_signal.as_mut_ptr(), embd_scale, total as i32,
        );

        // 2. BF16 matvec: ple_model_proj @ embedding → proj, scale × 1/√hidden_dim
        let mut proj_ple = vec![0.0f32; total];
        matmul::bf16_matvec(
            model.ple_model_proj,
            &self.x[..hd],
            &mut proj_ple,
            total,
            hd,
        );
        let proj_scale = 1.0 / (hd as f32).sqrt();
        ffi_inference::vec_scale_f32(
            proj_ple.as_ptr(), proj_ple.as_mut_ptr(), proj_scale, total as i32,
        );

        // 3. RMSNorm each [ple_dim] slice with ple_proj_norm
        if !model.ple_proj_norm.is_null() {
            for il in 0..n_layers {
                let off = il * ple_dim;
                ffi_inference::gemma4_rmsnorm(
                    proj_ple[off..].as_ptr(),
                    model.ple_proj_norm,
                    proj_ple[off..].as_mut_ptr(),
                    ple_dim as i32,
                    model.rms_eps,
                );
            }
        }

        // 4. Add + scale: (raw + proj) / √2
        let inv_sqrt2 = 1.0 / 2.0f32.sqrt();
        ffi_inference::vec_fma_f32(
            self.ple_signal.as_ptr(), proj_ple.as_ptr(),
            self.ple_signal.as_mut_ptr(), inv_sqrt2, total as i32,
        );
    }

    /// Run one decode step. Returns logits slice.
    pub fn forward_one(&mut self, model: &Gemma4Model, token_id: u32, pool: &crate::inference::threadpool::ThreadPool) -> &[f32] {
        let hd = model.hidden_dim;
        let pos = self.cache.seq_len();
        let diag = diag_enabled();

        // ── Pre-loop: embed + scale ──────────────────────────────────
        dequant::q6k_embed_lookup(model.embed_weight, token_id as usize, &mut self.x, hd);
        let embed_scale = (hd as f32).sqrt();
        ffi_inference::vec_scale_f32(
            self.x.as_ptr(), self.x.as_mut_ptr(), embed_scale, hd as i32,
        );

        if diag {
            eprintln!("[gemma4] pos={pos} embed L2={:.4}", l2_norm(&self.x));
        }

        // PLE Phase A: compute per-layer signal
        self.prepare_ple(model, token_id);

        // ── Per-layer transformer blocks ─────────────────────────────
        for il in 0..model.n_layers {
            self.layer_forward(model, il, pos, diag, pool);
        }

        // ── Post-loop: final norm + output matmul + softcap ─────────
        ffi_inference::gemma4_rmsnorm(
            self.x.as_ptr(),
            model.norm_weight,
            self.x_norm.as_mut_ptr(),
            hd as i32,
            model.rms_eps,
        );

        if diag {
            eprintln!("[gemma4] result_norm L2={:.4} first4=[{:.4},{:.4},{:.4},{:.4}]",
                l2_norm(&self.x_norm[..hd]),
                self.x_norm[0], self.x_norm[1], self.x_norm[2], self.x_norm[3]);
        }

        matmul::quant_input(
            &self.x_norm,
            &mut self.q8_qs,
            &mut self.q8_d,
            &mut self.q8_bsums,
        );

        matmul::par_matvec(
            pool, model.embed_dtype, model.embed_weight,
            &self.q8_qs, &self.q8_d, &self.q8_bsums,
            &mut self.logits, &mut self.q6k_d_scratch,
            model.vocab_size, hd,
        );

        if model.logit_softcap > 0.0 {
            ffi_inference::softcap_f32(
                self.logits.as_mut_ptr(), model.vocab_size as i32, model.logit_softcap,
            );
        }

        self.cache.advance();
        &self.logits
    }

    /// Reset state for a new sequence.
    pub fn reset(&mut self) {
        self.cache.reset();
    }
}

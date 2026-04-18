//! Gemma 4 forward pass — decode (single token).
//!
//! Pipeline matches llama.cpp gemma4-iswa.cpp EXACTLY:
//!   embed * sqrt(n_embd) -> per-layer(attn_norm -> Q/K/V -> QK_norm -> V_bare_norm
//!   -> rope -> cache -> attn(scale=1.0) -> wo -> post_attn_norm -> +inpL
//!   -> ffn_norm -> gelu_gate*up -> down -> post_ffn_norm -> +attn_out
//!   -> out_scale) -> final_norm -> output_matmul -> softcap

use crate::inference::cache::KvCache;
use crate::inference::engine::{AttnType, Gemma4Model};
use crate::kernels::ffi_inference;

// ---------------------------------------------------------------------------
// Diagnostic helpers
// ---------------------------------------------------------------------------

pub(crate) fn timing_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("GEMMA4_TIMING").is_ok())
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

/// Standalone RoPE table computation into arbitrary slices (for per-thread scratch).
pub(crate) fn compute_rope_tables_into(
    cos: &mut [f32], sin: &mut [f32],
    pos: usize, n_rot: usize, theta: f32, freq_factors: Option<&[f32]>,
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
    pub q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,

    // Attention
    pub attn_out: Vec<f32>,
    pub(crate) attn_scores: Vec<f32>,
    pub(crate) kv_f32_scratch: Vec<f32>,
    pub(crate) attn_scores_stride: usize,
    pub(crate) kv_scratch_stride: usize,

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
    pub(crate) logit_rows: usize,

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

    // PLE batch buffers
    pub(crate) batch_ple_gate_out: Vec<f32>,
    pub(crate) batch_ple_proj_out: Vec<f32>,
    pub(crate) batch_ple_q8_qs: Vec<i8>,
    pub(crate) batch_ple_q8_d: Vec<f32>,
    pub(crate) batch_ple_q8_bsums: Vec<i16>,
    pub(crate) batch_ple_q8_a: Vec<u8>,

    // Per-thread scratch for parallelized norms/RoPE
    pub(crate) batch_head_scratch: Vec<f32>,   // [max_head_dim * n_threads]
    pub(crate) batch_cos_tables: Vec<f32>,     // [max_rope_half * n_threads]
    pub(crate) batch_sin_tables: Vec<f32>,     // [max_rope_half * n_threads]

    // ── Batched forward buffers (column-major [dim, max_batch]) ──
    pub(crate) batch_x: Vec<f32>,
    pub(crate) batch_x_norm: Vec<f32>,
    pub(crate) batch_q: Vec<f32>,
    pub(crate) batch_k: Vec<f32>,
    pub(crate) batch_v: Vec<f32>,
    pub(crate) batch_attn_out: Vec<f32>,
    pub(crate) batch_wo_out: Vec<f32>,
    pub(crate) batch_attn_res: Vec<f32>,
    pub(crate) batch_gate: Vec<f32>,
    pub(crate) batch_up: Vec<f32>,
    pub(crate) batch_down: Vec<f32>,
    pub(crate) batch_ple_signal: Vec<f32>,
    // [max_batch * (ple_dim * n_layers)] — projection scratch for phase-A
    // batched PLE (fill sequentially per weight row, then scale/norm/combine).
    pub(crate) batch_ple_proj_scratch: Vec<f32>,
    // Batch Q8K buffers for work-stealing GEMM (all N tokens quantized before barrier)
    pub(crate) batch_q8_qs: Vec<i8>,
    pub(crate) batch_q8_d: Vec<f32>,
    pub(crate) batch_q8_bsums: Vec<i16>,
    pub(crate) batch_ffn_q8_qs: Vec<i8>,
    pub(crate) batch_ffn_q8_d: Vec<f32>,
    pub(crate) batch_ffn_q8_bsums: Vec<i16>,
    // Q8K repacked A-side for GEMM (block_q8_Kx4 tiles, groups of 4 tokens)
    pub(crate) batch_q8_a: Vec<u8>,
    pub(crate) batch_ffn_q8_a: Vec<u8>,
    pub(crate) max_batch: usize,

    // KV cache
    pub cache: KvCache,
}

pub use super::forward_ple::{prepare_ple_into, prepare_ple_batch};

impl Gemma4State {
    pub fn new(
        model: &Gemma4Model,
        max_seq_len: usize,
        pool: &crate::inference::threadpool::ThreadPool,
    ) -> Self {
        let hd = model.hidden_dim;
        let max_head_k = *model.head_dim_k.iter().max().unwrap_or(&512);
        let max_head_v = *model.head_dim_v.iter().max().unwrap_or(&512);
        let max_head = max_head_k.max(max_head_v);
        let n_thread_slots = pool.thread_count();
        let max_qkv = model.n_heads * max_head_k;
        let max_kv = model.n_kv_heads * max_head;
        let max_ffn = *model.ffn_dim.iter().max().unwrap_or(&12288);
        let n_blocks_ffn = max_ffn / 256;
        let n_blocks_out = hd / 256;

        let max_batch = 512usize;
        let ple_dim = model.ple_dim.max(1);
        let ple_nb = (ple_dim / 256).max(1);

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
            attn_scores: vec![0.0; max_seq_len * n_thread_slots],
            kv_f32_scratch: vec![0.0; max_head * n_thread_slots],
            attn_scores_stride: max_seq_len,
            kv_scratch_stride: max_head,

            wo_out: vec![0.0; hd],

            gate: vec![0.0; max_ffn],
            up: vec![0.0; max_ffn],
            down: vec![0.0; hd],

            ffn_q8_qs: vec![0; max_ffn + 12],
            ffn_q8_d: vec![0.0; n_blocks_ffn],
            ffn_q8_bsums: vec![0; n_blocks_ffn * 16],

            logits: vec![0.0; model.vocab_size],
            logit_rows: model.vocab_size,
            q6k_d_scratch: vec![0.0; std::cmp::max(n_blocks_out, n_blocks_ffn) * 4 * n_thread_slots],

            cos_table: vec![0.0; max_head / 2],
            sin_table: vec![0.0; max_head / 2],

            batch_head_scratch: vec![0.0; max_head_k * n_thread_slots],
            batch_cos_tables: vec![0.0; (max_head / 2) * n_thread_slots],
            batch_sin_tables: vec![0.0; (max_head / 2) * n_thread_slots],

            attn_res: vec![0.0; hd],

            ple_signal: vec![0.0; model.ple_dim * model.n_layers],
            ple_gate: vec![0.0; model.ple_dim.max(1)],
            ple_out: vec![0.0; hd],
            ple_q8_qs: vec![0; model.ple_dim + 12],
            ple_q8_d: vec![0.0; (model.ple_dim / 256).max(1)],
            ple_q8_bsums: vec![0; ((model.ple_dim / 256).max(1)) * 16],

            batch_ple_gate_out: vec![0.0; ple_dim * max_batch],
            batch_ple_proj_out: vec![0.0; hd * max_batch],
            batch_ple_q8_qs: vec![0; (ple_dim + 12) * max_batch],
            batch_ple_q8_d: vec![0.0; ple_nb * max_batch],
            batch_ple_q8_bsums: vec![0; ple_nb * 16 * max_batch],
            batch_ple_q8_a: {
                let q8_a_groups = (max_batch + 3) / 4;
                vec![0u8; q8_a_groups * ple_nb * 1168]
            },

            batch_x: vec![0.0; hd * max_batch],
            batch_x_norm: vec![0.0; hd * max_batch],
            batch_q: vec![0.0; max_qkv * max_batch],
            batch_k: vec![0.0; max_kv * max_batch],
            batch_v: vec![0.0; max_kv * max_batch],
            batch_attn_out: vec![0.0; max_qkv * max_batch],
            batch_wo_out: vec![0.0; hd * max_batch],
            batch_attn_res: vec![0.0; hd * max_batch],
            batch_gate: vec![0.0; max_ffn * max_batch],
            batch_up: vec![0.0; max_ffn * max_batch],
            batch_down: vec![0.0; hd * max_batch],
            batch_ple_signal: vec![0.0; model.ple_dim * model.n_layers * max_batch],
            batch_ple_proj_scratch: vec![0.0; model.ple_dim * model.n_layers * max_batch],
            // Q8K stride: max(hd, max_qkv) + 12 per token (Wo quantizes attn_out)
            batch_q8_qs: vec![0; (max_qkv.max(hd) + 12) * max_batch],
            batch_q8_d: vec![0.0; (max_qkv.max(hd) / 256) * max_batch],
            batch_q8_bsums: vec![0; (max_qkv.max(hd) / 256) * 16 * max_batch],
            batch_ffn_q8_qs: vec![0; (max_ffn + 12) * max_batch],
            batch_ffn_q8_d: vec![0.0; n_blocks_ffn * max_batch],
            batch_ffn_q8_bsums: vec![0; n_blocks_ffn * 16 * max_batch],
            batch_q8_a: {
                let nb_q8_max = max_qkv.max(hd) / 256;
                let q8_a_groups = (max_batch + 3) / 4;
                vec![0u8; q8_a_groups * nb_q8_max * 1168]
            },
            batch_ffn_q8_a: {
                let q8_a_groups = (max_batch + 3) / 4;
                vec![0u8; q8_a_groups * n_blocks_ffn * 1168]
            },
            max_batch,

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
        let total = model.ple_dim * model.n_layers;
        if total == 0 { return; }
        let mut proj_scratch = vec![0.0f32; total];
        prepare_ple_into(
            model, token_id,
            &self.x[..model.hidden_dim],
            &mut self.ple_signal[..total],
            &mut proj_scratch,
        );
    }



    /// Run one decode step using graph-loop threading.
    /// Main thread participates as ith=0, workers 1..N-1 spin-barrier.
    /// Matches llama.cpp: kickoff → main runs compute_thread → return.
    pub fn forward_one_graph(&mut self, model: &Gemma4Model, token_id: u32, pool: &crate::inference::threadpool::GraphPool) -> &[f32] {
        let state_ptr = self as *mut Gemma4State as usize;
        let model_ptr = model as *const Gemma4Model as usize;

        pool.run_graph(&|tid, nth, barrier, chunk| {
            let state = unsafe { &mut *(state_ptr as *mut Gemma4State) };
            let model = unsafe { &*(model_ptr as *const Gemma4Model) };
            super::forward_graph::forward_one_inner(state, model, token_id, barrier, chunk, tid, nth);
        });

        &self.logits[..self.logit_rows]
    }

    /// Batched forward pass. Processes N tokens using gemm for all Q4K matmuls.
    /// Returns logits for the last token.
    ///
    /// When `OLORIN_PREFILL_UBATCH=K` is set (positive integer), prompts longer
    /// than K are split into K-token ubatch chunks, each run through the full
    /// layer stack in sequence. KV-cache accumulates across chunks so attention
    /// in later chunks sees all prior tokens. Intermediate chunks skip the
    /// last-token logit compute. Goal: keep each chunk's working set
    /// (weight + activation) within L3, preventing the large-N FFN-down
    /// cache-spill degradation.
    pub fn forward_batch(&mut self, model: &Gemma4Model, tokens: &[u32], pool: &crate::inference::threadpool::GraphPool) -> &[f32] {
        assert!(!tokens.is_empty());
        assert!(tokens.len() <= self.max_batch);

        let ubatch = crate::inference::threadpool::detect_prefill_ubatch()
            .min(tokens.len());

        let state_ptr = self as *mut Gemma4State as usize;
        let model_ptr = model as *const Gemma4Model as usize;

        let mut start = 0usize;
        while start < tokens.len() {
            let end = (start + ubatch).min(tokens.len());
            let chunk_tokens: &[u32] = &tokens[start..end];
            let is_last = end == tokens.len();

            pool.run_graph(&|tid, nth, barrier, chunk| {
                let state = unsafe { &mut *(state_ptr as *mut Gemma4State) };
                let model = unsafe { &*(model_ptr as *const Gemma4Model) };
                super::forward_batch::forward_batch_inner(
                    state, model, chunk_tokens, is_last,
                    barrier, chunk, tid, nth,
                );
            });

            start = end;
        }

        &self.logits[..self.logit_rows]
    }

    /// Reset state for a new sequence.
    pub fn reset(&mut self) {
        self.cache.reset();
    }
}

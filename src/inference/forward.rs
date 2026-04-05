//! Gemma 4 forward pass — decode (single token).
//!
//! Pipeline: embed -> (rmsnorm -> attn -> residual -> rmsnorm -> ffn -> residual) x N
//!           -> final rmsnorm -> output matmul -> logits

use crate::inference::cache::KvCache;
use crate::inference::engine::Gemma4Model;
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
    x: Vec<f32>,
    x_norm: Vec<f32>,

    // Q8K quantized input (for matmul)
    q8_qs: Vec<i8>,
    q8_d: Vec<f32>,
    q8_bsums: Vec<i16>,

    // QKV buffers
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,

    // Attention
    attn_out: Vec<f32>,
    attn_scores: Vec<f32>,
    // Scratch for f16->f32 conversion of one KV head row
    kv_f32_scratch: Vec<f32>,

    // FFN
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,

    // FFN Q8K (for quantizing FFN intermediate)
    ffn_q8_qs: Vec<i8>,
    ffn_q8_d: Vec<f32>,
    ffn_q8_bsums: Vec<i16>,

    // Output
    logits: Vec<f32>,

    // Q6K d_scratch for output matmul (if output is Q6K)
    q6k_d_scratch: Vec<f32>,

    // RoPE tables
    cos_table: Vec<f32>,
    sin_table: Vec<f32>,

    // KV cache
    pub cache: KvCache,
}

impl Gemma4State {
    pub fn new(model: &Gemma4Model, max_seq_len: usize) -> Self {
        let hd = model.hidden_dim;
        let qkv_dim = model.n_heads * model.head_dim;
        let kv_dim = model.n_kv_heads * model.head_dim;
        let n_blocks_hd = hd / 256;
        let ffn = model.ffn_dim;
        let n_blocks_ffn = ffn / 256;
        let n_blocks_out = hd / 256; // output matmul input is hidden_dim

        let cache = KvCache::new(
            model.n_layers,
            model.n_kv_heads,
            model.head_dim,
            model.sliding_window,
            max_seq_len,
            model.attn_types.clone(),
            model.kv_shared_source.clone(),
        );

        Self {
            x: vec![0.0; hd],
            x_norm: vec![0.0; hd],

            q8_qs: vec![0; hd + 12],
            q8_d: vec![0.0; n_blocks_hd],
            q8_bsums: vec![0; n_blocks_hd * 16],

            q: vec![0.0; qkv_dim],
            k: vec![0.0; kv_dim],
            v: vec![0.0; kv_dim],

            attn_out: vec![0.0; qkv_dim],
            attn_scores: vec![0.0; max_seq_len],
            kv_f32_scratch: vec![0.0; model.head_dim],

            gate: vec![0.0; ffn],
            up: vec![0.0; ffn],
            down: vec![0.0; hd],

            ffn_q8_qs: vec![0; ffn + 12],
            ffn_q8_d: vec![0.0; n_blocks_ffn],
            ffn_q8_bsums: vec![0; n_blocks_ffn * 16],

            logits: vec![0.0; model.vocab_size],
            q6k_d_scratch: vec![0.0; n_blocks_out * 4],

            cos_table: vec![0.0; model.head_dim / 2],
            sin_table: vec![0.0; model.head_dim / 2],

            cache,
        }
    }

    /// Run one decode step. Returns logits slice.
    pub fn forward_one(&mut self, model: &Gemma4Model, token_id: u32) -> &[f32] {
        let hd = model.hidden_dim;
        let head_dim = model.head_dim;
        let n_heads = model.n_heads;
        let n_kv_heads = model.n_kv_heads;
        let gqa_ratio = n_heads / n_kv_heads;
        let kv_dim = n_kv_heads * head_dim;
        let pos = self.cache.seq_len();
        let diag = diag_enabled();
        let scale = 1.0 / (head_dim as f32).sqrt();

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

        // 2. RoPE tables for current position
        compute_rope_tables(
            &mut self.cos_table,
            &mut self.sin_table,
            pos,
            head_dim,
            model.rope_theta,
        );

        // 3. Per-layer transformer blocks
        for layer in 0..model.n_layers {
            let lw = &model.layers[layer];

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

            // b-d. QKV projections
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
                &mut self.v, kv_dim, hd,
            );

            // e-f. RoPE on Q and K
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
            self.cache.store(layer, &self.k, &self.v);

            // h. Attention (GQA, single-token decode)
            let attn_len = self.cache.attn_len(layer);
            let k_ptr = self.cache.k_ptr(layer);
            let v_ptr = self.cache.v_ptr(layer);

            self.attention_decode(
                n_heads, n_kv_heads, gqa_ratio, head_dim,
                kv_dim, attn_len, scale, k_ptr, v_ptr,
            );

            // i. Output projection
            matmul::quant_input(
                &self.attn_out,
                &mut self.q8_qs,
                &mut self.q8_d,
                &mut self.q8_bsums,
            );
            matmul::q4k_matvec(
                lw.wo,
                &self.q8_qs, &self.q8_d, &self.q8_bsums,
                &mut self.down, hd, n_heads * head_dim,
            );

            // j. Residual add
            for i in 0..hd {
                self.x[i] += self.down[i];
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

            // l. FFN (GeGLU)
            self.ffn(model, layer);

            // m. Residual add
            for i in 0..hd {
                self.x[i] += self.down[i];
            }

            if diag {
                eprintln!(
                    "[gemma4] L{layer} post-ffn L2={:.4}",
                    l2_norm(&self.x[..hd])
                );
            }
        }

        // 4. Final RMSNorm
        ffi_inference::gemma4_rmsnorm(
            self.x.as_ptr(),
            model.norm_weight,
            self.x_norm.as_mut_ptr(),
            hd as i32,
            model.rms_eps,
        );

        // 5. Output logits
        matmul::quant_input(
            &self.x_norm,
            &mut self.q8_qs,
            &mut self.q8_d,
            &mut self.q8_bsums,
        );

        if model.output_dtype == matmul::GGML_TYPE_Q6_K {
            matmul::q6k_matvec(
                model.output_weight,
                &self.q8_qs, &self.q8_d, &self.q8_bsums,
                &mut self.logits,
                &mut self.q6k_d_scratch,
                model.vocab_size, hd,
            );
        } else {
            matmul::q4k_matvec(
                model.output_weight,
                &self.q8_qs, &self.q8_d, &self.q8_bsums,
                &mut self.logits, model.vocab_size, hd,
            );
        }

        // Advance cache position
        self.cache.advance();

        &self.logits
    }

    /// GQA attention decode: for each Q head, dot with cached K, softmax, weighted V sum.
    fn attention_decode(
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
        let stride = kv_dim; // n_kv_heads * head_dim per position

        for h in 0..n_heads {
            let kv_h = h / gqa_ratio;
            let q_off = h * head_dim;
            let q_slice = &self.q[q_off..q_off + head_dim];

            // Compute attention scores: Q dot K for each cached position
            for p in 0..attn_len {
                let k_offset = p * stride + kv_h * head_dim;
                let k_src = unsafe { k_ptr.add(k_offset) };
                // Convert f16 -> f32
                unsafe {
                    ffi_inference::f16_to_f32(
                        k_src,
                        self.kv_f32_scratch.as_mut_ptr(),
                        head_dim as i32,
                    );
                }
                // Dot product
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_slice[d] * self.kv_f32_scratch[d];
                }
                self.attn_scores[p] = dot;
            }

            // Softmax with scale = 1/sqrt(head_dim)
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
    fn ffn(&mut self, model: &Gemma4Model, layer: usize) {
        let hd = model.hidden_dim;
        let ffn_dim = model.ffn_dim;
        let lw = &model.layers[layer];

        // Quantize x_norm for gate/up matmul
        matmul::quant_input(
            &self.x_norm,
            &mut self.q8_qs,
            &mut self.q8_d,
            &mut self.q8_bsums,
        );

        // Fused gate + up projection
        matmul::q4k_matvec_dual(
            lw.w_gate,
            lw.w_up,
            &self.q8_qs, &self.q8_d, &self.q8_bsums,
            &mut self.gate, &mut self.up,
            ffn_dim, hd,
        );

        // GELU(gate) * up -> gate buffer
        ffi_inference::gelu_mul(
            self.gate.as_ptr(),
            self.up.as_ptr(),
            self.gate.as_mut_ptr(),
            ffn_dim as i32,
        );

        // Quantize gate (ffn_dim) for down projection
        matmul::quant_input(
            &self.gate,
            &mut self.ffn_q8_qs,
            &mut self.ffn_q8_d,
            &mut self.ffn_q8_bsums,
        );

        // Down projection: ffn_dim -> hidden_dim
        matmul::q4k_matvec(
            lw.w_down,
            &self.ffn_q8_qs, &self.ffn_q8_d, &self.ffn_q8_bsums,
            &mut self.down, hd, ffn_dim,
        );
    }

    /// Reset state for a new sequence.
    pub fn reset(&mut self) {
        self.cache.reset();
    }
}

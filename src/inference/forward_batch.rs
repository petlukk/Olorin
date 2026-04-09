//! Batched prompt-eval orchestration.
//!
//! `forward_batch` drives the N-token prompt-eval pipeline: embed all N
//! tokens column-major into `batch_x`, stash per-token PLE signals, run
//! each layer through `layer_forward_batch`, then do the final norm +
//! lm_head on the last column only. Per-layer body lives in
//! `forward_attn_batched.rs`.

use crate::inference::engine::Gemma4Model;
use crate::inference::matmul;
use crate::kernels::ffi_inference;

use super::forward::Gemma4State;

impl Gemma4State {
    /// Run a prompt-eval forward pass over `tokens`. Returns the final-token logits.
    ///
    /// Per-column Q4K matmuls still use unrepacked weights against the
    /// existing `par_matvec` path — the repacked 8x8 gemm swap is Task 20.
    pub fn forward_batch(
        &mut self,
        model: &Gemma4Model,
        tokens: &[u32],
        pool: &crate::inference::threadpool::ThreadPool,
    ) -> &[f32] {
        assert!(!tokens.is_empty(), "forward_batch requires at least one token");
        let n = tokens.len();
        assert!(
            n <= self.max_batch,
            "batch size {} exceeds max_batch {}",
            n, self.max_batch,
        );
        let hd = model.hidden_dim;
        let pos_base = self.cache.seq_len();

        // SWA causal-cap assumption inside attention_decode_batch: for the
        // last column we need pos_base + n - 1 < window_size. Longer prompts
        // must be chunked by the caller (prompt chunking lands with Task 22).

        // ── 1. Embed + scale each token into batch_x (column k) ───────
        let embed_scale = (hd as f32).sqrt();
        for k in 0..n {
            let off = k * hd;
            crate::inference::dequant::q6k_embed_lookup(
                model.embed_weight,
                tokens[k] as usize,
                &mut self.batch_x[off..off + hd],
                hd,
            );
            ffi_inference::vec_scale_f32(
                self.batch_x[off..off + hd].as_ptr(),
                self.batch_x[off..off + hd].as_mut_ptr(),
                embed_scale,
                hd as i32,
            );
        }

        // ── 2. Per-token PLE signals (stashed into batch_ple_signal) ──
        // prepare_ple reads from self.x and writes self.ple_signal. Stage
        // each token's embedding into self.x, call prepare_ple, copy the
        // result into the per-token slot.
        if model.ple_dim > 0 && !model.ple_token_embd.is_null() {
            let per_tok = model.ple_dim * model.n_layers;
            for k in 0..n {
                let off = k * hd;
                self.x[..hd].copy_from_slice(&self.batch_x[off..off + hd]);
                self.prepare_ple(model, tokens[k]);
                let dst_off = k * per_tok;
                self.batch_ple_signal[dst_off..dst_off + per_tok]
                    .copy_from_slice(&self.ple_signal[..per_tok]);
            }
        }

        // ── 3. Per-layer batched forward ──────────────────────────────
        for il in 0..model.n_layers {
            self.layer_forward_batch(model, il, pos_base, n, pool);
        }

        // ── 4. Final norm + lm_head (last column only) ────────────────
        let last_off = (n - 1) * hd;
        ffi_inference::gemma4_rmsnorm(
            self.batch_x[last_off..].as_ptr(),
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
        matmul::par_matvec(
            pool,
            model.embed_dtype,
            model.embed_weight,
            &self.q8_qs,
            &self.q8_d,
            &self.q8_bsums,
            &mut self.logits,
            &mut self.q6k_d_scratch,
            model.vocab_size,
            hd,
        );
        if model.logit_softcap > 0.0 {
            ffi_inference::softcap_f32(
                self.logits.as_mut_ptr(),
                model.vocab_size as i32,
                model.logit_softcap,
            );
        }

        self.cache.advance_by(n);
        &self.logits
    }
}

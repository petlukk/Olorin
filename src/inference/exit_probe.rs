//! Offline early-exit logit probe.
//!
//! Takes a residual snapshot captured mid-forward, re-applies the model's
//! final RMSNorm + output head + softcap to produce the logits you *would*
//! have gotten if you exited at that layer. Compared against the true
//! final-layer logits, tells us at which depth the residual stream has
//! already committed to its answer.
//!
//! This is the single most decision-relevant measurement for speculative
//! decoding viability and Adaptive Quant per-layer sensitivity.
//!
//! Single-threaded by design — this is offline analysis, not hot-path.

use crate::inference::engine::Gemma4Model;
use crate::inference::matmul::{self, Q4K_BLOCK_SIZE};
use crate::kernels::{ffi_inference, ffi as kffi};
use crate::inference::matmul_graph;
use std::sync::atomic::AtomicI32;

/// Apply final RMSNorm + output head + softcap to a residual state vector.
/// Returns the full-vocab logits (length `model.vocab_size`).
pub fn reproject_residual(residual: &[f32], model: &Gemma4Model) -> Vec<f32> {
    let _ = kffi::init(); // defensive — must be called before any kernel use
    let hd = residual.len();
    assert_eq!(hd, model.hidden_dim,
        "residual length {} does not match model.hidden_dim {}",
        hd, model.hidden_dim);
    assert_eq!(hd % Q4K_BLOCK_SIZE, 0,
        "hidden_dim {hd} must be multiple of 256 for Q8K quant");

    // 1. Final RMSNorm on the residual.
    let mut x_norm = vec![0.0f32; hd];
    ffi_inference::gemma4_rmsnorm(
        residual.as_ptr(), model.norm_weight,
        x_norm.as_mut_ptr(), hd as i32, model.rms_eps,
    );

    // 2. Quantize input to Q8K (required by Q6K matvec).
    let n_blocks = hd / Q4K_BLOCK_SIZE;
    let mut q8_qs = vec![0i8; hd + 12];
    let mut q8_d = vec![0.0f32; n_blocks];
    let mut q8_bsums = vec![0i16; n_blocks * 16];
    matmul::quant_input(&x_norm, &mut q8_qs, &mut q8_d, &mut q8_bsums);

    // 3. Output-head matmul.
    let vocab = model.vocab_size;
    let mut logits = vec![0.0f32; vocab];
    // d_scratch sized for 1 thread (ith=0, nth=1): n_blocks * 4 floats.
    let mut d_scratch = vec![0.0f32; n_blocks * 4];
    let chunk = AtomicI32::new(0);
    if let (Some(q6k_buf), Some(d_arr)) =
        (model.embed_q6k_repacked.as_ref(), model.embed_q6k_d_arr.as_ref())
    {
        matmul_graph::q6k_repacked_batch_ws_pre_d(
            q6k_buf.as_ptr(), d_arr.as_ptr(),
            q8_qs.as_ptr(), q8_d.as_ptr(), q8_bsums.as_ptr(),
            logits.as_mut_ptr(), d_scratch.as_mut_ptr(),
            vocab, hd, 1, vocab,
            &chunk, 0, 1,
        );
    } else {
        matmul_graph::matvec_ws(
            model.embed_dtype, model.embed_weight,
            q8_qs.as_ptr(), q8_d.as_ptr(), q8_bsums.as_ptr(),
            logits.as_mut_ptr(), d_scratch.as_mut_ptr(),
            vocab, hd,
            &chunk, 0, 1,
        );
    }

    // 4. Softcap (if the model uses one).
    if model.logit_softcap > 0.0 {
        ffi_inference::softcap_f32(
            logits.as_mut_ptr(), vocab as i32, model.logit_softcap,
        );
    }
    logits
}

/// Comparison of a probe (partial) logit vector against the final-layer ground-truth.
#[derive(Debug, Clone)]
pub struct LayerProbeResult {
    /// Top-1 token argmax from the probe.
    pub probe_top1: u32,
    /// Top-1 token argmax from the final-layer logits.
    pub final_top1: u32,
    /// Whether they agree.
    pub top1_agreement: bool,
    /// Overlap size between top-5 sets (0..=5).
    pub top5_overlap: usize,
    /// KL(final || probe) — low means probe matches the final distribution.
    pub kl_final_given_probe: f32,
}

/// Argmax over logits.
fn argmax(v: &[f32]) -> u32 {
    let mut m = f32::NEG_INFINITY;
    let mut mi = 0u32;
    for (i, &x) in v.iter().enumerate() {
        if x > m {
            m = x;
            mi = i as u32;
        }
    }
    mi
}

/// Indices of the top-K entries (not necessarily sorted).
fn top_k(v: &[f32], k: usize) -> Vec<u32> {
    // Simple O(n*k) partial sort — fine for k=5 and vocab~262k.
    let mut out = Vec::with_capacity(k);
    let mut seen = vec![false; v.len()];
    for _ in 0..k {
        let mut best = f32::NEG_INFINITY;
        let mut best_i = 0usize;
        for (i, &x) in v.iter().enumerate() {
            if !seen[i] && x > best {
                best = x;
                best_i = i;
            }
        }
        seen[best_i] = true;
        out.push(best_i as u32);
    }
    out
}

/// Compute KL(p || q) where p,q are softmax of the given logits.
/// Uses log-sum-exp trick for numerical stability. O(n) in vocab size.
pub fn kl_divergence_softmax(p_logits: &[f32], q_logits: &[f32]) -> f32 {
    assert_eq!(p_logits.len(), q_logits.len());
    let p_max = p_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let q_max = q_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p_z = 0.0f32;
    let mut q_z = 0.0f32;
    for i in 0..p_logits.len() {
        p_z += (p_logits[i] - p_max).exp();
        q_z += (q_logits[i] - q_max).exp();
    }
    let log_p_z = p_z.ln();
    let log_q_z = q_z.ln();
    let mut kl = 0.0f32;
    for i in 0..p_logits.len() {
        let lp = p_logits[i] - p_max - log_p_z; // log p
        let lq = q_logits[i] - q_max - log_q_z;
        let p = lp.exp();
        if p > 1e-20 {
            kl += p * (lp - lq);
        }
    }
    kl
}

/// Compare a partial-layer probe against the final logits.
pub fn compare(probe_logits: &[f32], final_logits: &[f32]) -> LayerProbeResult {
    let probe_top1 = argmax(probe_logits);
    let final_top1 = argmax(final_logits);
    let probe_top5 = top_k(probe_logits, 5);
    let final_top5 = top_k(final_logits, 5);
    let top5_overlap = probe_top5.iter()
        .filter(|id| final_top5.contains(id))
        .count();
    // KL(final || probe) — "does the probe distribution cover the real one?"
    let kl = kl_divergence_softmax(final_logits, probe_logits);
    LayerProbeResult {
        probe_top1,
        final_top1,
        top1_agreement: probe_top1 == final_top1,
        top5_overlap,
        kl_final_given_probe: kl,
    }
}

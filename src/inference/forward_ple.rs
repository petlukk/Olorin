//! PLE phase-A computation — extracted from forward.rs.

use crate::inference::engine::Gemma4Model;
use crate::inference::matmul;
use crate::inference::dequant;
use crate::kernels::ffi_inference;

/// Core PLE phase-A computation into explicit output buffers.
/// Input: `x_in` is the current token's embedded+scaled vector (len = hidden_dim).
/// Output: `ple_out` (len = ple_dim × n_layers) receives the prepared signal.
/// `proj_scratch` must be the same length as `ple_out`.
///
/// Used by both the single-token wrapper `Gemma4State::prepare_ple` and the
/// parallel prefill pre-loop in `forward_batch`, where each thread supplies
/// its own `proj_scratch` and writes into a disjoint slice of
/// `state.batch_ple_signal`.
pub fn prepare_ple_into(
    model: &Gemma4Model,
    token_id: u32,
    x_in: &[f32],
    ple_out: &mut [f32],
    proj_scratch: &mut [f32],
) {
    let ple_dim = model.ple_dim;
    if ple_dim == 0 || model.ple_token_embd.is_null() {
        return;
    }
    let n_layers = model.n_layers;
    let hd = model.hidden_dim;
    let total = ple_dim * n_layers;

    // 1. Q6K dequant: ple_token_embd[token_id] → raw signal, scale × √ple_dim
    dequant::q6k_dequant_row(model.ple_token_embd, token_id as usize, ple_out, total);
    let embd_scale = (ple_dim as f32).sqrt();
    ffi_inference::vec_scale_f32(
        ple_out.as_ptr(), ple_out.as_mut_ptr(), embd_scale, total as i32,
    );

    // 2. BF16 matvec: ple_model_proj @ x_in → proj, scale × 1/√hidden_dim
    matmul::bf16_matvec(model.ple_model_proj, &x_in[..hd], proj_scratch, total, hd);
    let proj_scale = 1.0 / (hd as f32).sqrt();
    ffi_inference::vec_scale_f32(
        proj_scratch.as_ptr(), proj_scratch.as_mut_ptr(), proj_scale, total as i32,
    );

    // 3. RMSNorm each [ple_dim] slice with ple_proj_norm
    if !model.ple_proj_norm.is_null() {
        for il in 0..n_layers {
            let off = il * ple_dim;
            ffi_inference::gemma4_rmsnorm(
                proj_scratch[off..].as_ptr(),
                model.ple_proj_norm,
                proj_scratch[off..].as_mut_ptr(),
                ple_dim as i32,
                model.rms_eps,
            );
        }
    }

    // 4. Add + scale: ple_out = (ple_out + proj) / √2
    let inv_sqrt2 = 1.0 / 2.0f32.sqrt();
    ffi_inference::vec_fma_f32(
        ple_out.as_ptr(), proj_scratch.as_ptr(),
        ple_out.as_mut_ptr(), inv_sqrt2, total as i32,
    );
}

/// Batched PLE phase-A: runs all `n_tokens` tokens through phase-A in three
/// passes, with a barrier between each. Parallelized across `nth` threads
/// by splitting the weight-row axis (steps 1-2) and the token axis (step 3).
///
/// Buffers (all indexed by token t):
///   - `batch_x[t * hd .. (t+1) * hd]`           — input (embedded+scaled)
///   - `batch_ple_signal[t * total ..]`          — output (raw signal, then combined)
///   - `proj_scratch[t * total ..]`              — temporary projection
///
/// Bit-exact with calling `prepare_ple_into` sequentially per token, because
/// each (token, row) dot product uses the same column-reduction sequence
/// as `bf16_dot_f32`.
#[allow(clippy::too_many_arguments)]
pub fn prepare_ple_batch(
    model: &Gemma4Model,
    tokens: &[u32],
    batch_x: &[f32],
    batch_ple_signal: &mut [f32],
    proj_scratch: &mut [f32],
    barrier: &crate::inference::threadpool::SpinBarrier,
    ith: usize,
    nth: usize,
) {
    let ple_dim = model.ple_dim;
    if ple_dim == 0 || model.ple_token_embd.is_null() {
        return;
    }
    let n_layers = model.n_layers;
    let hd = model.hidden_dim;
    let total = ple_dim * n_layers;
    let n = tokens.len();
    let embd_scale = (ple_dim as f32).sqrt();
    let proj_scale = 1.0 / (hd as f32).sqrt();
    let inv_sqrt2 = 1.0 / 2.0f32.sqrt();

    // ── Step 1: Q6K dequant + scale per token (token-parallel) ────
    let per_t = (n + nth - 1) / nth;
    let t0 = ith * per_t;
    let t1 = (t0 + per_t).min(n);
    for t in t0..t1 {
        let out = &mut batch_ple_signal[t * total..(t + 1) * total];
        dequant::q6k_dequant_row(model.ple_token_embd, tokens[t] as usize, out, total);
        ffi_inference::vec_scale_f32(
            out.as_ptr(), out.as_mut_ptr(), embd_scale, total as i32,
        );
    }
    barrier.wait();

    // ── Step 2: Batched BF16 matvec (row-parallel across total rows) ──
    // Each thread handles a disjoint row range. For each row, dot against
    // all n_tokens inputs in one kernel call — the weight row stays L1-hot
    // across tokens, cutting ~60× DRAM reads of ple_model_proj to ~1×.
    let per_r = (total + nth - 1) / nth;
    let r0 = ith * per_r;
    let r1 = (r0 + per_r).min(total);
    let weight_u16 = model.ple_model_proj as *const u16;
    let mut scratch = [0i32; 8];
    for r in r0..r1 {
        unsafe {
            ffi_inference::bf16_dot_multi_input(
                weight_u16.add(r * hd),
                batch_x.as_ptr(),
                proj_scratch.as_mut_ptr().add(r),
                scratch.as_mut_ptr(),
                n as i32,
                hd as i32,
                hd as i32,
                total as i32,
            );
        }
    }
    barrier.wait();

    // ── Step 3: scale + RMSNorm + FMA combine, token-parallel ────
    let proj_norm = model.ple_proj_norm;
    for t in t0..t1 {
        let proj = &mut proj_scratch[t * total..(t + 1) * total];
        ffi_inference::vec_scale_f32(
            proj.as_ptr(), proj.as_mut_ptr(), proj_scale, total as i32,
        );
        if !proj_norm.is_null() {
            for il in 0..n_layers {
                let off = il * ple_dim;
                ffi_inference::gemma4_rmsnorm(
                    proj[off..].as_ptr(),
                    proj_norm,
                    proj[off..].as_mut_ptr(),
                    ple_dim as i32,
                    model.rms_eps,
                );
            }
        }
        let ple = &mut batch_ple_signal[t * total..(t + 1) * total];
        ffi_inference::vec_fma_f32(
            ple.as_ptr(), proj.as_ptr(),
            ple.as_mut_ptr(), inv_sqrt2, total as i32,
        );
    }
}

//! Per-head attention compute: Q/K/V RMSNorms and attention_decode.
//!
//! Split from forward_attn.rs to keep that file under the 500-line limit
//! and to localize parallelization changes. All functions in this module
//! are currently serial; parallel dispatch is added in later tasks of the
//! attention-head-parallelism plan.

use super::forward::{bare_rmsnorm, Gemma4State};
use crate::kernels::ffi_inference;

pub(crate) fn q_norm_per_head(
    state: &mut Gemma4State,
    q_norm: *const f32,
    n_heads: usize,
    head_dim: usize,
    rms_eps: f32,
) {
    for h in 0..n_heads {
        let off = h * head_dim;
        ffi_inference::gemma4_rmsnorm(
            state.q.as_ptr().wrapping_add(off),
            q_norm,
            state.kv_f32_scratch.as_mut_ptr(),
            head_dim as i32,
            rms_eps,
        );
        state.q[off..off + head_dim]
            .copy_from_slice(&state.kv_f32_scratch[..head_dim]);
    }
}

pub(crate) fn k_norm_per_head(
    state: &mut Gemma4State,
    k_norm: *const f32,
    n_kv_heads: usize,
    head_dim: usize,
    rms_eps: f32,
) {
    for h in 0..n_kv_heads {
        let off = h * head_dim;
        ffi_inference::gemma4_rmsnorm(
            state.k.as_ptr().wrapping_add(off),
            k_norm,
            state.kv_f32_scratch.as_mut_ptr(),
            head_dim as i32,
            rms_eps,
        );
        state.k[off..off + head_dim]
            .copy_from_slice(&state.kv_f32_scratch[..head_dim]);
    }
}

pub(crate) fn v_bare_norm_per_head(
    state: &mut Gemma4State,
    n_kv_heads: usize,
    head_dim_v: usize,
    rms_eps: f32,
) {
    for h in 0..n_kv_heads {
        let off = h * head_dim_v;
        bare_rmsnorm(&mut state.v[off..off + head_dim_v], rms_eps);
    }
}

pub(crate) fn attention_decode(
    state: &mut Gemma4State,
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
    let stride = kv_dim;

    for h in 0..n_heads {
        let kv_h = h / gqa_ratio;
        let q_off = h * head_dim;
        let q_slice = &state.q[q_off..q_off + head_dim];

        // Q dot K for each cached position
        for p in 0..attn_len {
            let k_offset = p * stride + kv_h * head_dim;
            let k_src = unsafe { k_ptr.add(k_offset) };
            unsafe {
                ffi_inference::f16_to_f32(
                    k_src,
                    state.kv_f32_scratch.as_mut_ptr(),
                    head_dim as i32,
                );
            }
            state.attn_scores[p] = ffi_inference::f32_dot(
                q_slice.as_ptr(),
                state.kv_f32_scratch.as_ptr(),
                head_dim as i32,
            );
        }

        // Softmax with scale (1.0 for Gemma4)
        unsafe {
            ffi_inference::softmax_f32(
                state.attn_scores.as_mut_ptr(),
                attn_len as i32,
                scale,
            );
        }

        // Weighted V sum
        let out_off = q_off;
        state.attn_out[out_off..out_off + head_dim].fill(0.0);
        for p in 0..attn_len {
            let v_offset = p * stride + kv_h * head_dim;
            let v_src = unsafe { v_ptr.add(v_offset) };
            unsafe {
                ffi_inference::f16_to_f32(
                    v_src,
                    state.kv_f32_scratch.as_mut_ptr(),
                    head_dim as i32,
                );
            }
            let s = state.attn_scores[p];
            ffi_inference::f32_dot_acc(
                state.attn_out[out_off..].as_mut_ptr(),
                state.kv_f32_scratch.as_ptr(),
                s,
                head_dim as i32,
            );
        }
    }
}

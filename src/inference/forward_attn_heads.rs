//! Per-head attention compute: Q/K/V RMSNorms and attention_decode.
//!
//! Split from forward_attn.rs to keep that file under the 500-line limit
//! and to localize parallelization changes. All functions in this module
//! are currently serial; parallel dispatch is added in later tasks of the
//! attention-head-parallelism plan.

use super::forward::{bare_rmsnorm, Gemma4State};
use crate::kernels::ffi_inference;

// Pointers are passed across pool closure boundaries by casting to `usize`
// (trivially Send + Sync) and casting back inside the worker. Wrapper structs
// don't survive Rust 2021 disjoint closure captures — the closure captures the
// inner raw pointer field, not the wrapper. Caller must ensure each thread
// accesses a disjoint range of the underlying allocation.

pub(crate) fn q_norm_per_head(
    state: &mut Gemma4State,
    q_norm: *const f32,
    n_heads: usize,
    head_dim: usize,
    rms_eps: f32,
    pool: &crate::inference::threadpool::ThreadPool,
) {
    let kv_scratch_stride = state.kv_scratch_stride;
    let q_addr = state.q.as_mut_ptr() as usize;
    let scratch_addr = state.kv_f32_scratch.as_mut_ptr() as usize;
    let q_norm_addr = q_norm as usize;

    let n_workers = n_heads.min(pool.thread_count()).max(1);

    pool.run(n_workers, |tid, nt| {
        let per = (n_heads + nt - 1) / nt;
        let h_start = tid * per;
        let h_end = ((tid + 1) * per).min(n_heads);
        if h_start >= h_end { return; }

        let q_ptr = q_addr as *mut f32;
        let q_norm_ptr = q_norm_addr as *const f32;
        let scratch_base = unsafe {
            (scratch_addr as *mut f32).add(tid * kv_scratch_stride)
        };

        for h in h_start..h_end {
            let off = h * head_dim;
            let q_head_in = unsafe { q_ptr.add(off) as *const f32 };
            ffi_inference::gemma4_rmsnorm(
                q_head_in,
                q_norm_ptr,
                scratch_base,
                head_dim as i32,
                rms_eps,
            );
            unsafe {
                std::ptr::copy_nonoverlapping(
                    scratch_base as *const f32,
                    q_ptr.add(off),
                    head_dim,
                );
            }
        }
    });
}

pub(crate) fn k_norm_per_head(
    state: &mut Gemma4State,
    k_norm: *const f32,
    n_kv_heads: usize,
    head_dim: usize,
    rms_eps: f32,
    pool: &crate::inference::threadpool::ThreadPool,
) {
    let kv_scratch_stride = state.kv_scratch_stride;
    let k_addr = state.k.as_mut_ptr() as usize;
    let scratch_addr = state.kv_f32_scratch.as_mut_ptr() as usize;
    let k_norm_addr = k_norm as usize;

    let n_workers = n_kv_heads.min(pool.thread_count()).max(1);

    pool.run(n_workers, |tid, nt| {
        let per = (n_kv_heads + nt - 1) / nt;
        let h_start = tid * per;
        let h_end = ((tid + 1) * per).min(n_kv_heads);
        if h_start >= h_end { return; }

        let k_ptr = k_addr as *mut f32;
        let k_norm_ptr = k_norm_addr as *const f32;
        let scratch_base = unsafe {
            (scratch_addr as *mut f32).add(tid * kv_scratch_stride)
        };

        for h in h_start..h_end {
            let off = h * head_dim;
            let k_head_in = unsafe { k_ptr.add(off) as *const f32 };
            ffi_inference::gemma4_rmsnorm(
                k_head_in,
                k_norm_ptr,
                scratch_base,
                head_dim as i32,
                rms_eps,
            );
            unsafe {
                std::ptr::copy_nonoverlapping(
                    scratch_base as *const f32,
                    k_ptr.add(off),
                    head_dim,
                );
            }
        }
    });
}

pub(crate) fn v_bare_norm_per_head(
    state: &mut Gemma4State,
    n_kv_heads: usize,
    head_dim_v: usize,
    rms_eps: f32,
    pool: &crate::inference::threadpool::ThreadPool,
) {
    let v_addr = state.v.as_mut_ptr() as usize;
    let n_workers = n_kv_heads.min(pool.thread_count()).max(1);

    pool.run(n_workers, |tid, nt| {
        let per = (n_kv_heads + nt - 1) / nt;
        let h_start = tid * per;
        let h_end = ((tid + 1) * per).min(n_kv_heads);
        if h_start >= h_end { return; }

        let v_ptr = v_addr as *mut f32;
        for h in h_start..h_end {
            let off = h * head_dim_v;
            let slice = unsafe {
                std::slice::from_raw_parts_mut(v_ptr.add(off), head_dim_v)
            };
            bare_rmsnorm(slice, rms_eps);
        }
    });
}

pub fn attention_decode(
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
    pool: &crate::inference::threadpool::ThreadPool,
) {
    let stride_kv = kv_dim;
    let kv_scratch_stride = state.kv_scratch_stride;
    let attn_scores_stride = state.attn_scores_stride;

    let q_addr = state.q.as_ptr() as usize;
    let attn_out_addr = state.attn_out.as_mut_ptr() as usize;
    let kv_scratch_addr = state.kv_f32_scratch.as_mut_ptr() as usize;
    let attn_scores_addr = state.attn_scores.as_mut_ptr() as usize;
    let k_addr = k_ptr as usize;
    let v_addr = v_ptr as usize;

    let n_workers = n_heads.min(pool.thread_count()).max(1);

    pool.run(n_workers, |tid, nt| {
        // Distribute heads across threads contiguously.
        let per = (n_heads + nt - 1) / nt;
        let h_start = tid * per;
        let h_end = ((tid + 1) * per).min(n_heads);
        if h_start >= h_end { return; }

        // Recover raw pointers from usize. Disjointness invariants hold because
        // each tid uses a unique scratch slab and writes a unique head range.
        let q_ptr = q_addr as *const f32;
        let attn_out_ptr = attn_out_addr as *mut f32;
        let k_ptr = k_addr as *const u16;
        let v_ptr = v_addr as *const u16;

        // This thread's private scratch slots — disjoint by tid.
        let kv_scratch_base = unsafe {
            (kv_scratch_addr as *mut f32).add(tid * kv_scratch_stride)
        };
        let attn_scores_base = unsafe {
            (attn_scores_addr as *mut f32).add(tid * attn_scores_stride)
        };

        for h in h_start..h_end {
            let kv_h = h / gqa_ratio;
            let q_off = h * head_dim;
            let q_slice_ptr = unsafe { q_ptr.add(q_off) };

            // Q · K for each cached position
            for p in 0..attn_len {
                let k_offset = p * stride_kv + kv_h * head_dim;
                let k_src = unsafe { k_ptr.add(k_offset) };
                unsafe {
                    ffi_inference::f16_to_f32(k_src, kv_scratch_base, head_dim as i32);
                }
                let dot = ffi_inference::f32_dot(
                    q_slice_ptr,
                    kv_scratch_base as *const f32,
                    head_dim as i32,
                );
                unsafe { *attn_scores_base.add(p) = dot; }
            }

            // Softmax with scale (1.0 for Gemma4)
            unsafe {
                ffi_inference::softmax_f32(attn_scores_base, attn_len as i32, scale);
            }

            // Weighted V sum into attn_out[h*head_dim..(h+1)*head_dim] (disjoint per head)
            let out_base = unsafe { attn_out_ptr.add(q_off) };
            unsafe {
                std::ptr::write_bytes(out_base, 0, head_dim);
            }
            for p in 0..attn_len {
                let v_offset = p * stride_kv + kv_h * head_dim;
                let v_src = unsafe { v_ptr.add(v_offset) };
                unsafe {
                    ffi_inference::f16_to_f32(v_src, kv_scratch_base, head_dim as i32);
                }
                let s = unsafe { *attn_scores_base.add(p) };
                ffi_inference::f32_dot_acc(
                    out_base,
                    kv_scratch_base as *const f32,
                    s,
                    head_dim as i32,
                );
            }
        }
    });
}

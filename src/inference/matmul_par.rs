//! Parallel (ThreadPool-based) dtype kernels for quantized matmul.
//!
//! Each `par_*` function splits the 4-row-batched work across pool workers
//! and falls back to the corresponding single-threaded kernel from
//! `matmul_seq.rs` when the pool has ≤1 thread or fewer quads than workers.
//! Dispatch entry point is `par_matvec` in `matmul.rs`.

use crate::kernels::ffi_inference;
use crate::inference::threadpool::ThreadPool;
use super::matmul::{
    Q4K_BLOCK_SIZE, Q4K_BLOCK_BYTES,
    Q5K_BLOCK_SIZE, Q5K_BLOCK_BYTES,
    Q6K_BLOCK_SIZE, Q6K_BLOCK_BYTES,
    pow2_table,
};
use super::matmul_seq::{
    q4k_matvec, q4k_matvec_dual,
    q5k_matvec, q6k_matvec, q6k_extract_d,
};

// ---------------------------------------------------------------------------
// Send-wrapping raw pointer helpers (for pool.run closures)
// ---------------------------------------------------------------------------

/// Wrapper for raw pointers to cross thread boundary in pool.run().
/// Safety: caller ensures pointer validity for the lifetime of the pool dispatch.
#[derive(Clone, Copy)]
struct SendPtr<T>(*const T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    #[inline] fn ptr(self) -> *const T { self.0 }
    #[inline] unsafe fn add(self, n: usize) -> *const T { self.0.add(n) }
}

#[derive(Clone, Copy)]
struct SendMutPtr<T>(*mut T);
unsafe impl<T> Send for SendMutPtr<T> {}
unsafe impl<T> Sync for SendMutPtr<T> {}
impl<T> SendMutPtr<T> {
    #[inline] unsafe fn add(self, n: usize) -> *mut T { self.0.add(n) }
}

// ---------------------------------------------------------------------------
// Parallel Q4K
// ---------------------------------------------------------------------------

pub(super) fn par_q4k_matvec(
    pool: &ThreadPool,
    weight: *const u8,
    input_qs: &[i8], input_d: &[f32], input_bsums: &[i16],
    output: &mut [f32],
    n_rows: usize, n_cols: usize,
) {
    let n_blocks = n_cols / Q4K_BLOCK_SIZE;
    let row_bytes = n_blocks * Q4K_BLOCK_BYTES;
    let full_quads = n_rows / 4;
    let n_threads = pool.thread_count().min(full_quads);
    if n_threads <= 1 {
        q4k_matvec(weight, input_qs, input_d, input_bsums, output, n_rows, n_cols);
        return;
    }

    let pow2 = pow2_table();
    let q8 = SendPtr(input_qs.as_ptr());
    let bsums = SendPtr(input_bsums.as_ptr());
    let q8_d = SendPtr(input_d.as_ptr());
    let pow2_ptr = SendPtr(pow2.as_ptr());
    let out_ptr = SendMutPtr(output.as_mut_ptr());
    let w = SendPtr(weight);

    pool.run(n_threads, move |tid, nt| {
        let start_quad = tid * full_quads / nt;
        let end_quad = (tid + 1) * full_quads / nt;
        unsafe {
            for quad in start_quad..end_quad {
                let base_row = quad * 4;
                ffi_inference::q4k_dot_q8k_4row(
                    w.add(base_row * row_bytes),
                    w.add((base_row + 1) * row_bytes),
                    w.add((base_row + 2) * row_bytes),
                    w.add((base_row + 3) * row_bytes),
                    q8.ptr(), bsums.ptr(),
                    out_ptr.add(base_row),
                    n_blocks as i32, q8_d.ptr(), pow2_ptr.ptr(),
                );
            }
        }
    });

    let remainder = n_rows % 4;
    if remainder > 0 {
        let base = full_quads * 4;
        let q8 = input_qs.as_ptr();
        let bsums = input_bsums.as_ptr();
        let q8_d = input_d.as_ptr();
        let pow2_ptr = pow2.as_ptr();
        for i in 0..remainder {
            let row = base + i;
            unsafe {
                output[row] = ffi_inference::q4k_dot_q8k(
                    weight.add(row * row_bytes), q8, bsums,
                    n_blocks as i32, q8_d, pow2_ptr,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Parallel Q5K
// ---------------------------------------------------------------------------

pub(super) fn par_q5k_matvec(
    pool: &ThreadPool,
    weight: *const u8,
    input_qs: &[i8], input_d: &[f32], input_bsums: &[i16],
    output: &mut [f32],
    n_rows: usize, n_cols: usize,
) {
    let n_blocks = n_cols / Q5K_BLOCK_SIZE;
    let row_bytes = n_blocks * Q5K_BLOCK_BYTES;
    let full_quads = n_rows / 4;
    let n_threads = pool.thread_count().min(full_quads);
    if n_threads <= 1 {
        q5k_matvec(weight, input_qs, input_d, input_bsums, output, n_rows, n_cols);
        return;
    }

    let pow2 = pow2_table();
    let q8 = SendPtr(input_qs.as_ptr());
    let bsums = SendPtr(input_bsums.as_ptr());
    let q8_d = SendPtr(input_d.as_ptr());
    let pow2_ptr = SendPtr(pow2.as_ptr());
    let out_ptr = SendMutPtr(output.as_mut_ptr());
    let w = SendPtr(weight);

    pool.run(n_threads, move |tid, nt| {
        let start_quad = tid * full_quads / nt;
        let end_quad = (tid + 1) * full_quads / nt;
        unsafe {
            for quad in start_quad..end_quad {
                let base_row = quad * 4;
                ffi_inference::q5k_dot_q8k_4row(
                    w.add(base_row * row_bytes),
                    w.add((base_row + 1) * row_bytes),
                    w.add((base_row + 2) * row_bytes),
                    w.add((base_row + 3) * row_bytes),
                    q8.ptr(), bsums.ptr(),
                    out_ptr.add(base_row),
                    n_blocks as i32, q8_d.ptr(), pow2_ptr.ptr(),
                );
            }
        }
    });

    let remainder = n_rows % 4;
    if remainder > 0 {
        let base = full_quads * 4;
        let q8 = input_qs.as_ptr();
        let bsums = input_bsums.as_ptr();
        let q8_d = input_d.as_ptr();
        let pow2_ptr = pow2.as_ptr();
        for i in 0..remainder {
            let row = base + i;
            unsafe {
                output[row] = ffi_inference::q5k_dot_q8k(
                    weight.add(row * row_bytes), q8, bsums,
                    n_blocks as i32, q8_d, pow2_ptr,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Parallel Q6K
// ---------------------------------------------------------------------------

pub(super) fn par_q6k_matvec(
    pool: &ThreadPool,
    weight: *const u8,
    input_qs: &[i8], input_d: &[f32], input_bsums: &[i16],
    output: &mut [f32],
    _d_scratch: &mut [f32],
    n_rows: usize, n_cols: usize,
) {
    let n_blocks = n_cols / Q6K_BLOCK_SIZE;
    let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
    let full_quads = n_rows / 4;
    let n_threads = pool.thread_count().min(full_quads);
    if n_threads <= 1 {
        q6k_matvec(weight, input_qs, input_d, input_bsums, output, _d_scratch, n_rows, n_cols);
        return;
    }

    let q8 = SendPtr(input_qs.as_ptr());
    let bsums = SendPtr(input_bsums.as_ptr());
    let out_ptr = SendMutPtr(output.as_mut_ptr());
    let w = SendPtr(weight);
    let id = SendPtr(input_d.as_ptr());

    pool.run(n_threads, move |tid, nt| {
        let mut d_local = vec![0.0f32; n_blocks * 4];
        let start_quad = tid * full_quads / nt;
        let end_quad = (tid + 1) * full_quads / nt;
        // Rebuild input_d slice from pointer for q6k_extract_d
        let input_d_slice = unsafe { std::slice::from_raw_parts(id.ptr(), n_blocks) };
        unsafe {
            for quad in start_quad..end_quad {
                let base_row = quad * 4;
                let w0 = w.add(base_row * row_bytes);
                let w1 = w.add((base_row + 1) * row_bytes);
                let w2 = w.add((base_row + 2) * row_bytes);
                let w3 = w.add((base_row + 3) * row_bytes);
                let (d0, rest) = d_local.split_at_mut(n_blocks);
                let (d1, rest) = rest.split_at_mut(n_blocks);
                let (d2, d3) = rest.split_at_mut(n_blocks);
                q6k_extract_d(w0, n_blocks, input_d_slice, d0);
                q6k_extract_d(w1, n_blocks, input_d_slice, d1);
                q6k_extract_d(w2, n_blocks, input_d_slice, d2);
                q6k_extract_d(w3, n_blocks, input_d_slice, d3);
                ffi_inference::q6k_dot_q8k_4row(
                    w0, w1, w2, w3,
                    q8.ptr(), bsums.ptr(),
                    out_ptr.add(base_row),
                    n_blocks as i32,
                    d0.as_ptr(), d1.as_ptr(), d2.as_ptr(), d3.as_ptr(),
                );
            }
        }
    });

    let remainder = n_rows % 4;
    if remainder > 0 {
        let base = full_quads * 4;
        let d0 = &mut _d_scratch[..n_blocks];
        let q8 = input_qs.as_ptr();
        let bsums = input_bsums.as_ptr();
        for i in 0..remainder {
            let row = base + i;
            unsafe {
                let w = weight.add(row * row_bytes);
                q6k_extract_d(w, n_blocks, input_d, d0);
                output[row] = ffi_inference::q6k_dot_q8k(
                    w, q8, bsums, n_blocks as i32, d0.as_ptr(),
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Parallel Q4K dual gate+up
// ---------------------------------------------------------------------------

/// Parallel Q4K dual gate+up matvec. Called directly from `forward_attn.rs`
/// for the fused FFN path (gate projection + up projection share Q8K input).
pub fn par_q4k_matvec_dual(
    pool: &ThreadPool,
    gate_weight: *const u8,
    up_weight: *const u8,
    input_qs: &[i8], input_d: &[f32], input_bsums: &[i16],
    gate_output: &mut [f32],
    up_output: &mut [f32],
    n_rows: usize, n_cols: usize,
) {
    let n_blocks = n_cols / Q4K_BLOCK_SIZE;
    let row_bytes = n_blocks * Q4K_BLOCK_BYTES;
    let full_quads = n_rows / 4;
    let n_threads = pool.thread_count().min(full_quads);
    if n_threads <= 1 {
        q4k_matvec_dual(gate_weight, up_weight, input_qs, input_d, input_bsums,
                        gate_output, up_output, n_rows, n_cols);
        return;
    }

    let pow2 = pow2_table();
    let q8 = SendPtr(input_qs.as_ptr());
    let bsums = SendPtr(input_bsums.as_ptr());
    let q8_d = SendPtr(input_d.as_ptr());
    let pow2_ptr = SendPtr(pow2.as_ptr());
    let gate_ptr = SendMutPtr(gate_output.as_mut_ptr());
    let up_ptr = SendMutPtr(up_output.as_mut_ptr());
    let gw = SendPtr(gate_weight);
    let uw = SendPtr(up_weight);

    pool.run(n_threads, move |tid, nt| {
        let start_quad = tid * full_quads / nt;
        let end_quad = (tid + 1) * full_quads / nt;
        unsafe {
            for quad in start_quad..end_quad {
                let base_row = quad * 4;
                ffi_inference::q4k_dot_q8k_4row_dual(
                    gw.add(base_row * row_bytes),
                    gw.add((base_row + 1) * row_bytes),
                    gw.add((base_row + 2) * row_bytes),
                    gw.add((base_row + 3) * row_bytes),
                    uw.add(base_row * row_bytes),
                    uw.add((base_row + 1) * row_bytes),
                    uw.add((base_row + 2) * row_bytes),
                    uw.add((base_row + 3) * row_bytes),
                    q8.ptr(), bsums.ptr(),
                    gate_ptr.add(base_row),
                    up_ptr.add(base_row),
                    n_blocks as i32, q8_d.ptr(), pow2_ptr.ptr(),
                );
            }
        }
    });

    let remainder = n_rows % 4;
    if remainder > 0 {
        let base = full_quads * 4;
        let q8 = input_qs.as_ptr();
        let bsums = input_bsums.as_ptr();
        let q8_d = input_d.as_ptr();
        let pow2_ptr = pow2.as_ptr();
        for i in 0..remainder {
            let row = base + i;
            unsafe {
                gate_output[row] = ffi_inference::q4k_dot_q8k(
                    gate_weight.add(row * row_bytes), q8, bsums, n_blocks as i32, q8_d, pow2_ptr,
                );
                up_output[row] = ffi_inference::q4k_dot_q8k(
                    up_weight.add(row * row_bytes), q8, bsums, n_blocks as i32, q8_d, pow2_ptr,
                );
            }
        }
    }
}

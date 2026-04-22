//! Primitive helpers shared by the graph decode layer forward.

use std::sync::atomic::AtomicI32;
use crate::inference::matmul_graph;
use crate::kernels::ffi_inference;

/// Parallel Q8K quantization across threads, split by 256-element blocks.
#[inline]
pub(super) fn parallel_quant_decode(
    src: *const f32, qs: *mut i8, d: *mut f32, bsums: *mut i16,
    dim: usize, ith: usize, nth: usize,
) {
    let nb = dim / 256;
    let per = (nb + nth - 1) / nth;
    let start = ith * per;
    let end = (start + per).min(nb);
    if start < nb {
        let n = (end - start) * 256;
        unsafe {
            ffi_inference::quant_f32_q8k(
                src.add(start * 256), qs.add(start * 256),
                d.add(start), bsums.add(start * 16), n as i32,
            );
        }
    }
}

/// Dispatch a single matvec_ws call through either the repacked 8x8 path
/// (Q4K), the repacked 4-row pre-d path (Q6K), or the standard matvec_ws
/// path, depending on whether the weight has been repacked at model load
/// time. `q6k_repacked` + `q6k_d_arr` must both be Some together
/// (populated by engine_helpers::populate_q4k_repacked).
#[inline]
#[allow(clippy::too_many_arguments)]
pub(super) fn matvec_step(
    dtype: u32,
    weight: *const u8,
    repacked: Option<&[u8]>,
    q6k_repacked: Option<&[u8]>,
    q6k_d_arr: Option<&[f32]>,
    q8: *const i8,
    q8_d: *const f32,
    bsums: *const i16,
    output: *mut f32,
    d_scratch: *mut f32,
    n_rows: usize,
    n_cols: usize,
    current_chunk: &AtomicI32,
    ith: usize,
    nth: usize,
) {
    if let Some(p) = repacked {
        matmul_graph::q4k_matvec_8x8_ws(
            p.as_ptr(), q8, q8_d, bsums, output,
            n_rows, n_cols, current_chunk, ith, nth,
        );
        return;
    }
    if let (Some(p), Some(d)) = (q6k_repacked, q6k_d_arr) {
        matmul_graph::q6k_repacked_batch_ws_pre_d(
            p.as_ptr(), d.as_ptr(),
            q8, q8_d, bsums,
            output, d_scratch,
            n_rows, n_cols, 1, n_rows,
            current_chunk, ith, nth,
        );
        return;
    }
    matmul_graph::matvec_ws(
        dtype, weight, q8, q8_d, bsums, output, d_scratch,
        n_rows, n_cols, current_chunk, ith, nth,
    );
}

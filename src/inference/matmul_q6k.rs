//! Q6_K matmul dispatch: reads block layout, calls SIMD dot-product kernels.

use crate::kernels::ffi_inference as ffi;
use crate::inference::matmul::f16_to_f32;
use crate::inference::ptr::{SendPtr, SendMutPtr};

/// Bytes per Q6_K super-block (256 elements).
pub(crate) const Q6K_BLOCK_BYTES: usize = 210;

const Q6K_QL_OFF: usize = 0;
const Q6K_QH_OFF: usize = 128;
const Q6K_SC_OFF: usize = 192;
const Q6K_D_OFF: usize = 208;


/// Max blocks on stack (same as Q4K).
pub(crate) const MAX_BLOCKS: usize = 128;

/// Pre-compute per-block d_arr (f16→f32 × q8_d) for one Q6K weight row.
unsafe fn q6k_unpack_d(
    weight: *const u8, n_blocks: usize, q8_d: *const f32,
    d_arr: &mut [f32],
) {
    for blk in 0..n_blocks {
        let bp = weight.add(blk * Q6K_BLOCK_BYTES);
        d_arr[blk] = f16_to_f32(*(bp.add(Q6K_D_OFF) as *const u16)) * *q8_d.add(blk);
    }
}

/// Dot product of one Q6_K weight row against Q8_K activations.
pub(crate) unsafe fn q6k_row_dot(
    weight: *const u8, n_blocks: usize,
    q8_qs: *const i8, q8_d: *const f32, q8_bsums: *const i32,
) -> f32 {
    debug_assert!(n_blocks <= MAX_BLOCKS);
    let mut d_arr = [0f32; MAX_BLOCKS];
    q6k_unpack_d(weight, n_blocks, q8_d, &mut d_arr);
    ffi::q6k_dot_q8k(
        weight, q8_qs, q8_bsums,
        n_blocks as i32, d_arr.as_ptr(),
    )
}

/// 4-row Q6_K x Q8_K dot product with shared activations.
pub(crate) unsafe fn q6k_4row_dot(
    w0: *const u8, w1: *const u8, w2: *const u8, w3: *const u8,
    n_blocks: usize,
    q8_qs: *const i8, q8_d: *const f32, q8_bsums: *const i32,
    scores: &mut [f32; 4],
) {
    debug_assert!(n_blocks <= MAX_BLOCKS);
    let mut da = [[0f32; MAX_BLOCKS]; 4];
    let ws = [w0, w1, w2, w3];
    for i in 0..4 {
        q6k_unpack_d(ws[i], n_blocks, q8_d, &mut da[i]);
    }
    ffi::q6k_dot_q8k_4row(
        w0, w1, w2, w3,
        q8_qs, q8_bsums, scores.as_mut_ptr(), n_blocks as i32,
        da[0].as_ptr(), da[1].as_ptr(), da[2].as_ptr(), da[3].as_ptr(),
    );
}

/// Multi-threaded Q6_K x Q8_K matrix multiplication.
pub(crate) fn q6k_matmul_mt(
    weight: *const u8, row_stride: usize, n_blocks: usize,
    q8_qs: *const i8, q8_d: *const f32, q8_bsums: *const i32,
    out: &mut [f32], out_dim: usize,
    pool: &crate::inference::threadpool::ThreadPool,
) {
    let n_thr = pool.thread_count().min(out_dim / 4).max(1);

    if n_thr <= 1 {
        let mut scores4 = [0.0f32; 4];
        let mut r = 0;
        unsafe {
            while r + 4 <= out_dim {
                q6k_4row_dot(weight.add(r * row_stride), weight.add((r+1) * row_stride),
                    weight.add((r+2) * row_stride), weight.add((r+3) * row_stride),
                    n_blocks, q8_qs, q8_d, q8_bsums, &mut scores4);
                for j in 0..4 { out[r+j] = scores4[j]; }
                r += 4;
            }
            while r < out_dim {
                out[r] = q6k_row_dot(weight.add(r * row_stride), n_blocks, q8_qs, q8_d, q8_bsums);
                r += 1;
            }
        }
        return;
    }

    let w = SendPtr(weight);
    let qs = SendPtr(q8_qs);
    let qd = SendPtr(q8_d);
    let qb = SendPtr(q8_bsums);
    let o = SendMutPtr(out.as_mut_ptr());

    pool.run(n_thr, move |tid, _n| {
        unsafe {
            q6k_matmul_work(w.ptr(), row_stride, n_blocks,
                qs.ptr(), qd.ptr(), qb.ptr(), o.ptr(), out_dim, tid, n_thr);
        }
    });
}

/// Per-thread work function for Q6_K matmul.
/// Buffers allocated once per thread, reused across row iterations.
pub(crate) unsafe fn q6k_matmul_work(
    weight: *const u8, row_stride: usize, n_blocks: usize,
    q8_qs: *const i8, q8_d: *const f32, q8_bsums: *const i32,
    out: *mut f32, out_dim: usize, tid: usize, n_threads: usize,
) {
    let chunk = ((out_dim + n_threads - 1) / n_threads + 3) & !3;
    let start = tid * chunk;
    let end = (start + chunk).min(out_dim);
    if start >= end { return; }
    let count = end - start;
    let out_slice = std::slice::from_raw_parts_mut(out.add(start), count);

    let mut da = [[0f32; MAX_BLOCKS]; 4];
    let mut scores4 = [0.0f32; 4];

    let mut r = 0;
    while r + 4 <= count {
        let row = start + r;
        let ws = [weight.add(row * row_stride), weight.add((row+1) * row_stride),
                  weight.add((row+2) * row_stride), weight.add((row+3) * row_stride)];
        for i in 0..4 {
            q6k_unpack_d(ws[i], n_blocks, q8_d, &mut da[i]);
        }
        ffi::q6k_dot_q8k_4row(
            ws[0], ws[1], ws[2], ws[3],
            q8_qs, q8_bsums, scores4.as_mut_ptr(), n_blocks as i32,
            da[0].as_ptr(), da[1].as_ptr(), da[2].as_ptr(), da[3].as_ptr(),
        );
        for j in 0..4 { out_slice[r+j] = scores4[j]; }
        r += 4;
    }
    while r < count {
        let row = start + r;
        q6k_unpack_d(weight.add(row * row_stride), n_blocks, q8_d, &mut da[0]);
        out_slice[r] = ffi::q6k_dot_q8k(
            weight.add(row * row_stride), q8_qs, q8_bsums,
            n_blocks as i32, da[0].as_ptr(),
        );
        r += 1;
    }
}

/// Per-thread matmul + residual add. Computes out[r] += matmul(w, q8)[r].
/// Eliminates the separate vecadd pass after projection.
pub(crate) unsafe fn q6k_matmul_residual_work(
    weight: *const u8, row_stride: usize, n_blocks: usize,
    q8_qs: *const i8, q8_d: *const f32, q8_bsums: *const i32,
    out: *mut f32, out_dim: usize, tid: usize, n_threads: usize,
) {
    let chunk = ((out_dim + n_threads - 1) / n_threads + 3) & !3;
    let start = tid * chunk;
    let end = (start + chunk).min(out_dim);
    if start >= end { return; }
    let count = end - start;

    let mut da = [[0f32; MAX_BLOCKS]; 4];
    let mut scores4 = [0.0f32; 4];

    let mut r = 0;
    while r + 4 <= count {
        let row = start + r;
        let ws = [weight.add(row * row_stride), weight.add((row+1) * row_stride),
                  weight.add((row+2) * row_stride), weight.add((row+3) * row_stride)];
        for i in 0..4 {
            q6k_unpack_d(ws[i], n_blocks, q8_d, &mut da[i]);
        }
        ffi::q6k_dot_q8k_4row(
            ws[0], ws[1], ws[2], ws[3],
            q8_qs, q8_bsums, scores4.as_mut_ptr(), n_blocks as i32,
            da[0].as_ptr(), da[1].as_ptr(), da[2].as_ptr(), da[3].as_ptr(),
        );
        for j in 0..4 { *out.add(start + r + j) += scores4[j]; }
        r += 4;
    }
    while r < count {
        let row = start + r;
        q6k_unpack_d(weight.add(row * row_stride), n_blocks, q8_d, &mut da[0]);
        *out.add(start + r) += ffi::q6k_dot_q8k(
            weight.add(row * row_stride), q8_qs, q8_bsums,
            n_blocks as i32, da[0].as_ptr(),
        );
        r += 1;
    }
}

/// Dequantize a single embedding row from Q6_K block data to f32.
/// Matches llama.cpp's `dequantize_row_q6_K` element ordering exactly.
pub(crate) fn q6k_embed_lookup(
    embed_data: *const u8, token: u32, out: &mut [f32], hidden_dim: usize,
) {
    let n_blocks = hidden_dim / 256;
    let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
    let row_ptr = unsafe { embed_data.add(token as usize * row_bytes) };

    for blk in 0..n_blocks {
        let block = unsafe { row_ptr.add(blk * Q6K_BLOCK_BYTES) };
        let d = f16_to_f32(unsafe { *(block.add(Q6K_D_OFF) as *const u16) });
        let mut ql = unsafe { block.add(Q6K_QL_OFF) };
        let mut qh = unsafe { block.add(Q6K_QH_OFF) };
        let mut sc = unsafe { block.add(Q6K_SC_OFF) as *const i8 };
        let mut y = blk * 256;

        // Two halves of 128 elements each (matching llama.cpp's n += 128 loop)
        for _half in 0..2 {
            for l in 0..32usize {
                let is = l / 16;
                let ql0 = unsafe { *ql.add(l) };
                let ql32 = unsafe { *ql.add(l + 32) };
                let qh_byte = unsafe { *qh.add(l) };

                let q1 = ((ql0 & 0xF) | (((qh_byte >> 0) & 3) << 4)) as i8 as f32 - 32.0;
                let q2 = ((ql32 & 0xF) | (((qh_byte >> 2) & 3) << 4)) as i8 as f32 - 32.0;
                let q3 = ((ql0 >> 4) | (((qh_byte >> 4) & 3) << 4)) as i8 as f32 - 32.0;
                let q4 = ((ql32 >> 4) | (((qh_byte >> 6) & 3) << 4)) as i8 as f32 - 32.0;

                let s0 = unsafe { *sc.add(is as usize) } as f32;
                let s2 = unsafe { *sc.add(is as usize + 2) } as f32;
                let s4 = unsafe { *sc.add(is as usize + 4) } as f32;
                let s6 = unsafe { *sc.add(is as usize + 6) } as f32;

                out[y + l] = d * s0 * q1;
                out[y + l + 32] = d * s2 * q2;
                out[y + l + 64] = d * s4 * q3;
                out[y + l + 96] = d * s6 * q4;
            }
            y += 128;
            ql = unsafe { ql.add(64) };
            qh = unsafe { qh.add(32) };
            sc = unsafe { sc.add(8) };
        }
    }
}

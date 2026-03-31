//! GEMM-style batched Q6_K matmul: load weight once, multiply against N tokens.

use crate::inference::gemm_q4k::BatchQ8K;
use crate::inference::matmul_q6k::{q6k_4row_dot, q6k_row_dot};
use crate::inference::ptr::{SendPtr, SendMutPtr};

use crate::inference::threadpool::ThreadPool;

/// GEMM: Q6_K weight[out_dim] x batch[n_tokens] -> out[n_tokens * out_dim]
pub(crate) fn q6k_gemm_mt(
    weight: *const u8, row_stride: usize, n_blocks: usize,
    batch: &BatchQ8K, out: &mut [f32], out_dim: usize,
    pool: &ThreadPool,
) {
    let nt = batch.n_tokens;
    let total = pool.thread_count().min(out_dim / 4).max(1);
    let qs: Vec<usize> = (0..nt).map(|t| batch.qs_ptr(t) as usize).collect();
    let ds: Vec<usize> = (0..nt).map(|t| batch.d_ptr(t) as usize).collect();
    let bs: Vec<usize> = (0..nt).map(|t| batch.bsums_ptr(t) as usize).collect();
    let chunk = ((out_dim + total - 1) / total + 3) & !3;
    let w = SendPtr(weight);
    let o = SendMutPtr(out.as_mut_ptr());

    pool.run(total, move |tid, _n| {
            let start = tid * chunk;
            let end = (start + chunk).min(out_dim);
            if start >= end { return; }
                let mut scores = [0.0f32; 4];
                let mut r = start;
                unsafe {
                    while r + 4 <= end {
                        let w0 = w.ptr().add(r * row_stride);
                        let w1 = w.ptr().add((r+1) * row_stride);
                        let w2 = w.ptr().add((r+2) * row_stride);
                        let w3 = w.ptr().add((r+3) * row_stride);
                        for t in 0..nt {
                            q6k_4row_dot(w0, w1, w2, w3, n_blocks,
                                qs[t] as _, ds[t] as _, bs[t] as _, &mut scores);
                            let base = o.ptr().add(t * out_dim + r);
                            *base = scores[0]; *base.add(1) = scores[1];
                            *base.add(2) = scores[2]; *base.add(3) = scores[3];
                        }
                        r += 4;
                    }
                    while r < end {
                        let wr = w.ptr().add(r * row_stride);
                        for t in 0..nt {
                            let v = q6k_row_dot(wr, n_blocks, qs[t] as _, ds[t] as _, bs[t] as _);
                            *o.ptr().add(t * out_dim + r) = v;
                        }
                        r += 1;
                    }
                }
    });
}

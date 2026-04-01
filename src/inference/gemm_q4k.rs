//! GEMM-style batched Q4_K matmul: load weight once, multiply against N tokens.

use crate::kernels::ffi_inference as ffi;
use crate::inference::matmul_q4k::{q4k_row_dot, Q4K_BLOCK_BYTES, MAX_BLOCKS};
use crate::inference::matmul::f16_to_f32;
use crate::inference::ptr::{SendPtr, SendMutPtr};

use crate::inference::threadpool::ThreadPool;

/// Batched Q8K activation data for N tokens.
pub(crate) struct BatchQ8K {
    pub n_tokens: usize,
    pub dim: usize,
    pub n_blocks: usize,
    pub qs_stride: usize,
    pub qs: Vec<i8>,
    pub d: Vec<f32>,
    pub bsums: Vec<i32>,
}

impl BatchQ8K {
    pub fn new(n_tokens: usize, dim: usize) -> Self {
        let n_blocks = dim / 256;
        let qs_stride = dim + 16;
        BatchQ8K {
            n_tokens, dim, n_blocks, qs_stride,
            qs: vec![0i8; n_tokens * qs_stride],
            d: vec![0.0f32; n_tokens * n_blocks],
            bsums: vec![0i32; n_tokens * n_blocks * 16],
        }
    }

    pub fn quantize(&mut self, t: usize, src: &[f32]) {
        unsafe {
            ffi::quant_f32_q8k(
                src.as_ptr(),
                self.qs.as_mut_ptr().add(t * self.qs_stride),
                self.d.as_mut_ptr().add(t * self.n_blocks),
                self.bsums.as_mut_ptr().add(t * self.n_blocks * 16),
                self.dim as i32,
            );
        }
    }

    pub fn qs_ptr(&self, t: usize) -> *const i8 {
        unsafe { self.qs.as_ptr().add(t * self.qs_stride) }
    }
    pub fn d_ptr(&self, t: usize) -> *const f32 {
        unsafe { self.d.as_ptr().add(t * self.n_blocks) }
    }
    pub fn bsums_ptr(&self, t: usize) -> *const i32 {
        unsafe { self.bsums.as_ptr().add(t * self.n_blocks * 16) }
    }
}

/// GEMM: weight[out_dim] x batch[n_tokens] -> out[n_tokens * out_dim]
pub(crate) fn q4k_gemm_mt(
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
                let mut scores4 = [0.0f32; 4];
                // Pre-cache f16→f32 weight scales ONCE (invariant across tokens)
                let mut da_w = [[0.0f32; MAX_BLOCKS]; 4];
                let mut dma_w = [[0.0f32; MAX_BLOCKS]; 4];
                let mut r = start;
                unsafe {
                    while r + 4 <= end {
                        let ws = [
                            w.ptr().add(r * row_stride),
                            w.ptr().add((r+1) * row_stride),
                            w.ptr().add((r+2) * row_stride),
                            w.ptr().add((r+3) * row_stride),
                        ];
                        for ri in 0..4 {
                            for blk in 0..n_blocks {
                                let bp = ws[ri].add(blk * Q4K_BLOCK_BYTES);
                                da_w[ri][blk] = f16_to_f32(*(bp as *const u16));
                                dma_w[ri][blk] = f16_to_f32(*(bp.add(2) as *const u16));
                            }
                        }
                        for t in 0..nt {
                            // Fused: kernel multiplies d_w * q8_d inline
                            ffi::q4k_dot_q8k_4row_fused(
                                ws[0], ws[1], ws[2], ws[3],
                                qs[t] as _, bs[t] as _,
                                scores4.as_mut_ptr(), n_blocks as i32,
                                da_w[0].as_ptr(), da_w[1].as_ptr(), da_w[2].as_ptr(), da_w[3].as_ptr(),
                                dma_w[0].as_ptr(), dma_w[1].as_ptr(), dma_w[2].as_ptr(), dma_w[3].as_ptr(),
                                ds[t] as _,
                            );
                            let base = o.ptr().add(t * out_dim + r);
                            for j in 0..4 { *base.add(j) = scores4[j]; }
                        }
                        r += 4;
                    }
                    while r < end {
                        let wr = w.ptr().add(r * row_stride);
                        for t in 0..nt {
                            let v = q4k_row_dot(wr, n_blocks, qs[t] as _, ds[t] as _, bs[t] as _);
                            *o.ptr().add(t * out_dim + r) = v;
                        }
                        r += 1;
                    }
                }
    });
}

/// Fused gate+up+SiLU GEMM: compute silu(gate) * up for all tokens.
/// Weight-stationary tiling: load 4 gate + 4 up rows once, multiply all tokens.
pub(crate) fn q4k_fused_silu_gemm_mt(
    w_gate: *const u8, w_up: *const u8,
    row_stride: usize, n_blocks: usize,
    batch: &BatchQ8K, out: &mut [f32], out_dim: usize,
    pool: &ThreadPool,
) {
    let nt = batch.n_tokens;
    let total = pool.thread_count().min(out_dim / 4).max(1);
    let qs: Vec<usize> = (0..nt).map(|t| batch.qs_ptr(t) as usize).collect();
    let ds: Vec<usize> = (0..nt).map(|t| batch.d_ptr(t) as usize).collect();
    let bs: Vec<usize> = (0..nt).map(|t| batch.bsums_ptr(t) as usize).collect();
    let chunk = ((out_dim + total - 1) / total + 3) & !3;
    let wg = SendPtr(w_gate);
    let wu = SendPtr(w_up);
    let o = SendMutPtr(out.as_mut_ptr());

    pool.run(total, move |tid, _n| {
        let start = tid * chunk;
        let end = (start + chunk).min(out_dim);
        if start >= end { return; }
        // Pre-cache f16→f32 weight scales ONCE per 4-row group
        let mut gd_w = [[0f32; MAX_BLOCKS]; 4];
        let mut gdm_w = [[0f32; MAX_BLOCKS]; 4];
        let mut ud_w = [[0f32; MAX_BLOCKS]; 4];
        let mut udm_w = [[0f32; MAX_BLOCKS]; 4];
        let mut g_scores = [0.0f32; 4];
        let mut u_scores = [0.0f32; 4];
        let mut r = start;
        unsafe {
            while r + 4 <= end {
                let gws = [wg.ptr().add(r * row_stride), wg.ptr().add((r+1) * row_stride),
                           wg.ptr().add((r+2) * row_stride), wg.ptr().add((r+3) * row_stride)];
                let uws = [wu.ptr().add(r * row_stride), wu.ptr().add((r+1) * row_stride),
                           wu.ptr().add((r+2) * row_stride), wu.ptr().add((r+3) * row_stride)];
                for i in 0..4 {
                    for blk in 0..n_blocks {
                        let gbp = gws[i].add(blk * Q4K_BLOCK_BYTES);
                        let ubp = uws[i].add(blk * Q4K_BLOCK_BYTES);
                        gd_w[i][blk] = f16_to_f32(*(gbp as *const u16));
                        gdm_w[i][blk] = f16_to_f32(*(gbp.add(2) as *const u16));
                        ud_w[i][blk] = f16_to_f32(*(ubp as *const u16));
                        udm_w[i][blk] = f16_to_f32(*(ubp.add(2) as *const u16));
                    }
                }
                for t in 0..nt {
                    // Fused: kernel multiplies d_w * q8_d inline (no Rust pre-multiply)
                    ffi::q4k_dot_q8k_4row_fused(
                        gws[0], gws[1], gws[2], gws[3],
                        qs[t] as _, bs[t] as _,
                        g_scores.as_mut_ptr(), n_blocks as i32,
                        gd_w[0].as_ptr(), gd_w[1].as_ptr(), gd_w[2].as_ptr(), gd_w[3].as_ptr(),
                        gdm_w[0].as_ptr(), gdm_w[1].as_ptr(), gdm_w[2].as_ptr(), gdm_w[3].as_ptr(),
                        ds[t] as _,
                    );
                    ffi::q4k_dot_q8k_4row_fused(
                        uws[0], uws[1], uws[2], uws[3],
                        qs[t] as _, bs[t] as _,
                        u_scores.as_mut_ptr(), n_blocks as i32,
                        ud_w[0].as_ptr(), ud_w[1].as_ptr(), ud_w[2].as_ptr(), ud_w[3].as_ptr(),
                        udm_w[0].as_ptr(), udm_w[1].as_ptr(), udm_w[2].as_ptr(), udm_w[3].as_ptr(),
                        ds[t] as _,
                    );
                    let base = o.ptr().add(t * out_dim + r);
                    for i in 0..4 {
                        let g = g_scores[i];
                        *base.add(i) = (g / (1.0 + (-g).exp())) * u_scores[i];
                    }
                }
                r += 4;
            }
            while r < end {
                let gw = wg.ptr().add(r * row_stride);
                let uw = wu.ptr().add(r * row_stride);
                for t in 0..nt {
                    let g = q4k_row_dot(gw, n_blocks, qs[t] as _, ds[t] as _, bs[t] as _);
                    let u = q4k_row_dot(uw, n_blocks, qs[t] as _, ds[t] as _, bs[t] as _);
                    *o.ptr().add(t * out_dim + r) = (g / (1.0 + (-g).exp())) * u;
                }
                r += 1;
            }
        }
    });
}

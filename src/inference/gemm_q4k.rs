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
                let mut scores16 = [0.0f32; 16];
                let mut dw = [[0.0f32; MAX_BLOCKS]; 4];
                let mut dmw = [[0.0f32; MAX_BLOCKS]; 4];
                let mut r = start;
                unsafe {
                    while r + 4 <= end {
                        let ws = [
                            w.ptr().add(r * row_stride),
                            w.ptr().add((r+1) * row_stride),
                            w.ptr().add((r+2) * row_stride),
                            w.ptr().add((r+3) * row_stride),
                        ];
                        // f16→f32 once per 4-row group
                        for ri in 0..4 {
                            for blk in 0..n_blocks {
                                let bp = ws[ri].add(blk * Q4K_BLOCK_BYTES);
                                dw[ri][blk] = f16_to_f32(*(bp as *const u16));
                                dmw[ri][blk] = f16_to_f32(*(bp.add(2) as *const u16));
                            }
                        }
                        // Tiled 4×4: 4 tokens at a time
                        let mut t = 0;
                        while t + 4 <= nt {
                            ffi::q4k_gemm_4x4(
                                ws[0], ws[1], ws[2], ws[3],
                                qs[t] as _, qs[t+1] as _, qs[t+2] as _, qs[t+3] as _,
                                bs[t] as _, bs[t+1] as _, bs[t+2] as _, bs[t+3] as _,
                                dw[0].as_ptr(), dw[1].as_ptr(), dw[2].as_ptr(), dw[3].as_ptr(),
                                dmw[0].as_ptr(), dmw[1].as_ptr(), dmw[2].as_ptr(), dmw[3].as_ptr(),
                                ds[t] as _, ds[t+1] as _, ds[t+2] as _, ds[t+3] as _,
                                scores16.as_mut_ptr(), n_blocks as i32,
                            );
                            for ri in 0..4 {
                                for ti in 0..4 {
                                    *o.ptr().add((t + ti) * out_dim + r + ri) = scores16[ri * 4 + ti];
                                }
                            }
                            t += 4;
                        }
                        // Remainder tokens: per-token 4-row
                        while t < nt {
                            use crate::inference::matmul_q4k::unpack_d;
                            let mut da = [[0f32; MAX_BLOCKS]; 4];
                            let mut dma = [[0f32; MAX_BLOCKS]; 4];
                            for ri in 0..4 {
                                unpack_d(ws[ri], n_blocks, ds[t] as _, &mut da[ri], &mut dma[ri]);
                            }
                            ffi::q4k_dot_q8k_4row(
                                ws[0], ws[1], ws[2], ws[3],
                                qs[t] as _, bs[t] as _,
                                scores4.as_mut_ptr(), n_blocks as i32,
                                da[0].as_ptr(), da[1].as_ptr(), da[2].as_ptr(), da[3].as_ptr(),
                                dma[0].as_ptr(), dma[1].as_ptr(), dma[2].as_ptr(), dma[3].as_ptr(),
                            );
                            let base = o.ptr().add(t * out_dim + r);
                            for j in 0..4 { *base.add(j) = scores4[j]; }
                            t += 1;
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
        // Pre-cache f16→f32 weight scales: computed ONCE per 4-row group,
        // then multiplied by per-token q8_d in the inner loop.
        let mut gd_w = [[0f32; MAX_BLOCKS]; 4];  // gate d (weight-only)
        let mut gdm_w = [[0f32; MAX_BLOCKS]; 4]; // gate dmin
        let mut ud_w = [[0f32; MAX_BLOCKS]; 4];  // up d
        let mut udm_w = [[0f32; MAX_BLOCKS]; 4]; // up dmin
        let mut gd = [[0f32; MAX_BLOCKS]; 4];
        let mut gdm = [[0f32; MAX_BLOCKS]; 4];
        let mut ud = [[0f32; MAX_BLOCKS]; 4];
        let mut udm = [[0f32; MAX_BLOCKS]; 4];
        let mut g_scores = [0.0f32; 4];
        let mut u_scores = [0.0f32; 4];
        let mut r = start;
        unsafe {
            while r + 4 <= end {
                let gws = [wg.ptr().add(r * row_stride), wg.ptr().add((r+1) * row_stride),
                           wg.ptr().add((r+2) * row_stride), wg.ptr().add((r+3) * row_stride)];
                let uws = [wu.ptr().add(r * row_stride), wu.ptr().add((r+1) * row_stride),
                           wu.ptr().add((r+2) * row_stride), wu.ptr().add((r+3) * row_stride)];
                // f16→f32 ONCE per 4-row group (invariant across tokens)
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
                    // Multiply cached weight-d by per-token q8_d (cheap: just n_blocks muls)
                    let q8_d_ptr = ds[t] as *const f32;
                    for i in 0..4 {
                        for blk in 0..n_blocks {
                            let q = *q8_d_ptr.add(blk);
                            gd[i][blk] = gd_w[i][blk] * q;
                            gdm[i][blk] = gdm_w[i][blk] * q;
                            ud[i][blk] = ud_w[i][blk] * q;
                            udm[i][blk] = udm_w[i][blk] * q;
                        }
                    }
                    ffi::q4k_dot_q8k_4row_dual(
                        gws[0], gws[1], gws[2], gws[3],
                        uws[0], uws[1], uws[2], uws[3],
                        qs[t] as _, bs[t] as _,
                        g_scores.as_mut_ptr(), u_scores.as_mut_ptr(), n_blocks as i32,
                        gd[0].as_ptr(), gd[1].as_ptr(), gd[2].as_ptr(), gd[3].as_ptr(),
                        gdm[0].as_ptr(), gdm[1].as_ptr(), gdm[2].as_ptr(), gdm[3].as_ptr(),
                        ud[0].as_ptr(), ud[1].as_ptr(), ud[2].as_ptr(), ud[3].as_ptr(),
                        udm[0].as_ptr(), udm[1].as_ptr(), udm[2].as_ptr(), udm[3].as_ptr(),
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

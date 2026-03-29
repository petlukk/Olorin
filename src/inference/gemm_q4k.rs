//! GEMM-style batched Q4_K matmul: load weight once, multiply against N tokens.

use crate::kernels::ffi_inference as ffi;
use crate::inference::matmul_q4k::{q4k_4row_dot, q4k_row_dot, q4k_dual_4row_dot, Q4K_BLOCK_BYTES, unpack_q4k_scales};
use crate::inference::matmul::f16_to_f32;
use crate::inference::ptr::{SendPtr, SendMutPtr};

fn n_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

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
) {
    let nt = batch.n_tokens;
    let total = n_threads().min(out_dim / 4).max(1);
    let qs: Vec<usize> = (0..nt).map(|t| batch.qs_ptr(t) as usize).collect();
    let ds: Vec<usize> = (0..nt).map(|t| batch.d_ptr(t) as usize).collect();
    let bs: Vec<usize> = (0..nt).map(|t| batch.bsums_ptr(t) as usize).collect();
    let chunk = ((out_dim + total - 1) / total + 3) & !3;
    let w = SendPtr(weight);
    let o = SendMutPtr(out.as_mut_ptr());

    std::thread::scope(|s| {
        for tid in 0..total {
            let start = tid * chunk;
            let end = (start + chunk).min(out_dim);
            if start >= end { continue; }
            let qs = &qs;
            let ds = &ds;
            let bs = &bs;
            s.spawn(move || {
                let mut scores4 = [0.0f32; 4];
                let mut scores16 = [0.0f32; 16];
                // Per-row pre-unpacked scale/min/d/dm arrays
                let mut sc = [vec![0u8; n_blocks * 8], vec![0u8; n_blocks * 8],
                              vec![0u8; n_blocks * 8], vec![0u8; n_blocks * 8]];
                let mut mn = [vec![0u8; n_blocks * 8], vec![0u8; n_blocks * 8],
                              vec![0u8; n_blocks * 8], vec![0u8; n_blocks * 8]];
                let mut da = [vec![0.0f32; n_blocks], vec![0.0f32; n_blocks],
                              vec![0.0f32; n_blocks], vec![0.0f32; n_blocks]];
                let mut dma = [vec![0.0f32; n_blocks], vec![0.0f32; n_blocks],
                               vec![0.0f32; n_blocks], vec![0.0f32; n_blocks]];
                let mut sc_tmp = [0u8; 8];
                let mut mn_tmp = [0u8; 8];
                let mut r = start;
                unsafe {
                    while r + 4 <= end {
                        let ws = [
                            w.ptr().add(r * row_stride),
                            w.ptr().add((r+1) * row_stride),
                            w.ptr().add((r+2) * row_stride),
                            w.ptr().add((r+3) * row_stride),
                        ];
                        // Pre-unpack for tiled kernel
                        for ri in 0..4 {
                            for blk in 0..n_blocks {
                                let bp = ws[ri].add(blk * Q4K_BLOCK_BYTES);
                                da[ri][blk] = f16_to_f32(*(bp as *const u16));
                                dma[ri][blk] = f16_to_f32(*(bp.add(2) as *const u16));
                                unpack_q4k_scales(
                                    std::slice::from_raw_parts(bp.add(4), 12),
                                    &mut sc_tmp, &mut mn_tmp);
                                sc[ri][blk*8..blk*8+8].copy_from_slice(&sc_tmp);
                                mn[ri][blk*8..blk*8+8].copy_from_slice(&mn_tmp);
                            }
                        }
                        // Tiled: 4 tokens at a time via SIMD kernel
                        let mut t = 0;
                        while t + 4 <= nt {
                            ffi::q4k_gemm_4x4(
                                ws[0], ws[1], ws[2], ws[3],
                                qs[t] as _, qs[t+1] as _, qs[t+2] as _, qs[t+3] as _,
                                bs[t] as _, bs[t+1] as _, bs[t+2] as _, bs[t+3] as _,
                                sc[0].as_ptr(), sc[1].as_ptr(), sc[2].as_ptr(), sc[3].as_ptr(),
                                mn[0].as_ptr(), mn[1].as_ptr(), mn[2].as_ptr(), mn[3].as_ptr(),
                                da[0].as_ptr(), da[1].as_ptr(), da[2].as_ptr(), da[3].as_ptr(),
                                dma[0].as_ptr(), dma[1].as_ptr(), dma[2].as_ptr(), dma[3].as_ptr(),
                                ds[t] as _, ds[t+1] as _, ds[t+2] as _, ds[t+3] as _,
                                scores16.as_mut_ptr(), n_blocks as i32,
                            );
                            // scores16: [r0t0, r0t1, r0t2, r0t3, r1t0, ..., r3t3]
                            for ri in 0..4 {
                                for ti in 0..4 {
                                    *o.ptr().add((t + ti) * out_dim + r + ri) = scores16[ri * 4 + ti];
                                }
                            }
                            t += 4;
                        }
                        // Remainder tokens
                        while t < nt {
                            q4k_4row_dot(ws[0], ws[1], ws[2], ws[3], n_blocks,
                                qs[t] as _, ds[t] as _, bs[t] as _, &mut scores4);
                            let base = o.ptr().add(t * out_dim + r);
                            *base = scores4[0]; *base.add(1) = scores4[1];
                            *base.add(2) = scores4[2]; *base.add(3) = scores4[3];
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
    });
}

/// Fused gate+up+SiLU GEMM: compute silu(gate) * up for all tokens.
pub(crate) fn q4k_fused_silu_gemm_mt(
    w_gate: *const u8, w_up: *const u8,
    row_stride: usize, n_blocks: usize,
    batch: &BatchQ8K, out: &mut [f32], out_dim: usize,
) {
    let nt = batch.n_tokens;
    let total = n_threads().min(out_dim / 4).max(1);
    let qs: Vec<usize> = (0..nt).map(|t| batch.qs_ptr(t) as usize).collect();
    let ds: Vec<usize> = (0..nt).map(|t| batch.d_ptr(t) as usize).collect();
    let bs: Vec<usize> = (0..nt).map(|t| batch.bsums_ptr(t) as usize).collect();
    let chunk = ((out_dim + total - 1) / total + 3) & !3;
    let wg = SendPtr(w_gate);
    let wu = SendPtr(w_up);
    let o = SendMutPtr(out.as_mut_ptr());

    std::thread::scope(|s| {
        for tid in 0..total {
            let start = tid * chunk;
            let end = (start + chunk).min(out_dim);
            if start >= end { continue; }
            let qs = &qs;
            let ds = &ds;
            let bs = &bs;
            s.spawn(move || {
                let mut g_scores = [0.0f32; 4];
                let mut u_scores = [0.0f32; 4];
                let mut r = start;
                unsafe {
                    while r + 4 <= end {
                        for t in 0..nt {
                            q4k_dual_4row_dot(
                                wg.ptr().add(r * row_stride), wg.ptr().add((r+1) * row_stride),
                                wg.ptr().add((r+2) * row_stride), wg.ptr().add((r+3) * row_stride),
                                wu.ptr().add(r * row_stride), wu.ptr().add((r+1) * row_stride),
                                wu.ptr().add((r+2) * row_stride), wu.ptr().add((r+3) * row_stride),
                                n_blocks, qs[t] as _, ds[t] as _, bs[t] as _,
                                &mut g_scores, &mut u_scores,
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
    });
}

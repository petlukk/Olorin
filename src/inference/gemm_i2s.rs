//! GEMM-style batched I2S matmul: load weight once, multiply against N tokens.

use crate::kernels::ffi_inference as ffi;
use crate::inference::ptr::{SendPtr, SendMutPtr};

fn n_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Batched i8 activation data for N tokens (I2S quantization).
pub(crate) struct BatchI8 {
    pub n_tokens: usize,
    pub dim: usize,
    pub stride: usize,
    pub scales: Vec<f32>,
    pub sums: Vec<i32>,
    pub qs: Vec<i8>,
}

impl BatchI8 {
    pub fn new(n_tokens: usize, dim: usize) -> Self {
        let stride = dim + 12;
        BatchI8 {
            n_tokens, dim, stride,
            scales: vec![0.0f32; n_tokens],
            sums: vec![0i32; n_tokens],
            qs: vec![0i8; n_tokens * stride],
        }
    }

    pub fn quantize(&mut self, t: usize, src: &[f32]) {
        unsafe {
            ffi::quant_f32_i8(
                src.as_ptr(),
                self.qs.as_mut_ptr().add(t * self.stride),
                self.scales.as_mut_ptr().add(t),
                self.sums.as_mut_ptr().add(t),
                self.dim as i32,
            );
        }
    }

    pub fn qs_ptr(&self, t: usize) -> *const i8 {
        unsafe { self.qs.as_ptr().add(t * self.stride) }
    }
    pub fn scale(&self, t: usize) -> f32 { self.scales[t] }
    pub fn sum(&self, t: usize) -> i32 { self.sums[t] }
}

/// GEMM: weight[out_dim x in_dim] x batch[n_tokens] -> out[n_tokens * out_dim]
pub(crate) fn i2s_gemm_mt(
    weight: *const u8, weight_scale: f32,
    batch: &BatchI8, out: &mut [f32],
    out_dim: usize, in_dim: usize,
) {
    let nt = batch.n_tokens;
    let row_bytes = in_dim / 4;
    let total = n_threads().min(out_dim / 4).max(1);
    let qs: Vec<usize> = (0..nt).map(|t| batch.qs_ptr(t) as usize).collect();
    let scales: Vec<f32> = (0..nt).map(|t| batch.scale(t)).collect();
    let sums: Vec<i32> = (0..nt).map(|t| batch.sum(t)).collect();
    let chunk = ((out_dim + total - 1) / total + 3) & !3;
    let w = SendPtr(weight);
    let o = SendMutPtr(out.as_mut_ptr());

    std::thread::scope(|s| {
        for tid in 0..total {
            let start = tid * chunk;
            let end = (start + chunk).min(out_dim);
            if start >= end { continue; }
            let qs = &qs;
            let scales = &scales;
            let sums = &sums;
            s.spawn(move || {
                let mut raw4 = [0i32; 4];
                let mut r = start;
                unsafe {
                    while r + 4 <= end {
                        let w0 = w.ptr().add(r * row_bytes);
                        let w1 = w.ptr().add((r+1) * row_bytes);
                        let w2 = w.ptr().add((r+2) * row_bytes);
                        let w3 = w.ptr().add((r+3) * row_bytes);
                        for t in 0..nt {
                            let combined = (scales[t] / 127.0) * weight_scale;
                            ffi::i2_dot_i8_4row(w0, w1, w2, w3,
                                qs[t] as *const i8, raw4.as_mut_ptr(), in_dim as i32);
                            let base = o.ptr().add(t * out_dim + r);
                            for j in 0..4 {
                                *base.add(j) = (raw4[j] - sums[t]) as f32 * combined;
                            }
                        }
                        r += 4;
                    }
                    while r < end {
                        for t in 0..nt {
                            let combined = (scales[t] / 127.0) * weight_scale;
                            let v = ffi::i2_dot_i8(
                                w.ptr().add(r * row_bytes), qs[t] as *const i8, in_dim as i32);
                            *o.ptr().add(t * out_dim + r) = (v - sums[t]) as f32 * combined;
                        }
                        r += 1;
                    }
                }
            });
        }
    });
}

/// Fused gate+up+SquaredReLU GEMM for I2S.
pub(crate) fn i2s_fused_sqrelu_gemm_mt(
    w_gate: *const u8, scale_gate: f32,
    w_up: *const u8, scale_up: f32,
    batch: &BatchI8, out: &mut [f32],
    out_dim: usize, in_dim: usize,
) {
    let nt = batch.n_tokens;
    let row_bytes = in_dim / 4;
    let total = n_threads().min(out_dim / 4).max(1);
    let qs: Vec<usize> = (0..nt).map(|t| batch.qs_ptr(t) as usize).collect();
    let scales: Vec<f32> = (0..nt).map(|t| batch.scale(t)).collect();
    let sums: Vec<i32> = (0..nt).map(|t| batch.sum(t)).collect();
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
            let scales = &scales;
            let sums = &sums;
            s.spawn(move || {
                let mut g_raw = [0i32; 4];
                let mut u_raw = [0i32; 4];
                let mut r = start;
                unsafe {
                    while r + 4 <= end {
                        for t in 0..nt {
                            let g_combined = (scales[t] / 127.0) * scale_gate;
                            let u_combined = (scales[t] / 127.0) * scale_up;
                            ffi::i2_dot_i8_4row_dual(
                                wg.ptr().add(r * row_bytes), wg.ptr().add((r+1) * row_bytes),
                                wg.ptr().add((r+2) * row_bytes), wg.ptr().add((r+3) * row_bytes),
                                wu.ptr().add(r * row_bytes), wu.ptr().add((r+1) * row_bytes),
                                wu.ptr().add((r+2) * row_bytes), wu.ptr().add((r+3) * row_bytes),
                                qs[t] as *const i8, g_raw.as_mut_ptr(), u_raw.as_mut_ptr(),
                                in_dim as i32,
                            );
                            let base = o.ptr().add(t * out_dim + r);
                            for j in 0..4 {
                                let g = (g_raw[j] - sums[t]) as f32 * g_combined;
                                let u = (u_raw[j] - sums[t]) as f32 * u_combined;
                                *base.add(j) = if g > 0.0 { g * g * u } else { 0.0 };
                            }
                        }
                        r += 4;
                    }
                    while r < end {
                        for t in 0..nt {
                            let g_combined = (scales[t] / 127.0) * scale_gate;
                            let u_combined = (scales[t] / 127.0) * scale_up;
                            let gv = ffi::i2_dot_i8(wg.ptr().add(r * row_bytes), qs[t] as *const i8, in_dim as i32);
                            let uv = ffi::i2_dot_i8(wu.ptr().add(r * row_bytes), qs[t] as *const i8, in_dim as i32);
                            let g = (gv - sums[t]) as f32 * g_combined;
                            let u = (uv - sums[t]) as f32 * u_combined;
                            *o.ptr().add(t * out_dim + r) = if g > 0.0 { g * g * u } else { 0.0 };
                        }
                        r += 1;
                    }
                }
            });
        }
    });
}

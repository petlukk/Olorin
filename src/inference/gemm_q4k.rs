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
/// 16×16 block-tiling (like llama.cpp): 16 weight rows × 16 tokens per chunk.
/// Work-stealing via atomic counter across threads.
/// Scales read inline inside kernel — no pre-caching of weight d/dmin.
pub(crate) fn q4k_gemm_mt(
    weight: *const u8, row_stride: usize, n_blocks: usize,
    batch: &BatchQ8K, out: &mut [f32], out_dim: usize,
    pool: &ThreadPool,
) {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let nt = batch.n_tokens;
    let n_threads = pool.thread_count();
    let qs: Vec<usize> = (0..nt).map(|t| batch.qs_ptr(t) as usize).collect();
    let ds: Vec<usize> = (0..nt).map(|t| batch.d_ptr(t) as usize).collect();
    let bs: Vec<usize> = (0..nt).map(|t| batch.bsums_ptr(t) as usize).collect();
    let w = SendPtr(weight);
    let o = SendMutPtr(out.as_mut_ptr());

    // llama.cpp-style 2D tiling: chunk_size=16, token-outer/row-inner,
    // with rechunk heuristic when total chunks < threads*4.
    let chunk_size = 16usize;
    let mut nchunk_r = (out_dim + chunk_size - 1) / chunk_size;
    let mut nchunk_t = (nt + chunk_size - 1) / chunk_size;
    // Rechunk: if not enough chunks for good load-balancing, collapse to 1D
    if nchunk_r * nchunk_t < n_threads * 4 {
        if out_dim > nt {
            nchunk_r = n_threads;
            nchunk_t = 1;
        } else {
            nchunk_r = 1;
            nchunk_t = n_threads;
        }
    }
    let total_chunks = nchunk_r * nchunk_t;
    let counter = AtomicUsize::new(0);

    pool.run(n_threads.min(total_chunks), move |_tid, _n| {
        let mut scores4 = [0.0f32; 4];
        let pow2 = crate::inference::matmul_q4k::F16_POW2.as_ptr();

        loop {
            let chunk_id = counter.fetch_add(1, Ordering::Relaxed);
            if chunk_id >= total_chunks { break; }

            // 2D chunk → row/token ranges
            let cr = chunk_id % nchunk_r;
            let ct = chunk_id / nchunk_r;
            let dr = (out_dim + nchunk_r - 1) / nchunk_r;
            let dt = (nt + nchunk_t - 1) / nchunk_t;
            let r_start = cr * dr;
            let r_end = (r_start + dr).min(out_dim);
            let t_start = ct * dt;
            let t_end = (t_start + dt).min(nt);
            if r_start >= r_end || t_start >= t_end { continue; }

            // Token-outer, row-inner (llama.cpp order)
            unsafe {
                let mut iit = t_start;
                while iit < t_end {
                    let tile_t_end = (iit + chunk_size).min(t_end);
                    let mut iir = r_start;
                    while iir < r_end {
                        let tile_r_end = (iir + chunk_size).min(r_end);
                        // Process 4 rows at a time within this tile
                        let mut r = iir;
                        while r + 4 <= tile_r_end {
                            let ws = [
                                w.ptr().add(r * row_stride),
                                w.ptr().add((r+1) * row_stride),
                                w.ptr().add((r+2) * row_stride),
                                w.ptr().add((r+3) * row_stride),
                            ];
                            for t in iit..tile_t_end {
                                ffi::q4k_dot_q8k_4row(
                                    ws[0], ws[1], ws[2], ws[3],
                                    qs[t] as _, bs[t] as _,
                                    scores4.as_mut_ptr(), n_blocks as i32, ds[t] as _,
                                    pow2,
                                );
                                let base = o.ptr().add(t * out_dim + r);
                                for j in 0..4 { *base.add(j) = scores4[j]; }
                            }
                            r += 4;
                        }
                        while r < tile_r_end {
                            let wr = w.ptr().add(r * row_stride);
                            for t in iit..tile_t_end {
                                let v = q4k_row_dot(wr, n_blocks, qs[t] as _, ds[t] as _, bs[t] as _);
                                *o.ptr().add(t * out_dim + r) = v;
                            }
                            r += 1;
                        }
                        iir += chunk_size;
                    }
                    iit += chunk_size;
                }
            }
        }
    });
}

/// Fused gate+up+SiLU GEMM with 16×16 block-tiling and work-stealing.
/// Scales read inline inside kernel — no pre-caching.
pub(crate) fn q4k_fused_silu_gemm_mt(
    w_gate: *const u8, w_up: *const u8,
    row_stride: usize, n_blocks: usize,
    batch: &BatchQ8K, out: &mut [f32], out_dim: usize,
    pool: &ThreadPool,
) {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let nt = batch.n_tokens;
    let n_threads = pool.thread_count();
    let qs: Vec<usize> = (0..nt).map(|t| batch.qs_ptr(t) as usize).collect();
    let ds: Vec<usize> = (0..nt).map(|t| batch.d_ptr(t) as usize).collect();
    let bs: Vec<usize> = (0..nt).map(|t| batch.bsums_ptr(t) as usize).collect();
    let wg = SendPtr(w_gate);
    let wu = SendPtr(w_up);
    let o = SendMutPtr(out.as_mut_ptr());

    // llama.cpp-style 2D tiling with rechunk heuristic
    let chunk_size = 16usize;
    let mut nchunk_r = (out_dim + chunk_size - 1) / chunk_size;
    let mut nchunk_t = (nt + chunk_size - 1) / chunk_size;
    if nchunk_r * nchunk_t < n_threads * 4 {
        if out_dim > nt { nchunk_r = n_threads; nchunk_t = 1; }
        else { nchunk_r = 1; nchunk_t = n_threads; }
    }
    let total_chunks = nchunk_r * nchunk_t;
    let counter = AtomicUsize::new(0);

    pool.run(n_threads.min(total_chunks), move |_tid, _n| {
        let mut g_scores = [0.0f32; 4];
        let mut u_scores = [0.0f32; 4];
        let pow2 = crate::inference::matmul_q4k::F16_POW2.as_ptr();

        loop {
            let chunk_id = counter.fetch_add(1, Ordering::Relaxed);
            if chunk_id >= total_chunks { break; }

            let cr = chunk_id % nchunk_r;
            let ct = chunk_id / nchunk_r;
            let dr = (out_dim + nchunk_r - 1) / nchunk_r;
            let dt = (nt + nchunk_t - 1) / nchunk_t;
            let r_start = cr * dr;
            let r_end = (r_start + dr).min(out_dim);
            let t_start = ct * dt;
            let t_end = (t_start + dt).min(nt);
            if r_start >= r_end || t_start >= t_end { continue; }

            // Token-outer, row-inner (llama.cpp order)
            unsafe {
                let mut iit = t_start;
                while iit < t_end {
                    let tile_t_end = (iit + chunk_size).min(t_end);
                    let mut iir = r_start;
                    while iir < r_end {
                        let tile_r_end = (iir + chunk_size).min(r_end);
                        let mut r = iir;
                        while r + 4 <= tile_r_end {
                            let gws = [wg.ptr().add(r * row_stride), wg.ptr().add((r+1) * row_stride),
                                       wg.ptr().add((r+2) * row_stride), wg.ptr().add((r+3) * row_stride)];
                            let uws = [wu.ptr().add(r * row_stride), wu.ptr().add((r+1) * row_stride),
                                       wu.ptr().add((r+2) * row_stride), wu.ptr().add((r+3) * row_stride)];
                            for t in iit..tile_t_end {
                                ffi::q4k_dot_q8k_4row(
                                    gws[0], gws[1], gws[2], gws[3],
                                    qs[t] as _, bs[t] as _,
                                    g_scores.as_mut_ptr(), n_blocks as i32, ds[t] as _,
                                    pow2,
                                );
                                ffi::q4k_dot_q8k_4row(
                                    uws[0], uws[1], uws[2], uws[3],
                                    qs[t] as _, bs[t] as _,
                                    u_scores.as_mut_ptr(), n_blocks as i32, ds[t] as _,
                                    pow2,
                                );
                                let base = o.ptr().add(t * out_dim + r);
                                for i in 0..4 {
                                    let g = g_scores[i];
                                    *base.add(i) = (g / (1.0 + (-g).exp())) * u_scores[i];
                                }
                            }
                            r += 4;
                        }
                        while r < tile_r_end {
                            let gw = wg.ptr().add(r * row_stride);
                            let uw = wu.ptr().add(r * row_stride);
                            for t in iit..tile_t_end {
                                let g = q4k_row_dot(gw, n_blocks, qs[t] as _, ds[t] as _, bs[t] as _);
                                let u = q4k_row_dot(uw, n_blocks, qs[t] as _, ds[t] as _, bs[t] as _);
                                *o.ptr().add(t * out_dim + r) = (g / (1.0 + (-g).exp())) * u;
                            }
                            r += 1;
                        }
                        iir += chunk_size;
                    }
                    iit += chunk_size;
                }
            }
        }
    });
}

/// Fused GEMM: f32 activations × Q4K weights → f32 output.
/// No Q8K intermediate buffer — quantization happens inside the kernel.
/// Weight d/dmin scales cached once, reused across all tokens.
pub(crate) fn q4k_fused_gemm_f32_mt(
    weight: *const u8, row_stride: usize, n_blocks: usize,
    activations: &[f32], act_stride: usize, n_tokens: usize,
    out: &mut [f32], out_dim: usize,
    pool: &ThreadPool,
) {
    let total = pool.thread_count().min(out_dim / 4).max(1);
    let chunk = ((out_dim + total - 1) / total + 3) & !3;
    let w = SendPtr(weight);
    let act = SendPtr(activations.as_ptr());
    let o = SendMutPtr(out.as_mut_ptr());
    let nt = n_tokens;

    pool.run(total, move |tid, _n| {
        let start = tid * chunk;
        let end = (start + chunk).min(out_dim);
        if start >= end { return; }
        let mut scores4 = [0.0f32; 4];
        let mut scratch = vec![0.0f32; 8];  // for kernel max-reduce (heap — must survive FFI)
        let mut bs = vec![0i32; 16];        // for kernel bsums (heap — must survive FFI)
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
                    ffi::q4k_fused_dot_4row(
                        ws[0], ws[1], ws[2], ws[3],
                        act.ptr().add(t * act_stride),
                        scores4.as_mut_ptr(), n_blocks as i32,
                        da_w[0].as_ptr(), da_w[1].as_ptr(),
                        da_w[2].as_ptr(), da_w[3].as_ptr(),
                        dma_w[0].as_ptr(), dma_w[1].as_ptr(),
                        dma_w[2].as_ptr(), dma_w[3].as_ptr(),
                        scratch.as_mut_ptr(), bs.as_mut_ptr(),
                    );
                    let base = o.ptr().add(t * out_dim + r);
                    for j in 0..4 { *base.add(j) = scores4[j]; }
                }
                r += 4;
            }
            while r < end {
                let wr = w.ptr().add(r * row_stride);
                let mut dw = [0f32; MAX_BLOCKS];
                let mut dmw = [0f32; MAX_BLOCKS];
                for blk in 0..n_blocks {
                    let bp = wr.add(blk * Q4K_BLOCK_BYTES);
                    dw[blk] = f16_to_f32(*(bp as *const u16));
                    dmw[blk] = f16_to_f32(*(bp.add(2) as *const u16));
                }
                for t in 0..nt {
                    let v = ffi::q4k_fused_dot(
                        wr, act.ptr().add(t * act_stride),
                        n_blocks as i32, dw.as_ptr(), dmw.as_ptr(),
                        scratch.as_mut_ptr(), bs.as_mut_ptr(),
                    );
                    *o.ptr().add(t * out_dim + r) = v;
                }
                r += 1;
            }
        }
    });
}

/// Fused gate+up+SiLU GEMM with f32 activations (no Q8K buffer).
pub(crate) fn q4k_fused_silu_gemm_f32_mt(
    w_gate: *const u8, w_up: *const u8,
    row_stride: usize, n_blocks: usize,
    activations: &[f32], act_stride: usize, n_tokens: usize,
    out: &mut [f32], out_dim: usize,
    pool: &ThreadPool,
) {
    let total = pool.thread_count().min(out_dim / 4).max(1);
    let chunk = ((out_dim + total - 1) / total + 3) & !3;
    let wg = SendPtr(w_gate);
    let wu = SendPtr(w_up);
    let act = SendPtr(activations.as_ptr());
    let o = SendMutPtr(out.as_mut_ptr());
    let nt = n_tokens;

    pool.run(total, move |tid, _n| {
        let start = tid * chunk;
        let end = (start + chunk).min(out_dim);
        if start >= end { return; }
        let mut gd_w = [[0f32; MAX_BLOCKS]; 4];
        let mut gdm_w = [[0f32; MAX_BLOCKS]; 4];
        let mut ud_w = [[0f32; MAX_BLOCKS]; 4];
        let mut udm_w = [[0f32; MAX_BLOCKS]; 4];
        let mut g_scores = [0.0f32; 4];
        let mut u_scores = [0.0f32; 4];
        let mut scratch = vec![0.0f32; 8];
        let mut bs_buf = vec![0i32; 16];
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
                    let act_ptr = act.ptr().add(t * act_stride);
                    ffi::q4k_fused_dot_4row(
                        gws[0], gws[1], gws[2], gws[3],
                        act_ptr, g_scores.as_mut_ptr(), n_blocks as i32,
                        gd_w[0].as_ptr(), gd_w[1].as_ptr(),
                        gd_w[2].as_ptr(), gd_w[3].as_ptr(),
                        gdm_w[0].as_ptr(), gdm_w[1].as_ptr(),
                        gdm_w[2].as_ptr(), gdm_w[3].as_ptr(),
                        scratch.as_mut_ptr(), bs_buf.as_mut_ptr(),
                    );
                    ffi::q4k_fused_dot_4row(
                        uws[0], uws[1], uws[2], uws[3],
                        act_ptr, u_scores.as_mut_ptr(), n_blocks as i32,
                        ud_w[0].as_ptr(), ud_w[1].as_ptr(),
                        ud_w[2].as_ptr(), ud_w[3].as_ptr(),
                        udm_w[0].as_ptr(), udm_w[1].as_ptr(),
                        udm_w[2].as_ptr(), udm_w[3].as_ptr(),
                        scratch.as_mut_ptr(), bs_buf.as_mut_ptr(),
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
                let mut gdw = [0f32; MAX_BLOCKS];
                let mut gdmw = [0f32; MAX_BLOCKS];
                let mut udw = [0f32; MAX_BLOCKS];
                let mut udmw = [0f32; MAX_BLOCKS];
                for blk in 0..n_blocks {
                    let gbp = gw.add(blk * Q4K_BLOCK_BYTES);
                    let ubp = uw.add(blk * Q4K_BLOCK_BYTES);
                    gdw[blk] = f16_to_f32(*(gbp as *const u16));
                    gdmw[blk] = f16_to_f32(*(gbp.add(2) as *const u16));
                    udw[blk] = f16_to_f32(*(ubp as *const u16));
                    udmw[blk] = f16_to_f32(*(ubp.add(2) as *const u16));
                }
                for t in 0..nt {
                    let act_ptr = act.ptr().add(t * act_stride);
                    let g = ffi::q4k_fused_dot(gw, act_ptr, n_blocks as i32, gdw.as_ptr(), gdmw.as_ptr(),
                        scratch.as_mut_ptr(), bs_buf.as_mut_ptr());
                    let u = ffi::q4k_fused_dot(uw, act_ptr, n_blocks as i32, udw.as_ptr(), udmw.as_ptr(),
                        scratch.as_mut_ptr(), bs_buf.as_mut_ptr());
                    *o.ptr().add(t * out_dim + r) = (g / (1.0 + (-g).exp())) * u;
                }
                r += 1;
            }
        }
    });
}

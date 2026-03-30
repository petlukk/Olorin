//! Q4_K matmul dispatch: unpacks 6-bit scales, calls SIMD dot-product kernels.

use crate::kernels::ffi_inference as ffi;
use crate::inference::matmul::f16_to_f32;
use crate::inference::ptr::{SendPtr, SendMutPtr};

/// Bytes per Q4_K super-block (256 elements).
pub(crate) const Q4K_BLOCK_BYTES: usize = 144;

/// Max blocks we support on the stack (32k dim / 256 = 128).
const MAX_BLOCKS: usize = 128;

/// Unpack Q4_K 12-byte packed scales into 8 scales + 8 mins.
pub(crate) fn unpack_q4k_scales(packed: &[u8], scales: &mut [u8; 8], mins: &mut [u8; 8]) {
    for i in 0..4 {
        scales[i] = packed[i] & 0x3F;
        mins[i] = packed[4 + i] & 0x3F;
    }
    for i in 0..4 {
        scales[4 + i] = (packed[8 + i] & 0x0F) | ((packed[i] >> 6) << 4);
        mins[4 + i] = (packed[8 + i] >> 4) | ((packed[4 + i] >> 6) << 4);
    }
}

/// Pre-unpack one weight row into stack buffers.
unsafe fn unpack_row(
    weight: *const u8, n_blocks: usize, q8_d: *const f32,
    sc_buf: &mut [u8], mn_buf: &mut [u8],
    d_buf: &mut [f32], dm_buf: &mut [f32],
) {
    let mut sc_tmp = [0u8; 8];
    let mut mn_tmp = [0u8; 8];
    for blk in 0..n_blocks {
        let bp = weight.add(blk * Q4K_BLOCK_BYTES);
        let blk_q8_d = *q8_d.add(blk);
        d_buf[blk] = f16_to_f32(*(bp as *const u16)) * blk_q8_d;
        dm_buf[blk] = f16_to_f32(*(bp.add(2) as *const u16)) * blk_q8_d;
        unpack_q4k_scales(
            std::slice::from_raw_parts(bp.add(4), 12), &mut sc_tmp, &mut mn_tmp);
        sc_buf[blk * 8..blk * 8 + 8].copy_from_slice(&sc_tmp);
        mn_buf[blk * 8..blk * 8 + 8].copy_from_slice(&mn_tmp);
    }
}

/// Dot product of one Q4_K weight row against Q8_K activations.
pub(crate) unsafe fn q4k_row_dot(
    weight: *const u8, n_blocks: usize,
    q8_qs: *const i8, q8_d: *const f32, q8_bsums: *const i32,
) -> f32 {
    debug_assert!(n_blocks <= MAX_BLOCKS);
    let mut sc = [0u8; MAX_BLOCKS * 8];
    let mut mn = [0u8; MAX_BLOCKS * 8];
    let mut da = [0f32; MAX_BLOCKS];
    let mut dma = [0f32; MAX_BLOCKS];
    unpack_row(weight, n_blocks, q8_d, &mut sc, &mut mn, &mut da, &mut dma);
    ffi::q4k_dot_q8k(
        weight, q8_qs, q8_bsums,
        sc.as_ptr(), mn.as_ptr(), n_blocks as i32,
        da.as_ptr(), dma.as_ptr(),
    )
}

/// 4-row Q4_K dot product with shared Q8_K activations.
pub(crate) unsafe fn q4k_4row_dot(
    w0: *const u8, w1: *const u8, w2: *const u8, w3: *const u8,
    n_blocks: usize,
    q8_qs: *const i8, q8_d: *const f32, q8_bsums: *const i32,
    scores: &mut [f32; 4],
) {
    debug_assert!(n_blocks <= MAX_BLOCKS);
    let mut sc = [[0u8; MAX_BLOCKS * 8]; 4];
    let mut mn = [[0u8; MAX_BLOCKS * 8]; 4];
    let mut da = [[0f32; MAX_BLOCKS]; 4];
    let mut dma = [[0f32; MAX_BLOCKS]; 4];
    let ws = [w0, w1, w2, w3];
    for ri in 0..4 {
        unpack_row(ws[ri], n_blocks, q8_d,
            &mut sc[ri], &mut mn[ri], &mut da[ri], &mut dma[ri]);
    }
    ffi::q4k_dot_q8k_4row(
        w0, w1, w2, w3,
        q8_qs, q8_bsums,
        sc[0].as_ptr(), sc[1].as_ptr(), sc[2].as_ptr(), sc[3].as_ptr(),
        mn[0].as_ptr(), mn[1].as_ptr(), mn[2].as_ptr(), mn[3].as_ptr(),
        scores.as_mut_ptr(), n_blocks as i32,
        da[0].as_ptr(), da[1].as_ptr(), da[2].as_ptr(), da[3].as_ptr(),
        dma[0].as_ptr(), dma[1].as_ptr(), dma[2].as_ptr(), dma[3].as_ptr(),
    );
}

/// Multi-threaded Q4_K x Q8_K matrix multiplication.
pub(crate) fn q4k_matmul_mt(
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
                q4k_4row_dot(weight.add(r * row_stride), weight.add((r+1) * row_stride),
                    weight.add((r+2) * row_stride), weight.add((r+3) * row_stride),
                    n_blocks, q8_qs, q8_d, q8_bsums, &mut scores4);
                for j in 0..4 { out[r+j] = scores4[j]; }
                r += 4;
            }
            while r < out_dim {
                out[r] = q4k_row_dot(weight.add(r * row_stride), n_blocks, q8_qs, q8_d, q8_bsums);
                r += 1;
            }
        }
        return;
    }

    let chunk = ((out_dim + n_thr - 1) / n_thr + 3) & !3;
    let w = SendPtr(weight);
    let qs = SendPtr(q8_qs);
    let qd = SendPtr(q8_d);
    let qb = SendPtr(q8_bsums);
    let o = SendMutPtr(out.as_mut_ptr());

    pool.run(n_thr, move |tid, _n| {
        let start = tid * chunk;
        let end = (start + chunk).min(out_dim);
        if start >= end { return; }
        let count = end - start;
        let out_slice = unsafe { std::slice::from_raw_parts_mut(o.ptr().add(start), count) };
        let mut scores4 = [0.0f32; 4];
        let mut r = 0;
        unsafe {
            while r + 4 <= count {
                let row = start + r;
                q4k_4row_dot(w.ptr().add(row * row_stride), w.ptr().add((row+1) * row_stride),
                    w.ptr().add((row+2) * row_stride), w.ptr().add((row+3) * row_stride),
                    n_blocks, qs.ptr(), qd.ptr(), qb.ptr(), &mut scores4);
                for j in 0..4 { out_slice[r+j] = scores4[j]; }
                r += 4;
            }
            while r < count {
                let row = start + r;
                out_slice[r] = q4k_row_dot(w.ptr().add(row * row_stride), n_blocks, qs.ptr(), qd.ptr(), qb.ptr());
                r += 1;
            }
        }
    });
}

/// Dequantize a single embedding row from Q4_K block data to f32.
pub(crate) fn q4k_embed_lookup(
    embed_data: *const u8, token: u32, out: &mut [f32], hidden_dim: usize,
) {
    let n_blocks = hidden_dim / 256;
    let row_bytes = n_blocks * Q4K_BLOCK_BYTES;
    let row_ptr = unsafe { embed_data.add(token as usize * row_bytes) };
    let mut scales = [0u8; 8];
    let mut mins = [0u8; 8];

    for blk in 0..n_blocks {
        let block = unsafe { row_ptr.add(blk * Q4K_BLOCK_BYTES) };
        let d = f16_to_f32(unsafe { *(block as *const u16) });
        let dmin = f16_to_f32(unsafe { *(block.add(2) as *const u16) });
        let scales_raw = unsafe { std::slice::from_raw_parts(block.add(4), 12) };
        unpack_q4k_scales(scales_raw, &mut scales, &mut mins);
        let qs = unsafe { block.add(16) };

        for j in 0..4 {
            let d1 = d * scales[2 * j] as f32;
            let m1 = dmin * mins[2 * j] as f32;
            let d2 = d * scales[2 * j + 1] as f32;
            let m2 = dmin * mins[2 * j + 1] as f32;
            for k in 0..32 {
                let byte = unsafe { *qs.add(j * 32 + k) };
                out[blk * 256 + j * 64 + k] = d1 * (byte & 0xF) as f32 - m1;
                out[blk * 256 + j * 64 + 32 + k] = d2 * (byte >> 4) as f32 - m2;
            }
        }
    }
}

/// Per-thread work function for Q4_K matmul.
pub(crate) unsafe fn q4k_matmul_work(
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
    let mut scores4 = [0.0f32; 4];
    let mut r = 0;
    while r + 4 <= count {
        let row = start + r;
        q4k_4row_dot(weight.add(row * row_stride), weight.add((row+1) * row_stride),
            weight.add((row+2) * row_stride), weight.add((row+3) * row_stride),
            n_blocks, q8_qs, q8_d, q8_bsums, &mut scores4);
        for j in 0..4 { out_slice[r+j] = scores4[j]; }
        r += 4;
    }
    while r < count {
        let row = start + r;
        out_slice[r] = q4k_row_dot(weight.add(row * row_stride), n_blocks, q8_qs, q8_d, q8_bsums);
        r += 1;
    }
}

/// Fused 4-row gate+up dot product using dual kernel.
pub(crate) unsafe fn q4k_dual_4row_dot(
    gw0: *const u8, gw1: *const u8, gw2: *const u8, gw3: *const u8,
    uw0: *const u8, uw1: *const u8, uw2: *const u8, uw3: *const u8,
    n_blocks: usize,
    q8_qs: *const i8, q8_d: *const f32, q8_bsums: *const i32,
    g_scores: &mut [f32; 4], u_scores: &mut [f32; 4],
) {
    debug_assert!(n_blocks <= MAX_BLOCKS);
    let mut gsc = [[0u8; MAX_BLOCKS * 8]; 4];
    let mut gmn = [[0u8; MAX_BLOCKS * 8]; 4];
    let mut usc = [[0u8; MAX_BLOCKS * 8]; 4];
    let mut umn = [[0u8; MAX_BLOCKS * 8]; 4];
    let mut gd = [[0f32; MAX_BLOCKS]; 4];
    let mut gdm = [[0f32; MAX_BLOCKS]; 4];
    let mut ud = [[0f32; MAX_BLOCKS]; 4];
    let mut udm = [[0f32; MAX_BLOCKS]; 4];
    let gws = [gw0, gw1, gw2, gw3];
    let uws = [uw0, uw1, uw2, uw3];
    for i in 0..4 {
        unpack_row(gws[i], n_blocks, q8_d,
            &mut gsc[i], &mut gmn[i], &mut gd[i], &mut gdm[i]);
        unpack_row(uws[i], n_blocks, q8_d,
            &mut usc[i], &mut umn[i], &mut ud[i], &mut udm[i]);
    }
    ffi::q4k_dot_q8k_4row_dual(
        gw0, gw1, gw2, gw3,
        uw0, uw1, uw2, uw3,
        q8_qs, q8_bsums,
        gsc[0].as_ptr(), gsc[1].as_ptr(), gsc[2].as_ptr(), gsc[3].as_ptr(),
        gmn[0].as_ptr(), gmn[1].as_ptr(), gmn[2].as_ptr(), gmn[3].as_ptr(),
        usc[0].as_ptr(), usc[1].as_ptr(), usc[2].as_ptr(), usc[3].as_ptr(),
        umn[0].as_ptr(), umn[1].as_ptr(), umn[2].as_ptr(), umn[3].as_ptr(),
        g_scores.as_mut_ptr(), u_scores.as_mut_ptr(), n_blocks as i32,
        gd[0].as_ptr(), gd[1].as_ptr(), gd[2].as_ptr(), gd[3].as_ptr(),
        gdm[0].as_ptr(), gdm[1].as_ptr(), gdm[2].as_ptr(), gdm[3].as_ptr(),
        ud[0].as_ptr(), ud[1].as_ptr(), ud[2].as_ptr(), ud[3].as_ptr(),
        udm[0].as_ptr(), udm[1].as_ptr(), udm[2].as_ptr(), udm[3].as_ptr(),
    );
}

/// Fused gate+up+SiLU per-thread work function.
pub(crate) unsafe fn q4k_fused_gate_up_silu_work(
    w_gate: *const u8, w_up: *const u8,
    row_stride: usize, n_blocks: usize,
    q8_qs: *const i8, q8_d: *const f32, q8_bsums: *const i32,
    hidden_out: *mut f32, out_dim: usize, tid: usize, n_threads: usize,
) {
    let chunk = ((out_dim + n_threads - 1) / n_threads + 3) & !3;
    let start = tid * chunk;
    let end = (start + chunk).min(out_dim);
    if start >= end { return; }
    let count = end - start;
    let out = std::slice::from_raw_parts_mut(hidden_out.add(start), count);
    let mut g_scores = [0.0f32; 4];
    let mut u_scores = [0.0f32; 4];
    let mut r = 0;
    while r + 4 <= count {
        let row = start + r;
        q4k_dual_4row_dot(
            w_gate.add(row * row_stride), w_gate.add((row+1) * row_stride),
            w_gate.add((row+2) * row_stride), w_gate.add((row+3) * row_stride),
            w_up.add(row * row_stride), w_up.add((row+1) * row_stride),
            w_up.add((row+2) * row_stride), w_up.add((row+3) * row_stride),
            n_blocks, q8_qs, q8_d, q8_bsums, &mut g_scores, &mut u_scores,
        );
        for i in 0..4 {
            let g = g_scores[i];
            out[r + i] = (g / (1.0 + (-g).exp())) * u_scores[i];
        }
        r += 4;
    }
    while r < count {
        let row = start + r;
        let g = q4k_row_dot(w_gate.add(row * row_stride), n_blocks, q8_qs, q8_d, q8_bsums);
        let u = q4k_row_dot(w_up.add(row * row_stride), n_blocks, q8_qs, q8_d, q8_bsums);
        out[r] = (g / (1.0 + (-g).exp())) * u;
        r += 1;
    }
}

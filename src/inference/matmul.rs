//! Matrix multiplication helpers for BitNet inference (I2S ternary).

use crate::kernels::ffi_inference as ffi;
use crate::inference::ptr::{SendPtr, SendMutPtr};
use crate::inference::threadpool::ThreadPool;

pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    if exp == 0 {
        if frac == 0 { return f32::from_bits(sign << 31); }
        let mut e = 0i32;
        let mut f = frac;
        while f & 0x400 == 0 { f <<= 1; e -= 1; }
        f &= 0x3ff;
        return f32::from_bits((sign << 31) | (((127 - 15 + 1 + e) as u32) << 23) | (f << 13));
    }
    if exp == 31 { return f32::from_bits((sign << 31) | (0xff << 23) | (frac << 13)); }
    f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13))
}

pub(crate) fn embed_f16_lookup(embed: *const u8, token: u32, out: &mut [f32], hidden_dim: usize) {
    let row = unsafe {
        std::slice::from_raw_parts(embed.add(token as usize * hidden_dim * 2) as *const u16, hidden_dim)
    };
    for i in 0..hidden_dim { out[i] = f16_to_f32(row[i]); }
}

/// i8 output matmul: quantize x to i8, then i8*u8 dot product for each vocab row.
pub(crate) fn i8_output_matmul_mt(
    embed_i8: &[u8], row_scales: &[f32],
    x: &[f32], out: &mut [f32],
    vocab_size: usize, hidden_dim: usize,
    pool: &ThreadPool,
) {
    let mut x_amax = 0.0f32;
    for &v in x.iter().take(hidden_dim) { let a = v.abs(); if a > x_amax { x_amax = a; } }
    let x_inv = if x_amax > 1e-10 { 127.0 / x_amax } else { 0.0 };
    let mut x_i8 = vec![0i8; hidden_dim];
    #[cfg(not(target_arch = "aarch64"))]
    let mut x_sum: i32 = 0;
    for d in 0..hidden_dim {
        let q = (x[d] * x_inv).round().clamp(-127.0, 127.0) as i8;
        x_i8[d] = q;
        #[cfg(not(target_arch = "aarch64"))]
        { x_sum += q as i32; }
    }
    let x_scale = x_amax / 127.0;
    let n_thr = pool.thread_count();
    let chunk = ((vocab_size + n_thr - 1) / n_thr + 3) & !3;

    let embed = SendPtr(embed_i8.as_ptr());
    let scales = SendPtr(row_scales.as_ptr());
    let act = SendPtr(x_i8.as_ptr());
    let out_base = SendMutPtr(out.as_mut_ptr());

    pool.run(n_thr, move |tid, _n| {
        let start = tid * chunk;
        let end = (start + chunk).min(vocab_size);
        if start >= end { return; }
        let count = end - start;
        let out_slice = unsafe { std::slice::from_raw_parts_mut(out_base.ptr().add(start), count) };
        let mut raw4 = [0i32; 4];
        let mut r = 0;
        while r + 4 <= count {
            let row = start + r;
            unsafe {
                ffi::i8dot_4row(act.ptr(), embed.ptr().add(row * hidden_dim),
                    embed.ptr().add((row+1) * hidden_dim), embed.ptr().add((row+2) * hidden_dim),
                    embed.ptr().add((row+3) * hidden_dim), raw4.as_mut_ptr(), hidden_dim as i32);
            }
            for j in 0..4 {
                let row_s = unsafe { *scales.ptr().add(row + j) };
                #[cfg(target_arch = "aarch64")]
                let corrected = raw4[j];
                #[cfg(not(target_arch = "aarch64"))]
                let corrected = raw4[j] - 128 * x_sum;
                out_slice[r + j] = corrected as f32 * x_scale * (row_s / 127.0);
            }
            r += 4;
        }
        while r < count {
            let row = start + r;
            let raw_val = unsafe { ffi::i8dot_1row(act.ptr(), embed.ptr().add(row * hidden_dim), hidden_dim as i32) };
            let row_s = unsafe { *scales.ptr().add(row) };
            #[cfg(target_arch = "aarch64")]
            let corrected = raw_val;
            #[cfg(not(target_arch = "aarch64"))]
            let corrected = raw_val - 128 * x_sum;
            out_slice[r] = corrected as f32 * x_scale * (row_s / 127.0);
            r += 1;
        }
    });
}

/// Pre-allocated work buffers for speculative output matmul.
#[cfg(target_arch = "aarch64")]
pub(crate) struct SpeculativeWork {
    pub x_i8: Vec<i8>,
    pub x_sketch: Vec<i8>,
    pub sketch_scores: Vec<i32>,
    pub indices: Vec<u32>,
}

#[cfg(target_arch = "aarch64")]
impl SpeculativeWork {
    pub fn new(vocab_size: usize, hidden_dim: usize, sketch_dim: usize) -> Self {
        SpeculativeWork {
            x_i8: vec![0i8; hidden_dim],
            x_sketch: vec![0i8; sketch_dim],
            sketch_scores: vec![0i32; vocab_size],
            indices: (0..vocab_size as u32).collect(),
        }
    }
}

/// Speculative output matmul: sketch-based pre-filter + full dot for candidates.
/// ARM-only: single pool.run — each thread does sketch scoring on its vocab chunk,
/// local top-k selection, then full dot on its candidates. One wake, zero allocs.
#[cfg(target_arch = "aarch64")]
pub(crate) fn i8_output_matmul_speculative(
    embed_i8: &[u8], row_scales: &[f32],
    sketch: &[u8], sketch_dim: usize,
    x: &[f32], out: &mut [f32],
    vocab_size: usize, hidden_dim: usize,
    pool: &ThreadPool, work: &mut SpeculativeWork,
) {
    let mut x_amax = 0.0f32;
    for &v in x.iter().take(hidden_dim) { let a = v.abs(); if a > x_amax { x_amax = a; } }
    let x_inv = if x_amax > 1e-10 { 127.0 / x_amax } else { 0.0 };
    for d in 0..hidden_dim {
        work.x_i8[d] = (x[d] * x_inv).round().clamp(-127.0, 127.0) as i8;
    }
    let x_scale = x_amax / 127.0;
    for s in 0..sketch_dim { work.x_sketch[s] = work.x_i8[s * 4]; }

    for r in 0..vocab_size { out[r] = f32::NEG_INFINITY; }

    // Single pool.run: sketch + local top-k + full dot per thread
    let n_thr = pool.thread_count().min(vocab_size).max(1);
    let chunk = (vocab_size + n_thr - 1) / n_thr;
    let top_k_per_thread = 4096usize.min(vocab_size) / n_thr.max(1) + 1;
    let sk_ptr = SendPtr(sketch.as_ptr());
    let xs_ptr = SendPtr(work.x_sketch.as_ptr());
    let ss_ptr = SendMutPtr(work.sketch_scores.as_mut_ptr());
    let idx_ptr = SendMutPtr(work.indices.as_mut_ptr());
    let act = SendPtr(work.x_i8.as_ptr());
    let emb = SendPtr(embed_i8.as_ptr());
    let sc = SendPtr(row_scales.as_ptr());
    let out_ptr = SendMutPtr(out.as_mut_ptr());

    pool.run(n_thr, move |tid, _| {
        let start = tid * chunk;
        let end = (start + chunk).min(vocab_size);
        if start >= end { return; }
        let count = end - start;

        // Phase 1: sketch score this chunk (4-row batched)
        let mut row = start;
        while row + 4 <= end {
            let mut raw4 = [0i32; 4];
            unsafe {
                ffi::i8dot_4row(
                    xs_ptr.ptr(),
                    sk_ptr.ptr().add(row * sketch_dim) as *const u8,
                    sk_ptr.ptr().add((row+1) * sketch_dim) as *const u8,
                    sk_ptr.ptr().add((row+2) * sketch_dim) as *const u8,
                    sk_ptr.ptr().add((row+3) * sketch_dim) as *const u8,
                    raw4.as_mut_ptr(), sketch_dim as i32,
                );
                *ss_ptr.ptr().add(row) = raw4[0];
                *ss_ptr.ptr().add(row+1) = raw4[1];
                *ss_ptr.ptr().add(row+2) = raw4[2];
                *ss_ptr.ptr().add(row+3) = raw4[3];
            }
            row += 4;
        }
        while row < end {
            let raw = unsafe {
                ffi::i8dot_1row(
                    xs_ptr.ptr(),
                    sk_ptr.ptr().add(row * sketch_dim) as *const u8,
                    sketch_dim as i32,
                )
            };
            unsafe { *ss_ptr.ptr().add(row) = raw; }
            row += 1;
        }

        // Phase 2: local top-k via partial partition
        let local_k = top_k_per_thread.min(count);
        let idx_slice = unsafe {
            std::slice::from_raw_parts_mut(idx_ptr.ptr().add(start), count)
        };
        for i in 0..count { idx_slice[i] = (start + i) as u32; }
        let scores = ss_ptr;
        idx_slice.select_nth_unstable_by(local_k, |&a, &b| {
            let sa = unsafe { *scores.ptr().add(a as usize) };
            let sb = unsafe { *scores.ptr().add(b as usize) };
            sb.cmp(&sa)
        });

        // Phase 3: full dot on local top-k candidates
        for ci in 0..local_k {
            let row = idx_slice[ci] as usize;
            let raw = unsafe {
                ffi::i8dot_1row(
                    act.ptr(), emb.ptr().add(row * hidden_dim),
                    hidden_dim as i32,
                )
            };
            let row_s = unsafe { *sc.ptr().add(row) };
            unsafe {
                *out_ptr.ptr().add(row) = raw as f32 * x_scale * (row_s / 127.0);
            }
        }
    });
}

/// Ternary matmul with configurable thread count.
pub(crate) fn ternary_matmul_mt_n(
    weight: *const u8, act: *const i8,
    act_scale: f32, act_sum: i32, weight_scale: f32,
    out: &mut [f32], out_dim: usize, in_dim: usize,
    pool: &ThreadPool, n_thr: usize,
) {
    let n_threads = n_thr.min(out_dim / 4).max(1);
    let row_bytes = in_dim / 4;
    let scale = (act_scale / 127.0) * weight_scale;

    if n_threads <= 1 {
        let mut raw4 = [0i32; 4];
        let mut r = 0;
        unsafe {
            while r + 4 <= out_dim {
                ffi::i2_dot_i8_4row(weight.add(r * row_bytes), weight.add((r+1) * row_bytes),
                    weight.add((r+2) * row_bytes), weight.add((r+3) * row_bytes),
                    act, raw4.as_mut_ptr(), in_dim as i32);
                for j in 0..4 { out[r+j] = (raw4[j] - act_sum) as f32 * scale; }
                r += 4;
            }
            while r < out_dim {
                let v = ffi::i2_dot_i8(weight.add(r * row_bytes), act, in_dim as i32);
                out[r] = (v - act_sum) as f32 * scale;
                r += 1;
            }
        }
        return;
    }

    let chunk = ((out_dim + n_threads - 1) / n_threads + 3) & !3;
    let w = SendPtr(weight);
    let a = SendPtr(act);
    let o = SendMutPtr(out.as_mut_ptr());

    pool.run(n_threads, move |tid, _n| {
        let start = tid * chunk;
        let end = (start + chunk).min(out_dim);
        if start >= end { return; }
        let count = end - start;
        let out_slice = unsafe { std::slice::from_raw_parts_mut(o.ptr().add(start), count) };
        let mut raw4 = [0i32; 4];
        let mut r = 0;
        unsafe {
            while r + 4 <= count {
                let row = start + r;
                ffi::i2_dot_i8_4row(w.ptr().add(row * row_bytes), w.ptr().add((row+1) * row_bytes),
                    w.ptr().add((row+2) * row_bytes), w.ptr().add((row+3) * row_bytes),
                    a.ptr(), raw4.as_mut_ptr(), in_dim as i32);
                for j in 0..4 { out_slice[r+j] = (raw4[j] - act_sum) as f32 * scale; }
                r += 4;
            }
            while r < count {
                let v = ffi::i2_dot_i8(w.ptr().add((start + r) * row_bytes), a.ptr(), in_dim as i32);
                out_slice[r] = (v - act_sum) as f32 * scale;
                r += 1;
            }
        }
    });
}

/// Ternary matmul using all available threads.
pub(crate) fn ternary_matmul_mt(
    weight: *const u8, act: *const i8,
    act_scale: f32, act_sum: i32, weight_scale: f32,
    out: &mut [f32], out_dim: usize, in_dim: usize,
    pool: &ThreadPool,
) {
    ternary_matmul_mt_n(weight, act, act_scale, act_sum, weight_scale,
        out, out_dim, in_dim, pool, pool.thread_count());
}

/// Run Q + K + V concurrently.
pub(crate) fn ternary_matmul_qkv(
    w_q: *const u8, scale_q: f32, out_q: &mut [f32], out_dim_q: usize,
    w_k: *const u8, scale_k: f32, out_k: &mut [f32], out_dim_kv: usize,
    w_v: *const u8, scale_v: f32, out_v: &mut [f32],
    act: *const i8, act_scale: f32, act_sum: i32, in_dim: usize,
    pool: &ThreadPool,
) {
    let total = pool.thread_count();
    if total < 3 {
        ternary_matmul_mt_n(w_q, act, act_scale, act_sum, scale_q, out_q, out_dim_q, in_dim, pool, total);
        ternary_matmul_mt_n(w_k, act, act_scale, act_sum, scale_k, out_k, out_dim_kv, in_dim, pool, total);
        ternary_matmul_mt_n(w_v, act, act_scale, act_sum, scale_v, out_v, out_dim_kv, in_dim, pool, total);
        return;
    }
    let q_threads = (total / 2).max(1);
    let remaining = total - q_threads;
    let k_threads = remaining / 2;
    let v_threads = remaining - k_threads;
    let row_bytes = in_dim / 4;

    let a = SendPtr(act);
    let oq = SendMutPtr(out_q.as_mut_ptr());
    let ok = SendMutPtr(out_k.as_mut_ptr());
    let ov = SendMutPtr(out_v.as_mut_ptr());
    let wq = SendPtr(w_q);
    let wk = SendPtr(w_k);
    let wv = SendPtr(w_v);
    let sc_q = (act_scale / 127.0) * scale_q;
    let sc_k = (act_scale / 127.0) * scale_k;
    let sc_v = (act_scale / 127.0) * scale_v;
    let chunk_q = ((out_dim_q + q_threads - 1) / q_threads + 3) & !3;
    let chunk_kv_k = ((out_dim_kv + k_threads - 1) / k_threads + 3) & !3;
    let chunk_kv_v = ((out_dim_kv + v_threads - 1) / v_threads + 3) & !3;

    pool.run_split3(
        q_threads, move |tid, _n| {
            let start = tid * chunk_q;
            let end = (start + chunk_q).min(out_dim_q);
            if start >= end { return; }
            let count = end - start;
            let out_slice = unsafe { std::slice::from_raw_parts_mut(oq.ptr().add(start), count) };
            let mut raw4 = [0i32; 4];
            let mut r = 0;
            unsafe {
                while r + 4 <= count {
                    let row = start + r;
                    ffi::i2_dot_i8_4row(
                        wq.ptr().add(row * row_bytes), wq.ptr().add((row+1) * row_bytes),
                        wq.ptr().add((row+2) * row_bytes), wq.ptr().add((row+3) * row_bytes),
                        a.ptr(), raw4.as_mut_ptr(), in_dim as i32);
                    for j in 0..4 { out_slice[r+j] = (raw4[j] - act_sum) as f32 * sc_q; }
                    r += 4;
                }
                while r < count {
                    let v = ffi::i2_dot_i8(wq.ptr().add((start+r) * row_bytes), a.ptr(), in_dim as i32);
                    out_slice[r] = (v - act_sum) as f32 * sc_q;
                    r += 1;
                }
            }
        },
        k_threads, move |tid, _n| {
            let start = tid * chunk_kv_k;
            let end = (start + chunk_kv_k).min(out_dim_kv);
            if start >= end { return; }
            let count = end - start;
            let out_slice = unsafe { std::slice::from_raw_parts_mut(ok.ptr().add(start), count) };
            let mut raw4 = [0i32; 4];
            let mut r = 0;
            unsafe {
                while r + 4 <= count {
                    let row = start + r;
                    ffi::i2_dot_i8_4row(
                        wk.ptr().add(row * row_bytes), wk.ptr().add((row+1) * row_bytes),
                        wk.ptr().add((row+2) * row_bytes), wk.ptr().add((row+3) * row_bytes),
                        a.ptr(), raw4.as_mut_ptr(), in_dim as i32);
                    for j in 0..4 { out_slice[r+j] = (raw4[j] - act_sum) as f32 * sc_k; }
                    r += 4;
                }
                while r < count {
                    let v = ffi::i2_dot_i8(wk.ptr().add((start+r) * row_bytes), a.ptr(), in_dim as i32);
                    out_slice[r] = (v - act_sum) as f32 * sc_k;
                    r += 1;
                }
            }
        },
        v_threads, move |tid, _n| {
            let start = tid * chunk_kv_v;
            let end = (start + chunk_kv_v).min(out_dim_kv);
            if start >= end { return; }
            let count = end - start;
            let out_slice = unsafe { std::slice::from_raw_parts_mut(ov.ptr().add(start), count) };
            let mut raw4 = [0i32; 4];
            let mut r = 0;
            unsafe {
                while r + 4 <= count {
                    let row = start + r;
                    ffi::i2_dot_i8_4row(
                        wv.ptr().add(row * row_bytes), wv.ptr().add((row+1) * row_bytes),
                        wv.ptr().add((row+2) * row_bytes), wv.ptr().add((row+3) * row_bytes),
                        a.ptr(), raw4.as_mut_ptr(), in_dim as i32);
                    for j in 0..4 { out_slice[r+j] = (raw4[j] - act_sum) as f32 * sc_v; }
                    r += 4;
                }
                while r < count {
                    let v = ffi::i2_dot_i8(wv.ptr().add((start+r) * row_bytes), a.ptr(), in_dim as i32);
                    out_slice[r] = (v - act_sum) as f32 * sc_v;
                    r += 1;
                }
            }
        },
    );
}

/// Fused gate+up matmul via i2_dot_i8_4row_dual kernel.
pub(crate) fn ternary_matmul_fused_pair(
    w_a: *const u8, scale_a: f32,
    w_b: *const u8, scale_b: f32,
    act: *const i8, act_scale: f32, act_sum: i32,
    out_a: &mut [f32], out_b: &mut [f32],
    out_dim: usize, in_dim: usize,
    pool: &ThreadPool,
) {
    let total = pool.thread_count();
    let row_bytes = in_dim / 4;
    let sca = (act_scale / 127.0) * scale_a;
    let scb = (act_scale / 127.0) * scale_b;
    let chunk = ((out_dim + total - 1) / total + 3) & !3;
    let wa = SendPtr(w_a); let wb = SendPtr(w_b); let a = SendPtr(act);
    let oa = SendMutPtr(out_a.as_mut_ptr()); let ob = SendMutPtr(out_b.as_mut_ptr());

    pool.run(total, move |tid, _n| {
        let start = tid * chunk;
        let end = (start + chunk).min(out_dim);
        if start >= end { return; }
        let count = end - start;
        let oa_s = unsafe { std::slice::from_raw_parts_mut(oa.ptr().add(start), count) };
        let ob_s = unsafe { std::slice::from_raw_parts_mut(ob.ptr().add(start), count) };
        let mut ra = [0i32; 4]; let mut rb = [0i32; 4];
        let mut r = 0;
        unsafe {
            while r + 4 <= count {
                let row = start + r;
                ffi::i2_dot_i8_4row_dual(
                    wa.ptr().add(row * row_bytes), wa.ptr().add((row+1) * row_bytes),
                    wa.ptr().add((row+2) * row_bytes), wa.ptr().add((row+3) * row_bytes),
                    wb.ptr().add(row * row_bytes), wb.ptr().add((row+1) * row_bytes),
                    wb.ptr().add((row+2) * row_bytes), wb.ptr().add((row+3) * row_bytes),
                    a.ptr(), ra.as_mut_ptr(), rb.as_mut_ptr(), in_dim as i32);
                for j in 0..4 {
                    oa_s[r+j] = (ra[j] - act_sum) as f32 * sca;
                    ob_s[r+j] = (rb[j] - act_sum) as f32 * scb;
                }
                r += 4;
            }
            while r < count {
                let row = start + r;
                let va = ffi::i2_dot_i8(wa.ptr().add(row * row_bytes), a.ptr(), in_dim as i32);
                let vb = ffi::i2_dot_i8(wb.ptr().add(row * row_bytes), a.ptr(), in_dim as i32);
                oa_s[r] = (va - act_sum) as f32 * sca;
                ob_s[r] = (vb - act_sum) as f32 * scb;
                r += 1;
            }
        }
    });
}

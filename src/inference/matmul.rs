//! Matrix multiplication helpers for BitNet inference (I2S ternary).

use crate::kernels::ffi_inference as ffi;
use crate::inference::ptr::{SendPtr, SendMutPtr};

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

fn n_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// i8 output matmul: quantize x to i8, then i8*u8 dot product for each vocab row.
pub(crate) fn i8_output_matmul_mt(
    embed_i8: &[u8], row_scales: &[f32],
    x: &[f32], out: &mut [f32],
    vocab_size: usize, hidden_dim: usize,
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
    let n_thr = n_threads();
    let chunk = ((vocab_size + n_thr - 1) / n_thr + 3) & !3;

    let embed = SendPtr(embed_i8.as_ptr());
    let scales = SendPtr(row_scales.as_ptr());
    let act = SendPtr(x_i8.as_ptr());
    let out_base = SendMutPtr(out.as_mut_ptr());

    std::thread::scope(|s| {
        for tid in 0..n_thr {
            let start = tid * chunk;
            let end = (start + chunk).min(vocab_size);
            if start >= end { continue; }
            let count = end - start;
            #[cfg(not(target_arch = "aarch64"))]
            let xs = x_sum;
            let xsc = x_scale;
            let h = hidden_dim;
            s.spawn(move || {
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_base.ptr().add(start), count) };
                let mut raw4 = [0i32; 4];
                let mut r = 0;
                while r + 4 <= count {
                    let row = start + r;
                    unsafe {
                        ffi::i8dot_4row(act.ptr(), embed.ptr().add(row * h),
                            embed.ptr().add((row+1) * h), embed.ptr().add((row+2) * h),
                            embed.ptr().add((row+3) * h), raw4.as_mut_ptr(), h as i32);
                    }
                    for j in 0..4 {
                        let row_s = unsafe { *scales.ptr().add(row + j) };
                        #[cfg(target_arch = "aarch64")]
                        let corrected = raw4[j];
                        #[cfg(not(target_arch = "aarch64"))]
                        let corrected = raw4[j] - 128 * xs;
                        out_slice[r + j] = corrected as f32 * xsc * (row_s / 127.0);
                    }
                    r += 4;
                }
                while r < count {
                    let row = start + r;
                    let raw_val = unsafe { ffi::i8dot_1row(act.ptr(), embed.ptr().add(row * h), h as i32) };
                    let row_s = unsafe { *scales.ptr().add(row) };
                    #[cfg(target_arch = "aarch64")]
                    let corrected = raw_val;
                    #[cfg(not(target_arch = "aarch64"))]
                    let corrected = raw_val - 128 * xs;
                    out_slice[r] = corrected as f32 * xsc * (row_s / 127.0);
                    r += 1;
                }
            });
        }
    });
}

#[cfg(target_arch = "aarch64")]
pub(crate) fn i8_output_matmul_speculative(
    embed_i8: &[u8], row_scales: &[f32],
    sketch: &[u8], sketch_dim: usize,
    x: &[f32], out: &mut [f32],
    vocab_size: usize, hidden_dim: usize,
) {
    const TOP_K: usize = 512;
    const SKETCH_STRIDE: usize = 4;
    let mut x_amax = 0.0f32;
    for &v in x.iter().take(hidden_dim) { let a = v.abs(); if a > x_amax { x_amax = a; } }
    let x_inv = if x_amax > 1e-10 { 127.0 / x_amax } else { 0.0 };
    let mut x_i8 = vec![0i8; hidden_dim];
    for d in 0..hidden_dim { x_i8[d] = (x[d] * x_inv).round().clamp(-127.0, 127.0) as i8; }
    let x_scale = x_amax / 127.0;
    let mut act_sketch = vec![0i8; sketch_dim];
    for s in 0..sketch_dim { act_sketch[s] = x_i8[s * SKETCH_STRIDE]; }

    let mut rough_scores = vec![0.0f32; vocab_size];
    let n_thr = n_threads();
    let chunk = ((vocab_size + n_thr - 1) / n_thr + 3) & !3;
    let sk = SendPtr(sketch.as_ptr());
    let act_sk = SendPtr(act_sketch.as_ptr());
    let rough = SendMutPtr(rough_scores.as_mut_ptr());
    let sc = SendPtr(row_scales.as_ptr());
    let xsc = x_scale;
    let sd = sketch_dim;
    std::thread::scope(|s| {
        for tid in 0..n_thr {
            let start = tid * chunk;
            let end = (start + chunk).min(vocab_size);
            if start >= end { continue; }
            s.spawn(move || {
                let mut raw4 = [0i32; 4];
                let mut r = start;
                while r + 4 <= end {
                    unsafe {
                        ffi::i8dot_4row(act_sk.ptr(),
                            sk.ptr().add(r * sd), sk.ptr().add((r+1) * sd),
                            sk.ptr().add((r+2) * sd), sk.ptr().add((r+3) * sd),
                            raw4.as_mut_ptr(), sd as i32);
                        for j in 0..4 {
                            let rs = *sc.ptr().add(r + j);
                            *rough.ptr().add(r + j) = raw4[j] as f32 * xsc * (rs / 127.0);
                        }
                    }
                    r += 4;
                }
                while r < end {
                    let v = unsafe { ffi::i8dot_1row(act_sk.ptr(), sk.ptr().add(r * sd), sd as i32) };
                    let rs = unsafe { *sc.ptr().add(r) };
                    unsafe { *rough.ptr().add(r) = v as f32 * xsc * (rs / 127.0); }
                    r += 1;
                }
            });
        }
    });
    let mut indices: Vec<u32> = (0..vocab_size as u32).collect();
    indices.select_nth_unstable_by(TOP_K, |&a, &b| {
        rough_scores[b as usize].partial_cmp(&rough_scores[a as usize]).unwrap()
    });
    let top_indices = &indices[..TOP_K];
    for v in out.iter_mut() { *v = f32::NEG_INFINITY; }
    let act_ptr = x_i8.as_ptr();
    let embed_ptr = embed_i8.as_ptr();
    let h = hidden_dim;
    for &idx in top_indices {
        let row = idx as usize;
        let raw = unsafe { ffi::i8dot_1row(act_ptr, embed_ptr.add(row * h), h as i32) };
        out[row] = raw as f32 * x_scale * (row_scales[row] / 127.0);
    }
}

/// Ternary matmul with configurable thread count.
pub(crate) fn ternary_matmul_mt_n(
    weight: *const u8, act: *const i8,
    act_scale: f32, act_sum: i32, weight_scale: f32,
    out: &mut [f32], out_dim: usize, in_dim: usize, n_thr: usize,
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
    std::thread::scope(|s| {
        for tid in 0..n_threads {
            let start = tid * chunk;
            let end = (start + chunk).min(out_dim);
            if start >= end { continue; }
            let count = end - start;
            s.spawn(move || {
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
    });
}

/// Ternary matmul using all available threads.
pub(crate) fn ternary_matmul_mt(
    weight: *const u8, act: *const i8,
    act_scale: f32, act_sum: i32, weight_scale: f32,
    out: &mut [f32], out_dim: usize, in_dim: usize,
) {
    ternary_matmul_mt_n(weight, act, act_scale, act_sum, weight_scale, out, out_dim, in_dim, n_threads());
}

/// Run Q + K + V concurrently.
pub(crate) fn ternary_matmul_qkv(
    w_q: *const u8, scale_q: f32, out_q: &mut [f32], out_dim_q: usize,
    w_k: *const u8, scale_k: f32, out_k: &mut [f32], out_dim_kv: usize,
    w_v: *const u8, scale_v: f32, out_v: &mut [f32],
    act: *const i8, act_scale: f32, act_sum: i32, in_dim: usize,
) {
    let total = n_threads();
    if total < 3 {
        ternary_matmul_mt_n(w_q, act, act_scale, act_sum, scale_q, out_q, out_dim_q, in_dim, total);
        ternary_matmul_mt_n(w_k, act, act_scale, act_sum, scale_k, out_k, out_dim_kv, in_dim, total);
        ternary_matmul_mt_n(w_v, act, act_scale, act_sum, scale_v, out_v, out_dim_kv, in_dim, total);
        return;
    }
    let q_threads = (total / 2).max(1);
    let remaining = total - q_threads;
    let k_threads = remaining / 2;
    let v_threads = remaining - k_threads;
    let row_bytes = in_dim / 4;

    struct Work { w: SendPtr<u8>, o: SendMutPtr<f32>, dim: usize, sc: f32, nt: usize }
    unsafe impl Send for Work {}
    unsafe impl Sync for Work {}

    let a = SendPtr(act);
    let works = [
        Work { w: SendPtr(w_q), o: SendMutPtr(out_q.as_mut_ptr()), dim: out_dim_q, sc: scale_q, nt: q_threads },
        Work { w: SendPtr(w_k), o: SendMutPtr(out_k.as_mut_ptr()), dim: out_dim_kv, sc: scale_k, nt: k_threads },
        Work { w: SendPtr(w_v), o: SendMutPtr(out_v.as_mut_ptr()), dim: out_dim_kv, sc: scale_v, nt: v_threads },
    ];

    std::thread::scope(|s| {
        for work in &works {
            for tid in 0..work.nt {
                let chunk = ((work.dim + work.nt - 1) / work.nt + 3) & !3;
                let start = tid * chunk;
                let end = (start + chunk).min(work.dim);
                if start >= end { continue; }
                let count = end - start;
                let combined_scale = (act_scale / 127.0) * work.sc;
                let w = work.w;
                let o = work.o;
                s.spawn(move || {
                    let out_slice = unsafe { std::slice::from_raw_parts_mut(o.ptr().add(start), count) };
                    let mut raw4 = [0i32; 4];
                    let mut r = 0;
                    unsafe {
                        while r + 4 <= count {
                            let row = start + r;
                            ffi::i2_dot_i8_4row(
                                w.ptr().add(row * row_bytes), w.ptr().add((row+1) * row_bytes),
                                w.ptr().add((row+2) * row_bytes), w.ptr().add((row+3) * row_bytes),
                                a.ptr(), raw4.as_mut_ptr(), in_dim as i32);
                            for j in 0..4 { out_slice[r+j] = (raw4[j] - act_sum) as f32 * combined_scale; }
                            r += 4;
                        }
                        while r < count {
                            let v = ffi::i2_dot_i8(w.ptr().add((start+r) * row_bytes), a.ptr(), in_dim as i32);
                            out_slice[r] = (v - act_sum) as f32 * combined_scale;
                            r += 1;
                        }
                    }
                });
            }
        }
    });
}

/// Fused gate+up matmul via i2_dot_i8_4row_dual kernel.
pub(crate) fn ternary_matmul_fused_pair(
    w_a: *const u8, scale_a: f32,
    w_b: *const u8, scale_b: f32,
    act: *const i8, act_scale: f32, act_sum: i32,
    out_a: &mut [f32], out_b: &mut [f32],
    out_dim: usize, in_dim: usize,
) {
    let total = n_threads();
    let row_bytes = in_dim / 4;
    let sca = (act_scale / 127.0) * scale_a;
    let scb = (act_scale / 127.0) * scale_b;
    let chunk = ((out_dim + total - 1) / total + 3) & !3;
    let wa = SendPtr(w_a); let wb = SendPtr(w_b); let a = SendPtr(act);
    let oa = SendMutPtr(out_a.as_mut_ptr()); let ob = SendMutPtr(out_b.as_mut_ptr());

    std::thread::scope(|s| {
        for tid in 0..total {
            let start = tid * chunk;
            let end = (start + chunk).min(out_dim);
            if start >= end { continue; }
            let count = end - start;
            s.spawn(move || {
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
    });
}

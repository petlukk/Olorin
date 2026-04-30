//! Q3_K quantization — port of llama.cpp `quantize_row_q3_K_impl` +
//! `make_qx_quants` (rmse_type=1, no imatrix). Used by:
//!   - `tests/requant_q4k_to_q3k.rs` — offline tool that requantizes a GGUF
//!   - The smoke test in that same file (synthetic round-trip)
//!
//! Lives in `tests/common/` so both consumers can share without duplication.
//!
//! The _impl + make_qx_quants path is what llama-quantize uses with --imatrix;
//! without imatrix llama.cpp falls back to _ref (ggml-quants.c:1379-1381).
//! Empirically the _impl/make_qx_quants chain still produces noticeably better
//! quality than _ref even without an imatrix because it does a 19-point iscale
//! grid search per sub-block AND a second-level optimization across scales.

#![allow(dead_code)]

pub const QK_K: usize = 256;
pub const Q3K_BLOCK_BYTES: usize = 110;
const GROUP_MAX_EPS: f32 = 1e-15;

#[inline]
pub fn nearest_int(fval: f32) -> i32 {
    // Same fast trick as the Q4K tool (avoids round() ties-to-even fenv reliance).
    let v = fval + 12_582_912.0_f32;
    let i = v.to_bits() as i32;
    (i & 0x007f_ffff) - 0x0040_0000
}

/// Port of llama.cpp `make_qx_quants(n, nmax, x, L, rmse_type, qw)`.
/// Always called with rmse_type=1 from quantize_row_q3_K_impl. We hardcode that.
/// Returns the per-sub-block scale; writes L in `[0..2*nmax]` (after `+ nmax` shift).
pub fn make_qx_quants(nmax: i32, x: &[f32], l: &mut [i8], qw: &[f32]) -> f32 {
    let n = x.len();
    debug_assert_eq!(l.len(), n);
    debug_assert_eq!(qw.len(), n);

    let mut max = 0.0_f32;
    let mut amax = 0.0_f32;
    for i in 0..n {
        let ax = x[i].abs();
        if ax > amax { amax = ax; max = x[i]; }
    }
    if amax < GROUP_MAX_EPS {
        for li in l.iter_mut() { *li = 0; }
        return 0.0;
    }
    let mut iscale = -(nmax as f32) / max;
    let mut sumlx = 0.0_f32;
    let mut suml2 = 0.0_f32;
    for i in 0..n {
        let mut li = nearest_int(iscale * x[i]);
        if li < -nmax { li = -nmax; }
        if li > nmax - 1 { li = nmax - 1; }
        l[i] = (li + nmax) as i8;
        let w = qw[i];
        sumlx += w * x[i] * li as f32;
        suml2 += w * (li as f32) * (li as f32);
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    let mut best = scale * sumlx;

    // 19-point iscale grid search around the initial estimate (skip is=0 = original).
    for is in -9..=9 {
        if is == 0 { continue; }
        let trial_iscale = -((nmax as f32) + 0.1 * is as f32) / max;
        let mut t_sumlx = 0.0_f32;
        let mut t_suml2 = 0.0_f32;
        for i in 0..n {
            let mut li = nearest_int(trial_iscale * x[i]);
            if li < -nmax { li = -nmax; }
            if li > nmax - 1 { li = nmax - 1; }
            let w = qw[i];
            t_sumlx += w * x[i] * li as f32;
            t_suml2 += w * (li as f32) * (li as f32);
        }
        if t_suml2 > 0.0 && t_sumlx * t_sumlx > best * t_suml2 {
            for i in 0..n {
                let mut li = nearest_int(trial_iscale * x[i]);
                if li < -nmax { li = -nmax; }
                if li > nmax - 1 { li = nmax - 1; }
                l[i] = (li + nmax) as i8;
            }
            scale = t_sumlx / t_suml2;
            best = scale * t_sumlx;
            iscale = trial_iscale;
        }
    }
    let _ = iscale;  // kept assigned for clarity even though final value is unused.
    scale
}

/// Port of llama.cpp `quantize_row_q3_K_impl(..., quant_weights=NULL)`.
/// With NULL quant_weights, the per-sub-block weight is just `x[i]^2`.
/// Two-level optimization: per-sub-block scales via make_qx_quants(nmax=4),
/// then scales themselves are quantized by make_qx_quants(nmax=32) with sub-block
/// weights = sum(x^2) per sub-block.
pub fn quantize_row_q3_k(x: &[f32], dst: &mut [u8]) {
    assert!(x.len() % QK_K == 0);
    assert_eq!(dst.len(), (x.len() / QK_K) * Q3K_BLOCK_BYTES);
    let nb = x.len() / QK_K;
    let mut l = [0i8; QK_K];
    let mut scales = [0f32; QK_K / 16];
    let mut sw = [0f32; QK_K / 16];
    let mut ls = [0i8; QK_K / 16];
    let mut weight = [0f32; 16];

    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let block = &mut dst[i * Q3K_BLOCK_BYTES..(i + 1) * Q3K_BLOCK_BYTES];
        for b in block.iter_mut() { *b = 0; }

        for j in 0..(QK_K / 16) {
            let xs = &xb[j * 16..(j + 1) * 16];
            let mut sumw = 0.0_f32;
            for k in 0..16 {
                weight[k] = xs[k] * xs[k];
                sumw += weight[k];
            }
            sw[j] = sumw;
            scales[j] = make_qx_quants(4, xs, &mut l[j * 16..(j + 1) * 16], &weight);
        }

        let d_block = make_qx_quants(32, &scales, &mut ls, &sw);

        for j in 0..(QK_K / 16) {
            let li = ls[j] as i32;
            if j < 8 {
                block[96 + j] = (li & 0xF) as u8;
            } else {
                block[96 + j - 8] |= ((li & 0xF) as u8) << 4;
            }
            let li_high = li >> 4;
            block[96 + (j % 4) + 8] |= (li_high as u8) << (2 * (j / 4));
        }
        block[108..110].copy_from_slice(&f32_to_f16(d_block).to_le_bytes());

        // Re-extract effective per-sub-block scales and re-quant L — necessary
        // because the scale quantization above may have changed sc[j] from the
        // value used in the per-sub-block quant.
        for j in 0..(QK_K / 16) {
            let sc_low = if j < 8 { block[96 + j] & 0xF } else { block[96 + j - 8] >> 4 };
            let sc_high = (block[96 + 8 + (j % 4)] >> (2 * (j / 4))) & 0x3;
            let sc = ((sc_low | (sc_high << 4)) as i32) - 32;
            let d = d_block * sc as f32;
            if d == 0.0 {
                // Match llama.cpp's `continue;` — preserves the L values that
                // make_qx_quants(nmax=4) wrote during the first pass.
                continue;
            }
            for ii in 0..16 {
                let mut li = nearest_int(xb[16 * j + ii] / d);
                if li < -4 { li = -4; }
                if li > 3 { li = 3; }
                l[16 * j + ii] = (li + 4) as i8;
            }
        }

        // Build hmask
        let mut m = 0;
        let mut hm: u8 = 1;
        for j in 0..QK_K {
            if l[j] > 3 {
                block[m] |= hm;
                l[j] -= 4;
            }
            m += 1;
            if m == QK_K / 8 {
                m = 0;
                hm <<= 1;
            }
        }

        // Pack qs: 4 values per byte at shifts 0,2,4,6.
        for j in (0..QK_K).step_by(128) {
            for ll in 0..32 {
                let v = (l[j + ll] as u8)
                      | ((l[j + ll + 32] as u8) << 2)
                      | ((l[j + ll + 64] as u8) << 4)
                      | ((l[j + ll + 96] as u8) << 6);
                block[32 + j / 4 + ll] = v;
            }
        }
    }
}

pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;
    if exp == 255 {
        return (sign | 0x7C00 | if mantissa != 0 { 0x0200 } else { 0 }) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return (sign | 0x7C00) as u16;
    }
    if new_exp <= 0 {
        if new_exp < -10 { return sign as u16; }
        let m = mantissa | 0x0080_0000;
        let shift = (1 - new_exp) as u32;
        let half = (m >> (shift + 13 - 1)) & 1;
        let result = m >> (shift + 13);
        return (sign | (result + half)) as u16;
    }
    let half = (mantissa >> 12) & 1;
    let result = ((new_exp as u32) << 10) | (mantissa >> 13);
    (sign | (result + half)) as u16
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 { sign } else {
            let mut m = mant; let mut e = 0i32;
            while (m & 0x400) == 0 { m <<= 1; e -= 1; }
            m &= 0x3FF;
            sign | (((127 - 15 + e + 1) as u32) << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | 0x7F800000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

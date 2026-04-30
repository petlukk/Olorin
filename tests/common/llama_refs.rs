//! Scalar Rust ports of llama.cpp reference implementations, used as ground
//! truth in olorin's kernel parity tests. Each impl matches a specific
//! ggml_vec_dot_* / ggml_compute_* generic function bit-for-bit (modulo Olorin's
//! sign convention for Q8K d).

#![allow(dead_code)]

/// llama.cpp's quantize_row_q8_K_ref (ggml-quants.c:2692).
/// Returns (qs, d, bsums) for one 256-block.
pub fn llama_q8k_ref_block(x: &[f32]) -> (Vec<i8>, f32, Vec<i16>) {
    assert_eq!(x.len(), 256);
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &v in x {
        let av = v.abs();
        if av > amax { amax = av; max = v; }
    }
    if amax == 0.0 {
        return (vec![0; 256], 0.0, vec![0; 16]);
    }
    let iscale = -127.0f32 / max;
    // Magic-number nearest-int (round-half-to-even) — bit-equivalent to ggml's nearest_int.
    let nearest = |fval: f32| -> i32 {
        let val = fval + 12582912.0f32;
        let bits: u32 = val.to_bits();
        ((bits & 0x007fffff) as i32) - 0x00400000
    };
    let mut qs = vec![0i8; 256];
    for j in 0..256 {
        let v = nearest(iscale * x[j]);
        qs[j] = v.min(127) as i8;
    }
    let mut bsums = vec![0i16; 16];
    for g in 0..16 {
        let mut s = 0i32;
        for k in 0..16 { s += qs[g*16 + k] as i32; }
        bsums[g] = s as i16;
    }
    (qs, 1.0 / iscale, bsums)
}

/// Scalar Q4K dot matching llama.cpp's ggml_vec_dot_q4_K_q8_K_generic exactly.
/// q4_raw: pointer to raw Q4K block bytes (144 bytes per block)
/// q8_qs/q8_d/q8_bsums: Olorin-convention Q8K (positive d)
pub fn llama_q4k_dot_ref(
    q4_raw: *const u8, q8_qs: &[i8], q8_d: &[f32], q8_bsums: &[i16], n_blocks: usize,
) -> f32 {
    use olorin::inference::matmul::f16_to_f32_scalar;

    let kmask1: u32 = 0x3f3f3f3f;
    let kmask2: u32 = 0x0f0f0f0f;
    let kmask3: u32 = 0x03030303;

    let mut sums = [0.0f32; 8];
    let mut sumf = 0.0f32;

    for i in 0..n_blocks {
        let bp = i * 144;
        let q4 = unsafe { q4_raw.add(bp) };

        let d_f16 = unsafe { *(q4 as *const u16) };
        let dmin_f16 = unsafe { *((q4 as *const u16).add(1)) };
        let x_d = f16_to_f32_scalar(d_f16);
        let x_dmin = f16_to_f32_scalar(dmin_f16);

        let d = q8_d[i] * x_d;
        let dmin = q8_d[i] * x_dmin;

        let mut utmp = [0u32; 4];
        unsafe {
            std::ptr::copy_nonoverlapping(q4.add(4), utmp.as_mut_ptr() as *mut u8, 12);
        }

        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        let uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        let scales = unsafe { &*(&utmp[0..2] as *const [u32] as *const [u8; 8]) };
        let mins = unsafe { &*(&utmp[2..4] as *const [u32] as *const [u8; 8]) };

        let bs_base = i * 16;
        let mut sumi_mins = 0i32;
        for g in 0..8 {
            let paired = q8_bsums[bs_base + g * 2] as i32 + q8_bsums[bs_base + g * 2 + 1] as i32;
            sumi_mins += paired * (mins[g] as i32);
        }
        sumf -= dmin * (sumi_mins as f32);

        let qs_base = bp + 16;
        let q8_base = i * 256;
        let mut aux8 = [0i8; 256];
        for j in 0..4 {
            for l in 0..32 {
                let byte = unsafe { *q4_raw.add(qs_base + j * 32 + l) };
                aux8[j * 64 + l] = (byte & 0xf) as i8;
                aux8[j * 64 + 32 + l] = (byte >> 4) as i8;
            }
        }

        let mut aux32 = [0i32; 8];
        let mut a_idx = 0usize;
        let mut q8_idx = q8_base;
        for is in 0..8 {
            let sc = scales[is] as i32;
            for _ in 0..4 {
                for l in 0..8 {
                    aux32[l] += sc * (q8_qs[q8_idx + l] as i32) * (aux8[a_idx + l] as i32);
                }
                a_idx += 8;
                q8_idx += 8;
            }
        }

        for l in 0..8 {
            sums[l] += d * (aux32[l] as f32);
        }
    }

    for l in 0..8 { sumf += sums[l]; }
    sumf
}

/// Scalar Q5K dot matching llama.cpp's ggml_vec_dot_q5_K_q8_K_generic exactly.
pub fn llama_q5k_dot_ref(
    q5_raw: *const u8, q8_qs: &[i8], q8_d: &[f32], q8_bsums: &[i16], n_blocks: usize,
) -> f32 {
    use olorin::inference::matmul::f16_to_f32_scalar;

    let kmask1: u32 = 0x3f3f3f3f;
    let kmask2: u32 = 0x0f0f0f0f;
    let kmask3: u32 = 0x03030303;

    let mut sums = [0.0f32; 8];
    let mut sumf = 0.0f32;

    for i in 0..n_blocks {
        let bp = i * 176;
        let blk = unsafe { q5_raw.add(bp) };

        let d_f16 = unsafe { *(blk as *const u16) };
        let dmin_f16 = unsafe { *((blk as *const u16).add(1)) };
        let x_d = f16_to_f32_scalar(d_f16);
        let x_dmin = f16_to_f32_scalar(dmin_f16);
        let d = q8_d[i] * x_d;
        let dmin = q8_d[i] * x_dmin;

        let mut utmp = [0u32; 4];
        unsafe { std::ptr::copy_nonoverlapping(blk.add(4), utmp.as_mut_ptr() as *mut u8, 12); }
        utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
        let uaux = utmp[1] & kmask1;
        utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
        utmp[2] = uaux;
        utmp[0] &= kmask1;

        let scales = unsafe { &*(&utmp[0..2] as *const [u32] as *const [u8; 8]) };
        let mins = unsafe { &*(&utmp[2..4] as *const [u32] as *const [u8; 8]) };

        let bs_base = i * 16;
        let mut sumi_mins = 0i32;
        for g in 0..8 {
            let paired = q8_bsums[bs_base + g*2] as i32 + q8_bsums[bs_base + g*2+1] as i32;
            sumi_mins += paired * (mins[g] as i32);
        }
        sumf -= dmin * (sumi_mins as f32);

        let qh_ptr = unsafe { blk.add(16) };
        let qs_ptr = unsafe { blk.add(48) };

        let mut aux8 = [0i8; 256];
        for j in 0..4 {
            for l in 0..32 {
                let qs_byte = unsafe { *qs_ptr.add(j * 32 + l) };
                let qh_byte = unsafe { *qh_ptr.add(l) };
                let h_lo = ((qh_byte >> (2*j)) & 1) << 4;
                aux8[j*64 + l] = ((qs_byte & 0xf) | h_lo) as i8;
                let h_hi = ((qh_byte >> (2*j+1)) & 1) << 4;
                aux8[j*64 + 32 + l] = ((qs_byte >> 4) | h_hi) as i8;
            }
        }

        let q8_base = i * 256;
        let mut aux32 = [0i32; 8];
        let mut a_idx = 0usize;
        let mut q8_idx = q8_base;
        for is in 0..8 {
            let sc = scales[is] as i32;
            for _ in 0..4 {
                for l in 0..8 {
                    aux32[l] += sc * (q8_qs[q8_idx + l] as i32) * (aux8[a_idx + l] as i32);
                }
                a_idx += 8;
                q8_idx += 8;
            }
        }

        for l in 0..8 {
            sums[l] += d * (aux32[l] as f32);
        }
    }

    for l in 0..8 { sumf += sums[l]; }
    sumf
}

/// Scalar Q6K dot — direct port of llama.cpp's ggml_vec_dot_q6_K_q8_K_generic.
/// Q6K block (210 bytes): ql[128]@0, qh[64]@128, scales[16]@192, d(f16)@208.
pub fn llama_q6k_dot_ref(
    q6_raw: *const u8, q8_qs: &[i8], q8_d: &[f32], _q8_bsums: &[i16], n_blocks: usize,
) -> f32 {
    use olorin::inference::matmul::f16_to_f32_scalar;

    let mut sums = [0.0f32; 8];
    let mut sumf = 0.0f32;

    for i in 0..n_blocks {
        let bp = i * 210;
        let blk = unsafe { q6_raw.add(bp) };

        let mut aux8 = [0i8; 256];
        let mut q4_off = 0usize;
        let mut qh_off = 128usize;
        let mut a_idx = 0usize;

        for _j in (0..256).step_by(128) {
            for l in 0..32 {
                let q4_0 = unsafe { *blk.add(q4_off + l) };
                let q4_32 = unsafe { *blk.add(q4_off + 32 + l) };
                let qh_l = unsafe { *blk.add(qh_off + l) };
                aux8[a_idx + l]      = ((q4_0 & 0xf)  | (((qh_l >> 0) & 3) << 4)) as i8 - 32;
                aux8[a_idx + l + 32] = ((q4_32 & 0xf) | (((qh_l >> 2) & 3) << 4)) as i8 - 32;
                aux8[a_idx + l + 64] = ((q4_0 >> 4)   | (((qh_l >> 4) & 3) << 4)) as i8 - 32;
                aux8[a_idx + l + 96] = ((q4_32 >> 4)  | (((qh_l >> 6) & 3) << 4)) as i8 - 32;
            }
            a_idx += 128;
            q4_off += 64;
            qh_off += 32;
        }

        let mut aux32 = [0i32; 8];
        let mut a_pos = 0usize;
        let mut q8_pos = i * 256;
        let scales = unsafe { std::slice::from_raw_parts(blk.add(192) as *const i8, 16) };

        for is in 0..16 {
            let sc = scales[is] as i32;
            for l in 0..8 {
                aux32[l] += sc * (q8_qs[q8_pos + l] as i32) * (aux8[a_pos + l] as i32);
            }
            q8_pos += 8; a_pos += 8;
            for l in 0..8 {
                aux32[l] += sc * (q8_qs[q8_pos + l] as i32) * (aux8[a_pos + l] as i32);
            }
            q8_pos += 8; a_pos += 8;
        }

        let d_f16 = unsafe { u16::from_le_bytes([*blk.add(208), *blk.add(209)]) };
        let d = f16_to_f32_scalar(d_f16) * q8_d[i];
        for l in 0..8 { sums[l] += d * (aux32[l] as f32); }
    }
    for l in 0..8 { sumf += sums[l]; }
    sumf
}

/// Scalar RMSNorm matching llama.cpp exactly: double-precision sum, then mul(weight).
pub fn llama_rmsnorm_ref(x: &[f32], weight: *const f32, out: &mut [f32], eps: f32) {
    let n = x.len();
    let sum: f64 = x.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let mean = sum / (n as f64);
    let scale = 1.0f32 / ((mean as f32) + eps).sqrt();
    for i in 0..n {
        let w = unsafe { *weight.add(i) };
        out[i] = x[i] * scale * w;
    }
}

/// llama.cpp RoPE cache_init: multiplicative theta accumulation.
pub fn llama_rope_cache(
    pos: usize, freq_base: f32, n_dims: usize, freq_factors: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>) {
    let half = n_dims / 2;
    let theta_scale = freq_base.powf(-2.0 / n_dims as f32);
    let mut cos = vec![0.0f32; half];
    let mut sin = vec![0.0f32; half];
    let mut theta = pos as f32;
    for d in 0..half {
        let ff = freq_factors.map(|f| f[d]).unwrap_or(1.0);
        let angle = theta / ff;
        cos[d] = angle.cos();
        sin[d] = angle.sin();
        theta *= theta_scale;
    }
    (cos, sin)
}

/// llama.cpp GELU: 0.5*x*(1 + tanhf(SQRT_2_OVER_PI * x * (1 + 0.044715*x*x)))
pub fn llama_gelu_f32(x: f32) -> f32 {
    0.5f32 * x * (1.0f32 + (0.7978845608f32 * x * (1.0f32 + 0.044715f32 * x * x)).tanh())
}

/// llama.cpp f32→f16 (ggml_compute_fp32_to_fp16) — FP-based round-to-nearest-even.
pub fn llama_f32_to_f16(f: f32) -> u16 {
    let scale_to_inf: f32 = f32::from_bits(0x77800000);
    let scale_to_zero: f32 = f32::from_bits(0x08800000);
    let base_val = (f.abs() * scale_to_inf) * scale_to_zero;

    let w = f.to_bits();
    let shl1_w = w.wrapping_add(w);
    let sign = w & 0x80000000u32;
    let mut bias = shl1_w & 0xFF000000u32;
    if bias < 0x71000000u32 {
        bias = 0x71000000u32;
    }

    let base = f32::from_bits((bias >> 1) + 0x07800000u32) + base_val;
    let bits = base.to_bits();
    let exp_bits = (bits >> 13) & 0x00007C00u32;
    let mantissa_bits = bits & 0x00000FFFu32;
    let nonsign = exp_bits + mantissa_bits;
    ((sign >> 16) | if shl1_w > 0xFF000000u32 { 0x7E00u32 } else { nonsign }) as u16
}

/// Olorin's f32→f16 from cache.rs — replicated here to compare against llama_f32_to_f16.
pub fn olorin_f32_to_f16(x: f32) -> u16 {
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
        if new_exp < -10 {
            return sign as u16;
        }
        let m = mantissa | 0x0080_0000;
        let shift = 1 - new_exp;
        let half = (m >> (shift + 13 - 1)) & 1;
        let result = m >> (shift + 13);
        return (sign | (result + half)) as u16;
    }

    let half = (mantissa >> 12) & 1;
    let result = ((new_exp as u32) << 10) | (mantissa >> 13);
    (sign | result + half) as u16
}

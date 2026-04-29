//! ARM-only: run my Rust scalar mirror AND my q3k_8x8_q8k_gemm kernel on
//! the same data. The scalar uses the verified-correct logic from
//! q3k_repack_scalar_gemm.rs (on host this matches q3k_dot_q8k bit-exact).
//! If on Pi the scalar matches q3k_dot_q8k but the kernel doesn't, the bug
//! is purely in the kernel.

#![cfg(target_arch = "aarch64")]

use std::path::Path;

fn model_path() -> String {
    let home = std::env::var("HOME").unwrap();
    format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M-q3kffnup.gguf")
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    let f32_bits = if exp == 0 {
        if frac == 0 { sign << 31 } else {
            let mut e = 1u32;
            let mut m = frac;
            while m & 0x400 == 0 { m <<= 1; e += 1; }
            (sign << 31) | ((127 - 15 - e + 1) << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | (0xFFu32 << 23) | (frac << 13)
    } else {
        (sign << 31) | ((exp + (127 - 15)) << 23) | (frac << 13)
    };
    f32::from_bits(f32_bits)
}

#[test]
fn q3k_pi_scalar_vs_kernel() {
    if !Path::new(&model_path()).exists() {
        eprintln!("SKIP: no model"); return;
    }
    let gguf = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    let lw = &model.layers[0];
    assert_eq!(lw.w_up_dtype, olorin::inference::matmul::GGML_TYPE_Q3_K);

    let n = model.hidden_dim;
    let nc = model.ffn_dim[0];
    let nb = n / 256;
    let pow2 = olorin::inference::matmul::pow2_table();
    let row_bytes_src = nb * 110;

    // Test on first 8 weight rows × 4-token batch (token 0 used for parity).
    let n_rows_test = 8usize;

    // Build synthetic q8 for token 0
    let mut input = vec![0.0f32; n];
    for i in 0..n {
        input[i] = 0.01 * ((i * 3) % 89) as f32 - 0.4;
    }
    let mut qs = vec![0i8; n + 12];
    let mut q8d = vec![0.0f32; nb];
    let mut bsums = vec![0i16; nb * 16];
    unsafe {
        olorin::kernels::ffi_inference::quant_f32_q8k(
            input.as_ptr(), qs.as_mut_ptr(), q8d.as_mut_ptr(), bsums.as_mut_ptr(), n as i32,
        );
    }

    // Reference q3k_dot_q8k
    let mut ref_out = [0.0f32; 8];
    for r in 0..8 {
        ref_out[r] = unsafe {
            olorin::kernels::ffi_inference::q3k_dot_q8k(
                lw.w_up.add(r * row_bytes_src),
                qs.as_ptr(), bsums.as_ptr(),
                nb as i32, q8d.as_ptr(), pow2.as_ptr(),
            )
        };
    }
    eprintln!("REF:    {:?}", ref_out);

    // Repack 8 rows × full row width
    let mut src_8r = vec![0u8; n_rows_test * nb * 110];
    unsafe {
        for r in 0..n_rows_test {
            std::ptr::copy_nonoverlapping(
                lw.w_up.add(r * row_bytes_src),
                src_8r.as_mut_ptr().add(r * nb * 110),
                nb * 110,
            );
        }
    }
    let repacked = olorin::inference::repack::q3k_repack_8x8(
        src_8r.as_ptr(), n_rows_test, n,
    );

    // Build ARM-format q8_a (4 tokens of same data)
    let mut row_d = vec![0.0f32; nb * 4];
    for b in 0..nb {
        for c in 0..4 { row_d[b * 4 + c] = q8d[b]; }
    }
    let mut q8_a = vec![0u8; nb * 1168];
    unsafe {
        olorin::kernels::ffi_inference::q8k_repack_4(
            qs.as_ptr(), qs.as_ptr(), qs.as_ptr(), qs.as_ptr(),
            row_d.as_ptr(),
            bsums.as_ptr(), bsums.as_ptr(), bsums.as_ptr(), bsums.as_ptr(),
            q8_a.as_mut_ptr(), nb as i32,
        );
    }

    // SCALAR mirror (using my layout formulas)
    let mut scalar_out = [0.0f32; 8];
    for r in 0..8usize {
        for sb_idx in 0..nb {
            let bp = sb_idx * 1168;
            let ab = sb_idx * 1168;
            let raw = u16::from_le_bytes([repacked[bp + r * 2], repacked[bp + r * 2 + 1]]);
            let d_super = f16_to_f32(raw);
            let q8d_super = f32::from_le_bytes([
                q8_a[ab + 0], q8_a[ab + 1], q8_a[ab + 2], q8_a[ab + 3],
            ]);
            let mut sumi = 0i64;
            for sb in 0..16usize {
                let sc = (repacked[bp + 16 + sb * 8 + r] as i8) as i64;
                let mut sub_dot = 0i64;
                for pos in 0..16usize {
                    let sp = sb / 2;
                    let is_hi = (sb % 2) == 1;
                    let k = pos / 4;
                    let p = pos % 4;
                    let chunk_base = bp + 144 + sp * 128 + k * 32;
                    let dst_byte_base = if r < 4 { chunk_base + r * 4 } else { chunk_base + 16 + (r - 4) * 4 };
                    let byte = repacked[dst_byte_base + p] as i8 as i32;
                    let q3_signed: i32 = if is_hi { byte >> 4 } else { ((byte as i32) << 28) >> 28 };
                    let g = sb * 4 + (pos / 4);
                    let g_byte_off = ab + 16 + g * 16 + 0 * 4 + (pos % 4);
                    let q8_byte = q8_a[g_byte_off] as i8 as i32;
                    sub_dot += (q3_signed as i64) * (q8_byte as i64);
                }
                sumi += sc * sub_dot;
            }
            scalar_out[r] += d_super * q8d_super * (sumi as f32);
        }
    }
    eprintln!("SCALAR: {:?}", scalar_out);

    // KERNEL: run q3k_8x8_q8k_gemm with these inputs.
    // Need to repack ONLY the first tile here — model has more rows, but repack covered just 8.
    let mut gemm_out = vec![0.0f32; 4 * nc];
    let mut scratch = vec![0u8; 512];
    // We only care about the first tile (8 rows × 8 cols actually wait — this repack only has 1 tile).
    // The kernel iterates x = 0..nc/8. Our repacked is 1 tile only. To avoid OOB, build a dummy full repacked.
    // Simpler: just check if my 8 rows match the kernel's first tile output by using full-size repack.
    let full_repacked = unsafe {
        let mut buf = vec![0u8; (nc / 8) * nb * 1168];
        olorin::kernels::ffi_inference::q3k_repack_8x8(
            lw.w_up, buf.as_mut_ptr(), nc as i32, n as i32,
        );
        buf
    };
    unsafe {
        olorin::kernels::ffi_inference::q3k_8x8_q8k_gemm(
            full_repacked.as_ptr(),
            q8_a.as_ptr(),
            scratch.as_mut_ptr(),
            gemm_out.as_mut_ptr(),
            nc as i32, n as i32, 4i32, nc as i32,
        );
    }
    eprintln!("KERNEL: {:?}", &gemm_out[0..8]);

    eprintln!("\nrow | ref         | scalar      | kernel      | scalar-ref | kernel-ref");
    for r in 0..8 {
        let s = scalar_out[r];
        let k = gemm_out[0 * nc + r];
        let rf = ref_out[r];
        eprintln!("{}   | {:>+11.6} | {:>+11.6} | {:>+11.6} | {:>+10.3e} | {:>+10.3e}",
                  r, rf, s, k, s - rf, k - rf);
    }
}

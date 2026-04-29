//! Scalar Rust mirror of q3k_8x8_q8k_gemm_arm. Validates whether my mental
//! model of the layout/kernel is correct, independently of LLVM codegen.
//! If this matches the reference (q3k_dot_q8k), the model is right and any
//! Pi mismatches must be kernel-codegen issues. If this doesn't match, the
//! model itself is wrong.

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
            // subnormal — convert to f32 normalized
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
fn q3k_8x8_scalar_matches_q3k_dot_q8k() {
    if !Path::new(&model_path()).exists() {
        eprintln!("SKIP: no model");
        return;
    }
    let gguf = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    let lw = &model.layers[0];
    assert_eq!(lw.w_up_dtype, olorin::inference::matmul::GGML_TYPE_Q3_K);
    let n = model.hidden_dim;
    let nb = n / 256;
    let pow2 = olorin::inference::matmul::pow2_table();
    let row_bytes_src = nb * 110;

    // Test on first 8 weight rows × 1 token activation.
    let n_rows_test = 8;
    let n_cols_test = n;
    let nb_test = nb;

    // Build synthetic q8 input (token 0 of standard test pattern).
    let mut input = vec![0.0f32; n];
    for i in 0..n {
        input[i] = 0.01 * ((0 * 11 + i * 3) % 89) as f32 - 0.4;
    }
    let mut qs = vec![0i8; n + 12];
    let mut q8d = vec![0.0f32; nb];
    let mut bsums = vec![0i16; nb * 16];
    unsafe {
        olorin::kernels::ffi_inference::quant_f32_q8k(
            input.as_ptr(), qs.as_mut_ptr(), q8d.as_mut_ptr(), bsums.as_mut_ptr(), n as i32,
        );
    }

    // Reference: q3k_dot_q8k for each of 8 rows.
    let mut ref_out = [0.0f32; 8];
    let mut ref_blk0 = [0.0f32; 8]; // just super-block 0 contribution
    for r in 0..8 {
        ref_out[r] = unsafe {
            olorin::kernels::ffi_inference::q3k_dot_q8k(
                lw.w_up.add(r * row_bytes_src),
                qs.as_ptr(), bsums.as_ptr(),
                nb as i32, q8d.as_ptr(), pow2.as_ptr(),
            )
        };
        // Single-block reference for row r
        ref_blk0[r] = unsafe {
            olorin::kernels::ffi_inference::q3k_dot_q8k(
                lw.w_up.add(r * row_bytes_src),
                qs.as_ptr(), bsums.as_ptr(),
                1i32, q8d.as_ptr(), pow2.as_ptr(),
            )
        };
    }
    eprintln!("REF (full):  {:?}", ref_out);
    eprintln!("REF (blk0):  {:?}", ref_blk0);

    // Build the repacked Q3Kx8 tile for these 8 rows.
    let mut src_8r = vec![0u8; n_rows_test * nb_test * 110];
    unsafe {
        for r in 0..n_rows_test {
            std::ptr::copy_nonoverlapping(
                lw.w_up.add(r * row_bytes_src),
                src_8r.as_mut_ptr().add(r * nb_test * 110),
                nb_test * 110,
            );
        }
    }
    let repacked = olorin::inference::repack::q3k_repack_8x8(
        src_8r.as_ptr(), n_rows_test, n_cols_test,
    );

    // Build ARM-format block_q8_Kx4 manually (4-byte-granular interleave that
    // matches my kernel's vdot_lane_i32 access pattern). The host x86
    // q8k_repack_4 uses a DIFFERENT physical layout (8-byte interleave with
    // Q4K-style sub-block segments), so we can't use the kernel here.
    // ARM layout: per super-block b (1168 B):
    //   +0..15: 4 × f32 d (one per token)
    //   +16..1039: 64 groups × 16 B; group g byte (r*4 + p) = row r's qs[g*4+p]
    //   +1040..1167: bsums (unused for Q3K)
    let mut q8_a = vec![0u8; nb * 1168];
    for b in 0..nb {
        let off = b * 1168;
        for c in 0..4usize {
            let d_bytes = q8d[b].to_le_bytes();
            for i in 0..4 { q8_a[off + c * 4 + i] = d_bytes[i]; }
        }
        for g in 0..64usize {
            for c in 0..4usize {
                for j in 0..4usize {
                    let elem = g * 4 + j;
                    q8_a[off + 16 + g * 16 + c * 4 + j] = qs[b * 256 + elem] as u8;
                }
            }
        }
    }

    // VERIFY Q8K LAYOUT: my (sb, pos) → byte offset must produce exactly the
    // same q8 byte values as the original qs array for token 0, super-block 0.
    let mut q8_mismatches = 0;
    for sb in 0..16 {
        for pos in 0..16 {
            let g = sb * 4 + (pos / 4);
            let g_byte_off = 16 + g * 16 + 0 * 4 + (pos % 4); // ab=0
            let from_q8a = q8_a[g_byte_off] as i8;
            let from_qs = qs[sb * 16 + pos];
            if from_q8a != from_qs {
                q8_mismatches += 1;
                if q8_mismatches <= 5 {
                    eprintln!("Q8 mismatch sb={} pos={}: q8_a={} qs={}", sb, pos, from_q8a, from_qs);
                }
            }
        }
    }
    let q8d_check = f32::from_le_bytes([q8_a[0], q8_a[1], q8_a[2], q8_a[3]]);
    eprintln!("Q8 mismatches blk0: {} | q8d check: q8a={} q8d={}", q8_mismatches, q8d_check, q8d[0]);

    // SIMPLEST POSSIBLE SCALAR using my layout formulas — for each output row,
    // for each super-block, for each sub-block, for each element, dequant via my
    // formula and dot with q8 (read from the BLOCK_Q8_KX4 layout). No vdot, no
    // chunks. If this matches reference, my layout/dequant is right and the
    // earlier scalar's bug is in the chunk-loop structure. If this doesn't
    // match, the layout/dequant is wrong.
    let mut simple_out = [0.0f32; 8];
    for r in 0..8usize {
        for sb_idx in 0..(nb as usize) {
            let bp = sb_idx * 1168;
            let ab = sb_idx * 1168;
            // Row d (f16) for this super-block
            let raw = u16::from_le_bytes([repacked[bp + r * 2], repacked[bp + r * 2 + 1]]);
            let d_super = f16_to_f32(raw);
            // q8 d for token 0 (col 0)
            let q8d_super = f32::from_le_bytes([
                q8_a[ab + 0], q8_a[ab + 1], q8_a[ab + 2], q8_a[ab + 3],
            ]);
            // Sum over 16 sub-blocks
            let mut sumi = 0i64;
            for sb in 0..16usize {
                let sc = (repacked[bp + 16 + sb * 8 + r] as i8) as i64;
                let mut sub_dot = 0i64;
                for pos in 0..16usize {
                    // Decode q3_signed via my layout formula
                    let sp = sb / 2;
                    let is_hi = (sb % 2) == 1;
                    let k = pos / 4;
                    let p = pos % 4;
                    let chunk_base = bp + 144 + sp * 128 + k * 32;
                    let dst_byte_base = if r < 4 { chunk_base + r * 4 } else { chunk_base + 16 + (r - 4) * 4 };
                    let byte = repacked[dst_byte_base + p] as i8 as i32;
                    let q3_signed: i32 = if is_hi { byte >> 4 } else { ((byte as i32) << 28) >> 28 };
                    // q8 byte for sub-block sb element pos, for token 0 (col 0)
                    // sub-block sb's 16 elements span groups [sb*4..sb*4+3] in block_q8_Kx4
                    // group g, byte position c*4 + (pos % 4) within the group's 16 bytes
                    let g = sb * 4 + (pos / 4);   // group index in Q8K
                    let g_byte_off = ab + 16 + g * 16 + 0 * 4 + (pos % 4); // c=0
                    let q8_byte = q8_a[g_byte_off] as i8 as i32;
                    sub_dot += (q3_signed as i64) * (q8_byte as i64);
                }
                sumi += sc * sub_dot;
            }
            simple_out[r] += d_super * q8d_super * (sumi as f32);
        }
    }
    eprintln!("SIMPLE (full):  {:?}", simple_out);
    eprintln!("REF (full):     {:?}", ref_out);

    // Compute simple just for blk 0
    let mut simple_blk0 = [0.0f32; 8];
    for r in 0..8usize {
        let bp = 0;
        let ab = 0;
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
        simple_blk0[r] = d_super * q8d_super * (sumi as f32);
    }
    eprintln!("SIMPLE (blk0):  {:?}", simple_blk0);
    eprintln!("REF (blk0):     {:?}", ref_blk0);
    let mut max_diff_simple = 0.0f32;
    for r in 0..8 {
        let d = (simple_out[r] - ref_out[r]).abs();
        if d > max_diff_simple { max_diff_simple = d; }
    }
    eprintln!("simple-vs-ref max_diff={:.3e}", max_diff_simple);

    let mut scalar_out = [0.0f32; 8];

    // Tile 0 base in repacked = 0; nb tiles × 1168 bytes per super-block.
    for sb_idx in 0..(nb as usize) {
        let bp = sb_idx * 1168; // we have 1 tile, multiple super-blocks
        let ab = sb_idx * 1168;

        // Load 8 row d's for this super-block
        let mut row_d_f32 = [0.0f32; 8];
        for r in 0..8 {
            let raw = u16::from_le_bytes([repacked[bp + r * 2], repacked[bp + r * 2 + 1]]);
            row_d_f32[r] = f16_to_f32(raw);
        }
        // Load 4 q8 d's for this super-block from block_q8_Kx4
        let mut q8d_local = [0.0f32; 4];
        for c in 0..4 {
            q8d_local[c] = f32::from_le_bytes([
                q8_a[ab + c * 4], q8_a[ab + c * 4 + 1],
                q8_a[ab + c * 4 + 2], q8_a[ab + c * 4 + 3]
            ]);
        }

        // For each pair sp of sub-blocks (2*sp, 2*sp+1)
        for sp in 0..8 {
            let q3_base = bp + 144 + sp * 128;
            let q8_lo_base = ab + 16 + sp * 128;
            let q8_hi_base = q8_lo_base + 64;

            // Load scales: sub-block 2*sp and 2*sp+1, each 8 row scales.
            let mut sc_lo = [0i32; 8];
            let mut sc_hi = [0i32; 8];
            for r in 0..8 {
                sc_lo[r] = (repacked[bp + 16 + (sp * 2) * 8 + r] as i8) as i32;
                sc_hi[r] = (repacked[bp + 16 + (sp * 2 + 1) * 8 + r] as i8) as i32;
            }

            // For each k chunk (4 elements per row per sub-half)
            // Accumulate per-row × per-q8-col i32 partial dots
            let mut al = [[0i32; 4]; 8]; // al[row][q8_col]
            let mut ah = [[0i32; 4]; 8];

            for k in 0..4 {
                // Load weight bytes: 16 bytes for rows 0..3, 16 for rows 4..7
                let w_bytes_03 = &repacked[q3_base + k * 32 .. q3_base + k * 32 + 16];
                let w_bytes_47 = &repacked[q3_base + k * 32 + 16 .. q3_base + k * 32 + 32];
                let w_bytes_all = [w_bytes_03, w_bytes_47];

                // Load Q8K bytes
                let q8_lo = &q8_a[q8_lo_base + k * 16 .. q8_lo_base + k * 16 + 16];
                let q8_hi = &q8_a[q8_hi_base + k * 16 .. q8_hi_base + k * 16 + 16];

                for half in 0..2 { // 0=rows 0..3, 1=rows 4..7
                    let w_bytes = w_bytes_all[half];
                    let row_offset = half * 4;
                    for rr in 0..4 {
                        let r = row_offset + rr;
                        // Each row's 4 bytes hold 8 element-pairs (low+high nibble)
                        let mut nib_lo = [0i32; 4];
                        let mut nib_hi = [0i32; 4];
                        for j in 0..4 {
                            let b = w_bytes[rr * 4 + j] as i8;
                            // sign-extend low nibble: (b << 4) >> 4
                            nib_lo[j] = ((b as i32) << 28) >> 28;
                            // sign-extend high nibble: b >> 4
                            nib_hi[j] = (b >> 4) as i32;
                        }
                        // Compute partial dot for each q8 col
                        for c in 0..4 {
                            let q8_lo_lane = &q8_lo[c * 4 .. c * 4 + 4];
                            let q8_hi_lane = &q8_hi[c * 4 .. c * 4 + 4];
                            let mut acc_lo = 0i32;
                            let mut acc_hi = 0i32;
                            for j in 0..4 {
                                acc_lo += nib_lo[j] * (q8_lo_lane[j] as i8 as i32);
                                acc_hi += nib_hi[j] * (q8_hi_lane[j] as i8 as i32);
                            }
                            al[r][c] += acc_lo;
                            ah[r][c] += acc_hi;
                        }
                    }
                }
            }

            // Apply scales and FMA into scalar_out (only col 0 for parity test)
            for r in 0..8 {
                // contribution from sub-block 2*sp (lo) at col 0:
                let lo_contrib = (sc_lo[r] * al[r][0]) as f32;
                let hi_contrib = (sc_hi[r] * ah[r][0]) as f32;
                let d_q8d = row_d_f32[r] * q8d_local[0];
                scalar_out[r] += d_q8d * (lo_contrib + hi_contrib);
            }
        }
    }
    eprintln!("SCALAR: {:?}", scalar_out);

    // Compare scalar to reference
    let mut max_diff = 0.0f32;
    for r in 0..8 {
        let d = (scalar_out[r] - ref_out[r]).abs();
        if d > max_diff { max_diff = d; }
        eprintln!("row {}: ref={:.6} scalar={:.6} diff={:.3e}",
                  r, ref_out[r], scalar_out[r], d);
    }
    let pass = max_diff < 1e-3;
    eprintln!("max_diff={:.3e}  {}", max_diff, if pass { "PASS" } else { "FAIL" });
    assert!(pass, "scalar mirror differs from q3k_dot_q8k by max {max_diff:.3e}");
}

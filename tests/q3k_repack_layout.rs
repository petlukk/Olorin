//! Host-runnable verification that q3k_repack_8x8 produces the bytes that
//! q3k_dot_8x8_gemm_arm.ea expects, by independently dequantizing each
//! element from both the source Q3K and the repacked tile and comparing.
//!
//! This isolates layout bugs from kernel bugs — runs on x86 host.

use std::path::Path;

fn model_path() -> String {
    let home = std::env::var("HOME").unwrap();
    format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M-q3kffnup.gguf")
}

#[test]
fn q3k_repack_layout_matches_dequant() {
    if !Path::new(&model_path()).exists() {
        eprintln!("SKIP: no model at {}", model_path());
        return;
    }
    let gguf = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    let lw = &model.layers[0];
    assert_eq!(lw.w_up_dtype, olorin::inference::matmul::GGML_TYPE_Q3_K);

    let n = model.hidden_dim;
    let nc = model.ffn_dim[0];
    let nb = n / 256;
    eprintln!("Layout test: nc={nc}, n={n}, nb={nb}");

    // Restrict to first 8 rows × first super-block (1 tile, 1 super-block).
    let n_rows_test: usize = 8;
    let n_cols_test: usize = 256;
    let row_bytes_src = nb * 110;
    let nb_test = 1;

    // Allocate a small source buffer holding 8 rows of Q3K data, contiguous
    // (mimicking row-major layout). Copy from layer's actual weight.
    let mut src = vec![0u8; n_rows_test * nb_test * 110];
    unsafe {
        for r in 0..n_rows_test {
            std::ptr::copy_nonoverlapping(
                lw.w_up.add(r * row_bytes_src),
                src.as_mut_ptr().add(r * 110),
                110,
            );
        }
    }

    // Repack via the kernel.
    let repacked = olorin::inference::repack::q3k_repack_8x8(
        src.as_ptr(), n_rows_test, n_cols_test,
    );
    assert_eq!(repacked.len(), 1168);

    // Dequant tile bytes: pull (row, sb, elem) → q3_signed expected, both from
    // source (canonical) and from repacked (via my layout formulas), and compare.
    let tile = &repacked[..];

    // Per-row d (f16 little-endian at offset 0..15)
    let mut d_repack = [0u16; 8];
    for r in 0..8 {
        d_repack[r] = u16::from_le_bytes([tile[r * 2], tile[r * 2 + 1]]);
    }
    let mut d_src = [0u16; 8];
    for r in 0..8 {
        d_src[r] = u16::from_le_bytes([src[r * 110 + 108], src[r * 110 + 109]]);
    }
    assert_eq!(d_repack, d_src, "f16 d values mismatch");

    // Reference Q3K dequant (full block) and compare to repacked-extracted values.
    // For each (row, sb=0..15, elem=0..15):
    let mut errors = Vec::new();
    for r in 0..8 {
        let blk_off = r * 110;
        // Source-side dequant for sub-block sb at within-sb pos:
        let q3_signed_src = |sb: usize, pos: usize| -> i32 {
            let shift = ((sb / 2) % 4) * 2;
            let mbit = sb / 2;
            let j_off = (sb / 8) * 32;
            let ab_off = (sb % 2) * 16;
            let qs_byte = src[blk_off + 32 + j_off + ab_off + pos] as i32;
            let hm_byte = src[blk_off + ab_off + pos] as i32;
            let q3u = ((qs_byte >> shift) & 3) | (((hm_byte >> mbit) & 1) << 2);
            q3u - 4
        };
        // Repacked-tile-side decode using my layout formulas:
        let q3_signed_tile = |sb: usize, pos: usize| -> i32 {
            // sb is in pair sp = sb / 2; within-pair we are "lo" if sb is even, "hi" if odd
            let sp = sb / 2;
            let is_hi = (sb % 2) == 1;
            // pos within sub-block: 0..15. chunk k = pos / 4, p = pos % 4.
            let k = pos / 4;
            let p = pos % 4;
            // Row r at chunk k of pair sp: byte offset within tile.
            let chunk_base = 144 + sp * 128 + k * 32;
            let dst_byte_base = if r < 4 { chunk_base + r * 4 } else { chunk_base + 16 + (r - 4) * 4 };
            let byte = tile[dst_byte_base + p] as i8 as i32;
            // Extract nibble: low for is_hi=false, high for is_hi=true.
            // Sign-extend 4-bit:
            if is_hi {
                byte >> 4
            } else {
                ((byte << 4) >> 4) as i32 & 0x0F | if (byte & 0x08) != 0 { -16i32 } else { 0 }
                // Equivalently: (byte << 28) >> 28  (i32 4-bit sign-extend)
            }
        };
        for sb in 0..16 {
            for pos in 0..16 {
                let exp = q3_signed_src(sb, pos);
                let got = q3_signed_tile(sb, pos);
                if exp != got {
                    errors.push((r, sb, pos, exp, got));
                    if errors.len() <= 10 {
                        eprintln!("MISMATCH r={r} sb={sb} pos={pos}: src={exp} tile={got}");
                    }
                }
            }
        }
    }
    assert_eq!(errors.len(), 0, "{} layout mismatches found out of {}",
               errors.len(), 8 * 16 * 16);

    // Also verify scales: byte 16 + sb*8 + r should equal the (sb, r) signed scale [-32..31].
    let mut scale_errors = Vec::new();
    for r in 0..8 {
        let blk_off = r * 110;
        // Reference: unpack 16 scales from source's scales[12]
        let a0 = u32::from_le_bytes([src[blk_off + 96], src[blk_off + 97], src[blk_off + 98], src[blk_off + 99]]);
        let a1 = u32::from_le_bytes([src[blk_off + 100], src[blk_off + 101], src[blk_off + 102], src[blk_off + 103]]);
        let a2 = u32::from_le_bytes([src[blk_off + 104], src[blk_off + 105], src[blk_off + 106], src[blk_off + 107]]);
        let kmask1 = 0x03030303u32;
        let kmask2 = 0x0f0f0f0fu32;
        let u0 = (a0 & kmask2) | (((a2 >> 0) & kmask1) << 4);
        let u1 = (a1 & kmask2) | (((a2 >> 2) & kmask1) << 4);
        let u2 = ((a0 >> 4) & kmask2) | (((a2 >> 4) & kmask1) << 4);
        let u3 = ((a1 >> 4) & kmask2) | (((a2 >> 6) & kmask1) << 4);
        let mut exp_scales = [0i32; 16];
        exp_scales[0] = ((u0 & 0xff) as i32) - 32;
        exp_scales[1] = (((u0 >> 8) & 0xff) as i32) - 32;
        exp_scales[2] = (((u0 >> 16) & 0xff) as i32) - 32;
        exp_scales[3] = (((u0 >> 24) & 0xff) as i32) - 32;
        exp_scales[4] = ((u1 & 0xff) as i32) - 32;
        exp_scales[5] = (((u1 >> 8) & 0xff) as i32) - 32;
        exp_scales[6] = (((u1 >> 16) & 0xff) as i32) - 32;
        exp_scales[7] = (((u1 >> 24) & 0xff) as i32) - 32;
        exp_scales[8] = ((u2 & 0xff) as i32) - 32;
        exp_scales[9] = (((u2 >> 8) & 0xff) as i32) - 32;
        exp_scales[10] = (((u2 >> 16) & 0xff) as i32) - 32;
        exp_scales[11] = (((u2 >> 24) & 0xff) as i32) - 32;
        exp_scales[12] = ((u3 & 0xff) as i32) - 32;
        exp_scales[13] = (((u3 >> 8) & 0xff) as i32) - 32;
        exp_scales[14] = (((u3 >> 16) & 0xff) as i32) - 32;
        exp_scales[15] = (((u3 >> 24) & 0xff) as i32) - 32;
        for sb in 0..16 {
            let got = tile[16 + sb * 8 + r] as i8 as i32;
            if got != exp_scales[sb] {
                scale_errors.push((r, sb, exp_scales[sb], got));
                if scale_errors.len() <= 10 {
                    eprintln!("SCALE MISMATCH r={r} sb={sb}: src={} tile={}", exp_scales[sb], got);
                }
            }
        }
    }
    assert_eq!(scale_errors.len(), 0, "{} scale layout mismatches", scale_errors.len());

    eprintln!("PASS: 8 rows × 16 sub-blocks × 16 elements verified");
}

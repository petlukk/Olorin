//! Byte-layout test for q8k_repack_4.
//!
//! Builds 4 rows of synthetic Q8K input with non-constant values at every
//! (row, block, position) combination, runs the kernel, and checks every
//! output byte against the expected block_q8_Kx4 layout.
//!
//! Layout confirmed in:
//!   docs/superpowers/research/2026-04-11-q4k-8x8-gemm-ea-template.md
//!
//! The layout is 8-byte-granular row-interleaved (NOT row-major).
//! See kernels/q8k_repack_4.ea header comments for the formulas.

#[test]
fn q8k_repack_4_deltas() {
    olorin::kernels::ffi::init().unwrap();

    let nb = 3;
    let qk = 256;

    let (row_qs, row_d, row_bsums) = build_synthetic_input(nb, qk);
    let dst = run_repack(&row_qs, &row_d, &row_bsums, nb);

    // Section 1: deltas — dst[bloff + 0..16] = 4 × f32
    for b in 0..nb {
        let bloff = b * 1168;
        for r in 0..4 {
            let expected = row_d[b * 4 + r];
            let off = bloff + r * 4;
            let actual = f32::from_le_bytes([dst[off], dst[off + 1], dst[off + 2], dst[off + 3]]);
            assert_eq!(
                actual.to_bits(), expected.to_bits(),
                "b={b}, row={r}: delta mismatch. expected {expected}, got {actual}",
            );
        }
    }
    eprintln!("PASS: deltas verified for {nb} super-blocks");
}

#[test]
fn q8k_repack_4_qs_interleaved() {
    olorin::kernels::ffi::init().unwrap();

    let nb = 3;
    let qk = 256;

    let (row_qs, row_d, row_bsums) = build_synthetic_input(nb, qk);
    let dst = run_repack(&row_qs, &row_d, &row_bsums, nb);

    // Section 2: qs — 8-byte-granular row-interleaved layout.
    // Formula (from kernel header):
    //   dst_byte_off = bloff + 16 + (s/2)*256 + (s%2)*128 + (p/8)*32 + r*8 + (p%8)
    // Source: row_qs[r][b*256 + s*32 + p]
    for b in 0..nb {
        let bloff = b * 1168;
        for r in 0..4usize {
            for s in 0..8usize {
                for p in 0..32usize {
                    let src_val = row_qs[r][(b * qk) + s * 32 + p];
                    let dst_off = bloff + 16
                        + (s / 2) * 256
                        + (s % 2) * 128
                        + (p / 8) * 32
                        + r * 8
                        + (p % 8);
                    assert_eq!(
                        dst[dst_off] as i8, src_val,
                        "b={b}, row={r}, sb={s}, p={p}: qs mismatch at dst[{dst_off}]",
                    );
                }
            }
        }
    }
    eprintln!("PASS: qs interleaved layout verified for {nb} super-blocks");
}

#[test]
fn q8k_repack_4_bsums_interleaved() {
    olorin::kernels::ffi::init().unwrap();

    let nb = 3;
    let qk = 256;

    let (row_qs, row_d, row_bsums) = build_synthetic_input(nb, qk);
    let dst = run_repack(&row_qs, &row_d, &row_bsums, nb);

    // Section 3: bsums — row-group interleaved layout.
    // Formula (from kernel header):
    //   i16_off = (s/2)*16 + (r/2)*8 + (r%2)*4 + (s%2)*2 + h
    // Source: row_bsums[r][b*16 + s*2 + h]  (8 sub-blocks × 2 halves = 16 i16 per row)
    for b in 0..nb {
        let bloff = b * 1168;
        let bsums_base = bloff + 1040;
        for r in 0..4usize {
            for s in 0..8usize {
                for h in 0..2usize {
                    let src_val = row_bsums[r][b * 16 + s * 2 + h];
                    let i16_off = (s / 2) * 16
                        + (r / 2) * 8
                        + (r % 2) * 4
                        + (s % 2) * 2
                        + h;
                    let byte_off = bsums_base + i16_off * 2;
                    let actual = i16::from_le_bytes([dst[byte_off], dst[byte_off + 1]]);
                    assert_eq!(
                        actual, src_val,
                        "b={b}, row={r}, sb={s}, h={h}: bsum mismatch at i16_off={i16_off}",
                    );
                }
            }
        }
    }
    eprintln!("PASS: bsums interleaved layout verified for {nb} super-blocks");
}

// --- Helpers ---

fn build_synthetic_input(nb: usize, qk: usize) -> ([Vec<i8>; 4], Vec<f32>, [Vec<i16>; 4]) {
    let mut row_qs: [Vec<i8>; 4] = [vec![], vec![], vec![], vec![]];
    let mut row_bsums: [Vec<i16>; 4] = [vec![], vec![], vec![], vec![]];
    let mut row_d = Vec::new();

    for r in 0..4 {
        row_qs[r] = vec![0i8; nb * qk];
        row_bsums[r] = vec![0i16; nb * 16];
        for b in 0..nb {
            for i in 0..qk {
                row_qs[r][b * qk + i] = (((r * 7 + b * 11 + i) as i32) % 127 - 63) as i8;
            }
            for j in 0..16 {
                row_bsums[r][b * 16 + j] = (((r * 3 + b * 5 + j) as i16) % 31) - 15;
            }
        }
    }

    for b in 0..nb {
        for r in 0..4 {
            row_d.push(0.01 + (r as f32) * 0.001 + (b as f32) * 0.0001);
        }
    }

    (row_qs, row_d, row_bsums)
}

fn run_repack(
    row_qs: &[Vec<i8>; 4],
    row_d: &[f32],
    row_bsums: &[Vec<i16>; 4],
    nb: usize,
) -> Vec<u8> {
    let mut dst = vec![0u8; nb * 1168];
    unsafe {
        olorin::kernels::ffi_inference::q8k_repack_4(
            row_qs[0].as_ptr(),
            row_qs[1].as_ptr(),
            row_qs[2].as_ptr(),
            row_qs[3].as_ptr(),
            row_d.as_ptr(),
            row_bsums[0].as_ptr(),
            row_bsums[1].as_ptr(),
            row_bsums[2].as_ptr(),
            row_bsums[3].as_ptr(),
            dst.as_mut_ptr(),
            nb as i32,
        );
    }
    dst
}

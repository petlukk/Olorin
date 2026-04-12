//! Batched matmul dispatch: Q4K 8x8 gemm for N input columns.

use crate::kernels::ffi_inference;

/// Run Q4K 8x8 gemm: repacked_weights[nc, n_inner] × q8k_input[n_inner, N] → out[nc, N].
///
/// Input: N independently Q8K-quantized columns stored contiguously in qs/d/bsums arrays.
/// Layout for N columns:
///   qs: column k's qs starts at offset k * qs_stride where qs_stride = n_inner + 12
///   d:  column k's d starts at offset k * nb where nb = n_inner / 256
///   bsums: column k's bsums starts at offset k * nb * 16
///
/// N must be a multiple of 4 (caller must zero-pad if needed).
///
/// `q8_a_buf` is caller-owned scratch for repacked A-side tiles.
/// `gemm_scratch` is caller-owned scratch (128 bytes min).
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemm_q4k_8x8(
    repacked_weights: *const u8,
    qs: *const i8,
    d: *const f32,
    bsums: *const i16,
    q8_a_buf: *mut u8,
    gemm_scratch: *mut u8,
    out: *mut f32,
    n_inner: usize,
    nc: usize,
    n: usize,
) {
    let nb = n_inner / 256;
    let qs_stride = n_inner + 12;
    let block_q8_kx4_size = nb * 1168;

    for group in 0..(n / 4) {
        let r0 = group * 4;

        // Interleave d values: [d_r0_b0, d_r1_b0, d_r2_b0, d_r3_b0, d_r0_b1, ...]
        let mut row_d = vec![0.0f32; nb * 4];
        for b in 0..nb {
            for r in 0..4 {
                row_d[b * 4 + r] = *d.add((r0 + r) * nb + b);
            }
        }

        let dst_off = group * block_q8_kx4_size;
        ffi_inference::q8k_repack_4(
            qs.add(r0 * qs_stride),
            qs.add((r0 + 1) * qs_stride),
            qs.add((r0 + 2) * qs_stride),
            qs.add((r0 + 3) * qs_stride),
            row_d.as_ptr(),
            bsums.add(r0 * nb * 16),
            bsums.add((r0 + 1) * nb * 16),
            bsums.add((r0 + 2) * nb * 16),
            bsums.add((r0 + 3) * nb * 16),
            q8_a_buf.add(dst_off),
            nb as i32,
        );
    }

    ffi_inference::q4k_8x8_q8k_gemm(
        repacked_weights,
        q8_a_buf as *const u8,
        gemm_scratch,
        out,
        nc as i32,
        n_inner as i32,
        n as i32,
        nc as i32,
    );
}

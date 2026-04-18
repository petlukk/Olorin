//! Q4K weight repacking — safe wrapper over the Ea `q4k_repack_8x8` kernel.
//!
//! Standard Q4K weights live in row-major blocks (256 quants, 144 bytes each).
//! `q4k_repack_8x8` interleaves 8 consecutive rows into `block_q4_Kx8` tiles
//! (1152 bytes per tile) so one SIMD pass computes 8 dot products at once.
//!
//! Byte-for-byte match with llama.cpp `block_q4_Kx8` layout (see
//! `llama.cpp/ggml/src/ggml-cpu/repack.cpp:2836` make_block_q4_Kx8).

use crate::kernels::ffi_inference;

/// Repack `n_rows × n_cols` Q4K weights from standard row-major layout into
/// 8-row interleaved `block_q4_Kx8` layout. Returns a newly allocated
/// `Vec<u8>` of the same total byte size as the input.
///
/// # Requirements
/// - `src` must point to at least `n_rows * (n_cols / 256) * 144` bytes of
///   valid Q4K block data.
/// - `n_rows` must be a multiple of 8.
/// - `n_cols` must be a multiple of 256.
/// - `olorin::kernels::ffi::init()` must have been called.
///
/// # Safety contract (documented, not enforced by type system)
/// The caller asserts `src` is valid for `n_rows * (n_cols / 256) * 144`
/// readable bytes. Dereferencing happens inside the Ea kernel. The function
/// signature is safe (takes `*const u8` by value) to match the call pattern
/// in `tests/repack_q4k.rs`.
pub fn q4k_repack_8x8(src: *const u8, n_rows: usize, n_cols: usize) -> Vec<u8> {
    debug_assert!(n_rows % 8 == 0,  "q4k_repack_8x8: n_rows ({n_rows}) must be a multiple of 8");
    debug_assert!(n_cols % 256 == 0, "q4k_repack_8x8: n_cols ({n_cols}) must be a multiple of 256");

    let nb = n_cols / 256;
    let row_bytes = nb * 144;
    let total = n_rows * row_bytes;

    let mut dst = vec![0u8; total];
    unsafe {
        ffi_inference::q4k_repack_8x8(
            src,
            dst.as_mut_ptr(),
            n_rows as i32,
            n_cols as i32,
        );
    }
    dst
}

/// Repack Q6K weights: interleave 4 consecutive rows into contiguous tiles.
/// Each tile = 4 × 210 = 840 bytes per superblock column.
///
/// # Requirements
/// - `n_rows` must be a multiple of 4.
/// - `n_cols` must be a multiple of 256.
pub fn q6k_repack_4row(src: *const u8, n_rows: usize, n_cols: usize) -> Vec<u8> {
    debug_assert!(n_rows % 4 == 0, "q6k_repack_4row: n_rows ({n_rows}) must be a multiple of 4");
    debug_assert!(n_cols % 256 == 0, "q6k_repack_4row: n_cols ({n_cols}) must be a multiple of 256");

    let n_blocks = n_cols / 256;
    let row_bytes = n_blocks * 210;
    let tile_bytes = 4 * 210;
    let n_quads = n_rows / 4;
    let mut dst = vec![0u8; n_quads * n_blocks * tile_bytes];

    for quad in 0..n_quads {
        for blk in 0..n_blocks {
            for r in 0..4usize {
                let src_off = (quad * 4 + r) * row_bytes + blk * 210;
                let dst_off = (quad * n_blocks + blk) * tile_bytes + r * 210;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.add(src_off),
                        dst.as_mut_ptr().add(dst_off),
                        210,
                    );
                }
            }
        }
    }
    dst
}

/// Pre-compute `f16_to_f32(d)` for every (quad, block, row) of a Q6K weight.
///
/// The live `q6k_repacked_batch_ws` kernel pays a scattered f16 load + convert
/// on every token × every block × every row (4 per quad) in the hot path.
/// For the Gemma 4 output head (m=262144, k=1536) this is ~15 ms/decode-step
/// of arithmetic that only depends on the weight — invariant across tokens.
///
/// Moving it to load-time removes it from the hot path. Output layout is
/// indexed as `d_arr[(quad * n_blocks + blk) * 4 + r]` so the inference
/// kernel can read four contiguous floats per `(quad, blk)` and just
/// multiply by `q8_d[blk]` (the only token-specific scale).
///
/// # Requirements
/// - `n_rows` must be a multiple of 4.
/// - `n_cols` must be a multiple of 256.
/// - `src` must point to at least `n_rows * (n_cols/256) * 210` bytes of Q6K.
///
/// Memory cost: `(n_rows/4) * (n_cols/256) * 4 * sizeof(f32)` — e.g. 6.3 MB
/// for Gemma 4 output head vs 330 MB of weight. Negligible.
pub fn q6k_precompute_d_arr(src: *const u8, n_rows: usize, n_cols: usize) -> Vec<f32> {
    debug_assert!(n_rows % 4 == 0, "q6k_precompute_d_arr: n_rows ({n_rows}) must be a multiple of 4");
    debug_assert!(n_cols % 256 == 0, "q6k_precompute_d_arr: n_cols ({n_cols}) must be a multiple of 256");

    let n_blocks = n_cols / 256;
    let row_bytes = n_blocks * 210;
    let n_quads = n_rows / 4;
    let mut d_arr = vec![0.0f32; n_quads * n_blocks * 4];

    for quad in 0..n_quads {
        for blk in 0..n_blocks {
            for r in 0..4usize {
                // d lives at byte 208 of each Q6K block (f16)
                let byte_off = (quad * 4 + r) * row_bytes + blk * 210 + 208;
                let raw = unsafe {
                    let p = src.add(byte_off);
                    u16::from_le_bytes([*p, *p.add(1)])
                };
                d_arr[(quad * n_blocks + blk) * 4 + r] =
                    crate::inference::matmul::f16_to_f32_scalar(raw);
            }
        }
    }
    d_arr
}

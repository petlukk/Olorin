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

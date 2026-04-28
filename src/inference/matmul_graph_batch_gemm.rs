//! GEMM-style batch matmul drivers for the prefill graph loop.
//!
//! Work-stealing across output-row tiles. Each claimed tile invokes a
//! whole-matrix GEMM kernel that processes all `nr` tokens internally —
//! contrast with the matvec drivers in `matmul_graph_batch.rs`, which
//! claim per (row × token chunk) and call a 1-row dot kernel per token.

use std::sync::atomic::{AtomicI32, Ordering};
use crate::kernels::ffi_inference;
#[cfg(target_arch = "aarch64")]
use super::matmul::Q5K_BLOCK_BYTES;

/// Q4K 8×8 repacked batch GEMM: work-stealing across output row tiles.
/// Calls the GEMM kernel per claimed tile — weight data loaded once, all tokens processed.
/// Matches llama.cpp repack.cpp:4241 `gemm(ne00, dst + src0_start, ...)`.
///
/// `q8_a` must be pre-repacked into block_q8_Kx4 format via q8k_repack_4.
/// `nr` = number of tokens (must be multiple of 4, caller zero-pads).
/// `nc` = number of output rows (total, must be multiple of 8).
/// Output: `out[token * output_stride + row]`.
#[allow(clippy::too_many_arguments)]
pub fn q4k_gemm_8x8_batch_ws(
    packed: *const u8,
    q8_a: *const u8,
    output: *mut f32,
    n_inner: usize,
    nc: usize,
    nr: usize,
    output_stride: usize,
    current_chunk: &AtomicI32,
    ith: usize,
    _nth: usize,
) {
    debug_assert!(nc % 8 == 0);
    debug_assert!(n_inner % 256 == 0);
    debug_assert!(nr % 4 == 0);

    let nb = n_inner / 256;
    let tile_bytes = nb * 1152; // block_q4_Kx8 tile size
    let n_tiles = nc / 8;

    let mut scratch = [0u8; 128];

    let mut chunk = ith as i32;
    while (chunk as usize) < n_tiles {
        let tile = chunk as usize;
        let w_ptr = unsafe { packed.add(tile * tile_bytes) };
        let col_start = tile * 8;

        unsafe {
            ffi_inference::q4k_8x8_q8k_gemm(
                w_ptr,
                q8_a,
                scratch.as_mut_ptr(),
                output.add(col_start),
                output_stride as i32,
                n_inner as i32,
                nr as i32,
                8i32,
            );
        }

        chunk = current_chunk.fetch_add(1, Ordering::Relaxed);
    }
}

/// Q5K 8×8 repacked batch GEMM: work-stealing across output row tiles.
/// Mirrors `q4k_gemm_8x8_batch_ws` but for Q5K (block_q5_Kx8, 1408 B/sb tiles).
///
/// `packed` must be pre-repacked via `q5k_repack_8x8`.
/// `q8_a` must be pre-repacked into block_q8_Kx4 format via q8k_repack_4.
/// `nr` = number of tokens (must be multiple of 4, caller zero-pads).
/// `nc` = number of output rows (total, must be multiple of 8).
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub fn q5k_gemm_8x8_batch_ws(
    packed: *const u8,
    q8_a: *const u8,
    output: *mut f32,
    n_inner: usize,
    nc: usize,
    nr: usize,
    output_stride: usize,
    current_chunk: &AtomicI32,
    ith: usize,
    _nth: usize,
) {
    debug_assert!(nc % 8 == 0);
    debug_assert!(n_inner % 256 == 0);
    debug_assert!(nr % 4 == 0);

    let nb = n_inner / 256;
    let tile_bytes = nb * 1408; // block_q5_Kx8 tile size
    let n_tiles = nc / 8;

    let mut scratch = [0u8; 512];

    let mut chunk = ith as i32;
    while (chunk as usize) < n_tiles {
        let tile = chunk as usize;
        let w_ptr = unsafe { packed.add(tile * tile_bytes) };
        let col_start = tile * 8;

        unsafe {
            ffi_inference::q5k_8x8_q8k_gemm(
                w_ptr,
                q8_a,
                scratch.as_mut_ptr(),
                output.add(col_start),
                output_stride as i32,
                n_inner as i32,
                nr as i32,
                8i32,
            );
        }

        chunk = current_chunk.fetch_add(1, Ordering::Relaxed);
    }
}

/// Q5K GEMM batch — work-stealing across weight row chunks.
/// Uses q5k_gemm kernel: processes all nr tokens per weight chunk.
/// ARM-only — no x86 q5k_gemm kernel exists; x86 prefill uses matvec_batch_ws.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub fn q5k_gemm_batch_ws(
    weight: *const u8,
    q8_a: *const u8,
    output: *mut f32,
    n_inner: usize,
    nc: usize,        // weight rows (output dim)
    nr: usize,        // tokens (must be %4==0)
    output_stride: usize,
    current_chunk: &AtomicI32,
    ith: usize,
    _nth: usize,
) {
    let chunk_size = 8usize;
    let n_chunks = (nc + chunk_size - 1) / chunk_size;
    let row_bytes = (n_inner / 256) * Q5K_BLOCK_BYTES;

    let mut scratch = [0u8; 256];

    let mut chunk = ith as i32;
    while (chunk as usize) < n_chunks {
        let row_start = (chunk as usize) * chunk_size;
        let row_end = (row_start + chunk_size).min(nc);
        let n_rows = row_end - row_start;

        unsafe {
            ffi_inference::q5k_gemm(
                weight.add(row_start * row_bytes),
                q8_a,
                scratch.as_mut_ptr(),
                output.add(row_start),
                output_stride as i32,
                n_inner as i32,
                nr as i32,
                n_rows as i32,
            );
        }

        chunk = current_chunk.fetch_add(1, Ordering::Relaxed);
    }
}

/// Q6K GEMM batch — work-stealing across weight row chunks.
/// Uses q6k_gemm kernel: processes all nr tokens per weight chunk.
/// ARM-only — no x86 q6k_gemm kernel exists; x86 prefill uses matvec_batch_ws.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub fn q6k_gemm_batch_ws(
    weight: *const u8,
    q8_a: *const u8,
    output: *mut f32,
    n_inner: usize,
    nc: usize,        // weight rows (output dim)
    nr: usize,        // tokens (must be %4==0)
    output_stride: usize,
    current_chunk: &AtomicI32,
    ith: usize,
    _nth: usize,
) {
    // Chunk by weight rows. Each thread processes a slice of rows against all tokens.
    let chunk_size = 8usize; // process 8 weight rows per chunk
    let n_chunks = (nc + chunk_size - 1) / chunk_size;
    let row_bytes = (n_inner / 256) * 210; // Q6K block bytes per row

    let mut scratch = [0u8; 256];

    let mut chunk = ith as i32;
    while (chunk as usize) < n_chunks {
        let row_start = (chunk as usize) * chunk_size;
        let row_end = (row_start + chunk_size).min(nc);
        let n_rows = row_end - row_start;

        unsafe {
            ffi_inference::q6k_gemm(
                weight.add(row_start * row_bytes),
                q8_a,
                scratch.as_mut_ptr(),
                output.add(row_start),
                output_stride as i32,
                n_inner as i32,
                nr as i32,
                n_rows as i32,
            );
        }

        chunk = current_chunk.fetch_add(1, Ordering::Relaxed);
    }
}

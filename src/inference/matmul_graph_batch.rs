//! Batch/GEMM matmul ops for graph-loop threading (prefill path).
//!
//! Work-stealing across row tiles × token chunks, matching llama.cpp's
//! ggml_compute_forward_mul_mat chunking.

use std::sync::atomic::{AtomicI32, Ordering};
use crate::kernels::ffi_inference;
use super::matmul::*;

/// Q4K 8×8 repacked batch matvec: 2D work-stealing across (row_tiles × token_chunks).
/// Matches llama.cpp's ggml_compute_forward_mul_mat chunking: flattened 2D grid,
/// each thread claims (tile, token_range) pairs via atomic fetch_add.
#[allow(clippy::too_many_arguments)]
pub fn q4k_matvec_8x8_batch_ws(
    packed: *const u8, batch_q8_qs: *const i8, batch_q8_d: *const f32,
    batch_q8_bsums: *const i16, output: *mut f32,
    n_rows: usize, n_cols: usize, n_tokens: usize, output_stride: usize,
    current_chunk: &AtomicI32, ith: usize, _nth: usize,
) {
    debug_assert!(n_rows % 8 == 0);
    debug_assert!(n_cols % Q4K_BLOCK_SIZE == 0);

    let n_blocks = n_cols / Q4K_BLOCK_SIZE;
    let tile_bytes = n_blocks * 1152;
    let n_tiles = n_rows / 8;
    let qs_stride = n_cols + 12;
    let pow2 = pow2_table();
    let mut scratch = [0u8; 128];

    // 2D chunking: dim0 = row tiles (8 rows each), dim1 = token chunks
    let tok_chunk_size = 16usize;
    let n_tok_chunks = (n_tokens + tok_chunk_size - 1) / tok_chunk_size;
    let total_chunks = n_tiles * n_tok_chunks;

    let mut chunk = ith as i32;
    while (chunk as usize) < total_chunks {
        let tile = (chunk as usize) % n_tiles;
        let tok_idx = (chunk as usize) / n_tiles;
        let t_start = tok_idx * tok_chunk_size;
        let t_end = (t_start + tok_chunk_size).min(n_tokens);

        let w_ptr = unsafe { packed.add(tile * tile_bytes) };
        let out_col_off = tile * 8;

        for t in t_start..t_end {
            let q8 = unsafe { batch_q8_qs.add(t * qs_stride) };
            let q8_d = unsafe { batch_q8_d.add(t * n_blocks) };
            let bsums = unsafe { batch_q8_bsums.add(t * n_blocks * 16) };
            let out_ptr = unsafe { output.add(t * output_stride + out_col_off) };
            unsafe {
                ffi_inference::q4k_8x8_q8k_matvec(
                    w_ptr, q8, q8_d, bsums,
                    pow2.as_ptr(), scratch.as_mut_ptr(),
                    out_ptr, 8i32, n_cols as i32,
                );
            }
        }

        chunk = current_chunk.fetch_add(1, Ordering::Relaxed);
    }
}

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

/// Q6K GEMM batch — work-stealing across weight row chunks.
/// Uses q6k_gemm kernel: processes all nr tokens per weight chunk.
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

/// Q6K repacked batch matvec — work-stealing across row quads × token chunks.
#[allow(clippy::too_many_arguments)]
pub fn q6k_repacked_batch_ws(
    packed: *const u8, weight: *const u8,
    batch_q8_qs: *const i8, batch_q8_d: *const f32, batch_q8_bsums: *const i16,
    output: *mut f32, d_scratch: *mut f32,
    n_rows: usize, n_cols: usize, n_tokens: usize, output_stride: usize,
    current_chunk: &AtomicI32, ith: usize, _nth: usize,
) {
    let n_blocks = n_cols / Q6K_BLOCK_SIZE;
    let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
    let qs_stride = n_cols + 12;
    let full_quads = n_rows / 4;
    let tile_bytes = n_blocks * 840;

    let scratch_per = n_blocks * 4;
    let my_scratch = unsafe { d_scratch.add(ith * scratch_per) };

    let tok_chunk_size = 16usize;
    let n_tok_chunks = (n_tokens + tok_chunk_size - 1) / tok_chunk_size;
    let total_chunks = full_quads * n_tok_chunks;

    let mut chunk = ith as i32;
    while (chunk as usize) < total_chunks {
        let quad = (chunk as usize) % full_quads;
        let tok_idx = (chunk as usize) / full_quads;
        let t_start = tok_idx * tok_chunk_size;
        let t_end = (t_start + tok_chunk_size).min(n_tokens);
        let base_row = quad * 4;

        for t in t_start..t_end {
            let q8 = unsafe { batch_q8_qs.add(t * qs_stride) };
            let q8_d = unsafe { batch_q8_d.add(t * n_blocks) };
            let bsums = unsafe { batch_q8_bsums.add(t * n_blocks * 16) };
            let out_ptr = unsafe { output.add(t * output_stride + base_row) };

            unsafe {
                for blk in 0..n_blocks {
                    for r in 0..4usize {
                        let w = weight.add((base_row + r) * row_bytes + blk * Q6K_BLOCK_BYTES + 208);
                        let raw = u16::from_le_bytes([*w, *w.add(1)]);
                        *my_scratch.add(blk * 4 + r) = f16_to_f32_scalar(raw) * *q8_d.add(blk);
                    }
                }
                ffi_inference::q6k_dot_q8k_4row_repacked(
                    packed.add(quad * tile_bytes),
                    q8, bsums, out_ptr,
                    n_blocks as i32, my_scratch,
                );
            }
        }

        chunk = current_chunk.fetch_add(1, Ordering::Relaxed);
    }
}

/// Batch matvec fallback (any dtype): 2D work-stealing across (4-row chunks × token chunks).
#[allow(clippy::too_many_arguments)]
pub fn matvec_batch_ws(
    dtype: u32, weight: *const u8, batch_q8_qs: *const i8,
    batch_q8_d: *const f32, batch_q8_bsums: *const i16,
    output: *mut f32, d_scratch: *mut f32,
    n_rows: usize, n_cols: usize, n_tokens: usize, output_stride: usize,
    current_chunk: &AtomicI32, ith: usize, _nth: usize,
) {
    let n_blocks = n_cols / Q4K_BLOCK_SIZE;
    let qs_stride = n_cols + 12;
    let full_quads = n_rows / 4;
    let pow2 = pow2_table();

    let scratch_per = n_blocks * 4;
    let my_scratch = unsafe { d_scratch.add(ith * scratch_per) };

    // 2D chunking: dim0 = row quads (4 rows each), dim1 = token chunks
    let tok_chunk_size = 16usize;
    let n_tok_chunks = (n_tokens + tok_chunk_size - 1) / tok_chunk_size;
    let total_chunks = full_quads * n_tok_chunks;

    let mut chunk = ith as i32;
    while (chunk as usize) < total_chunks {
        let quad = (chunk as usize) % full_quads;
        let tok_idx = (chunk as usize) / full_quads;
        let t_start = tok_idx * tok_chunk_size;
        let t_end = (t_start + tok_chunk_size).min(n_tokens);
        let base_row = quad * 4;

        for t in t_start..t_end {
            let q8 = unsafe { batch_q8_qs.add(t * qs_stride) };
            let q8_d = unsafe { batch_q8_d.add(t * n_blocks) };
            let bsums = unsafe { batch_q8_bsums.add(t * n_blocks * 16) };
            let out_ptr = unsafe { output.add(t * output_stride + base_row) };

            match dtype {
                GGML_TYPE_Q4_K => {
                    let row_bytes = n_blocks * Q4K_BLOCK_BYTES;
                    unsafe {
                        ffi_inference::q4k_dot_q8k_4row(
                            weight.add(base_row * row_bytes),
                            weight.add((base_row + 1) * row_bytes),
                            weight.add((base_row + 2) * row_bytes),
                            weight.add((base_row + 3) * row_bytes),
                            q8, bsums, out_ptr,
                            n_blocks as i32, q8_d, pow2.as_ptr(),
                        );
                    }
                }
                GGML_TYPE_Q5_K => {
                    let row_bytes = n_blocks * Q5K_BLOCK_BYTES;
                    unsafe {
                        ffi_inference::q5k_dot_q8k_4row(
                            weight.add(base_row * row_bytes),
                            weight.add((base_row + 1) * row_bytes),
                            weight.add((base_row + 2) * row_bytes),
                            weight.add((base_row + 3) * row_bytes),
                            q8, bsums, out_ptr,
                            n_blocks as i32, q8_d, pow2.as_ptr(),
                        );
                    }
                }
                GGML_TYPE_Q6_K => {
                    let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
                    unsafe {
                        let d0 = my_scratch;
                        let d1 = my_scratch.add(n_blocks);
                        let d2 = my_scratch.add(n_blocks * 2);
                        let d3 = my_scratch.add(n_blocks * 3);
                        for blk in 0..n_blocks {
                            let off = 208;
                            for (row_off, d_ptr) in [(0, d0), (1, d1), (2, d2), (3, d3)] {
                                let w = weight.add((base_row + row_off) * row_bytes + blk * Q6K_BLOCK_BYTES + off);
                                let raw = u16::from_le_bytes([*w, *w.add(1)]);
                                *d_ptr.add(blk) = f16_to_f32_scalar(raw) * *q8_d.add(blk);
                            }
                        }
                        ffi_inference::q6k_dot_q8k_4row(
                            weight.add(base_row * row_bytes),
                            weight.add((base_row + 1) * row_bytes),
                            weight.add((base_row + 2) * row_bytes),
                            weight.add((base_row + 3) * row_bytes),
                            q8, bsums, out_ptr,
                            n_blocks as i32, d0, d1, d2, d3,
                        );
                    }
                }
                _ => panic!("unsupported weight dtype {dtype}"),
            }
        }

        chunk = current_chunk.fetch_add(1, Ordering::Relaxed);
    }

    // Remainder rows (n_rows % 4), thread 0 only
    if ith == 0 {
        let base = full_quads * 4;
        for r in 0..(n_rows % 4) {
            let row = base + r;
            for t in 0..n_tokens {
                let q8 = unsafe { batch_q8_qs.add(t * qs_stride) };
                let q8_d = unsafe { batch_q8_d.add(t * n_blocks) };
                let bsums = unsafe { batch_q8_bsums.add(t * n_blocks * 16) };
                let out_ptr = unsafe { output.add(t * output_stride + row) };
                match dtype {
                    GGML_TYPE_Q4_K => {
                        let row_bytes = n_blocks * Q4K_BLOCK_BYTES;
                        unsafe {
                            *out_ptr = ffi_inference::q4k_dot_q8k(
                                weight.add(row * row_bytes), q8, bsums,
                                n_blocks as i32, q8_d, pow2.as_ptr(),
                            );
                        }
                    }
                    GGML_TYPE_Q5_K => {
                        let row_bytes = n_blocks * Q5K_BLOCK_BYTES;
                        unsafe {
                            *out_ptr = ffi_inference::q5k_dot_q8k(
                                weight.add(row * row_bytes), q8, bsums,
                                n_blocks as i32, q8_d, pow2.as_ptr(),
                            );
                        }
                    }
                    GGML_TYPE_Q6_K => {
                        let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
                        unsafe {
                            let d0 = my_scratch;
                            for blk in 0..n_blocks {
                                let w = weight.add(row * row_bytes + blk * Q6K_BLOCK_BYTES + 208);
                                let raw = u16::from_le_bytes([*w, *w.add(1)]);
                                *d0.add(blk) = f16_to_f32_scalar(raw) * *q8_d.add(blk);
                            }
                            *out_ptr = ffi_inference::q6k_dot_q8k(
                                weight.add(row * row_bytes), q8, bsums,
                                n_blocks as i32, d0,
                            );
                        }
                    }
                    _ => panic!("unsupported weight dtype {dtype}"),
                }
            }
        }
    }
}

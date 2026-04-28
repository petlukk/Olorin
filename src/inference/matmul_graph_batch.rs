//! Batch matvec drivers for graph-loop threading (prefill path).
//!
//! Work-stealing across row quads × token chunks, matching llama.cpp's
//! ggml_compute_forward_mul_mat chunking. The 1-row dot kernel is invoked
//! per token. GEMM-style drivers (whole-tile work-stealing, kernel handles
//! all tokens internally) live in `matmul_graph_batch_gemm.rs`.

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

/// Q6K repacked batch matvec with pre-computed d_arr — work-stealing variant
/// intended for the output head (Gemma 4's 262K × 1536 Q6K weight).
///
/// Unlike `q6k_repacked_batch_ws`, this takes a `d_arr_base` pointer into a
/// Vec<f32> pre-computed at model load by `repack::q6k_precompute_d_arr`.
/// That moves the per-(quad, blk, row) `f16_to_f32(d)` conversion out of the
/// hot path — which was previously ~15 ms/decode-step on Gemma 4's output
/// head. Here we just multiply the pre-computed f32 `d` by the token's
/// `q8_d[blk]` and feed the resulting 4-row scale array into the tile kernel.
///
/// Layout contract: `d_arr_base[(quad * n_blocks + blk) * 4 + r]` — contiguous
/// in `r`, so four f32 multiplies share one cache line per block.
#[allow(clippy::too_many_arguments)]
pub fn q6k_repacked_batch_ws_pre_d(
    packed: *const u8, d_arr_base: *const f32,
    batch_q8_qs: *const i8, batch_q8_d: *const f32, batch_q8_bsums: *const i16,
    output: *mut f32, d_scratch: *mut f32,
    n_rows: usize, n_cols: usize, n_tokens: usize, output_stride: usize,
    current_chunk: &AtomicI32, ith: usize, _nth: usize,
) {
    let n_blocks = n_cols / Q6K_BLOCK_SIZE;
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

        // Pre-computed d row for this quad: 4 f32s per block, laid out contiguously.
        let quad_d = unsafe { d_arr_base.add(quad * n_blocks * 4) };

        for t in t_start..t_end {
            let q8 = unsafe { batch_q8_qs.add(t * qs_stride) };
            let q8_d = unsafe { batch_q8_d.add(t * n_blocks) };
            let bsums = unsafe { batch_q8_bsums.add(t * n_blocks * 16) };
            let out_ptr = unsafe { output.add(t * output_stride + base_row) };

            unsafe {
                // Hot inner loop: just 4 f32 multiplies per block, no
                // scattered loads and no f16 conversion.
                for blk in 0..n_blocks {
                    let qd = *q8_d.add(blk);
                    let src = quad_d.add(blk * 4);
                    let dst = my_scratch.add(blk * 4);
                    *dst.add(0) = *src.add(0) * qd;
                    *dst.add(1) = *src.add(1) * qd;
                    *dst.add(2) = *src.add(2) * qd;
                    *dst.add(3) = *src.add(3) * qd;
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
                GGML_TYPE_Q3_K => {
                    // 1-row kernel called 4× per chunk. The naive 4-row wrapper
                    // regressed decode in benching — see q3k_matvec doc.
                    let row_bytes = n_blocks * Q3K_BLOCK_BYTES;
                    unsafe {
                        for r in 0..4 {
                            *out_ptr.add(r) = ffi_inference::q3k_dot_q8k(
                                weight.add((base_row + r) * row_bytes), q8, bsums,
                                n_blocks as i32, q8_d, pow2.as_ptr(),
                            );
                        }
                    }
                }
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
                    GGML_TYPE_Q3_K => {
                        let row_bytes = n_blocks * Q3K_BLOCK_BYTES;
                        unsafe {
                            *out_ptr = ffi_inference::q3k_dot_q8k(
                                weight.add(row * row_bytes), q8, bsums,
                                n_blocks as i32, q8_d, pow2.as_ptr(),
                            );
                        }
                    }
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

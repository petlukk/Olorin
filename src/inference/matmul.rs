//! Matrix-vector multiplication wrappers for Q4K and Q6K Ea kernels.
//!
//! All dot-product compute goes through Ea SIMD kernels via FFI.
//! This module provides safe-ish wrappers that handle block layout,
//! row stride, and 4-row batching.

use crate::kernels::ffi_inference;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

pub const Q4K_BLOCK_SIZE: usize = 256; // elements per Q4K superblock
pub const Q4K_BLOCK_BYTES: usize = 144; // bytes per Q4K superblock
pub const Q5K_BLOCK_SIZE: usize = 256; // elements per Q5K superblock
pub const Q5K_BLOCK_BYTES: usize = 176; // bytes per Q5K superblock
pub const Q6K_BLOCK_SIZE: usize = 256; // elements per Q6K superblock
pub const Q6K_BLOCK_BYTES: usize = 210; // bytes per Q6K superblock

// GGUF dtype codes
pub const GGML_TYPE_Q4_K: u32 = 12;
pub const GGML_TYPE_Q5_K: u32 = 13;
pub const GGML_TYPE_Q6_K: u32 = 14;

/// Byte size of a Q4K tensor's repacked 8x8 representation.
pub fn q4k_packed_size(n_rows: usize, n_cols: usize) -> usize {
    (n_cols / 256) * n_rows * 144 // sizeof(block_q4_K) == 144
}

// ---------------------------------------------------------------------------
// pow2 table for Q4K f16→f32 inline conversion
// ---------------------------------------------------------------------------

/// Precomputed 2^(exp-15) for f16 exponent fields 0..31.
/// The Q4K kernel uses this to convert f16 scale/dmin without F16C.
fn build_pow2_table() -> [f32; 32] {
    let mut t = [0.0f32; 32];
    let mut i = 1u32;
    while i < 32 {
        // 2^(i - 15)
        let bits: u32 = (i.wrapping_add(127 - 15)) << 23;
        t[i as usize] = f32::from_bits(bits);
        i += 1;
    }
    t
}

pub fn pow2_table() -> &'static [f32; 32] {
    use std::sync::OnceLock;
    static TABLE: OnceLock<[f32; 32]> = OnceLock::new();
    TABLE.get_or_init(build_pow2_table)
}

// ---------------------------------------------------------------------------
// f16 scalar helper (for extracting d from Q6K blocks)
// ---------------------------------------------------------------------------

pub fn f16_to_f32_scalar(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;

    if exp == 0 {
        // Subnormal or zero
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        // Subnormal: normalize
        let mut m = mant;
        let mut e = 0i32;
        while m & 0x400 == 0 {
            m <<= 1;
            e -= 1;
        }
        let f32_exp = ((127 - 15 + 1) as i32 + e) as u32;
        let f32_mant = (m & 0x3ff) << 13;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mant);
    }
    if exp == 31 {
        // Inf / NaN
        let f32_mant = mant << 13;
        return f32::from_bits((sign << 31) | (0xff << 23) | f32_mant);
    }
    // Normal
    let f32_exp = (exp as i32 + 127 - 15) as u32;
    let f32_mant = mant << 13;
    f32::from_bits((sign << 31) | (f32_exp << 23) | f32_mant)
}

// ---------------------------------------------------------------------------
// Quantize input: f32 → Q8K
// ---------------------------------------------------------------------------

/// Quantize f32 input vector to Q8K format for use with dot-product kernels.
///
/// - `src`: input f32 values, length `n` (must be multiple of 256)
/// - `dst_qs`: output i8 quantized values, length `n + 12` (padding)
/// - `dst_d`: output per-block scale, length `n / 256`
/// - `dst_bsums`: output per-block sums, length `n / 256 * 16`
pub fn quant_input(
    src: &[f32],
    dst_qs: &mut [i8],
    dst_d: &mut [f32],
    dst_bsums: &mut [i16],
) {
    let n = src.len();
    debug_assert!(n % Q4K_BLOCK_SIZE == 0, "input length must be multiple of 256");
    let n_blocks = n / Q4K_BLOCK_SIZE;
    debug_assert!(dst_qs.len() >= n + 12);
    debug_assert!(dst_d.len() >= n_blocks);
    debug_assert!(dst_bsums.len() >= n_blocks * 16);

    unsafe {
        ffi_inference::quant_f32_q8k(
            src.as_ptr(),
            dst_qs.as_mut_ptr(),
            dst_d.as_mut_ptr(),
            dst_bsums.as_mut_ptr(),
            n as i32,
        );
    }
}

// ---------------------------------------------------------------------------
// Q4K matrix-vector multiply
// ---------------------------------------------------------------------------

/// Q4K matrix-vector: weight (n_rows × n_cols, Q4K) × input (Q8K) → output (f32).
///
/// - `weight`: pointer to packed Q4K weight data (n_rows * row_bytes)
/// - `input_qs`: Q8K quantized input, length n_cols + 12
/// - `input_d`: per-block input scale, length n_blocks
/// - `input_bsums`: per-block input sums, length n_blocks * 16
/// - `output`: result buffer, length >= n_rows
/// - `n_rows`: number of output rows
/// - `n_cols`: number of input columns (must be multiple of 256)
pub fn q4k_matvec(
    weight: *const u8,
    input_qs: &[i8],
    input_d: &[f32],
    input_bsums: &[i16],
    output: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) {
    debug_assert!(n_cols % Q4K_BLOCK_SIZE == 0);
    let n_blocks = n_cols / Q4K_BLOCK_SIZE;
    let row_bytes = n_blocks * Q4K_BLOCK_BYTES;
    let pow2 = pow2_table();

    let q8 = input_qs.as_ptr();
    let bsums = input_bsums.as_ptr();
    let q8_d = input_d.as_ptr();
    let pow2_ptr = pow2.as_ptr();

    // Process 4 rows at a time
    let full_quads = n_rows / 4;
    let remainder = n_rows % 4;

    unsafe {
        for quad in 0..full_quads {
            let base_row = quad * 4;
            let rw0 = weight.add(base_row * row_bytes);
            let rw1 = weight.add((base_row + 1) * row_bytes);
            let rw2 = weight.add((base_row + 2) * row_bytes);
            let rw3 = weight.add((base_row + 3) * row_bytes);

            ffi_inference::q4k_dot_q8k_4row(
                rw0, rw1, rw2, rw3,
                q8, bsums,
                output.as_mut_ptr().add(base_row),
                n_blocks as i32,
                q8_d, pow2_ptr,
            );
        }

        // Remainder rows (1-3)
        for i in 0..remainder {
            let row = full_quads * 4 + i;
            let rw = weight.add(row * row_bytes);
            output[row] = ffi_inference::q4k_dot_q8k(
                rw, q8, bsums,
                n_blocks as i32,
                q8_d, pow2_ptr,
            );
        }
    }
}

/// Q4K dual matrix-vector: gate + up projection in one pass.
///
/// Computes gate_weight × input and up_weight × input simultaneously,
/// sharing Q8K input across both. 4 rows at a time for each pair.
pub fn q4k_matvec_dual(
    gate_weight: *const u8,
    up_weight: *const u8,
    input_qs: &[i8],
    input_d: &[f32],
    input_bsums: &[i16],
    gate_output: &mut [f32],
    up_output: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) {
    debug_assert!(n_cols % Q4K_BLOCK_SIZE == 0);
    let n_blocks = n_cols / Q4K_BLOCK_SIZE;
    let row_bytes = n_blocks * Q4K_BLOCK_BYTES;
    let pow2 = pow2_table();

    let q8 = input_qs.as_ptr();
    let bsums = input_bsums.as_ptr();
    let q8_d = input_d.as_ptr();
    let pow2_ptr = pow2.as_ptr();

    let full_quads = n_rows / 4;
    let remainder = n_rows % 4;

    unsafe {
        for quad in 0..full_quads {
            let base_row = quad * 4;
            let gw0 = gate_weight.add(base_row * row_bytes);
            let gw1 = gate_weight.add((base_row + 1) * row_bytes);
            let gw2 = gate_weight.add((base_row + 2) * row_bytes);
            let gw3 = gate_weight.add((base_row + 3) * row_bytes);
            let uw0 = up_weight.add(base_row * row_bytes);
            let uw1 = up_weight.add((base_row + 1) * row_bytes);
            let uw2 = up_weight.add((base_row + 2) * row_bytes);
            let uw3 = up_weight.add((base_row + 3) * row_bytes);

            ffi_inference::q4k_dot_q8k_4row_dual(
                gw0, gw1, gw2, gw3,
                uw0, uw1, uw2, uw3,
                q8, bsums,
                gate_output.as_mut_ptr().add(base_row),
                up_output.as_mut_ptr().add(base_row),
                n_blocks as i32,
                q8_d, pow2_ptr,
            );
        }

        // Remainder rows — fall back to separate single-row calls
        for i in 0..remainder {
            let row = full_quads * 4 + i;
            let gw = gate_weight.add(row * row_bytes);
            let uw = up_weight.add(row * row_bytes);
            gate_output[row] = ffi_inference::q4k_dot_q8k(
                gw, q8, bsums, n_blocks as i32, q8_d, pow2_ptr,
            );
            up_output[row] = ffi_inference::q4k_dot_q8k(
                uw, q8, bsums, n_blocks as i32, q8_d, pow2_ptr,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Q5K matrix-vector multiply
// ---------------------------------------------------------------------------

/// Q5K matrix-vector: weight (n_rows × n_cols, Q5K) × input (Q8K) → output (f32).
///
/// Same interface as q4k_matvec. Q5K has same scale/mins format as Q4K,
/// just 5 bits per weight instead of 4 (extra bit in qh field).
pub fn q5k_matvec(
    weight: *const u8,
    input_qs: &[i8],
    input_d: &[f32],
    input_bsums: &[i16],
    output: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) {
    debug_assert!(n_cols % Q5K_BLOCK_SIZE == 0);
    let n_blocks = n_cols / Q5K_BLOCK_SIZE;
    let row_bytes = n_blocks * Q5K_BLOCK_BYTES;
    let pow2 = pow2_table();

    let q8 = input_qs.as_ptr();
    let bsums = input_bsums.as_ptr();
    let q8_d = input_d.as_ptr();
    let pow2_ptr = pow2.as_ptr();

    let full_quads = n_rows / 4;
    let remainder = n_rows % 4;

    unsafe {
        for quad in 0..full_quads {
            let base_row = quad * 4;
            let rw0 = weight.add(base_row * row_bytes);
            let rw1 = weight.add((base_row + 1) * row_bytes);
            let rw2 = weight.add((base_row + 2) * row_bytes);
            let rw3 = weight.add((base_row + 3) * row_bytes);

            ffi_inference::q5k_dot_q8k_4row(
                rw0, rw1, rw2, rw3,
                q8, bsums,
                output.as_mut_ptr().add(base_row),
                n_blocks as i32,
                q8_d, pow2_ptr,
            );
        }

        for i in 0..remainder {
            let row = full_quads * 4 + i;
            let rw = weight.add(row * row_bytes);
            output[row] = ffi_inference::q5k_dot_q8k(
                rw, q8, bsums,
                n_blocks as i32,
                q8_d, pow2_ptr,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Q6K matrix-vector multiply
// ---------------------------------------------------------------------------

/// Extract per-block d values from Q6K weight data and pre-multiply with Q8K d.
///
/// Q6K block layout: d (f16) is at offset 208 within each 210-byte block.
/// The kernel expects `d_arr[blk] = d_q6k * d_q8k`.
#[inline]
fn q6k_extract_d(weight: *const u8, n_blocks: usize, q8_d: &[f32], d_arr: &mut [f32]) {
    for blk in 0..n_blocks {
        let block_ptr = unsafe { weight.add(blk * Q6K_BLOCK_BYTES + 208) };
        let raw = unsafe { u16::from_le_bytes([*block_ptr, *block_ptr.add(1)]) };
        d_arr[blk] = f16_to_f32_scalar(raw) * q8_d[blk];
    }
}

/// Q6K matrix-vector: weight (n_rows × n_cols, Q6K) × input (Q8K) → output (f32).
///
/// - `weight`: pointer to packed Q6K weight data
/// - `input_qs`: Q8K quantized input
/// - `input_d`: per-block input scale
/// - `input_bsums`: per-block input sums
/// - `output`: result buffer
/// - `d_scratch`: scratch buffer for pre-multiplied d values, length >= n_blocks * 4
///   (needs 4× for the 4-row variant which needs 4 independent d arrays)
/// - `n_rows`, `n_cols`: dimensions
pub fn q6k_matvec(
    weight: *const u8,
    input_qs: &[i8],
    input_d: &[f32],
    input_bsums: &[i16],
    output: &mut [f32],
    d_scratch: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) {
    debug_assert!(n_cols % Q6K_BLOCK_SIZE == 0);
    let n_blocks = n_cols / Q6K_BLOCK_SIZE;
    let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
    debug_assert!(d_scratch.len() >= n_blocks * 4);

    let q8 = input_qs.as_ptr();
    let bsums = input_bsums.as_ptr();

    let full_quads = n_rows / 4;
    let remainder = n_rows % 4;

    unsafe {
        for quad in 0..full_quads {
            let base_row = quad * 4;
            let w0 = weight.add(base_row * row_bytes);
            let w1 = weight.add((base_row + 1) * row_bytes);
            let w2 = weight.add((base_row + 2) * row_bytes);
            let w3 = weight.add((base_row + 3) * row_bytes);

            // Extract d arrays for each row into scratch
            let (d0, rest) = d_scratch.split_at_mut(n_blocks);
            let (d1, rest) = rest.split_at_mut(n_blocks);
            let (d2, d3) = rest.split_at_mut(n_blocks);
            q6k_extract_d(w0, n_blocks, input_d, d0);
            q6k_extract_d(w1, n_blocks, input_d, d1);
            q6k_extract_d(w2, n_blocks, input_d, d2);
            q6k_extract_d(w3, n_blocks, input_d, d3);

            ffi_inference::q6k_dot_q8k_4row(
                w0, w1, w2, w3,
                q8, bsums,
                output.as_mut_ptr().add(base_row),
                n_blocks as i32,
                d0.as_ptr(), d1.as_ptr(), d2.as_ptr(), d3.as_ptr(),
            );
        }

        // Remainder rows
        let d0 = &mut d_scratch[..n_blocks];
        for i in 0..remainder {
            let row = full_quads * 4 + i;
            let w = weight.add(row * row_bytes);
            q6k_extract_d(w, n_blocks, input_d, d0);
            output[row] = ffi_inference::q6k_dot_q8k(
                w, q8, bsums, n_blocks as i32, d0.as_ptr(),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// BF16 matrix-vector multiply
// ---------------------------------------------------------------------------

/// BF16 matrix-vector: weight (n_rows x n_cols, BF16) x input (f32) -> output (f32).
///
/// BF16 weight data is 2 bytes per element. Row stride = n_cols * 2 bytes.
pub fn bf16_matvec(
    weight: *const u8,
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) {
    let row_bytes = n_cols * 2;
    let mut scratch = [0i32; 8]; // scratch for BF16->f32 reinterpretation

    let full_quads = n_rows / 4;
    let remainder = n_rows % 4;

    unsafe {
        for quad in 0..full_quads {
            let base_row = quad * 4;
            let w0 = weight.add(base_row * row_bytes) as *const u16;
            let w1 = weight.add((base_row + 1) * row_bytes) as *const u16;
            let w2 = weight.add((base_row + 2) * row_bytes) as *const u16;
            let w3 = weight.add((base_row + 3) * row_bytes) as *const u16;

            ffi_inference::bf16_dot_f32_4row(
                w0, w1, w2, w3,
                input.as_ptr(),
                output.as_mut_ptr().add(base_row),
                scratch.as_mut_ptr(),
                n_cols as i32,
            );
        }

        for i in 0..remainder {
            let row = full_quads * 4 + i;
            let w = weight.add(row * row_bytes) as *const u16;
            output[row] = ffi_inference::bf16_dot_f32(
                w, input.as_ptr(), scratch.as_mut_ptr(), n_cols as i32,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Generic dtype-dispatching matvec
// ---------------------------------------------------------------------------

/// Dispatch matvec based on GGML dtype code.
///
/// `d_scratch` is only used for Q6K (needs per-block d pre-multiplication).
/// Pass a slice of length >= n_blocks * 4 for Q6K, or empty for Q4K/Q5K.
pub fn matvec(
    dtype: u32,
    weight: *const u8,
    input_qs: &[i8],
    input_d: &[f32],
    input_bsums: &[i16],
    output: &mut [f32],
    d_scratch: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) {
    match dtype {
        GGML_TYPE_Q4_K => q4k_matvec(weight, input_qs, input_d, input_bsums, output, n_rows, n_cols),
        GGML_TYPE_Q5_K => q5k_matvec(weight, input_qs, input_d, input_bsums, output, n_rows, n_cols),
        GGML_TYPE_Q6_K => q6k_matvec(weight, input_qs, input_d, input_bsums, output, d_scratch, n_rows, n_cols),
        _ => panic!("unsupported weight dtype {dtype}"),
    }
}

/// Row bytes for a given dtype and column count.
#[inline]
pub fn row_bytes_for_dtype(dtype: u32, n_cols: usize) -> usize {
    match dtype {
        GGML_TYPE_Q4_K => q4k_row_bytes(n_cols),
        GGML_TYPE_Q5_K => q5k_row_bytes(n_cols),
        GGML_TYPE_Q6_K => q6k_row_bytes(n_cols),
        _ => panic!("unsupported weight dtype {dtype}"),
    }
}

// ---------------------------------------------------------------------------
// Row byte helpers (for external callers computing offsets)
// ---------------------------------------------------------------------------

/// Bytes per row for Q4K weight matrix with given column count.
#[inline]
pub fn q4k_row_bytes(n_cols: usize) -> usize {
    (n_cols / Q4K_BLOCK_SIZE) * Q4K_BLOCK_BYTES
}

/// Bytes per row for Q5K weight matrix with given column count.
#[inline]
pub fn q5k_row_bytes(n_cols: usize) -> usize {
    (n_cols / Q5K_BLOCK_SIZE) * Q5K_BLOCK_BYTES
}

/// Bytes per row for Q6K weight matrix with given column count.
#[inline]
pub fn q6k_row_bytes(n_cols: usize) -> usize {
    (n_cols / Q6K_BLOCK_SIZE) * Q6K_BLOCK_BYTES
}

// ---------------------------------------------------------------------------
// Parallel matvec — splits quad loop across ThreadPool workers
// ---------------------------------------------------------------------------

use crate::inference::threadpool::ThreadPool;

/// Wrapper for raw pointers to cross thread boundary in pool.run().
/// Safety: caller ensures pointer validity for the lifetime of the pool dispatch.
#[derive(Clone, Copy)]
struct SendPtr<T>(*const T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    #[inline] fn ptr(self) -> *const T { self.0 }
    #[inline] unsafe fn add(self, n: usize) -> *const T { self.0.add(n) }
}

#[derive(Clone, Copy)]
struct SendMutPtr<T>(*mut T);
unsafe impl<T> Send for SendMutPtr<T> {}
unsafe impl<T> Sync for SendMutPtr<T> {}
impl<T> SendMutPtr<T> {
    #[inline] unsafe fn add(self, n: usize) -> *mut T { self.0.add(n) }
}

/// Parallel dtype-dispatching matvec.
pub fn par_matvec(
    pool: &ThreadPool,
    dtype: u32,
    weight: *const u8,
    input_qs: &[i8],
    input_d: &[f32],
    input_bsums: &[i16],
    output: &mut [f32],
    d_scratch: &mut [f32],
    n_rows: usize,
    n_cols: usize,
) {
    let n_threads = pool.thread_count();
    if n_threads <= 1 {
        matvec(dtype, weight, input_qs, input_d, input_bsums, output, d_scratch, n_rows, n_cols);
        return;
    }
    match dtype {
        GGML_TYPE_Q4_K => par_q4k_matvec(pool, weight, input_qs, input_d, input_bsums, output, n_rows, n_cols),
        GGML_TYPE_Q5_K => par_q5k_matvec(pool, weight, input_qs, input_d, input_bsums, output, n_rows, n_cols),
        GGML_TYPE_Q6_K => par_q6k_matvec(pool, weight, input_qs, input_d, input_bsums, output, d_scratch, n_rows, n_cols),
        _ => panic!("unsupported weight dtype {dtype}"),
    }
}

fn par_q4k_matvec(
    pool: &ThreadPool,
    weight: *const u8,
    input_qs: &[i8], input_d: &[f32], input_bsums: &[i16],
    output: &mut [f32],
    n_rows: usize, n_cols: usize,
) {
    let n_blocks = n_cols / Q4K_BLOCK_SIZE;
    let row_bytes = n_blocks * Q4K_BLOCK_BYTES;
    let full_quads = n_rows / 4;
    let n_threads = pool.thread_count().min(full_quads);
    if n_threads <= 1 {
        q4k_matvec(weight, input_qs, input_d, input_bsums, output, n_rows, n_cols);
        return;
    }

    let pow2 = pow2_table();
    let q8 = SendPtr(input_qs.as_ptr());
    let bsums = SendPtr(input_bsums.as_ptr());
    let q8_d = SendPtr(input_d.as_ptr());
    let pow2_ptr = SendPtr(pow2.as_ptr());
    let out_ptr = SendMutPtr(output.as_mut_ptr());
    let w = SendPtr(weight);

    pool.run(n_threads, move |tid, nt| {
        let start_quad = tid * full_quads / nt;
        let end_quad = (tid + 1) * full_quads / nt;
        unsafe {
            for quad in start_quad..end_quad {
                let base_row = quad * 4;
                ffi_inference::q4k_dot_q8k_4row(
                    w.add(base_row * row_bytes),
                    w.add((base_row + 1) * row_bytes),
                    w.add((base_row + 2) * row_bytes),
                    w.add((base_row + 3) * row_bytes),
                    q8.ptr(), bsums.ptr(),
                    out_ptr.add(base_row),
                    n_blocks as i32, q8_d.ptr(), pow2_ptr.ptr(),
                );
            }
        }
    });

    let remainder = n_rows % 4;
    if remainder > 0 {
        let base = full_quads * 4;
        let q8 = input_qs.as_ptr();
        let bsums = input_bsums.as_ptr();
        let q8_d = input_d.as_ptr();
        let pow2_ptr = pow2.as_ptr();
        for i in 0..remainder {
            let row = base + i;
            unsafe {
                output[row] = ffi_inference::q4k_dot_q8k(
                    weight.add(row * row_bytes), q8, bsums,
                    n_blocks as i32, q8_d, pow2_ptr,
                );
            }
        }
    }
}

fn par_q5k_matvec(
    pool: &ThreadPool,
    weight: *const u8,
    input_qs: &[i8], input_d: &[f32], input_bsums: &[i16],
    output: &mut [f32],
    n_rows: usize, n_cols: usize,
) {
    let n_blocks = n_cols / Q5K_BLOCK_SIZE;
    let row_bytes = n_blocks * Q5K_BLOCK_BYTES;
    let full_quads = n_rows / 4;
    let n_threads = pool.thread_count().min(full_quads);
    if n_threads <= 1 {
        q5k_matvec(weight, input_qs, input_d, input_bsums, output, n_rows, n_cols);
        return;
    }

    let pow2 = pow2_table();
    let q8 = SendPtr(input_qs.as_ptr());
    let bsums = SendPtr(input_bsums.as_ptr());
    let q8_d = SendPtr(input_d.as_ptr());
    let pow2_ptr = SendPtr(pow2.as_ptr());
    let out_ptr = SendMutPtr(output.as_mut_ptr());
    let w = SendPtr(weight);

    pool.run(n_threads, move |tid, nt| {
        let start_quad = tid * full_quads / nt;
        let end_quad = (tid + 1) * full_quads / nt;
        unsafe {
            for quad in start_quad..end_quad {
                let base_row = quad * 4;
                ffi_inference::q5k_dot_q8k_4row(
                    w.add(base_row * row_bytes),
                    w.add((base_row + 1) * row_bytes),
                    w.add((base_row + 2) * row_bytes),
                    w.add((base_row + 3) * row_bytes),
                    q8.ptr(), bsums.ptr(),
                    out_ptr.add(base_row),
                    n_blocks as i32, q8_d.ptr(), pow2_ptr.ptr(),
                );
            }
        }
    });

    let remainder = n_rows % 4;
    if remainder > 0 {
        let base = full_quads * 4;
        let q8 = input_qs.as_ptr();
        let bsums = input_bsums.as_ptr();
        let q8_d = input_d.as_ptr();
        let pow2_ptr = pow2.as_ptr();
        for i in 0..remainder {
            let row = base + i;
            unsafe {
                output[row] = ffi_inference::q5k_dot_q8k(
                    weight.add(row * row_bytes), q8, bsums,
                    n_blocks as i32, q8_d, pow2_ptr,
                );
            }
        }
    }
}

fn par_q6k_matvec(
    pool: &ThreadPool,
    weight: *const u8,
    input_qs: &[i8], input_d: &[f32], input_bsums: &[i16],
    output: &mut [f32],
    _d_scratch: &mut [f32],
    n_rows: usize, n_cols: usize,
) {
    let n_blocks = n_cols / Q6K_BLOCK_SIZE;
    let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
    let full_quads = n_rows / 4;
    let n_threads = pool.thread_count().min(full_quads);
    if n_threads <= 1 {
        q6k_matvec(weight, input_qs, input_d, input_bsums, output, _d_scratch, n_rows, n_cols);
        return;
    }

    let q8 = SendPtr(input_qs.as_ptr());
    let bsums = SendPtr(input_bsums.as_ptr());
    let out_ptr = SendMutPtr(output.as_mut_ptr());
    let w = SendPtr(weight);
    let id = SendPtr(input_d.as_ptr());

    pool.run(n_threads, move |tid, nt| {
        let mut d_local = vec![0.0f32; n_blocks * 4];
        let start_quad = tid * full_quads / nt;
        let end_quad = (tid + 1) * full_quads / nt;
        // Rebuild input_d slice from pointer for q6k_extract_d
        let input_d_slice = unsafe { std::slice::from_raw_parts(id.ptr(), n_blocks) };
        unsafe {
            for quad in start_quad..end_quad {
                let base_row = quad * 4;
                let w0 = w.add(base_row * row_bytes);
                let w1 = w.add((base_row + 1) * row_bytes);
                let w2 = w.add((base_row + 2) * row_bytes);
                let w3 = w.add((base_row + 3) * row_bytes);
                let (d0, rest) = d_local.split_at_mut(n_blocks);
                let (d1, rest) = rest.split_at_mut(n_blocks);
                let (d2, d3) = rest.split_at_mut(n_blocks);
                q6k_extract_d(w0, n_blocks, input_d_slice, d0);
                q6k_extract_d(w1, n_blocks, input_d_slice, d1);
                q6k_extract_d(w2, n_blocks, input_d_slice, d2);
                q6k_extract_d(w3, n_blocks, input_d_slice, d3);
                ffi_inference::q6k_dot_q8k_4row(
                    w0, w1, w2, w3,
                    q8.ptr(), bsums.ptr(),
                    out_ptr.add(base_row),
                    n_blocks as i32,
                    d0.as_ptr(), d1.as_ptr(), d2.as_ptr(), d3.as_ptr(),
                );
            }
        }
    });

    let remainder = n_rows % 4;
    if remainder > 0 {
        let base = full_quads * 4;
        let d0 = &mut _d_scratch[..n_blocks];
        let q8 = input_qs.as_ptr();
        let bsums = input_bsums.as_ptr();
        for i in 0..remainder {
            let row = base + i;
            unsafe {
                let w = weight.add(row * row_bytes);
                q6k_extract_d(w, n_blocks, input_d, d0);
                output[row] = ffi_inference::q6k_dot_q8k(
                    w, q8, bsums, n_blocks as i32, d0.as_ptr(),
                );
            }
        }
    }
}

/// Parallel Q4K dual gate+up matvec.
pub fn par_q4k_matvec_dual(
    pool: &ThreadPool,
    gate_weight: *const u8,
    up_weight: *const u8,
    input_qs: &[i8], input_d: &[f32], input_bsums: &[i16],
    gate_output: &mut [f32],
    up_output: &mut [f32],
    n_rows: usize, n_cols: usize,
) {
    let n_blocks = n_cols / Q4K_BLOCK_SIZE;
    let row_bytes = n_blocks * Q4K_BLOCK_BYTES;
    let full_quads = n_rows / 4;
    let n_threads = pool.thread_count().min(full_quads);
    if n_threads <= 1 {
        q4k_matvec_dual(gate_weight, up_weight, input_qs, input_d, input_bsums,
                        gate_output, up_output, n_rows, n_cols);
        return;
    }

    let pow2 = pow2_table();
    let q8 = SendPtr(input_qs.as_ptr());
    let bsums = SendPtr(input_bsums.as_ptr());
    let q8_d = SendPtr(input_d.as_ptr());
    let pow2_ptr = SendPtr(pow2.as_ptr());
    let gate_ptr = SendMutPtr(gate_output.as_mut_ptr());
    let up_ptr = SendMutPtr(up_output.as_mut_ptr());
    let gw = SendPtr(gate_weight);
    let uw = SendPtr(up_weight);

    pool.run(n_threads, move |tid, nt| {
        let start_quad = tid * full_quads / nt;
        let end_quad = (tid + 1) * full_quads / nt;
        unsafe {
            for quad in start_quad..end_quad {
                let base_row = quad * 4;
                ffi_inference::q4k_dot_q8k_4row_dual(
                    gw.add(base_row * row_bytes),
                    gw.add((base_row + 1) * row_bytes),
                    gw.add((base_row + 2) * row_bytes),
                    gw.add((base_row + 3) * row_bytes),
                    uw.add(base_row * row_bytes),
                    uw.add((base_row + 1) * row_bytes),
                    uw.add((base_row + 2) * row_bytes),
                    uw.add((base_row + 3) * row_bytes),
                    q8.ptr(), bsums.ptr(),
                    gate_ptr.add(base_row),
                    up_ptr.add(base_row),
                    n_blocks as i32, q8_d.ptr(), pow2_ptr.ptr(),
                );
            }
        }
    });

    let remainder = n_rows % 4;
    if remainder > 0 {
        let base = full_quads * 4;
        let q8 = input_qs.as_ptr();
        let bsums = input_bsums.as_ptr();
        let q8_d = input_d.as_ptr();
        let pow2_ptr = pow2.as_ptr();
        for i in 0..remainder {
            let row = base + i;
            unsafe {
                gate_output[row] = ffi_inference::q4k_dot_q8k(
                    gate_weight.add(row * row_bytes), q8, bsums, n_blocks as i32, q8_d, pow2_ptr,
                );
                up_output[row] = ffi_inference::q4k_dot_q8k(
                    up_weight.add(row * row_bytes), q8, bsums, n_blocks as i32, q8_d, pow2_ptr,
                );
            }
        }
    }
}

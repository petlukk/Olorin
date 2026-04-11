//! Matrix-vector multiplication dispatchers and shared helpers.
//!
//! Actual dtype kernels live in sibling modules:
//! - `matmul_seq.rs` — single-threaded Q4K/Q5K/Q6K/BF16 kernels
//! - `matmul_par.rs` — ThreadPool-based parallel wrappers
//! - `matmul_graph.rs` — graph-threaded work-stealing variant
//!
//! All dot-product compute goes through Ea SIMD kernels via FFI. This
//! module holds the shared constants, dtype-dispatch entry points
//! (`matvec`, `par_matvec`), and numeric helpers (`pow2_table`,
//! `f16_to_f32_scalar`, `quant_input`) used across the other variants.

use crate::kernels::ffi_inference;
use crate::inference::threadpool::ThreadPool;

// Re-export kernel functions so external callers keep the existing paths
// (`olorin::inference::matmul::q4k_matvec`, etc.). Nothing outside this
// module should care which file the implementation actually lives in.
pub use super::matmul_seq::{
    q4k_matvec, q4k_matvec_dual,
    q5k_matvec, q6k_matvec,
    bf16_matvec,
};
pub use super::matmul_par::par_q4k_matvec_dual;

// Private imports for the dispatchers below.
use super::matmul_par::{par_q4k_matvec, par_q5k_matvec, par_q6k_matvec};

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
// f16 scalar helper (for extracting d from Q6K blocks, and matmul_graph)
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

// ---------------------------------------------------------------------------
// Parallel dtype-dispatching matvec — splits quad loop across ThreadPool workers
// ---------------------------------------------------------------------------

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

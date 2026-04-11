//! GGUF tensor helper functions for model loading.

use crate::inference::gguf::GgufFile;

/// Convert BF16 tensor data to a new Vec<f32>. Returns the vec.
pub(crate) fn bf16_to_f32_vec(data: &[u8]) -> Vec<f32> {
    let n = data.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let bits = u16::from_le_bytes([data[i * 2], data[i * 2 + 1]]);
        out.push(f32::from_bits((bits as u32) << 16));
    }
    out
}

/// Load a norm weight pointer. If the tensor is BF16, convert to f32 and
/// store the buffer in `bufs` (keeping it alive). Returns null if missing.
pub(crate) fn load_norm_ptr(
    gguf: &GgufFile,
    name: &str,
    bufs: &mut Vec<Vec<f32>>,
) -> *const f32 {
    let idx = match gguf.tensor_map.get(name) {
        Some(&i) => i,
        None => return std::ptr::null(),
    };
    let ti = &gguf.tensors[idx];
    match ti.dtype {
        0 => {
            // F32 — point directly into mmap
            gguf.tensor_data(name)
                .map(|d| d.as_ptr() as *const f32)
                .unwrap_or(std::ptr::null())
        }
        30 => {
            // BF16 — convert to owned Vec<f32>
            match gguf.tensor_data(name) {
                Some(data) => {
                    let converted = bf16_to_f32_vec(data);
                    let ptr = converted.as_ptr();
                    bufs.push(converted);
                    ptr
                }
                None => std::ptr::null(),
            }
        }
        _ => {
            eprintln!("[gemma4] warning: {name} has unexpected dtype {}, treating as f32", ti.dtype);
            gguf.tensor_data(name)
                .map(|d| d.as_ptr() as *const f32)
                .unwrap_or(std::ptr::null())
        }
    }
}

// ---------------------------------------------------------------------------
// Phase B.1: Q4K 8x8 repack gating
// ---------------------------------------------------------------------------

/// Runtime check: is the `q4k_8x8_q8k_matvec` kernel supported on this CPU?
///
/// The repacked kernel uses architecture-specific SIMD intrinsics that
/// require either AVX2 (x86_64) or NEON dotprod (aarch64). On any other
/// architecture, or on a CPU that lacks those features, we must not
/// dispatch to the repacked path.
#[inline]
pub(crate) fn q4k_8x8_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    { return std::is_x86_feature_detected!("avx2"); }
    #[cfg(target_arch = "aarch64")]
    { return std::arch::is_aarch64_feature_detected!("dotprod"); }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    { return false; }
}

/// Attempt to repack a Q4K weight matrix into the `block_q4_Kx8` layout.
/// Returns `None` if the weight is not eligible:
/// - `dtype` is not Q4K
/// - `n_rows` is not a multiple of 8
/// - `n_cols` is not a multiple of 256 (Q4K superblock size)
/// - the CPU does not support the required SIMD features
///
/// # Safety
/// The caller asserts `weight` is valid for `n_rows * (n_cols / 256) * 144`
/// readable bytes, and that `olorin::kernels::ffi::init()` has been called.
pub(crate) fn try_repack_q4k(
    weight: *const u8,
    dtype: u32,
    n_rows: usize,
    n_cols: usize,
) -> Option<Vec<u8>> {
    if dtype != crate::inference::matmul::GGML_TYPE_Q4_K { return None; }
    if n_rows % 8 != 0 { return None; }
    if n_cols % 256 != 0 { return None; }
    if !q4k_8x8_supported() { return None; }
    Some(crate::inference::repack::q4k_repack_8x8(weight, n_rows, n_cols))
}

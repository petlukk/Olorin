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

/// Populate all 7 Q4K _repacked fields on a LayerWeights instance.
///
/// Called by `Gemma4Model::from_gguf` once per layer, after the layer has
/// been constructed. Uses `try_repack_q4k` to gate per-weight; any weight
/// that fails the gate (e.g., non-Q4K dtype, odd row count, CPU lacks
/// required features) stays `None` and the forward path falls through to
/// the existing 4-row kernel for that weight.
///
/// # Shape parameters (from Gemma 4 layer metadata)
/// - `n_heads`, `n_kv_heads`: global attention head counts
/// - `head_dim_k`, `head_dim_v`: per-layer K/V head dims (SWA vs global)
/// - `hidden_dim`: global model width
/// - `ffn_dim`: per-layer FFN width
///
/// # Safety
/// Same contract as `try_repack_q4k` for each weight pointer.
pub(crate) fn populate_q4k_repacked(
    lw: &mut crate::inference::engine::LayerWeights,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim_k: usize,
    head_dim_v: usize,
    hidden_dim: usize,
    ffn_dim: usize,
    ple_dim: usize,
) {
    // Attention projections. Weight shape = [n_cols, n_rows] in GGUF, but
    // matmul call sites pass n_rows as "output dim" — match those shapes.
    lw.wq_repacked = try_repack_q4k(lw.wq, lw.wq_dtype, n_heads * head_dim_k, hidden_dim);
    lw.wk_repacked = try_repack_q4k(lw.wk, lw.wk_dtype, n_kv_heads * head_dim_k, hidden_dim);
    lw.wv_repacked = try_repack_q4k(lw.wv, lw.wv_dtype, n_kv_heads * head_dim_v, hidden_dim);
    lw.wo_repacked = try_repack_q4k(lw.wo, lw.wo_dtype, hidden_dim, n_heads * head_dim_k);
    // FFN projections
    lw.w_gate_repacked = try_repack_q4k(lw.w_gate, lw.w_gate_dtype, ffn_dim, hidden_dim);
    lw.w_up_repacked = try_repack_q4k(lw.w_up, lw.w_up_dtype, ffn_dim, hidden_dim);
    lw.w_down_repacked = try_repack_q4k(lw.w_down, lw.w_down_dtype, hidden_dim, ffn_dim);
    // PLE projections
    if ple_dim > 0 {
        lw.inp_gate_repacked = try_repack_q4k(lw.inp_gate, lw.inp_gate_dtype, ple_dim, hidden_dim);
        lw.proj_repacked = try_repack_q4k(lw.proj, lw.proj_dtype, hidden_dim, ple_dim);
    }
}

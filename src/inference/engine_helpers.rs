//! GGUF tensor helpers + per-layer Q4K repack.

use crate::inference::gguf::GgufFile;
use crate::inference::engine::LayerWeights;
use crate::inference::matmul::{self, q4k_packed_size};
use crate::kernels::ffi_inference;

/// Repack one layer's Q4K weights into `buf`. Returns (offsets, sizes)
/// for each of the 7 weights: [wq, wk, wv, wo, gate, up, down].
/// Non-Q4K weights get (0, 0) → the caller sees an empty slice and
/// falls back to per-column matvec.
#[allow(clippy::too_many_arguments)]
pub(crate) fn repack_layer(
    buf: &mut [u8],
    lw: &LayerWeights,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    head_dim_v: usize,
    hd: usize,
    ffn_dim: usize,
) -> ([usize; 7], [usize; 7]) {
    let q4k = matmul::GGML_TYPE_Q4_K;
    let specs: [(u32, *const u8, usize, usize); 7] = [
        (lw.wq_dtype,     lw.wq,     n_heads * head_dim,     hd),
        (lw.wk_dtype,     lw.wk,     n_kv_heads * head_dim,  hd),
        (lw.wv_dtype,     lw.wv,     n_kv_heads * head_dim_v, hd),
        (lw.wo_dtype,     lw.wo,     hd,                     n_heads * head_dim),
        (lw.w_gate_dtype, lw.w_gate, ffn_dim,                hd),
        (lw.w_up_dtype,   lw.w_up,   ffn_dim,                hd),
        (lw.w_down_dtype, lw.w_down, hd,                     ffn_dim),
    ];
    let mut offsets = [0usize; 7];
    let mut sizes = [0usize; 7];
    let mut cursor = 0usize;
    for (i, &(dtype, src, n_rows, n_cols)) in specs.iter().enumerate() {
        if dtype == q4k {
            let sz = q4k_packed_size(n_rows, n_cols);
            offsets[i] = cursor;
            sizes[i] = sz;
            unsafe {
                ffi_inference::q4k_repack_8x8(
                    src, buf[cursor..].as_mut_ptr(), n_rows as i32, n_cols as i32,
                );
            }
            cursor += sz;
        }
    }
    (offsets, sizes)
}

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

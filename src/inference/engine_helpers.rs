//! GGUF tensor helpers + Q4K weight repack.

use crate::inference::gguf::GgufFile;
use crate::inference::engine::LayerWeights;
use crate::inference::matmul::{q4k_packed_size, GGML_TYPE_Q4_K};

/// Repack all Q4K weights into one contiguous buffer (llama.cpp style).
/// Sets each layer's w*_packed pointers into the returned Vec.
#[allow(clippy::too_many_arguments)]
pub(crate) fn repack_all_q4k(
    layers: &mut [LayerWeights],
    n_heads: usize, n_kv_heads: usize,
    head_dim_k: &[usize], head_dim_v: &[usize],
    hd: usize, ffn_dim: &[usize],
) -> Vec<u8> {
    // Pass 1: total packed size
    let mut total = 0usize;
    for (i, lw) in layers.iter().enumerate() {
        for &(dt, nr, nc) in &weight_specs_size(lw, n_heads, n_kv_heads, head_dim_k[i], head_dim_v[i], hd, ffn_dim[i]) {
            if dt == GGML_TYPE_Q4_K { total += q4k_packed_size(nr, nc); }
        }
    }
    let mut buf = vec![0u8; total];
    let mut cursor = 0usize;
    let t = std::time::Instant::now();
    // Pass 2: repack each Q4K tensor
    for (i, lw) in layers.iter_mut().enumerate() {
        let specs = weight_specs_src(lw, n_heads, n_kv_heads, head_dim_k[i], head_dim_v[i], hd, ffn_dim[i]);
        let ptrs: [&mut *const u8; 7] = [
            &mut lw.wq_packed, &mut lw.wk_packed, &mut lw.wv_packed,
            &mut lw.wo_packed, &mut lw.w_gate_packed, &mut lw.w_up_packed,
            &mut lw.w_down_packed,
        ];
        for (j, &(dt, src, nr, nc)) in specs.iter().enumerate() {
            if dt == GGML_TYPE_Q4_K {
                let sz = q4k_packed_size(nr, nc);
                unsafe {
                    crate::kernels::ffi_inference::q4k_repack_8x8(
                        src, buf[cursor..].as_mut_ptr(), nr as i32, nc as i32,
                    );
                }
                *ptrs[j] = buf[cursor..].as_ptr();
                cursor += sz;
            }
        }
    }
    eprintln!("[gemma4] Q4K repack: {} MB in {:?}", total / 1_000_000, t.elapsed());
    buf
}

fn weight_specs_size(lw: &LayerWeights, nh: usize, nkv: usize, hdk: usize, hdv: usize, hd: usize, ffn: usize) -> [(u32, usize, usize); 7] {
    [(lw.wq_dtype, nh*hdk, hd), (lw.wk_dtype, nkv*hdk, hd), (lw.wv_dtype, nkv*hdv, hd),
     (lw.wo_dtype, hd, nh*hdk), (lw.w_gate_dtype, ffn, hd), (lw.w_up_dtype, ffn, hd), (lw.w_down_dtype, hd, ffn)]
}

fn weight_specs_src(lw: &LayerWeights, nh: usize, nkv: usize, hdk: usize, hdv: usize, hd: usize, ffn: usize) -> [(u32, *const u8, usize, usize); 7] {
    [(lw.wq_dtype, lw.wq, nh*hdk, hd), (lw.wk_dtype, lw.wk, nkv*hdk, hd), (lw.wv_dtype, lw.wv, nkv*hdv, hd),
     (lw.wo_dtype, lw.wo, hd, nh*hdk), (lw.w_gate_dtype, lw.w_gate, ffn, hd), (lw.w_up_dtype, lw.w_up, ffn, hd),
     (lw.w_down_dtype, lw.w_down, hd, ffn)]
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

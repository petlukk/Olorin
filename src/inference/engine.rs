//! Gemma 4 E2B model structure and weight loading from GGUF.
//!
//! Per-layer varying dimensions: SWA layers have head_dim 256, ffn 6144;
//! global layers have head_dim 512, ffn 12288. KV sharing across last N layers.

use crate::inference::gguf::{GgufFile, MetaValue};
use crate::inference::engine_helpers::{
    self, load_norm_ptr,
    get_meta_u32, get_meta_f32, get_meta_u32_array,
    tensor_ptr, tensor_dtype, tensor_ptr_opt, read_f32_scalar,
    compute_kv_shared,
};

// Attention type per layer

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttnType {
    SlidingWindow,
    Global,
}

// Per-layer weight pointers (raw into mmap)

pub struct LayerWeights {
    // Pre-attention RMSNorm
    pub attn_norm: *const f32,        // [hidden_dim]

    // Attention projections (Q4K quantized)
    pub wq: *const u8,                // [hidden_dim, n_heads * head_dim_k]
    pub wk: *const u8,                // [hidden_dim, n_kv_heads * head_dim_k]
    pub wv: *const u8,                // [hidden_dim, n_kv_heads * head_dim_v]
    pub wo: *const u8,                // [n_heads * head_dim_k, hidden_dim]

    // Per-tensor dtypes (GGML type codes)
    pub wq_dtype: u32,
    pub wk_dtype: u32,
    pub wv_dtype: u32,
    pub wo_dtype: u32,
    pub w_gate_dtype: u32,
    pub w_up_dtype: u32,
    pub w_down_dtype: u32,

    // QK norm (per-head, BF16->f32 converted at load)
    pub q_norm: *const f32,           // [head_dim_k]
    pub k_norm: *const f32,           // [head_dim_k]

    // Post-attention RMSNorm
    pub post_attn_norm: *const f32,   // [hidden_dim]

    // Pre-FFN RMSNorm
    pub ffn_norm: *const f32,         // [hidden_dim]

    // FFN (GeGLU, Q4K quantized)
    pub w_gate: *const u8,            // [hidden_dim, ffn_dim]
    pub w_up: *const u8,              // [hidden_dim, ffn_dim]
    pub w_down: *const u8,            // [ffn_dim, hidden_dim]

    // Post-FFN RMSNorm
    pub post_ffn_norm: *const f32,    // [hidden_dim]

    // PLE per-layer tensors
    pub inp_gate: *const u8,          // [hidden_dim, ple_dim]
    pub proj: *const u8,              // [ple_dim, hidden_dim]
    pub inp_gate_dtype: u32,
    pub proj_dtype: u32,
    pub post_norm: *const f32,        // [hidden_dim]
    pub layer_output_scale: f32,      // scalar

    // Phase B.1: Q4K 8x8 repacked buffers. `Some` if the weight was eligible
    // for the q4k_8x8_q8k_matvec fast path (Q4K dtype, rows%8==0, CPU supports
    // AVX2 or NEON dotprod); `None` means fall through to the 4-row path.
    pub wq_repacked: Option<Vec<u8>>,
    pub wk_repacked: Option<Vec<u8>>,
    pub wv_repacked: Option<Vec<u8>>,
    pub wo_repacked: Option<Vec<u8>>,
    pub w_gate_repacked: Option<Vec<u8>>,
    pub w_up_repacked: Option<Vec<u8>>,
    pub w_down_repacked: Option<Vec<u8>>,
    pub w_down_q6k_repacked: Option<Vec<u8>>,
    pub inp_gate_repacked: Option<Vec<u8>>,
    pub proj_repacked: Option<Vec<u8>>,
}

impl std::fmt::Debug for LayerWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerWeights").finish()
    }
}

// Gemma4Model

pub struct Gemma4Model {
    pub n_layers: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab_size: usize,
    pub rms_eps: f32,
    pub sliding_window: usize,
    pub logit_softcap: f32,
    pub ple_dim: usize,

    // Per-layer varying dimensions
    pub head_dim_k: Vec<usize>,       // per-layer: 256 (SWA) or 512 (global)
    pub head_dim_v: Vec<usize>,       // per-layer: 256 or 512
    pub ffn_dim: Vec<usize>,          // per-layer: 6144 or 12288
    pub is_swa: Vec<bool>,            // per-layer: true = sliding window
    pub kv_shared_source: Vec<Option<usize>>,

    // RoPE (dual frequencies)
    pub rope_theta_swa: f32,          // 10000.0
    pub rope_theta_global: f32,       // 1000000.0
    pub rope_dim_swa: usize,          // 256
    pub rope_dim_global: usize,       // 512
    pub rope_freqs: Option<Vec<f32>>, // global RoPE freq factors [rope_dim_global/2]

    // Layers
    pub layers: Vec<LayerWeights>,

    // Global tensors
    pub embed_weight: *const u8,
    pub embed_dtype: u32,
    pub embed_q6k_repacked: Option<Vec<u8>>,
    pub norm_weight: *const f32,

    // PLE global tensors
    pub ple_token_embd: *const u8,    // [ple_dim * n_layers, vocab_size] Q6K
    pub ple_model_proj: *const u8,    // [hidden_dim, ple_dim * n_layers] BF16
    pub ple_proj_norm: *const f32,    // [ple_dim]

    _bf16_bufs: Vec<Vec<f32>>,        // BF16→f32 conversion buffers (kept alive)
}

unsafe impl Send for Gemma4Model {}
unsafe impl Sync for Gemma4Model {}

// Model loading

impl Gemma4Model {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        // Phase B.1: layer loop needs SIMD kernels (for q4k_repack_8x8). Idempotent.
        crate::kernels::ffi::init().map_err(|e| format!("ffi init: {e}"))?;

        // 1. Architecture name
        let arch = gguf.get_str("general.architecture").unwrap_or("gemma4");
        eprintln!("[gemma4] architecture: {arch}");

        // 2. Core metadata
        let n_layers = get_meta_u32(gguf, &format!("{arch}.block_count"))
            .ok_or("missing block_count")? as usize;
        let hidden_dim = get_meta_u32(gguf, &format!("{arch}.embedding_length"))
            .ok_or("missing embedding_length")? as usize;
        let n_heads = get_meta_u32(gguf, &format!("{arch}.attention.head_count"))
            .ok_or("missing head_count")? as usize;
        let n_kv_heads = get_meta_u32(gguf, &format!("{arch}.attention.head_count_kv"))
            .ok_or("missing head_count_kv")? as usize;

        let rms_eps = get_meta_f32(gguf, &format!("{arch}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6);
        let sliding_window = get_meta_u32(gguf, &format!("{arch}.attention.sliding_window"))
            .map(|v| v as usize)
            .unwrap_or(512);
        let logit_softcap = get_meta_f32(gguf, &format!("{arch}.final_logit_softcapping"))
            .unwrap_or(0.0);

        // Per-layer head dimensions
        let key_len_global = get_meta_u32(gguf, &format!("{arch}.attention.key_length"))
            .unwrap_or(512) as usize;
        let key_len_swa = get_meta_u32(gguf, &format!("{arch}.attention.key_length_swa"))
            .unwrap_or(256) as usize;
        let val_len_global = get_meta_u32(gguf, &format!("{arch}.attention.value_length"))
            .unwrap_or(512) as usize;
        let val_len_swa = get_meta_u32(gguf, &format!("{arch}.attention.value_length_swa"))
            .unwrap_or(256) as usize;

        // Shared KV suffix length
        let shared_suffix = get_meta_u32(gguf, &format!("{arch}.attention.shared_kv_layers"))
            .map(|v| v as usize)
            .unwrap_or(0);

        // PLE dimension
        let ple_dim = get_meta_u32(gguf, &format!("{arch}.embedding_length_per_layer_input"))
            .map(|v| v as usize)
            .unwrap_or(0);

        // RoPE dual frequencies
        let rope_theta_global = get_meta_f32(gguf, &format!("{arch}.rope.freq_base"))
            .unwrap_or(1000000.0);
        let rope_theta_swa = get_meta_f32(gguf, &format!("{arch}.rope.freq_base_swa"))
            .unwrap_or(10000.0);
        let rope_dim_global = get_meta_u32(gguf, &format!("{arch}.rope.dimension_count"))
            .unwrap_or(key_len_global as u32) as usize;
        let rope_dim_swa = get_meta_u32(gguf, &format!("{arch}.rope.dimension_count_swa"))
            .unwrap_or(key_len_swa as u32) as usize;

        // 3. Sliding window pattern (per-layer array: 1=SWA, 0=global)
        let swp_key = format!("{arch}.attention.sliding_window_pattern");
        // Debug: check what metadata type the key has
        if let Some(mv) = gguf.metadata.get(&swp_key) {
            eprintln!("[gemma4] swp metadata type: {:?}", std::mem::discriminant(mv));
            if let MetaValue::Array(arr) = mv {
                eprintln!("[gemma4] swp array len={}, first elem type: {:?}",
                    arr.len(), arr.first().map(|v| std::mem::discriminant(v)));
            }
        } else {
            eprintln!("[gemma4] swp key not found in metadata!");
        }
        let is_swa: Vec<bool> = match get_meta_u32_array(gguf, &swp_key) {
            Some(pattern) => {
                if pattern.len() != n_layers {
                    return Err(format!(
                        "sliding_window_pattern has {} items, expected {n_layers}",
                        pattern.len()
                    ));
                }
                eprintln!("[gemma4] sliding_window_pattern loaded: first 5 = {:?}", &pattern[..5.min(pattern.len())]);
                pattern.iter().map(|&v| v == 1).collect()
            }
            None => {
                eprintln!("[gemma4] WARNING: sliding_window_pattern not found, using fallback");
                // Fallback: compute from global_layer_interval
                let interval = get_meta_u32(
                    gguf,
                    &format!("{arch}.attention.global_layer_interval"),
                )
                .map(|v| v as usize)
                .unwrap_or(5);
                (0..n_layers)
                    .map(|i| {
                        !(i == n_layers - 1
                            || (interval > 0 && i % interval == 0))
                    })
                    .collect()
            }
        };

        // 4. Per-layer FFN dimensions (array or scalar)
        let ffn_dim: Vec<usize> = match get_meta_u32_array(
            gguf,
            &format!("{arch}.feed_forward_length"),
        ) {
            Some(arr) if arr.len() == n_layers => {
                arr.iter().map(|&v| v as usize).collect()
            }
            Some(arr) if arr.len() == 1 => {
                vec![arr[0] as usize; n_layers]
            }
            _ => {
                // Scalar fallback
                let single = get_meta_u32(gguf, &format!("{arch}.feed_forward_length"))
                    .unwrap_or(6144) as usize;
                vec![single; n_layers]
            }
        };

        // 5. Per-layer head dims derived from SWA/global type
        let head_dim_k: Vec<usize> = is_swa
            .iter()
            .map(|&swa| if swa { key_len_swa } else { key_len_global })
            .collect();
        let head_dim_v: Vec<usize> = is_swa
            .iter()
            .map(|&swa| if swa { val_len_swa } else { val_len_global })
            .collect();

        // 6. KV shared source
        let kv_shared_source = compute_kv_shared(n_layers, shared_suffix, &is_swa);

        // Vocab size from embedding tensor dims: [hidden_dim, vocab_size]
        let vocab_size = gguf
            .tensor_map
            .get("token_embd.weight")
            .and_then(|&idx| gguf.tensors.get(idx))
            .and_then(|ti| ti.dims.get(1).copied())
            .ok_or("cannot determine vocab_size from token_embd.weight")?
            as usize;

        // 7. RoPE frequency factors (optional global tensor)
        let rope_freqs = gguf.tensor_data("rope_freqs.weight").map(|data| {
            // F32 tensor
            let n = data.len() / 4;
            let mut v = Vec::with_capacity(n);
            for i in 0..n {
                let off = i * 4;
                let bits = u32::from_le_bytes([
                    data[off], data[off + 1], data[off + 2], data[off + 3],
                ]);
                v.push(f32::from_bits(bits));
            }
            v
        });

        // 8. Per-layer weight pointers (BF16 norms converted to owned f32 bufs)
        let mut bf16_bufs: Vec<Vec<f32>> = Vec::new();
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let b = format!("blk.{i}");
            let lw = LayerWeights {
                attn_norm: tensor_ptr::<f32>(gguf, &format!("{b}.attn_norm.weight"))?,
                wq: tensor_ptr::<u8>(gguf, &format!("{b}.attn_q.weight"))?,
                wk: tensor_ptr::<u8>(gguf, &format!("{b}.attn_k.weight"))?,
                wv: tensor_ptr::<u8>(gguf, &format!("{b}.attn_v.weight"))?,
                wo: tensor_ptr::<u8>(gguf, &format!("{b}.attn_output.weight"))?,
                wq_dtype: tensor_dtype(gguf, &format!("{b}.attn_q.weight")),
                wk_dtype: tensor_dtype(gguf, &format!("{b}.attn_k.weight")),
                wv_dtype: tensor_dtype(gguf, &format!("{b}.attn_v.weight")),
                wo_dtype: tensor_dtype(gguf, &format!("{b}.attn_output.weight")),
                w_gate_dtype: tensor_dtype(gguf, &format!("{b}.ffn_gate.weight")),
                w_up_dtype: tensor_dtype(gguf, &format!("{b}.ffn_up.weight")),
                w_down_dtype: tensor_dtype(gguf, &format!("{b}.ffn_down.weight")),
                q_norm: load_norm_ptr(gguf, &format!("{b}.attn_q_norm.weight"), &mut bf16_bufs),
                k_norm: load_norm_ptr(gguf, &format!("{b}.attn_k_norm.weight"), &mut bf16_bufs),
                post_attn_norm: load_norm_ptr(gguf, &format!("{b}.post_attention_norm.weight"), &mut bf16_bufs),
                ffn_norm: tensor_ptr::<f32>(gguf, &format!("{b}.ffn_norm.weight"))?,
                w_gate: tensor_ptr::<u8>(gguf, &format!("{b}.ffn_gate.weight"))?,
                w_up: tensor_ptr::<u8>(gguf, &format!("{b}.ffn_up.weight"))?,
                w_down: tensor_ptr::<u8>(gguf, &format!("{b}.ffn_down.weight"))?,
                post_ffn_norm: load_norm_ptr(gguf, &format!("{b}.post_ffw_norm.weight"), &mut bf16_bufs),
                inp_gate: tensor_ptr_opt::<u8>(gguf, &format!("{b}.inp_gate.weight")),
                proj: tensor_ptr_opt::<u8>(gguf, &format!("{b}.proj.weight")),
                inp_gate_dtype: tensor_dtype(gguf, &format!("{b}.inp_gate.weight")),
                proj_dtype: tensor_dtype(gguf, &format!("{b}.proj.weight")),
                post_norm: load_norm_ptr(gguf, &format!("{b}.post_norm.weight"), &mut bf16_bufs),
                layer_output_scale: read_f32_scalar(gguf, &format!("{b}.layer_output_scale.weight")),
                // Phase B.1: repacked buffers populated after struct construction (Task 7).
                wq_repacked: None,
                wk_repacked: None,
                wv_repacked: None,
                wo_repacked: None,
                w_gate_repacked: None,
                w_up_repacked: None,
                w_down_repacked: None,
                w_down_q6k_repacked: None,
                inp_gate_repacked: None,
                proj_repacked: None,
            };
            layers.push(lw);
            engine_helpers::populate_q4k_repacked(
                layers.last_mut().unwrap(),
                n_heads, n_kv_heads, head_dim_k[i], head_dim_v[i], hidden_dim, ffn_dim[i],
                ple_dim,
            );
        }

        // 9. Global tensors
        let embed_weight = tensor_ptr::<u8>(gguf, "token_embd.weight")?;
        let embed_idx = gguf.tensor_map["token_embd.weight"];
        let embed_dtype = gguf.tensors[embed_idx].dtype;
        let norm_weight = tensor_ptr::<f32>(gguf, "output_norm.weight")?;

        // PLE global tensors (optional)
        let ple_token_embd = tensor_ptr_opt::<u8>(gguf, "per_layer_token_embd.weight");
        let ple_model_proj = tensor_ptr_opt::<u8>(gguf, "per_layer_model_proj.weight");

        // PLE projection norm — may be BF16 or F32
        let ple_proj_norm = load_norm_ptr(
            gguf,
            "per_layer_proj_norm.weight",
            &mut bf16_bufs,
        );

        // 10. Diagnostic summary
        let n_swa = is_swa.iter().filter(|&&s| s).count();
        eprintln!("[gemma4] {n_layers}L h={hidden_dim} heads={n_heads}/{n_kv_heads} swa={n_swa} global={}", n_layers - n_swa);
        eprintln!("[gemma4] head_k={key_len_swa}/{key_len_global} ffn={}/{} sw={sliding_window} ple={ple_dim}",
            ffn_dim.iter().min().unwrap_or(&0), ffn_dim.iter().max().unwrap_or(&0));
        eprintln!("[gemma4] rope={rope_theta_swa}/{rope_theta_global} dim={rope_dim_swa}/{rope_dim_global} softcap={logit_softcap} shared_kv={shared_suffix} bf16_bufs={}", bf16_bufs.len());

        Ok(Gemma4Model {
            n_layers,
            hidden_dim,
            n_heads,
            n_kv_heads,
            vocab_size,
            rms_eps,
            sliding_window,
            logit_softcap,
            ple_dim,
            head_dim_k,
            head_dim_v,
            ffn_dim,
            is_swa,
            kv_shared_source,
            rope_theta_swa,
            rope_theta_global,
            rope_dim_swa,
            rope_dim_global,
            rope_freqs,
            layers,
            embed_weight,
            embed_q6k_repacked: None, // repacking hurts for large vocab (d_arr extraction overhead)
            embed_dtype,
            norm_weight,
            ple_token_embd,
            ple_model_proj,
            ple_proj_norm,
            _bf16_bufs: bf16_bufs,
        })
    }
}


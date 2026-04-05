//! Gemma 4 model structure and weight loading from GGUF.

use crate::inference::gguf::{GgufFile, MetaValue};

// ---------------------------------------------------------------------------
// Attention type per layer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttnType {
    SlidingWindow,
    Global,
}

// ---------------------------------------------------------------------------
// Per-layer weight pointers (raw into mmap)
// ---------------------------------------------------------------------------

pub struct LayerWeights {
    pub attn_norm: *const f32,
    pub wq: *const u8,
    pub wk: *const u8,
    pub wv: *const u8,
    pub wo: *const u8,
    pub ffn_norm: *const f32,
    pub w_gate: *const u8,
    pub w_up: *const u8,
    pub w_down: *const u8,
}

impl std::fmt::Debug for LayerWeights {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LayerWeights").finish()
    }
}

// ---------------------------------------------------------------------------
// Gemma4Model
// ---------------------------------------------------------------------------

pub struct Gemma4Model {
    pub n_layers: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub sliding_window: usize,
    pub attn_types: Vec<AttnType>,
    pub kv_shared_source: Vec<Option<usize>>,
    pub ple_dim: usize,

    pub layers: Vec<LayerWeights>,
    pub embed_weight: *const u8,
    pub embed_dtype: u32,
    pub output_weight: *const u8,
    pub output_dtype: u32,
    pub norm_weight: *const f32,
}

unsafe impl Send for Gemma4Model {}
unsafe impl Sync for Gemma4Model {}

// ---------------------------------------------------------------------------
// Metadata helpers
// ---------------------------------------------------------------------------

fn get_meta_u32(gguf: &GgufFile, key: &str) -> Option<u32> {
    match gguf.metadata.get(key)? {
        MetaValue::U32(v) => Some(*v),
        MetaValue::I32(v) => Some(*v as u32),
        MetaValue::U64(v) => Some(*v as u32),
        MetaValue::I64(v) => Some(*v as u32),
        MetaValue::U16(v) => Some(*v as u32),
        MetaValue::U8(v) => Some(*v as u32),
        _ => None,
    }
}

fn get_meta_f32(gguf: &GgufFile, key: &str) -> Option<f32> {
    match gguf.metadata.get(key)? {
        MetaValue::F32(v) => Some(*v),
        MetaValue::F64(v) => Some(*v as f32),
        MetaValue::U32(v) => Some(*v as f32),
        _ => None,
    }
}

fn tensor_ptr<T>(gguf: &GgufFile, name: &str) -> Result<*const T, String> {
    let data = gguf
        .tensor_data(name)
        .ok_or_else(|| format!("missing tensor: {name}"))?;
    Ok(data.as_ptr() as *const T)
}

// ---------------------------------------------------------------------------
// Compute attention type pattern
// ---------------------------------------------------------------------------

fn compute_attn_types(n_layers: usize, global_every: usize) -> Vec<AttnType> {
    (0..n_layers)
        .map(|i| {
            if i == n_layers - 1 || (global_every > 0 && i % global_every == 0) {
                AttnType::Global
            } else {
                AttnType::SlidingWindow
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Compute shared KV source mapping
// ---------------------------------------------------------------------------

fn compute_kv_shared(n_layers: usize, shared_suffix_len: usize) -> Vec<Option<usize>> {
    if shared_suffix_len == 0 {
        return vec![None; n_layers];
    }
    let first_shared = n_layers.saturating_sub(shared_suffix_len);
    (0..n_layers)
        .map(|i| {
            if i >= first_shared && i > 0 {
                // Reuse KV from the layer just before the shared range
                Some(first_shared.saturating_sub(1))
            } else {
                None
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Model loading
// ---------------------------------------------------------------------------

impl Gemma4Model {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, String> {
        // 1. Architecture name
        let arch = gguf
            .get_str("general.architecture")
            .unwrap_or("gemma2");
        eprintln!("[gemma4] architecture: {arch}");

        // 2. Read metadata params
        let n_layers = get_meta_u32(gguf, &format!("{arch}.block_count"))
            .ok_or("missing block_count")? as usize;
        let hidden_dim = get_meta_u32(gguf, &format!("{arch}.embedding_length"))
            .ok_or("missing embedding_length")? as usize;
        let n_heads = get_meta_u32(gguf, &format!("{arch}.attention.head_count"))
            .ok_or("missing head_count")? as usize;
        let n_kv_heads = get_meta_u32(gguf, &format!("{arch}.attention.head_count_kv"))
            .ok_or("missing head_count_kv")? as usize;
        let ffn_dim = get_meta_u32(gguf, &format!("{arch}.feed_forward_length"))
            .ok_or("missing feed_forward_length")? as usize;
        let head_dim = hidden_dim / n_heads;

        let rope_theta = get_meta_f32(gguf, &format!("{arch}.rope.freq_base"))
            .unwrap_or(10000.0);
        let rms_eps = get_meta_f32(gguf, &format!("{arch}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6);
        let sliding_window = get_meta_u32(gguf, &format!("{arch}.attention.sliding_window"))
            .map(|v| v as usize)
            .unwrap_or(512);

        // Vocab size from embedding tensor dims
        let vocab_size = gguf
            .tensor_map
            .get("token_embd.weight")
            .and_then(|&idx| gguf.tensors.get(idx))
            .and_then(|ti| ti.dims.first().copied())
            .ok_or("cannot determine vocab_size from token_embd.weight")? as usize;

        // Global attention interval (default every 5th layer)
        let global_every = get_meta_u32(gguf, &format!("{arch}.attention.global_layer_interval"))
            .map(|v| v as usize)
            .unwrap_or(5);

        // Shared KV suffix length
        let shared_suffix = get_meta_u32(gguf, &format!("{arch}.attention.shared_kv_layers"))
            .map(|v| v as usize)
            .unwrap_or(0);

        // PLE dimension (0 if not present)
        let ple_dim = get_meta_u32(gguf, &format!("{arch}.ple.embedding_length"))
            .map(|v| v as usize)
            .unwrap_or(0);

        // 3. Attention types
        let attn_types = compute_attn_types(n_layers, global_every);

        // 4. Shared KV source
        let kv_shared_source = compute_kv_shared(n_layers, shared_suffix);

        // 5. Per-layer weight pointers
        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let prefix = format!("blk.{i}");
            let lw = LayerWeights {
                attn_norm: tensor_ptr::<f32>(gguf, &format!("{prefix}.attn_norm.weight"))?,
                wq: tensor_ptr::<u8>(gguf, &format!("{prefix}.attn_q.weight"))?,
                wk: tensor_ptr::<u8>(gguf, &format!("{prefix}.attn_k.weight"))?,
                wv: tensor_ptr::<u8>(gguf, &format!("{prefix}.attn_v.weight"))?,
                wo: tensor_ptr::<u8>(gguf, &format!("{prefix}.attn_output.weight"))?,
                ffn_norm: tensor_ptr::<f32>(gguf, &format!("{prefix}.ffn_norm.weight"))?,
                w_gate: tensor_ptr::<u8>(gguf, &format!("{prefix}.ffn_gate.weight"))?,
                w_up: tensor_ptr::<u8>(gguf, &format!("{prefix}.ffn_up.weight"))?,
                w_down: tensor_ptr::<u8>(gguf, &format!("{prefix}.ffn_down.weight"))?,
            };
            layers.push(lw);
        }

        // 6. Global tensors
        let embed_weight = tensor_ptr::<u8>(gguf, "token_embd.weight")?;
        let embed_idx = gguf.tensor_map["token_embd.weight"];
        let embed_dtype = gguf.tensors[embed_idx].dtype;

        // Output projection — may be tied to embedding
        let (output_weight, output_dtype) = if gguf.tensor_map.contains_key("output.weight") {
            let ptr = tensor_ptr::<u8>(gguf, "output.weight")?;
            let idx = gguf.tensor_map["output.weight"];
            (ptr, gguf.tensors[idx].dtype)
        } else {
            // Tied to embedding
            (embed_weight, embed_dtype)
        };

        let norm_weight = tensor_ptr::<f32>(gguf, "output_norm.weight")?;

        // 7. Diagnostic info
        let n_global = attn_types.iter().filter(|t| **t == AttnType::Global).count();
        let n_sliding = n_layers - n_global;
        eprintln!("[gemma4] layers={n_layers} hidden={hidden_dim} heads={n_heads} kv_heads={n_kv_heads} head_dim={head_dim}");
        eprintln!("[gemma4] ffn={ffn_dim} vocab={vocab_size} rope_theta={rope_theta} rms_eps={rms_eps}");
        eprintln!("[gemma4] sliding_window={sliding_window} global_every={global_every} ({n_global} global, {n_sliding} sliding)");
        eprintln!("[gemma4] shared_kv_suffix={shared_suffix} ple_dim={ple_dim}");
        eprintln!("[gemma4] embed_dtype={embed_dtype} output_dtype={output_dtype} output_tied={}", !gguf.tensor_map.contains_key("output.weight"));

        Ok(Gemma4Model {
            n_layers,
            hidden_dim,
            n_heads,
            n_kv_heads,
            head_dim,
            ffn_dim,
            vocab_size,
            rope_theta,
            rms_eps,
            sliding_window,
            attn_types,
            kv_shared_source,
            ple_dim,
            layers,
            embed_weight,
            embed_dtype,
            output_weight,
            output_dtype,
            norm_weight,
        })
    }
}

//! Offline weight-tensor inventory for Gemma 4 E2B.
//!
//! Iterates every tensor in a loaded `Gemma4Model`, reports dtype + byte
//! size per tensor role. Feeds Adaptive Quant baseline — tells us which
//! tensors are already Q6K (high precision) vs Q4K vs F32, and the
//! current bytes-per-tensor breakdown so we know where the bandwidth
//! actually lives.
//!
//! Pure offline analysis — no kernel calls, no forward pass. Safe to
//! call at any point after model load.

use crate::inference::engine::{Gemma4Model, LayerWeights};
use crate::inference::matmul::{
    GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
    Q4K_BLOCK_SIZE, Q4K_BLOCK_BYTES, Q5K_BLOCK_BYTES, Q6K_BLOCK_BYTES,
};

const GGML_TYPE_F32: u32 = 0;
const GGML_TYPE_F16: u32 = 1;
const GGML_TYPE_BF16: u32 = 30;

pub struct TensorSummary {
    pub role: &'static str,
    pub layer: Option<usize>,
    pub dtype_name: &'static str,
    pub n_elements: usize,
    pub n_bytes: usize,
}

pub struct ModelSummary {
    pub tensors: Vec<TensorSummary>,
    pub total_bytes: usize,
    pub total_elements: usize,
}

pub fn summarize(model: &Gemma4Model) -> ModelSummary {
    let mut tensors = Vec::new();

    // Per-layer tensors.
    for (li, lw) in model.layers.iter().enumerate() {
        let hd = model.hidden_dim;
        let ffn = model.ffn_dim[li];
        let hkv_q = model.n_heads * model.head_dim_k[li];
        let hkv_k = model.n_kv_heads * model.head_dim_k[li];
        let hkv_v = model.n_kv_heads * model.head_dim_v[li];

        // Attention projections
        push_tensor(&mut tensors, "attn.wq", Some(li), lw.wq_dtype, hd * hkv_q);
        push_tensor(&mut tensors, "attn.wk", Some(li), lw.wk_dtype, hd * hkv_k);
        push_tensor(&mut tensors, "attn.wv", Some(li), lw.wv_dtype, hd * hkv_v);
        push_tensor(&mut tensors, "attn.wo", Some(li), lw.wo_dtype, hkv_q * hd);

        // FFN projections
        push_tensor(&mut tensors, "ffn.gate", Some(li), lw.w_gate_dtype, hd * ffn);
        push_tensor(&mut tensors, "ffn.up",   Some(li), lw.w_up_dtype,   hd * ffn);
        push_tensor(&mut tensors, "ffn.down", Some(li), lw.w_down_dtype, ffn * hd);

        // PLE
        if model.ple_dim > 0 && !lw.inp_gate.is_null() {
            push_tensor(&mut tensors, "ple.inp_gate", Some(li), lw.inp_gate_dtype,
                hd * model.ple_dim);
            push_tensor(&mut tensors, "ple.proj",     Some(li), lw.proj_dtype,
                model.ple_dim * hd);
        }

        // Norms (f32)
        if !lw.attn_norm.is_null() { push_tensor(&mut tensors, "norm.attn_pre", Some(li), GGML_TYPE_F32, hd); }
        if !lw.post_attn_norm.is_null() { push_tensor(&mut tensors, "norm.attn_post", Some(li), GGML_TYPE_F32, hd); }
        if !lw.ffn_norm.is_null() { push_tensor(&mut tensors, "norm.ffn_pre", Some(li), GGML_TYPE_F32, hd); }
        if !lw.post_ffn_norm.is_null() { push_tensor(&mut tensors, "norm.ffn_post", Some(li), GGML_TYPE_F32, hd); }
    }

    // Global tensors.
    push_tensor(&mut tensors, "embed", None, model.embed_dtype,
        model.vocab_size * model.hidden_dim);
    push_tensor(&mut tensors, "norm.final", None, GGML_TYPE_F32, model.hidden_dim);

    let total_bytes: usize = tensors.iter().map(|t| t.n_bytes).sum();
    let total_elements: usize = tensors.iter().map(|t| t.n_elements).sum();
    ModelSummary { tensors, total_bytes, total_elements }
}

fn push_tensor(out: &mut Vec<TensorSummary>, role: &'static str, layer: Option<usize>,
    dtype: u32, n_elements: usize,
) {
    let (name, bytes) = match dtype {
        GGML_TYPE_F32  => ("F32",  n_elements * 4),
        GGML_TYPE_F16  => ("F16",  n_elements * 2),
        GGML_TYPE_BF16 => ("BF16", n_elements * 2),
        GGML_TYPE_Q4_K => ("Q4K",  (n_elements + Q4K_BLOCK_SIZE - 1) / Q4K_BLOCK_SIZE * Q4K_BLOCK_BYTES),
        GGML_TYPE_Q5_K => ("Q5K",  (n_elements + Q4K_BLOCK_SIZE - 1) / Q4K_BLOCK_SIZE * Q5K_BLOCK_BYTES),
        GGML_TYPE_Q6_K => ("Q6K",  (n_elements + Q4K_BLOCK_SIZE - 1) / Q4K_BLOCK_SIZE * Q6K_BLOCK_BYTES),
        _              => ("???",  0),
    };
    out.push(TensorSummary { role, layer, dtype_name: name, n_elements, n_bytes: bytes });
}

/// Aggregate by role + dtype, report total bytes and per-tensor-type share.
pub fn format_report(summary: &ModelSummary) -> String {
    use std::collections::BTreeMap;
    let mut by_role_dtype: BTreeMap<(&'static str, &'static str), (usize, usize, usize)> =
        BTreeMap::new();
    for t in &summary.tensors {
        let e = by_role_dtype.entry((t.role, t.dtype_name)).or_default();
        e.0 += 1;
        e.1 += t.n_bytes;
        e.2 += t.n_elements;
    }
    let mut out = String::new();
    out.push_str(&format!(
        "Model inventory: {} tensors, {:.1} MB total, {:.1}M elements\n",
        summary.tensors.len(),
        summary.total_bytes as f64 / 1_048_576.0,
        summary.total_elements as f64 / 1_000_000.0,
    ));
    out.push_str(&format!(
        "{:<18} {:>5} {:>5} {:>10} {:>8}\n",
        "role", "dtype", "count", "bytes_MB", "share_%",
    ));
    out.push_str(&"-".repeat(52));
    out.push('\n');
    for ((role, dtype), (count, bytes, _elements)) in &by_role_dtype {
        let share = 100.0 * *bytes as f64 / summary.total_bytes as f64;
        out.push_str(&format!(
            "{:<18} {:>5} {:>5} {:>10.2} {:>7.2}%\n",
            role, dtype, count,
            *bytes as f64 / 1_048_576.0,
            share,
        ));
    }
    out
}

/// Per-layer dtype fingerprint: returns a short string per layer like
/// "Q6K/Q4K/Q6K/Q4K/Q4K/Q4K" summarizing wq/wk/wv/wo/ffn_gate/ffn_up/ffn_down
/// dtype codes. Useful for spotting layers with unusual quant choices.
pub fn per_layer_dtype_fingerprint(model: &Gemma4Model) -> Vec<String> {
    model.layers.iter().enumerate().map(|(li, lw)| {
        format!("L{:02} attn({}/{}/{}/{}) ffn({}/{}/{}) ffn_dim={}",
            li,
            dtype_short(lw.wq_dtype),
            dtype_short(lw.wk_dtype),
            dtype_short(lw.wv_dtype),
            dtype_short(lw.wo_dtype),
            dtype_short(lw.w_gate_dtype),
            dtype_short(lw.w_up_dtype),
            dtype_short(lw.w_down_dtype),
            model.ffn_dim[li],
        )
    }).collect()
}

fn dtype_short(dtype: u32) -> &'static str {
    match dtype {
        GGML_TYPE_F32 => "F32",
        GGML_TYPE_F16 => "F16",
        GGML_TYPE_BF16 => "BF",
        GGML_TYPE_Q4_K => "Q4K",
        GGML_TYPE_Q5_K => "Q5K",
        GGML_TYPE_Q6_K => "Q6K",
        _ => "?",
    }
}

/// Silence the no-op LayerWeights warning: acknowledges the `_lw` parameter
/// would be unused if we ever drop the inventory logic.
#[allow(dead_code)]
fn _hint_unused(_lw: &LayerWeights) {}

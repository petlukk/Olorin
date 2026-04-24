//! Per-layer sensitivity-aware quantization recipe generator.
//!
//! Consumes telemetry (residual norms + FFN activation stats) to produce a
//! per-layer "how sensitive is this layer to quant error" score, then maps
//! scores to quant buckets (Q4K / Q5K / Q6K). Cross-references the current
//! model's per-tensor dtypes to produce an upgrade/downgrade delta table.
//!
//! Offline analysis — runs after a calibration forward pass populates the
//! activation_track tracker. Output drives requantization tooling (either
//! llama-quantize overrides or a future native Rust requant pass).

use crate::inference::engine::Gemma4Model;
use crate::inference::matmul::{GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K};

/// Raw stats for one decoder layer, aggregated across neurons and tokens.
#[derive(Debug, Clone)]
pub struct LayerProfile {
    pub layer: usize,
    pub ffn_dim: usize,
    /// Mean across neurons of the per-neuron mean |activation|.
    pub ffn_mean: f32,
    /// Std across neurons of the per-neuron mean |activation|.
    pub ffn_std: f32,
    /// Max across neurons of the per-neuron max |activation|.
    pub ffn_max: f32,
    /// Mean residual-stream L2 norm observed across decoded tokens.
    pub residual_mean: f32,
    /// Std of residual-stream L2 norm across tokens (how variable across prompts).
    pub residual_std: f32,
    /// Signed delta from previous layer's residual mean. Positive = layer adds signal.
    pub residual_delta: f32,
}

/// Build layer profiles from the tracker's existing snapshots.
pub fn compute_profiles(
    per_layer_mean_abs: &[Vec<f32>],
    per_layer_max_abs: &[Vec<f32>],
    residual_norms: &[Vec<f32>],
) -> Vec<LayerProfile> {
    let n = per_layer_mean_abs.len()
        .min(per_layer_max_abs.len())
        .min(residual_norms.len());
    let mut out = Vec::with_capacity(n);
    let mut prev_residual_mean = 0.0f32;
    for li in 0..n {
        let mean_abs = &per_layer_mean_abs[li];
        let max_abs = &per_layer_max_abs[li];
        let norms = &residual_norms[li];

        let (ffn_mean, ffn_std) = mean_std(mean_abs);
        let ffn_max = max_abs.iter().copied().fold(0.0f32, f32::max);
        let (residual_mean, residual_std) = mean_std(norms);
        let residual_delta = residual_mean - prev_residual_mean;
        prev_residual_mean = residual_mean;

        out.push(LayerProfile {
            layer: li,
            ffn_dim: mean_abs.len(),
            ffn_mean, ffn_std, ffn_max,
            residual_mean, residual_std, residual_delta,
        });
    }
    out
}

fn mean_std(v: &[f32]) -> (f32, f32) {
    if v.is_empty() { return (0.0, 0.0); }
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
    (mean, var.sqrt())
}

/// Normalize a series to [0, 1] via min-max scaling.
fn minmax_normalize(v: &[f32]) -> Vec<f32> {
    let lo = v.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = (hi - lo).max(1e-9);
    v.iter().map(|&x| (x - lo) / range).collect()
}

/// Sensitivity score per layer — 0..1, higher = quant-precision-critical.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantBucket { Q4K, Q5K, Q6K }

impl QuantBucket {
    pub fn as_ggml_type(self) -> u32 {
        match self {
            QuantBucket::Q4K => GGML_TYPE_Q4_K,
            QuantBucket::Q5K => GGML_TYPE_Q5_K,
            QuantBucket::Q6K => GGML_TYPE_Q6_K,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            QuantBucket::Q4K => "Q4K",
            QuantBucket::Q5K => "Q5K",
            QuantBucket::Q6K => "Q6K",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LayerSensitivity {
    pub layer: usize,
    pub score: f32,
    pub bucket: QuantBucket,
}

/// Compute sensitivity + bucket per layer. Score weights (tunable):
///   0.35 * residual_mean   — how much signal flows through the layer
///   0.25 * residual_std    — prompt-specific work (quant-sensitive)
///   0.15 * |residual_delta| — how much work this layer adds (absorbs ≠ adds)
///   0.15 * ffn_max         — outlier presence (clipping risk under aggressive quant)
///   0.10 * ffn_std         — FFN activation spread across neurons
///
/// Buckets use percentile thresholds on the final normalized score so the
/// distribution is self-calibrating:
///   top 25%  → Q6K   (precision-critical)
///   middle   → Q5K
///   bottom 25% → Q4K (quant-tolerant)
pub fn compute_sensitivity(profiles: &[LayerProfile]) -> Vec<LayerSensitivity> {
    let n = profiles.len();
    if n == 0 { return Vec::new(); }

    let r_mean: Vec<f32> = profiles.iter().map(|p| p.residual_mean).collect();
    let r_std: Vec<f32> = profiles.iter().map(|p| p.residual_std).collect();
    let r_delta: Vec<f32> = profiles.iter().map(|p| p.residual_delta.abs()).collect();
    let ffn_max: Vec<f32> = profiles.iter().map(|p| p.ffn_max).collect();
    let ffn_std: Vec<f32> = profiles.iter().map(|p| p.ffn_std).collect();

    let n_rm = minmax_normalize(&r_mean);
    let n_rs = minmax_normalize(&r_std);
    let n_rd = minmax_normalize(&r_delta);
    let n_fm = minmax_normalize(&ffn_max);
    let n_fs = minmax_normalize(&ffn_std);

    let mut scored: Vec<(usize, f32)> = (0..n).map(|i| {
        let s = 0.35 * n_rm[i]
              + 0.25 * n_rs[i]
              + 0.15 * n_rd[i]
              + 0.15 * n_fm[i]
              + 0.10 * n_fs[i];
        (i, s)
    }).collect();

    // Percentile buckets on final score.
    let mut sorted_scores: Vec<f32> = scored.iter().map(|(_, s)| *s).collect();
    sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p25 = sorted_scores[n / 4];
    let p75 = sorted_scores[(n * 3) / 4];

    scored.sort_by_key(|(i, _)| *i);
    scored.into_iter().map(|(i, s)| {
        let bucket = if s >= p75 { QuantBucket::Q6K }
                     else if s < p25 { QuantBucket::Q4K }
                     else { QuantBucket::Q5K };
        LayerSensitivity { layer: i, score: s, bucket }
    }).collect()
}

/// One recipe row — per-tensor recommendation with current-vs-recommended delta.
#[derive(Debug, Clone)]
pub struct TensorRecipe {
    pub layer: usize,
    pub role: &'static str,
    pub current_dtype: &'static str,
    pub recommended: QuantBucket,
    pub bytes_delta: i64, // negative = saves bytes
}

/// Rank of a dtype in "precision bits" order — higher = more precision.
fn dtype_rank(dtype: u32) -> u8 {
    match dtype {
        GGML_TYPE_Q4_K => 4,
        GGML_TYPE_Q5_K => 5,
        GGML_TYPE_Q6_K => 6,
        _              => 99, // F16/BF16/F32 stay untouched
    }
}

fn bucket_rank(b: QuantBucket) -> u8 {
    match b {
        QuantBucket::Q4K => 4,
        QuantBucket::Q5K => 5,
        QuantBucket::Q6K => 6,
    }
}

/// Generate tensor-level recipe from layer-level sensitivity buckets.
/// Only targets FFN tensors (gate/up/down) + attention (q/k/v/o) for now —
/// norms and PLE stay untouched.
///
/// Recommendation policy: within a layer, all big compute tensors get the
/// layer's bucket. If the recommendation equals the current dtype, row
/// is included with `bytes_delta = 0` for visibility. If different, the
/// delta captures the bandwidth impact.
pub fn generate_recipe(
    model: &Gemma4Model,
    sensitivity: &[LayerSensitivity],
) -> Vec<TensorRecipe> {
    use crate::inference::matmul::{
        Q4K_BLOCK_SIZE, Q4K_BLOCK_BYTES, Q5K_BLOCK_BYTES, Q6K_BLOCK_BYTES,
    };
    let bytes_for = |elements: usize, dtype: u32| -> usize {
        let blocks = (elements + Q4K_BLOCK_SIZE - 1) / Q4K_BLOCK_SIZE;
        match dtype {
            GGML_TYPE_Q4_K => blocks * Q4K_BLOCK_BYTES,
            GGML_TYPE_Q5_K => blocks * Q5K_BLOCK_BYTES,
            GGML_TYPE_Q6_K => blocks * Q6K_BLOCK_BYTES,
            _              => elements * 2, // fallback for F16-ish
        }
    };
    let dtype_label = |dtype: u32| -> &'static str {
        match dtype {
            GGML_TYPE_Q4_K => "Q4K",
            GGML_TYPE_Q5_K => "Q5K",
            GGML_TYPE_Q6_K => "Q6K",
            _              => "?",
        }
    };

    let mut rows = Vec::new();
    let hd = model.hidden_dim;
    for s in sensitivity {
        let li = s.layer;
        if li >= model.layers.len() { continue; }
        let lw = &model.layers[li];
        let ffn = model.ffn_dim[li];
        let hkv_q = model.n_heads * model.head_dim_k[li];
        let hkv_k = model.n_kv_heads * model.head_dim_k[li];
        let hkv_v = model.n_kv_heads * model.head_dim_v[li];

        let target = s.bucket.as_ggml_type();
        let push = |rows: &mut Vec<TensorRecipe>, role, current, n_elem| {
            let current_bytes = bytes_for(n_elem, current) as i64;
            let target_bytes = bytes_for(n_elem, target) as i64;
            rows.push(TensorRecipe {
                layer: li,
                role,
                current_dtype: dtype_label(current),
                recommended: s.bucket,
                bytes_delta: target_bytes - current_bytes,
            });
        };
        push(&mut rows, "attn.wq",   lw.wq_dtype,      hd * hkv_q);
        push(&mut rows, "attn.wk",   lw.wk_dtype,      hd * hkv_k);
        push(&mut rows, "attn.wv",   lw.wv_dtype,      hd * hkv_v);
        push(&mut rows, "attn.wo",   lw.wo_dtype,      hkv_q * hd);
        push(&mut rows, "ffn.gate",  lw.w_gate_dtype,  hd * ffn);
        push(&mut rows, "ffn.up",    lw.w_up_dtype,    hd * ffn);
        push(&mut rows, "ffn.down",  lw.w_down_dtype,  ffn * hd);
    }
    rows
}

/// Human-readable per-layer recipe summary with bytes-delta total.
pub fn format_recipe(
    sensitivity: &[LayerSensitivity],
    recipe: &[TensorRecipe],
) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    writeln!(out, "{:>3}  {:>6}  {:>6}  {:>8}  {:>10}",
        "L", "score", "bucket", "Δ_MB", "policy").unwrap();
    writeln!(out, "{:-<44}", "").unwrap();
    let mut total_delta = 0i64;
    for s in sensitivity {
        let layer_delta: i64 = recipe.iter()
            .filter(|r| r.layer == s.layer)
            .map(|r| r.bytes_delta)
            .sum();
        total_delta += layer_delta;
        let policy = match s.bucket {
            QuantBucket::Q6K => "keep-high",
            QuantBucket::Q5K => "moderate",
            QuantBucket::Q4K => "aggressive",
        };
        writeln!(out, "{:>3}  {:>6.3}  {:>6}  {:>8.2}  {:>10}",
            s.layer, s.score, s.bucket.label(),
            layer_delta as f64 / 1_048_576.0, policy).unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "Total bytes delta: {:.2} MB ({})",
        total_delta as f64 / 1_048_576.0,
        if total_delta < 0 { "savings" } else { "cost" }).unwrap();
    out
}

/// Per-tensor delta table — only rows where the recommendation differs from current.
pub fn format_recipe_deltas(recipe: &[TensorRecipe]) -> String {
    use std::fmt::Write;
    let mut out = String::new();
    writeln!(out, "{:>3}  {:<10}  {:>7}  {:>5}  {:>8}",
        "L", "role", "current", "new", "Δ_KB").unwrap();
    writeln!(out, "{:-<40}", "").unwrap();
    for r in recipe.iter().filter(|r| r.current_dtype != r.recommended.label()) {
        writeln!(out, "{:>3}  {:<10}  {:>7}  {:>5}  {:>+8.1}",
            r.layer, r.role, r.current_dtype, r.recommended.label(),
            r.bytes_delta as f64 / 1024.0).unwrap();
    }
    out
}

/// Filter a recipe to "apply these" rows: downgrade-only semantics.
/// Keeps a row only if the recommended bucket has LOWER precision than the
/// current dtype — i.e. this is a bandwidth saving, not a quality upgrade.
/// For llama.cpp's baseline, this gives a pure-savings recipe.
///
/// **Policy: never downgrade Q6K sources.** Q6K tensors in Gemma 4 E2B
/// (embeddings, wq/wv and w_down on a subset of layers) sit on paths with
/// dedicated pre-computed d-arrays and repack layouts. Downgrading them
/// forfeits both quality and optimized-path infrastructure.
pub fn filter_downgrades(recipe: &[TensorRecipe]) -> Vec<TensorRecipe> {
    recipe.iter()
        .filter(|r| {
            if r.current_dtype == "Q6K" { return false; }
            let cur = match r.current_dtype {
                "Q4K" => 4, "Q5K" => 5, "Q6K" => 6, _ => 99,
            };
            bucket_rank(r.recommended) < cur
        })
        .cloned()
        .collect()
}

/// Filter a recipe to "review these" rows: where current is LOWER precision
/// than our sensitivity says is ideal. These are candidates for QUALITY
/// improvement — typically tried AFTER downgrades ship, to see if perplexity
/// can be recovered. Don't apply together with downgrades unless you also
/// want the net bandwidth impact.
pub fn filter_upgrade_candidates(recipe: &[TensorRecipe]) -> Vec<TensorRecipe> {
    recipe.iter()
        .filter(|r| {
            let cur = match r.current_dtype {
                "Q4K" => 4, "Q5K" => 5, "Q6K" => 6, _ => 0,
            };
            cur != 0 && bucket_rank(r.recommended) > cur
        })
        .cloned()
        .collect()
}

/// Total bytes saved by applying only the downgrade rows.
pub fn downgrade_savings_bytes(recipe: &[TensorRecipe]) -> i64 {
    filter_downgrades(recipe).iter().map(|r| -r.bytes_delta).sum()
}

/// Map an Olorin role to the llama.cpp GGUF tensor-name pattern.
/// Returns `blk.{layer}.{tail}.weight` where `tail` matches what
/// `llama-quantize --tensor-type` expects as the pattern substring.
fn role_to_llamacpp_tail(role: &str) -> Option<&'static str> {
    match role {
        "attn.wq"   => Some("attn_q"),
        "attn.wk"   => Some("attn_k"),
        "attn.wv"   => Some("attn_v"),
        "attn.wo"   => Some("attn_output"),
        "ffn.gate"  => Some("ffn_gate"),
        "ffn.up"    => Some("ffn_up"),
        "ffn.down"  => Some("ffn_down"),
        _           => None,
    }
}

/// Emit a `--tensor-type-file` formatted file body for `llama-quantize`.
/// Each line is `tensor_name=ggml_type`, one per row. `llama-quantize`
/// parses the name as a regex and applies the type to all matching tensors.
///
/// We emit exact names (escaped `\.`) rather than regexes so each line
/// targets exactly one tensor — no accidental regex overlap.
pub fn format_llamacpp_tensor_types(recipe: &[TensorRecipe]) -> String {
    let mut out = String::new();
    for r in recipe {
        let Some(tail) = role_to_llamacpp_tail(r.role) else { continue };
        let type_name = match r.recommended {
            QuantBucket::Q4K => "q4_K",
            QuantBucket::Q5K => "q5_K",
            QuantBucket::Q6K => "q6_K",
        };
        // Escape dots so llama.cpp regex only matches this exact tensor name.
        // `blk\.17\.attn_q\.weight` → matches only blk.17.attn_q.weight.
        out.push_str(&format!(
            "blk\\.{}\\.{}\\.weight={}\n",
            r.layer, tail, type_name,
        ));
    }
    out
}

// Silence "unused dtype_rank" — kept for future use by callers that want to
// compare current-vs-recommended numerically.
#[allow(dead_code)]
const _USE_DTYPE_RANK: fn(u32) -> u8 = dtype_rank;

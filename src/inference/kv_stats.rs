//! KV-cache statistics — per-layer entropy proxy for Adaptive Quant.
//!
//! After a decode run, the cache holds K/V vectors for every layer. We
//! read them back (f16 → f32), compute L2 norms per (head, position),
//! then summarize:
//!
//! * mean / std / min / max of norm across positions — quant headroom
//! * normalized Shannon entropy of the norm distribution — low entropy
//!   means a few positions dominate attention (precision-critical);
//!   high entropy means diffuse attention (quant-tolerant).
//!
//! Offline — reads the cache after generation, no hot-path impact.

use crate::inference::cache::KvCache;
use crate::inference::matmul::f16_to_f32_scalar;

#[derive(Debug, Clone)]
pub struct KvLayerStats {
    pub layer: usize,
    pub shared: bool,
    pub attn_len: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub k_mean_norm: f32,
    pub k_std_norm: f32,
    pub k_min_norm: f32,
    pub k_max_norm: f32,
    pub k_norm_entropy: f32,   // Normalized in [0, 1]
    pub v_mean_norm: f32,
    pub v_std_norm: f32,
    pub v_min_norm: f32,
    pub v_max_norm: f32,
    pub v_norm_entropy: f32,
}

pub fn summarize(cache: &KvCache) -> Vec<KvLayerStats> {
    let n_layers = cache.n_layers();
    let n_kv_heads = cache.n_kv_heads();
    let mut out = Vec::with_capacity(n_layers);

    for li in 0..n_layers {
        let shared = cache.is_shared(li);
        let attn_len = cache.attn_len(li);
        let hd = cache.head_dim_v(li);
        let stride = n_kv_heads * hd;
        if shared || attn_len == 0 {
            out.push(KvLayerStats {
                layer: li, shared, attn_len,
                n_kv_heads, head_dim: hd,
                k_mean_norm: 0.0, k_std_norm: 0.0, k_min_norm: 0.0,
                k_max_norm: 0.0, k_norm_entropy: 0.0,
                v_mean_norm: 0.0, v_std_norm: 0.0, v_min_norm: 0.0,
                v_max_norm: 0.0, v_norm_entropy: 0.0,
            });
            continue;
        }

        let k_norms = read_position_norms(cache.k_ptr(li), attn_len, stride);
        let v_norms = read_position_norms(cache.v_ptr(li), attn_len, stride);

        let (k_mean, k_std, k_min, k_max) = minmax_stats(&k_norms);
        let (v_mean, v_std, v_min, v_max) = minmax_stats(&v_norms);
        let k_entropy = normalized_entropy(&k_norms);
        let v_entropy = normalized_entropy(&v_norms);

        out.push(KvLayerStats {
            layer: li, shared, attn_len,
            n_kv_heads, head_dim: hd,
            k_mean_norm: k_mean, k_std_norm: k_std,
            k_min_norm: k_min, k_max_norm: k_max,
            k_norm_entropy: k_entropy,
            v_mean_norm: v_mean, v_std_norm: v_std,
            v_min_norm: v_min, v_max_norm: v_max,
            v_norm_entropy: v_entropy,
        });
    }
    out
}

/// Read L2 norms of `attn_len` consecutive vectors each of length `stride`
/// from a buffer of f16 (stored as u16). Returns a Vec<f32> of length attn_len.
fn read_position_norms(ptr: *const u16, attn_len: usize, stride: usize) -> Vec<f32> {
    let mut norms = Vec::with_capacity(attn_len);
    for p in 0..attn_len {
        let mut sq = 0.0f32;
        for i in 0..stride {
            let raw = unsafe { *ptr.add(p * stride + i) };
            let v = f16_to_f32_scalar(raw);
            sq += v * v;
        }
        norms.push(sq.sqrt());
    }
    norms
}

fn minmax_stats(v: &[f32]) -> (f32, f32, f32, f32) {
    if v.is_empty() { return (0.0, 0.0, 0.0, 0.0); }
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = var.sqrt();
    let min = v.iter().copied().fold(f32::INFINITY, f32::min);
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    (mean, std, min, max)
}

/// Normalized Shannon entropy of a norm distribution. Converts norms to a
/// probability (norm_i / sum) and returns H / log(n) ∈ [0, 1]. 1 = uniform,
/// 0 = single dominant entry.
fn normalized_entropy(v: &[f32]) -> f32 {
    let n = v.len();
    if n <= 1 { return 0.0; }
    let sum: f32 = v.iter().sum();
    if sum <= f32::EPSILON { return 0.0; }
    let mut h = 0.0f32;
    for &x in v {
        let p = x / sum;
        if p > 1e-20 {
            h -= p * p.ln();
        }
    }
    h / (n as f32).ln()
}

pub fn format_report(stats: &[KvLayerStats]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "{:>3}  {:>6}  {:>4}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}\n",
        "L", "shared", "len", "k_mean", "k_max/min", "k_H%", "v_mean", "v_max/min", "v_H%",
    ));
    out.push_str(&"-".repeat(72));
    out.push('\n');
    for s in stats {
        let k_ratio = if s.k_min_norm > 0.0 {
            s.k_max_norm / s.k_min_norm
        } else { f32::INFINITY };
        let v_ratio = if s.v_min_norm > 0.0 {
            s.v_max_norm / s.v_min_norm
        } else { f32::INFINITY };
        let shared_str = if s.shared { "SHR" } else { "-" };
        if s.shared {
            out.push_str(&format!(
                "{:>3}  {:>6}  {:>4}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}\n",
                s.layer, shared_str, "-", "-", "-", "-", "-", "-", "-",
            ));
        } else {
            out.push_str(&format!(
                "{:>3}  {:>6}  {:>4}  {:>7.3}  {:>9.2}  {:>6.1}%  {:>7.3}  {:>9.2}  {:>6.1}%\n",
                s.layer, shared_str, s.attn_len,
                s.k_mean_norm, k_ratio, s.k_norm_entropy * 100.0,
                s.v_mean_norm, v_ratio, s.v_norm_entropy * 100.0,
            ));
        }
    }
    out
}

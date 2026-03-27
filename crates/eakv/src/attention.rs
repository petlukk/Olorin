//! Attention — scores and output with 4-way kernel dispatch.
//!
//! Port of csrc/attention.c. Queries are rotated before scoring against
//! pre-rotated K vectors; output is inverse-rotated after V summation.

use crate::cache::EakvCache;

/// Compute attention scores for all query heads against cached K vectors.
///
/// `queries`: `[n_q_heads * head_dim]` f32 (one token's query vectors).
/// `scores_out`: `[n_q_heads * seq_len]` f32, written by the kernel.
pub fn attention_scores(
    cache: &EakvCache,
    queries: &[f32],
    layer: i32,
    n_q_heads: i32,
    n_kv_heads: i32,
    scores_out: &mut [f32],
) {
    let hd = cache.head_dim();
    let q_elems = (n_q_heads * hd) as usize;
    let n_q_groups = (q_elems / 64) as i32;
    let groups_per_head = cache.max_seq_len() * (hd / 64);

    // Rotate queries to match pre-rotated K vectors.
    let mut rot_q = vec![0.0f32; q_elems];
    rot_q.copy_from_slice(&queries[..q_elems]);

    let kt = cache.kernels();
    let signs = cache.jl_signs();
    for g in 0..n_q_groups as usize {
        unsafe {
            (kt.rotate)(rot_q.as_mut_ptr().add(g * 64), signs.as_ptr(), 64);
        }
    }

    let k_weights = cache.weights(layer, 0);
    let k_scales = cache.scales(layer, 0);
    let k_biases = cache.biases(layer, 0);

    unsafe {
        if hd == 64 {
            if n_q_heads == n_kv_heads {
                (kt.k_score_mha_64)(
                    rot_q.as_ptr(),
                    k_weights.as_ptr(),
                    k_scales.as_ptr(),
                    k_biases.as_ptr(),
                    scores_out.as_mut_ptr(),
                    cache.seq_len(),
                    n_q_heads,
                    groups_per_head,
                );
            } else {
                (kt.k_score_gqa_64)(
                    rot_q.as_ptr(),
                    k_weights.as_ptr(),
                    k_scales.as_ptr(),
                    k_biases.as_ptr(),
                    scores_out.as_mut_ptr(),
                    cache.seq_len(),
                    n_q_heads,
                    n_kv_heads,
                    groups_per_head,
                );
            }
        } else if n_q_heads == n_kv_heads {
            (kt.k_score_mha)(
                rot_q.as_ptr(),
                k_weights.as_ptr(),
                k_scales.as_ptr(),
                k_biases.as_ptr(),
                scores_out.as_mut_ptr(),
                cache.seq_len(),
                n_q_heads,
                groups_per_head,
            );
        } else {
            (kt.k_score_gqa)(
                rot_q.as_ptr(),
                k_weights.as_ptr(),
                k_scales.as_ptr(),
                k_biases.as_ptr(),
                scores_out.as_mut_ptr(),
                cache.seq_len(),
                n_q_heads,
                n_kv_heads,
                groups_per_head,
            );
        }
    }
}

/// Compute attention output by summing cached V vectors weighted by scores.
///
/// `weights_in`: `[n_q_heads * seq_len]` f32 (softmax'd attention weights).
/// `output_out`: `[n_q_heads * head_dim]` f32, written by the kernel then
///               inverse-rotated in-place.
pub fn attention_output(
    cache: &EakvCache,
    weights_in: &[f32],
    layer: i32,
    n_q_heads: i32,
    n_kv_heads: i32,
    output_out: &mut [f32],
) {
    let hd = cache.head_dim();
    let groups_per_head = cache.max_seq_len() * (hd / 64);

    let v_weights = cache.weights(layer, 1);
    let v_scales = cache.scales(layer, 1);
    let v_biases = cache.biases(layer, 1);

    let kt = cache.kernels();

    unsafe {
        if hd == 64 {
            if n_q_heads == n_kv_heads {
                (kt.v_sum_mha_64)(
                    weights_in.as_ptr(),
                    v_weights.as_ptr(),
                    v_scales.as_ptr(),
                    v_biases.as_ptr(),
                    output_out.as_mut_ptr(),
                    cache.seq_len(),
                    n_q_heads,
                    groups_per_head,
                );
            } else {
                (kt.v_sum_gqa_64)(
                    weights_in.as_ptr(),
                    v_weights.as_ptr(),
                    v_scales.as_ptr(),
                    v_biases.as_ptr(),
                    output_out.as_mut_ptr(),
                    cache.seq_len(),
                    n_q_heads,
                    n_kv_heads,
                    groups_per_head,
                );
            }
        } else if n_q_heads == n_kv_heads {
            (kt.v_sum_mha)(
                weights_in.as_ptr(),
                v_weights.as_ptr(),
                v_scales.as_ptr(),
                v_biases.as_ptr(),
                output_out.as_mut_ptr(),
                cache.seq_len(),
                n_q_heads,
                groups_per_head,
            );
        } else {
            (kt.v_sum_gqa)(
                weights_in.as_ptr(),
                v_weights.as_ptr(),
                v_scales.as_ptr(),
                v_biases.as_ptr(),
                output_out.as_mut_ptr(),
                cache.seq_len(),
                n_q_heads,
                n_kv_heads,
                groups_per_head,
            );
        }
    }

    // Inverse-rotate output to undo the V pre-rotation.
    let out_elems = (n_q_heads * hd) as usize;
    let n_out_groups = out_elems / 64;
    let signs = cache.jl_signs();
    for g in 0..n_out_groups {
        unsafe {
            let ptr = output_out.as_mut_ptr().add(g * 64);
            (kt.fwht)(ptr, 64);
            (kt.sign_flip)(ptr, signs.as_ptr(), 64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels;

    fn load_kernels() -> Option<crate::KernelTable> {
        let dir = kernels::find_kernel_dir().ok()?;
        crate::KernelTable::load(&dir).ok()
    }

    #[test]
    fn test_attention_scores_nonzero() {
        let kt = match load_kernels() {
            Some(k) => k,
            None => {
                eprintln!("skipping — kernels not available");
                return;
            }
        };

        let n_layers = 1;
        let n_kv_heads = 2;
        let head_dim = 64;
        let max_seq = 32;
        let seq_len = 4;
        let n_q_heads = 2; // MHA path

        let mut cache = EakvCache::new(n_layers, n_kv_heads, head_dim, max_seq, kt).unwrap();

        // Load known non-zero K and V data.
        let elems = n_layers as usize * 2 * n_kv_heads as usize * seq_len as usize * head_dim as usize;
        let data: Vec<f32> = (0..elems).map(|i| (i as f32) * 0.01 + 0.1).collect();
        cache.load_raw(&data, seq_len).unwrap();

        // Query vectors — non-zero.
        let q_elems = n_q_heads as usize * head_dim as usize;
        let queries: Vec<f32> = (0..q_elems).map(|i| (i as f32) * 0.02 + 0.5).collect();

        let mut scores = vec![0.0f32; n_q_heads as usize * seq_len as usize];
        attention_scores(&cache, &queries, 0, n_q_heads, n_kv_heads, &mut scores);

        // At least one score must be non-zero.
        let any_nonzero = scores.iter().any(|&s| s != 0.0);
        assert!(any_nonzero, "all scores are zero: {scores:?}");
    }

    #[test]
    fn test_attention_output_nonzero() {
        let kt = match load_kernels() {
            Some(k) => k,
            None => {
                eprintln!("skipping — kernels not available");
                return;
            }
        };

        let n_layers = 1;
        let n_kv_heads = 2;
        let head_dim = 64;
        let max_seq = 32;
        let seq_len = 4;
        let n_q_heads = 2;

        let mut cache = EakvCache::new(n_layers, n_kv_heads, head_dim, max_seq, kt).unwrap();

        let elems = n_layers as usize * 2 * n_kv_heads as usize * seq_len as usize * head_dim as usize;
        let data: Vec<f32> = (0..elems).map(|i| (i as f32) * 0.01 + 0.1).collect();
        cache.load_raw(&data, seq_len).unwrap();

        // Uniform attention weights (like softmax output).
        let w = 1.0 / seq_len as f32;
        let weights = vec![w; n_q_heads as usize * seq_len as usize];

        let mut output = vec![0.0f32; n_q_heads as usize * head_dim as usize];
        attention_output(&cache, &weights, 0, n_q_heads, n_kv_heads, &mut output);

        let any_nonzero = output.iter().any(|&v| v != 0.0);
        assert!(any_nonzero, "all output values are zero: {:?}", &output[..8]);
    }
}

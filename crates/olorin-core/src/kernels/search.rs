use super::ffi;

/// Compute dot products of query against n vectors of given dimension.
pub fn batch_dot(query: &[f32], vecs: &[f32], dim: usize, n_vecs: usize) -> Vec<f32> {
    assert!(query.len() >= dim);
    assert!(vecs.len() >= dim * n_vecs);
    let mut scores = vec![0.0f32; n_vecs];
    unsafe {
        ffi::batch_dot(
            query.as_ptr(),
            vecs.as_ptr(),
            dim as i32,
            n_vecs as i32,
            scores.as_mut_ptr(),
        );
    }
    scores
}

/// Compute cosine similarity of query against n vectors.
/// query_norm is the precomputed L2 norm of the query.
pub fn batch_cosine(
    query: &[f32],
    query_norm: f32,
    vecs: &[f32],
    dim: usize,
    n_vecs: usize,
) -> Vec<f32> {
    assert!(query.len() >= dim);
    assert!(vecs.len() >= dim * n_vecs);
    let mut scores = vec![0.0f32; n_vecs];
    unsafe {
        ffi::batch_cosine(
            query.as_ptr(),
            query_norm,
            vecs.as_ptr(),
            dim as i32,
            n_vecs as i32,
            scores.as_mut_ptr(),
        );
    }
    scores
}

/// Compute squared L2 distances from query to n vectors.
pub fn batch_l2(query: &[f32], vecs: &[f32], dim: usize, n_vecs: usize) -> Vec<f32> {
    assert!(query.len() >= dim);
    assert!(vecs.len() >= dim * n_vecs);
    let mut scores = vec![0.0f32; n_vecs];
    unsafe {
        ffi::batch_l2(
            query.as_ptr(),
            vecs.as_ptr(),
            dim as i32,
            n_vecs as i32,
            scores.as_mut_ptr(),
        );
    }
    scores
}

/// In-place L2 normalization of n vectors.
pub fn normalize_vectors(vecs: &mut [f32], dim: usize, n_vecs: usize) {
    assert!(vecs.len() >= dim * n_vecs);
    unsafe {
        ffi::normalize_vectors(vecs.as_mut_ptr(), dim as i32, n_vecs as i32);
    }
}

/// Return indices of scores >= threshold.
pub fn threshold_filter(scores: &[f32], threshold: f32) -> Vec<i32> {
    let n = scores.len();
    let mut indices = vec![0i32; n];
    let mut count: i32 = 0;
    unsafe {
        ffi::threshold_filter(
            scores.as_ptr(),
            n as i32,
            threshold,
            indices.as_mut_ptr(),
            &mut count,
        );
    }
    indices.truncate(count as usize);
    indices
}

/// Return (indices, scores) of the k highest-scoring elements.
pub fn top_k(scores: &[f32], k: usize) -> (Vec<i32>, Vec<f32>) {
    let n = scores.len();
    let actual_k = k.min(n);
    let mut out_indices = vec![0i32; actual_k];
    let mut out_scores = vec![0.0f32; actual_k];
    unsafe {
        ffi::top_k(
            scores.as_ptr(),
            n as i32,
            actual_k as i32,
            out_indices.as_mut_ptr(),
            out_scores.as_mut_ptr(),
        );
    }
    (out_indices, out_scores)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn norm_scalar(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_batch_dot() {
        let dim = 16;
        let query: Vec<f32> = (0..dim).map(|i| i as f32 * 0.1).collect();
        let vecs: Vec<f32> = (0..dim * 3).map(|i| (i % 7) as f32 * 0.1).collect();
        let scores = batch_dot(&query, &vecs, dim, 3);
        for v in 0..3 {
            let expected = dot_scalar(&query, &vecs[v * dim..(v + 1) * dim]);
            assert!(
                approx_eq(scores[v], expected, 1e-4),
                "vec {v}: got {}, expected {}",
                scores[v],
                expected
            );
        }
    }

    #[test]
    fn test_batch_cosine() {
        let dim = 16;
        let query: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let qnorm = norm_scalar(&query);
        let vecs: Vec<f32> = (0..dim * 2).map(|i| ((i % 5) as f32 + 1.0) * 0.1).collect();
        let scores = batch_cosine(&query, qnorm, &vecs, dim, 2);
        for v in 0..2 {
            let vec_slice = &vecs[v * dim..(v + 1) * dim];
            let expected = dot_scalar(&query, vec_slice) / (qnorm * norm_scalar(vec_slice));
            assert!(
                approx_eq(scores[v], expected, 1e-4),
                "vec {v}: got {}, expected {}",
                scores[v],
                expected
            );
        }
    }

    #[test]
    fn test_batch_l2() {
        let dim = 8;
        let query = vec![1.0f32; dim];
        let vecs = vec![2.0f32; dim * 2];
        let scores = batch_l2(&query, &vecs, dim, 2);
        // Each dimension contributes (1-2)^2 = 1, so total = dim
        for v in 0..2 {
            assert!(
                approx_eq(scores[v], dim as f32, 1e-4),
                "vec {v}: got {}",
                scores[v]
            );
        }
    }

    #[test]
    fn test_normalize_vectors() {
        let dim = 8;
        let mut vecs = vec![3.0f32; dim * 2];
        normalize_vectors(&mut vecs, dim, 2);
        for v in 0..2 {
            let norm = norm_scalar(&vecs[v * dim..(v + 1) * dim]);
            assert!(
                approx_eq(norm, 1.0, 1e-4),
                "vec {v}: norm = {}",
                norm
            );
        }
    }

    #[test]
    fn test_threshold_filter() {
        let scores = vec![0.1, 0.5, 0.9, 0.3, 0.8, 0.2];
        let indices = threshold_filter(&scores, 0.5);
        assert_eq!(indices, vec![1, 2, 4]);
    }

    #[test]
    fn test_top_k() {
        let scores = vec![0.1, 0.9, 0.5, 0.8, 0.3, 0.7];
        let (indices, top_scores) = top_k(&scores, 3);
        // Top 3 should be indices 1(0.9), 3(0.8), 5(0.7) — unordered
        let mut pairs: Vec<(i32, f32)> =
            indices.into_iter().zip(top_scores.into_iter()).collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        assert_eq!(pairs[0].0, 1);
        assert!(approx_eq(pairs[0].1, 0.9, 1e-4));
        assert_eq!(pairs[1].0, 3);
        assert!(approx_eq(pairs[1].1, 0.8, 1e-4));
        assert_eq!(pairs[2].0, 5);
        assert!(approx_eq(pairs[2].1, 0.7, 1e-4));
    }

    #[test]
    fn test_top_k_smaller_than_n() {
        let scores = vec![0.5, 0.3];
        let (indices, top_scores) = top_k(&scores, 5);
        assert_eq!(indices.len(), 2);
        assert_eq!(top_scores.len(), 2);
    }
}

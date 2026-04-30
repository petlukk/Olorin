//! Token sampling from logits — top-k → top-p → min-p → temperature → softmax.
//!
//! Sampler chain matches llama.cpp's default order. `argmax` is the greedy
//! shortcut when temperature is effectively 0. xorshift64 supplies the
//! deterministic-given-seed random stream for the categorical draw.

/// Pick a token id from `logits` using the chained sampler.
///
/// `temperature < 1e-6` short-circuits to greedy `argmax`. Otherwise:
/// 1. Top-k selection (O(n) scan + min-heap)
/// 2. Sort top-k descending
/// 3. Softmax to probabilities
/// 4. Top-p truncation (cumulative cutoff)
/// 5. Min-p truncation (relative-to-max cutoff)
/// 6. Temperature scaling on log-probs, re-softmax
/// 7. Categorical draw via xorshift64
pub(super) fn sample(
    logits: &mut [f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    rng: &mut u64,
) -> u32 {
    let n = logits.len();

    if temperature < 1e-6 {
        return argmax(logits);
    }

    // 1. Top-k selection: O(n) scan with a min-heap of the top_k largest.
    let mut candidates: Vec<(u32, f32)> = Vec::with_capacity(top_k + 1);
    for i in 0..n as u32 {
        let v = logits[i as usize];
        if candidates.len() < top_k {
            candidates.push((i, v));
            let mut c = candidates.len() - 1;
            while c > 0 {
                let p = (c - 1) / 2;
                if candidates[c].1 < candidates[p].1 { candidates.swap(c, p); c = p; } else { break; }
            }
        } else if v > candidates[0].1 {
            candidates[0] = (i, v);
            let mut p = 0;
            loop {
                let l = 2 * p + 1;
                let r = 2 * p + 2;
                let mut s = p;
                if l < candidates.len() && candidates[l].1 < candidates[s].1 { s = l; }
                if r < candidates.len() && candidates[r].1 < candidates[s].1 { s = r; }
                if s == p { break; }
                candidates.swap(p, s);
                p = s;
            }
        }
    }

    // 2. Sort the top-k candidates descending (≤ 64 elements typical)
    candidates.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // 3. Softmax (top-p / min-p need probabilities)
    let cmax = candidates[0].1;
    let mut sum = 0.0f32;
    for c in candidates.iter_mut() {
        c.1 = (c.1 - cmax).exp();
        sum += c.1;
    }
    let inv_sum = 1.0 / sum;
    for c in candidates.iter_mut() {
        c.1 *= inv_sum;
    }

    // 4. Top-p
    let mut cumulative = 0.0f32;
    let mut cutoff = candidates.len();
    for (i, c) in candidates.iter().enumerate() {
        cumulative += c.1;
        if cumulative > top_p {
            cutoff = i + 1;
            break;
        }
    }
    candidates.truncate(cutoff);

    // 5. Min-p
    let pmax = candidates[0].1;
    let min_thresh = min_p * pmax;
    candidates.retain(|c| c.1 >= min_thresh);
    if candidates.is_empty() {
        return argmax(logits);
    }

    // 6. Temperature scaling on log-probs, re-softmax
    if (temperature - 1.0).abs() > 1e-6 {
        for c in candidates.iter_mut() {
            c.1 = c.1.ln() / temperature;
        }
        let cmax2 = candidates.iter().map(|c| c.1).fold(f32::NEG_INFINITY, f32::max);
        let mut s2 = 0.0f32;
        for c in candidates.iter_mut() {
            c.1 = (c.1 - cmax2).exp();
            s2 += c.1;
        }
        let inv2 = 1.0 / s2;
        for c in candidates.iter_mut() {
            c.1 *= inv2;
        }
    } else {
        // Re-normalize after min-p truncation
        let s2: f32 = candidates.iter().map(|c| c.1).sum();
        let inv2 = 1.0 / s2;
        for c in candidates.iter_mut() {
            c.1 *= inv2;
        }
    }

    // 7. Categorical draw
    let r = xorshift64(rng);
    let threshold = (r as f64) / (u64::MAX as f64);
    let mut acc = 0.0f64;
    for c in &candidates {
        acc += c.1 as f64;
        if acc >= threshold {
            return c.0;
        }
    }

    candidates.last().unwrap().0
}

pub(super) fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

pub(super) fn xorshift_seed() -> u64 {
    let mut s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    if s == 0 { s = 0xDEAD_BEEF_CAFE_BABE; }
    s
}

pub(super) fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

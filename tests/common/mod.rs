//! Shared test helpers for gemma4_verify and friends.
//!
//! Cargo's convention: files under `tests/common/` aren't compiled as
//! separate test binaries. Pull this in from each test file via
//! `mod common;` then `use common::*;` or path-qualified access.

#![allow(dead_code)]

use std::path::Path;

pub mod llama_refs;

pub fn model_path() -> String {
    if let Ok(p) = std::env::var("OLORIN_MODEL_PATH") {
        return p;
    }
    let home = std::env::var("HOME").unwrap();
    format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")
}

pub fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

pub fn sum(v: &[f32]) -> f64 {
    v.iter().map(|&x| x as f64).sum::<f64>()
}

pub fn first4(v: &[f32]) -> String {
    format!("[{:.4}, {:.4}, {:.4}, {:.4}]", v[0], v[1], v[2], v[3])
}

pub fn bare_rmsnorm(x: &mut [f32], eps: f32) {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>();
    let scale = 1.0 / ((ss / n as f32) + eps).sqrt();
    for v in x.iter_mut() { *v *= scale; }
}

pub fn compute_rope_tables(
    cos: &mut [f32], sin: &mut [f32],
    pos: usize, n_rot: usize, theta: f32, ff: Option<&[f32]>,
) {
    let half = n_rot / 2;
    for d in 0..half {
        let base_freq = 1.0 / theta.powf(2.0 * d as f32 / n_rot as f32);
        let freq = match ff { Some(f) => base_freq / f[d], None => base_freq };
        let angle = pos as f32 * freq;
        cos[d] = angle.cos();
        sin[d] = angle.sin();
    }
}

pub fn has_model() -> bool {
    Path::new(&model_path()).exists()
}

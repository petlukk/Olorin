//! Bit-exact verification of olorin's batched prompt-eval against llama.cpp.
//!
//! Run: cargo test --release --test gemma4_batch_verify -- --nocapture --test-threads=1
//!
//! Each test compares an olorin intermediate against a value captured from
//! llama-eval-callback dumps. Sums use f64 accumulation to avoid the f32
//! ordering trap that bit us in step6.

use std::path::Path;

fn model_path() -> String {
    let home = std::env::var("HOME").unwrap();
    format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")
}

fn has_model() -> bool {
    Path::new(&model_path()).exists()
}

#[allow(dead_code)]
fn sum_f64(v: &[f32]) -> f64 {
    v.iter().map(|&x| x as f64).sum::<f64>()
}

#[test]
fn batch0_skeleton() {
    if !has_model() {
        eprintln!("SKIP: no model");
        return;
    }
    eprintln!("=== batch0: skeleton — no batched code yet ===");
    // Sanity: model can be loaded and forward_batch over [BOS] equals forward_one(BOS).
    let gguf = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();
    let pool = olorin::inference::threadpool::ThreadPool::new();

    let mut a = olorin::inference::forward::Gemma4State::new(&model, 512, &pool);
    let logits_one = a.forward_one(&model, 2, &pool).to_vec();

    let mut b = olorin::inference::forward::Gemma4State::new(&model, 512, &pool);
    let logits_batch = b.forward_batch(&model, &[2u32], &pool).to_vec();

    assert_eq!(logits_one.len(), logits_batch.len());
    let max_abs_diff = logits_one
        .iter()
        .zip(logits_batch.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("max abs diff = {}", max_abs_diff);
    assert!(max_abs_diff < 1e-6, "skeleton forward_batch should equal forward_one for [BOS]");

    eprintln!("PASS: batch0");
}

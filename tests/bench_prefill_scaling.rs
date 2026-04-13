//! Prefill scaling benchmark: measures prefill t/s at different prompt lengths.
//! Run: cargo test --release --test bench_prefill_scaling -- --nocapture

use std::path::Path;
use std::time::Instant;

fn model_path() -> String {
    std::env::var("OLORIN_MODEL")
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_default();
            format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")
        })
}

#[test]
fn prefill_scaling() {
    if !Path::new(&model_path()).exists() {
        eprintln!("SKIP: model not present");
        return;
    }

    let gguf = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    let tokenizer = olorin::inference::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();
    let pool = olorin::inference::threadpool::ThreadPool::new();
    let graph_pool = olorin::inference::threadpool::GraphPool::new();
    let mut state = olorin::inference::forward::Gemma4State::new(&model, 512, &pool);

    // Build a long prompt by repeating text
    let base = "The quick brown fox jumps over the lazy dog and then runs across the field. ";
    let long_text = base.repeat(20);
    let all_ids: Vec<u32> = tokenizer.encode(&long_text);

    eprintln!("\n=== prefill scaling benchmark ===");
    eprintln!("total available tokens: {}", all_ids.len());

    for &n in &[8, 16, 32, 64, 128, 256] {
        if n > all_ids.len() { break; }
        state.reset();
        let mut prompt: Vec<u32> = vec![2]; // BOS
        prompt.extend_from_slice(&all_ids[..n]);

        let t0 = Instant::now();
        let _ = state.forward_batch(&model, &prompt, &graph_pool);
        let secs = t0.elapsed().as_secs_f64();
        let tps = prompt.len() as f64 / secs;
        eprintln!("  N={:>3}:  {:.2} ms  ({:.2} t/s)", prompt.len(), secs * 1000.0, tps);
    }
}

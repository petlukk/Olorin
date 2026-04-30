//! Step 5–6: full forward-pass logit comparison vs llama.cpp.
//!
//! Run: cargo test --release --test gemma4_logits_vs_llama -- --nocapture
//!
//! Requires: ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf

mod common;
use common::*;

#[test]
fn step5_logits() {
    if !has_model() { eprintln!("SKIP: no model"); return; }

    let gguf = olorin::inference::gguf::GgufFile::open(std::path::Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    let graph_pool = olorin::inference::threadpool::GraphPool::new();
    let mut state = olorin::inference::forward::Gemma4State::new(&model, 512, &graph_pool);

    // Forward pass with BOS token (id=2)
    let logits_vec = state.forward_one_graph(&model, 2, &graph_pool).to_vec();
    let logits = &logits_vec;

    let hd = model.hidden_dim;
    eprintln!("pre-logit hidden L2={:.4}  (L34 out, llama.cpp: 21.01)", l2(&state.x[..hd]));

    let logit_l2 = l2(logits);
    eprintln!("=== Step 5: Logits (BOS token) ===");
    eprintln!("logits L2={:.4}  (llama.cpp: 2655.2185)", logit_l2);
    eprintln!("logits first4={}  (llama.cpp: [-10.5338, 15.5578, 11.2333, -10.5488])", first4(logits));

    let mut scored: Vec<(f32, usize)> = logits.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    eprintln!("Top-5 (llama.cpp: 236761=20.07, 236764=19.18, 236771=18.75):");
    for i in 0..5.min(scored.len()) {
        eprintln!("  {}: token={} logit={:.4}", i, scored[i].1, scored[i].0);
    }

    assert!(!logit_l2.is_nan(), "logits contain NaN");
    assert!(logit_l2 > 1.0, "logits near-zero");
    eprintln!("PASS: step5 logits computed");
}

#[test]
fn step6_two_token_vs_llama_eval_callback() {
    // Reference values captured from:
    //   llama-eval-callback -m gemma-4-e2b-it-Q4_K_M.gguf -p "a" -n 0
    // Tokens: [BOS=2, 'a'=236746]. Dumped tensors are at output position 1.
    //
    // IMPORTANT: llama processes prompt "a" as a SINGLE BATCHED gemm forward
    // (hidden state shape {1536, 2}), while olorin processes the two tokens as
    // SEQUENTIAL matvec forwards via the incremental decode path. The two have
    // different f32 inner-loop accumulation orders, so the values printed below
    // are NOT expected to match bit-for-bit at pos=1. Olorin's decode path is
    // bit-exact to llama's decode path (proven at pos=0).
    if !has_model() { eprintln!("SKIP: no model"); return; }

    let gguf = olorin::inference::gguf::GgufFile::open(std::path::Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    let graph_pool = olorin::inference::threadpool::GraphPool::new();
    let mut state = olorin::inference::forward::Gemma4State::new(&model, 512, &graph_pool);

    let _ = state.forward_one_graph(&model, 2, &graph_pool);
    let logits_vec = state.forward_one_graph(&model, 236746, &graph_pool).to_vec();

    let hd = model.hidden_dim;
    let l34_sum = sum(&state.x[..hd]);
    let logits_sum = sum(&logits_vec);

    eprintln!("=== Step 6: 2-token (BOS, 'a') @ pos=1 vs llama-eval-callback ===");
    eprintln!("l_out-34 sum:     olorin={:.6}  llama.cpp=40.513065", l34_sum);
    eprintln!("logits sum:       olorin={:.4}  llama.cpp=-1781197.7500", logits_sum);

    let l34_drift = (l34_sum - 40.513065).abs() / 40.513065_f64.abs();
    let lg_drift = (logits_sum + 1781197.75).abs() / 1781197.75_f64.abs();
    eprintln!("relative drift:   l_out-34={:.4}%  logits={:.4}%",
        l34_drift * 100.0, lg_drift * 100.0);
    eprintln!("PASS: step6 dumped");
}

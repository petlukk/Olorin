//! Bit-exact gate: forward_batch(&[BOS]) must produce identical logits
//! to forward_one_graph(BOS). If this fails, forward_batch has a bug.
//!
//! Requires: ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf

use std::path::Path;

fn model_path() -> String {
    if let Ok(p) = std::env::var("OLORIN_MODEL_PATH") {
        return p;
    }
    let home = std::env::var("HOME").unwrap();
    format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")
}

#[test]
fn forward_batch_n1_matches_forward_one_graph() {
    let mp = model_path();
    if !Path::new(&mp).exists() {
        eprintln!("SKIP: model not found at {mp}");
        return;
    }

    // Spawn with 32 MB stack — forward_batch's call chain is deep.
    let handle = std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(move || run_comparison(&mp))
        .unwrap();
    handle.join().unwrap();
}

fn run_comparison(mp: &str) {
    // Force full vocab so both paths produce the same number of logits
    std::env::set_var("OLORIN_FULL_VOCAB", "1");
    olorin::kernels::ffi::init().unwrap();

    let gguf = olorin::inference::gguf::GgufFile::open(Path::new(mp)).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();

    let graph_pool = olorin::inference::threadpool::GraphPool::new();
    let max_seq = 2048;
    let bos: u32 = 2;

    // Path A: forward_one_graph with BOS
    let mut state_a = olorin::inference::forward::Gemma4State::new(&model, max_seq, &graph_pool);
    let logits_a = state_a.forward_one_graph(&model, bos, &graph_pool).to_vec();

    // Path B: forward_batch with &[BOS]
    let mut state_b = olorin::inference::forward::Gemma4State::new(&model, max_seq, &graph_pool);
    let logits_b = state_b.forward_batch(&model, &[bos], &graph_pool).to_vec();

    // Compare bit-exact
    assert_eq!(logits_a.len(), logits_b.len(), "logits length mismatch");
    let mut mismatches = 0;
    for i in 0..logits_a.len() {
        if logits_a[i].to_bits() != logits_b[i].to_bits() {
            if mismatches < 10 {
                eprintln!(
                    "MISMATCH logit[{i}]: graph={:.6} batch={:.6} (diff={:.2e})",
                    logits_a[i],
                    logits_b[i],
                    (logits_a[i] - logits_b[i]).abs()
                );
            }
            mismatches += 1;
        }
    }

    if mismatches > 0 {
        let l2: f32 = logits_a
            .iter()
            .zip(&logits_b)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt();
        eprintln!(
            "L2 distance: {l2:.6} ({mismatches} mismatches out of {} logits)",
            logits_a.len()
        );

        let mut a_sorted: Vec<(usize, f32)> = logits_a.iter().copied().enumerate().collect();
        let mut b_sorted: Vec<(usize, f32)> = logits_b.iter().copied().enumerate().collect();
        a_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        b_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        eprintln!("graph top-3: {:?}", &a_sorted[..3]);
        eprintln!("batch top-3: {:?}", &b_sorted[..3]);
    }

    assert_eq!(
        mismatches, 0,
        "forward_batch(N=1) must be bit-exact with forward_one_graph"
    );
    eprintln!(
        "PASS: forward_batch(N=1) bit-exact match ({} logits)",
        logits_a.len()
    );
}

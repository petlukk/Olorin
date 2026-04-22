//! Correctness: running forward_batch with OLORIN_PREFILL_UBATCH=K on a
//! prompt longer than K must produce logits bit-identical to the
//! non-ubatched single-shot path. If it doesn't, ubatch has a KV-cache
//! or cross-chunk attention bug.

use std::path::Path;

fn model_path() -> String {
    std::env::var("OLORIN_MODEL").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")
    })
}

#[test]
fn ubatch_logits_match_full_batch() {
    if !Path::new(&model_path()).exists() {
        eprintln!("SKIP: model not present");
        return;
    }

    let h = std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(inner)
        .unwrap();
    h.join().unwrap();
}

fn inner() {
    // Make sure env var is clean for baseline.
    unsafe { std::env::remove_var("OLORIN_PREFILL_UBATCH"); }

    let gguf = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    let tokenizer = olorin::inference::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    let graph_pool = olorin::inference::threadpool::GraphPool::new();

    // Build a ~260-token prompt so ubatch=64 forces 5 chunks
    // (64 + 64 + 64 + 64 + tail). Covers the N=257 regression path plus
    // multiple intermediate ubatch boundaries.
    let base = "The quick brown fox jumps over the lazy dog and then runs across the field. ";
    let text = base.repeat(20);
    let ids: Vec<u32> = tokenizer.encode(&text);
    let mut prompt: Vec<u32> = vec![2]; // BOS
    prompt.extend_from_slice(&ids[..260.min(ids.len())]);
    assert!(prompt.len() >= 200, "need a prompt >= 200 tokens, got {}", prompt.len());

    let mut state = olorin::inference::forward::Gemma4State::new(&model, 512, &graph_pool);

    // ── Baseline: single-shot forward_batch ───────────────────────────
    state.reset();
    let baseline = state.forward_batch(&model, &prompt, &graph_pool).to_vec();

    // ── Under test: ubatch=64 ─────────────────────────────────────────
    unsafe { std::env::set_var("OLORIN_PREFILL_UBATCH", "64"); }
    state.reset();
    let ubatched = state.forward_batch(&model, &prompt, &graph_pool).to_vec();
    unsafe { std::env::remove_var("OLORIN_PREFILL_UBATCH"); }

    assert_eq!(baseline.len(), ubatched.len(), "logit length mismatch");

    // Bit-exact check first. If that fails, print the top-8 argmax positions
    // of both so divergence is obvious.
    let differ: Vec<usize> = baseline.iter().zip(ubatched.iter())
        .enumerate()
        .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
        .map(|(i, _)| i)
        .take(8)
        .collect();
    if !differ.is_empty() {
        eprintln!("first diverging logit indices: {differ:?}");
        for &i in &differ {
            eprintln!("  idx {i:6}: baseline={:.6}  ubatch={:.6}  Δ={:+.6}",
                baseline[i], ubatched[i], ubatched[i] - baseline[i]);
        }
        let top = |v: &[f32]| {
            let mut idx: Vec<usize> = (0..v.len()).collect();
            idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
            idx[..8].to_vec()
        };
        eprintln!("baseline top-8 argmax: {:?}", top(&baseline));
        eprintln!("ubatch   top-8 argmax: {:?}", top(&ubatched));
    }
    assert!(differ.is_empty(), "ubatch logits diverge from baseline");
}

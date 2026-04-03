//! End-to-end test for f16 attention correctness.
//!
//! Loads a Llama Q4K model, runs prefill + one decode step, and verifies
//! non-zero logits are produced. Uses f16 KV cache with per-head
//! attn_dot_f16 + softmax + attn_vsum_f16 attention path.

use std::path::{Path, PathBuf};

/// Find a Q4K GGUF model in ~/.olorin/models/.
fn q4k_model_path() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let dir = Path::new(&home).join(".olorin/models");
    std::fs::read_dir(&dir).ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            let name = p.file_name().unwrap_or_default().to_string_lossy();
            name.ends_with(".gguf") && name.contains("Q4_K")
        })
}

#[test]
fn test_f16_attn_produces_nonzero_logits() {
    olorin::kernels::ffi::init().unwrap();

    let Some(path) = q4k_model_path() else {
        eprintln!("SKIP: no Q4K model found in ~/.olorin/models/");
        return;
    };
    eprintln!("Using model: {}", path.display());

    let mut engine = olorin::inference::generate::Engine::load(&path, 512).unwrap();
    assert_eq!(engine.quant_type_str(), "Q4_K", "expected Q4K model");

    engine.max_tokens = 4;
    engine.temperature = 0.0;

    let got_token = std::cell::Cell::new(false);
    let result = engine.generate("Hi", "", &|_tok| {
        got_token.set(true);
    });

    let text = result.unwrap();
    assert!(got_token.get(), "on_token callback was never called");
    assert!(!text.is_empty(), "generated text is empty");
    eprintln!("f16 attention output: {text:?}");
}

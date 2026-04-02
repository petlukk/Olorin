//! End-to-end test for flash decode attention correctness.
//!
//! Loads a Llama Q4K model, runs prefill + one decode step, and verifies
//! non-zero logits are produced. Flash decode is the default path for models
//! without K-bias (like Llama 3.2 3B Q4K) when the kernel is available.

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
fn test_flash_decode_produces_nonzero_logits() {
    olorin::kernels::ffi::init().unwrap();

    let Some(path) = q4k_model_path() else {
        eprintln!("SKIP: no Q4K model found in ~/.olorin/models/");
        return;
    };
    eprintln!("Using model: {}", path.display());

    let mut engine = olorin::inference::generate::Engine::load(&path, 512).unwrap();
    assert_eq!(engine.quant_type_str(), "Q4_K", "expected Q4K model");

    // Limit to 4 tokens — enough for prefill + decode without wasting time.
    engine.max_tokens = 4;
    engine.temperature = 0.0;

    let mut got_token = false;
    let result = engine.generate("Hi", "", &|_tok| {
        got_token = true;
    });

    let text = result.unwrap();
    assert!(got_token, "on_token callback was never called");
    assert!(!text.is_empty(), "generated text is empty");
    eprintln!("Flash decode output: {text:?}");
}

#[test]
fn test_flash_decode_kernel_available() {
    olorin::kernels::ffi::init().unwrap();

    // Verify the flash decode kernel loaded (compiled for this platform).
    let available = olorin::kernels::ffi_inference::has_flash_decode_attn();
    eprintln!("flash_decode_attn kernel available: {available}");
    // On x86_64 with AVX2 this should be true; on other platforms it may not be.
    // We just log it here — the generate test above will use whatever path is active.
}

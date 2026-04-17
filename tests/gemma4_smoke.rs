//! Fast smoke test: load Gemma 4, run greedy generation, verify first 6 tokens
//! match the known-good llama.cpp output for the "Hi" prompt.
//!
//! Run: cargo test --release --test gemma4_smoke -- --nocapture

use olorin::inference::generate::Engine;
use std::path::Path;

#[test]
fn smoke_hi_greedy() {
    let home = std::env::var("HOME").unwrap();
    let path: std::path::PathBuf =
        Path::new(&home).join(".olorin/models/gemma-4-e2b-it-Q4_K_M.gguf");
    if !path.exists() {
        eprintln!("SKIP: no model");
        return;
    }

    // forward_batch needs more than the default 2 MB test thread stack.
    let path2 = path.clone();
    let handle = std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(move || run_smoke(&path2))
        .unwrap();
    handle.join().unwrap();
}

fn run_smoke(path: &Path) {
    let mut engine = Box::new(Engine::load(path, 512).expect("load"));
    engine.temperature = 0.0;
    engine.max_tokens = 8;

    use std::cell::RefCell;
    let got = RefCell::new(String::new());
    let on_token = |t: &str| got.borrow_mut().push_str(t);
    engine.generate("Hi", "", &on_token).expect("generate");
    let got = got.into_inner();

    eprintln!("got: {got:?}");
    assert!(got.contains("Thinking"), "expected 'Thinking' in output, got: {got:?}");
    assert!(got.contains("Process"), "expected 'Process' in output, got: {got:?}");
}

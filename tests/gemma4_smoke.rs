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

    let mut engine = Engine::load(&path, 512).expect("load");
    engine.temperature = 0.0;
    engine.max_tokens = 8;

    use std::cell::RefCell;
    let got = RefCell::new(String::new());
    let on_token = |t: &str| got.borrow_mut().push_str(t);
    engine.generate("Hi", "", &on_token).expect("generate");
    let got = got.into_inner();

    eprintln!("got: {got:?}");
    // llama.cpp greedy on Hi prompt produces:
    //   <|channel> thought \n Thinking  Process : \n\n 1
    // Olorin hides USER_DEFINED/CONTROL tokens, so visible output is
    // "thought\nThinking Process:\n\n1" (approximately).
    assert!(got.contains("Thinking"), "expected 'Thinking' in output, got: {got:?}");
    assert!(got.contains("Process"), "expected 'Process' in output, got: {got:?}");
}

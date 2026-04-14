//! Greedy parity: speculative decoding must produce bit-identical tokens
//! to non-speculative under temperature = 0.
//!
//! Prompt-lookup speculative decoding is only correct if drafts are emitted
//! exactly when they equal argmax. Under greedy sampling, that means the
//! output stream MUST be bit-identical to the non-speculative path.
//!
//! Run: cargo test --release --test speculative_parity -- --test-threads=1 --nocapture

use olorin::inference::generate::{resolve_model, Engine};
use std::sync::Mutex;

fn load_engine(draft_k: usize) -> Option<Engine> {
    let model_path = resolve_model(Some("gemma4"))?;
    if !model_path.exists() {
        return None;
    }
    let mut engine = Engine::load(&model_path, 2048).expect("engine load");
    engine.temperature = 0.0;
    engine.max_tokens = 96;
    engine.draft_k = draft_k;
    Some(engine)
}

fn capture(engine: &mut Engine, prompt: &str) -> String {
    let buf = Mutex::new(String::new());
    let cb = |t: &str| {
        buf.lock().unwrap().push_str(t);
    };
    engine.generate(prompt, "", &cb).expect("generate ok");
    buf.into_inner().unwrap()
}

fn parity_for_prompt(prompt: &str) {
    let Some(mut base_engine) = load_engine(0) else {
        eprintln!("SKIP: no gemma4 model under ~/.olorin/models/");
        return;
    };
    let baseline = capture(&mut base_engine, prompt);
    drop(base_engine);

    let mut spec4_engine = load_engine(4).expect("engine load");
    let spec4 = capture(&mut spec4_engine, prompt);
    drop(spec4_engine);

    let mut spec8_engine = load_engine(8).expect("engine load");
    let spec8 = capture(&mut spec8_engine, prompt);
    drop(spec8_engine);

    assert_eq!(baseline, spec4, "parity failure draft_k=4 on {prompt:?}");
    assert_eq!(baseline, spec8, "parity failure draft_k=8 on {prompt:?}");
}

#[test]
fn parity_code_prompt() {
    parity_for_prompt("Write a Python hello world script.");
}

#[test]
fn parity_prose_prompt() {
    parity_for_prompt("In two sentences, what is Rust?");
}

#[test]
fn parity_json_prompt() {
    parity_for_prompt("Return a JSON object with keys a, b, c set to 1, 2, 3.");
}

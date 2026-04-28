//! Fast smoke test: load Gemma 4, run greedy generation, verify the model
//! produces a coherent greeting with thinking properly suppressed.
//!
//! Run: cargo test --release --test gemma4_smoke -- --nocapture

use olorin::inference::generate::{Engine, GenEvent};
use std::path::Path;
use std::sync::atomic::{AtomicU32, Ordering};

#[test]
fn smoke_hi_greedy() {
    let path: std::path::PathBuf = if let Ok(p) = std::env::var("OLORIN_MODEL_PATH") {
        std::path::PathBuf::from(p)
    } else {
        let home = std::env::var("HOME").unwrap();
        Path::new(&home).join(".olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")
    };
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
    // "Hi" must decode many tokens to exercise both thinking and answer phases;
    // max_tokens=8 would cut off mid-thought.
    let mut engine = Box::new(Engine::load(path, 1024).expect("load"));
    engine.temperature = 0.0;
    engine.max_tokens = 200;

    use std::cell::RefCell;
    let got = RefCell::new(String::new());
    let think_events = AtomicU32::new(0);
    let on_event = |ev: GenEvent| match ev {
        GenEvent::Token(t) => got.borrow_mut().push_str(t),
        GenEvent::Thinking(_) => { think_events.fetch_add(1, Ordering::Relaxed); }
    };
    engine.generate("Hi", "", &on_event).expect("generate");
    let got = got.into_inner();

    eprintln!("got: {got:?}");
    eprintln!("think events: {}", think_events.load(Ordering::Relaxed));

    // Thinking must be exercised (open + close = 2 events).
    assert!(
        think_events.load(Ordering::Relaxed) >= 2,
        "expected thinking to open and close at least once"
    );
    // Thinking text must NOT leak into user-visible output.
    assert!(
        !got.contains("Thinking Process"),
        "thinking content leaked into output: {got:?}"
    );
    // User-visible output should greet the user.
    let lower = got.to_lowercase();
    assert!(
        lower.contains("hi") || lower.contains("hello") || lower.contains("there"),
        "expected greeting in visible output, got: {got:?}"
    );
}

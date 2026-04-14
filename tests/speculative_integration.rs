//! Six-prompt integration regression under speculative decoding.
//! All six must produce non-empty output under draft_k=4, temperature=0 (deterministic).

use olorin::inference::generate::{Engine, resolve_model};
use std::sync::Mutex;

fn run(engine: &mut Engine, prompt: &str) -> String {
    let buf = Mutex::new(String::new());
    let cb = |t: &str| { buf.lock().unwrap().push_str(t); };
    engine.generate(prompt, "", &cb).unwrap();
    buf.into_inner().unwrap()
}

#[test]
#[ignore]
fn six_prompt_regression() {
    let Some(model_path) = resolve_model(Some("gemma4")) else {
        eprintln!("skipping — no gemma4 model");
        return;
    };
    let mut engine = Engine::load(&model_path, 2048).unwrap();
    engine.temperature = 0.0;
    engine.max_tokens = 200;
    engine.draft_k = 4;

    // Mix of short, medium, structured, and open-ended — all prompts the
    // bare engine can answer (no router tool intercepts).
    let prompts = [
        "the capital of France is?",
        "tell me a joke",
        "give me a short recipe for pancakes",
        "name three colors of the rainbow",
        "who wrote Hamlet?",
        "explain gravity in one sentence",
    ];

    for p in prompts {
        let out = run(&mut engine, p);
        assert!(!out.trim().is_empty(), "empty response on prompt: {p:?}");
        eprintln!("[integration] prompt={p:?} out_len={}", out.len());
    }
}

//! Wall-clock speedup for speculative decoding across three workload types.
//! Run explicitly: cargo test --release --test bench_speculative -- --ignored --nocapture

use olorin::inference::generate::{resolve_model, Engine};
use std::sync::Mutex;
use std::time::Instant;

fn run(engine: &mut Engine, prompt: &str) -> (String, u128) {
    let buf = Mutex::new(String::new());
    let cb = |t: &str| {
        buf.lock().unwrap().push_str(t);
    };
    let t0 = Instant::now();
    engine.generate(prompt, "", &cb).expect("generate");
    (buf.into_inner().unwrap(), t0.elapsed().as_millis())
}

fn bench_prompt(label: &str, prompt: &str) {
    let Some(model_path) = resolve_model(Some("gemma4")) else {
        eprintln!("[bench] skipping — no gemma4 model");
        return;
    };

    let mut baseline_out: Option<String> = None;
    let mut baseline_ms = 0u128;

    for &k in &[0usize, 4, 8] {
        let mut e = Engine::load(&model_path, 2048).unwrap();
        e.temperature = 0.0;
        e.max_tokens = 128;
        e.draft_k = k;
        let (out, ms) = run(&mut e, prompt);
        if k == 0 {
            baseline_ms = ms;
            baseline_out = Some(out);
            eprintln!("[bench] {label} K=0: {ms}ms (baseline)");
        } else {
            assert_eq!(
                baseline_out.as_deref(),
                Some(out.as_str()),
                "{label} K={k}: parity broke"
            );
            let speedup = baseline_ms as f64 / ms.max(1) as f64;
            eprintln!("[bench] {label} K={k}: {ms}ms, speedup {speedup:.2}x");
        }
    }
}

#[test]
#[ignore]
fn bench_all() {
    bench_prompt(
        "code",
        "Write a Python function that reverses a linked list.",
    );
    bench_prompt("chat", "What's your favorite color and why?");
    bench_prompt(
        "json",
        "Produce a JSON array of 5 objects, each with id and name fields.",
    );
}

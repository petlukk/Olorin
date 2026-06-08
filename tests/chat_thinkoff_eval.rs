//! Does turning thinking OFF for chat hold quality once the minimal system
//! prompt is in play? With the ~2.6 KB runes-block prefill gone (aarch64 uses
//! MINIMAL_SYSTEM_PROMPT), the residual chat latency is almost all thinking
//! tokens. This runs the production aarch64 chat prompt across a factual /
//! reasoning / code prompt with thinking ON vs OFF, measuring wall-time and
//! capturing the answer for a quality eyeball.
//!
//!   OLORIN_THREADS=3 \
//!   OLORIN_PROBE_MODEL=~/.olorin/models/gemma-4-e2b-it-Q4_K_M-q3kffnimpl.gguf \
//!   ./chat_thinkoff_eval --ignored --nocapture

use olorin::core::llm::MINIMAL_SYSTEM_PROMPT;
use olorin::inference::generate::{Engine, GenEvent};
use std::cell::RefCell;
use std::path::Path;
use std::time::Instant;

const PROMPTS: &[(&str, &str)] = &[
    ("factual",   "What is the capital of Australia?"),
    ("reasoning", "A train leaves at 14:30 and arrives at 17:15. It stopped \
                   twice for 8 minutes each. How many minutes was it actually moving?"),
    ("code",      "Write a Rust function that returns the nth Fibonacci number \
                   iteratively."),
];

#[test]
#[ignore = "model eval; run explicitly with --ignored --nocapture"]
fn chat_minimal_thinking_on_vs_off() {
    let model = std::env::var("OLORIN_PROBE_MODEL").unwrap_or_else(|_| {
        format!("{}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf",
            std::env::var("HOME").unwrap())
    });
    if !Path::new(&model).exists() { eprintln!("SKIP: no model at {model}"); return; }
    olorin::kernels::ffi::init().unwrap();
    let m = model.clone();
    std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(move || drive(&m))
        .unwrap()
        .join()
        .unwrap();
}

fn drive(model: &str) {
    let mut engine = Box::new(Engine::load(Path::new(model), 2048).expect("load"));
    let system = MINIMAL_SYSTEM_PROMPT; // production aarch64 chat prompt

    for (label, prompt) in PROMPTS {
        for thinking in [true, false] {
            engine.thinking = thinking;
            let buf = RefCell::new(String::new());
            let on = |ev: GenEvent| if let GenEvent::Token(t) = ev {
                buf.borrow_mut().push_str(t);
            };
            let t0 = Instant::now();
            let _ = engine.generate(prompt, system, &on);
            let secs = t0.elapsed().as_secs_f64();
            let out = buf.into_inner();
            let out = out.trim();
            println!(
                "\n===== {label}  thinking={thinking}  wall={secs:.1}s  answer_chars={} =====",
                out.len()
            );
            println!("{out}");
        }
    }
    println!("\n===== END =====");
}

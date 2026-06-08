//! Prototype: does a minimal chat system prompt cut latency without hurting
//! chat quality? The prefill probe showed the ~2.6 KB runes_prompt_block costs
//! ~30s of prefill per turn (vs ~2s with an empty system) — and on the Pi the
//! forward pass can't emit tool calls anyway, so that framing is dead weight.
//!
//! This runs three representative chat prompts through the REAL runes system
//! prompt vs MINIMAL_SYSTEM_PROMPT, thinking ON (chat's production default),
//! measuring wall-time and capturing the answer for a quality eyeball.
//!
//!   OLORIN_THREADS=3 \
//!   OLORIN_PROBE_MODEL=~/.olorin/models/gemma-4-e2b-it-Q4_K_M-q3kffnimpl.gguf \
//!   ./chat_minimal_system_eval --ignored --nocapture

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
fn chat_minimal_vs_full_system() {
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
    // Chat's production default is thinking ON; hold it fixed so we isolate the
    // system-prompt variable.
    engine.thinking = true;
    let systems: [(&str, &str); 2] = [
        ("full_runes", olorin::runes::runes_prompt_block()),
        ("minimal",    MINIMAL_SYSTEM_PROMPT),
    ];

    for (plabel, prompt) in PROMPTS {
        for (slabel, system) in systems {
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
                "\n===== {plabel} / {slabel}  wall={secs:.1}s  answer_chars={} =====",
                out.len()
            );
            println!("{out}");
        }
    }
    println!("\n===== END =====");
}

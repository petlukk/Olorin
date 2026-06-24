//! Experiment: how much of narration latency is Gemma 4's hidden chain-of-
//! thought? Runs the real narration input (an eatime spike summary +
//! NARRATION_SYSTEM_PROMPT) through generate() with thinking on vs off,
//! measuring wall-time and the answer, two rounds each.
//!
//! Run on the Pi for production-representative numbers:
//!   OLORIN_THREADS=3 \
//!   OLORIN_PROBE_MODEL=~/.olorin/models/gemma-4-e2b-it-Q4_K_M-q3kffnimpl.gguf \
//!   ./narration_thinking_eval --ignored --nocapture

use olorin::inference::generate::{Engine, GenEvent};
use std::cell::RefCell;
use std::path::Path;
use std::time::Instant;

// Matches router_tools::NARRATION_SYSTEM_PROMPT.
const NARRATION_SYSTEM: &str =
    "You are a helpful data analyst. Read the user's tool output and reply with \
     1-2 plain-English sentences naming the single most important finding. Be \
     concrete: give the actual date/time, category, or value and the magnitude \
     of any peak, spike, or anomaly — e.g. 'X peaked on <date> at about N× the \
     baseline' — not vague phrasing like 'a significant peak around a certain \
     time'. State the headline finding; do not reproduce the table or list \
     every row.";

// The exact shape build_narration_prompt produces: "Output of `<rune>`:\n\n<answer>".
const NARRATION_INPUT: &str = "Output of `eatime`:\n\n\
bytes:       45.02 MB\n\
timestamps:  436781\n\
buckets:     120\n\
scan:        14 µs\n\n\
span:        1995-07-11T00:00:00 .. 1995-07-15T23:00:00\n\
peak bucket: 1995-07-13T09:00:00 (14926 timestamps)\n\n\
anomalies:   4 spike(s) detected\n\
  1995-07-13T08:00:00 count=11567 (3.8× baseline 3080)\n\
  1995-07-13T09:00:00 count=14926 (4.8× baseline 3080)\n\
  1995-07-13T10:00:00 count=10214 (3.3× baseline 3080)\n\
  1995-07-13T11:00:00 count=9748 (3.2× baseline 3080)";

#[test]
#[ignore = "model eval; run explicitly with --ignored --nocapture"]
fn narration_thinking_on_vs_off() {
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
    engine.max_tokens = 768; // the production narration decode cap

    for round in 1..=2 {
        for thinking in [true, false] {
            engine.thinking = thinking;
            let buf = RefCell::new(String::new());
            let on = |ev: GenEvent| if let GenEvent::Token(t) = ev {
                buf.borrow_mut().push_str(t);
            };
            let t0 = Instant::now();
            let _ = engine.generate(NARRATION_INPUT, NARRATION_SYSTEM, &on);
            let secs = t0.elapsed().as_secs_f64();
            let out = buf.into_inner();
            let out = out.trim();
            println!(
                "\n===== round {round}  thinking={thinking}  wall={secs:.1}s  answer_chars={} =====",
                out.len()
            );
            println!("{out}");
        }
    }
    println!("\n===== END =====");
}

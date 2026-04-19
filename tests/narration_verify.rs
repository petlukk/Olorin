//! Verify fix: narrating structured tool output no longer collapses into
//! a repetition loop. Pre-fix (32K hot vocab default), Olorin degenerated
//! into `= \mathbf{"count"} = \mathbf{"count"} ...`. Post-fix (full vocab),
//! output tracks llama.cpp's coherent markdown narration.
//!
//! Run: cargo test --release --test narration_verify -- --ignored --nocapture

use olorin::inference::generate::{Engine, GenEvent};
use std::path::Path;

#[test]
#[ignore = "run explicitly with --ignored; needs model"]
fn narration_has_variety_and_no_loop() {
    let home = std::env::var("HOME").unwrap();
    let path: std::path::PathBuf =
        Path::new(&home).join(".olorin/models/gemma-4-e2b-it-Q4_K_M.gguf");
    if !path.exists() {
        eprintln!("SKIP: no model");
        return;
    }

    let path2 = path.clone();
    std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || run(&path2))
        .unwrap()
        .join()
        .unwrap();
}

fn run(path: &Path) {
    let mut engine = Box::new(Engine::load(path, 1024).expect("load"));
    engine.temperature = 0.0;
    engine.top_k = 1;
    engine.top_p = 1.0;
    engine.min_p = 0.0;
    engine.repetition_penalty = 1.0;
    engine.max_tokens = 160;

    let prompt = r#"I ran the eastat rune on the employees.csv file and got these column stats. Please explain them to me in plain English.

<rune_output rune="eastat" untrusted="true">
{
  "columns": [
    {"name": "age",          "count": 1500, "mean": 38.4,    "std": 9.7,     "min": 21,    "max": 67,     "p25": 31,    "p50": 37,    "p75": 45},
    {"name": "salary",       "count": 1500, "mean": 72843.5, "std": 21456.8, "min": 28000, "max": 198000, "p25": 58000, "p50": 70500, "p75": 87000},
    {"name": "tenure_years", "count": 1500, "mean": 6.2,     "std": 4.8,     "min": 0.0,   "max": 32.5,   "p25": 2.3,   "p50": 5.1,   "p75": 9.4}
  ]
}
</rune_output>"#;

    use std::cell::RefCell;
    let got = RefCell::new(String::new());
    let on_event = |ev: GenEvent| if let GenEvent::Token(t) = ev { got.borrow_mut().push_str(t); };
    engine.generate(prompt, "", &on_event).expect("generate");
    let text = got.into_inner();

    eprintln!("---- narration output ----\n{text}\n----");

    // Guard against the pre-fix failure mode — long runs of a single token
    // or n-gram repetition. Healthy narration includes the expected facts.
    assert!(text.contains("1500") || text.contains("1,500"),
        "expected the count '1500' to appear");
    assert!(text.contains("38.4"), "expected mean age '38.4' to appear");

    // Repetition-loop detector: no 5-word phrase should appear 3+ times.
    let words: Vec<&str> = text.split_whitespace().collect();
    for i in 0..words.len().saturating_sub(15) {
        let phrase = words[i..i + 5].join(" ");
        let mut hits = 0usize;
        for j in 0..words.len().saturating_sub(4) {
            if words[j..j + 5].join(" ") == phrase {
                hits += 1;
            }
        }
        assert!(hits < 3, "repetition loop detected — 5-gram {phrase:?} appears {hits}x");
    }
}

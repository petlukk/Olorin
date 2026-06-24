//! End-to-end proof of the narration grid-continuation gate on the real
//! production model. Wires the actual production functions —
//! `build_narration_prompt` + `narration::is_grid_continuation` — to a live
//! q3kffnimpl engine and asserts the gate's two-sided behavior:
//!   A. a 24-row numeric grid is SUPPRESSED on every draw (the model continues
//!      the grid; the gate discards it), and
//!   B. a 5-row grid is NARRATED (same row shape, but the model summarizes, so
//!      the gate lets it through) — proving the gate keys on the OUTPUT, not on
//!      input length or "is it a grid".
//! Pi-only; skips when the production model is absent.
//!
//! Run: OLORIN_THREADS=3 cargo test --release --test narration_e2e_gate -- --ignored --nocapture

use olorin::inference::generate::{Engine, GenEvent};
use olorin::runes::{build_narration_prompt, narration::is_grid_continuation, OutputSafety, RuneResult};
use std::path::Path;

// Mirror NARRATION_SYSTEM_PROMPT / NARRATION_DECODE_TOKEN_CAP (both pub(crate),
// so unreachable from an integration test). Kept at production temperature
// (Engine default 1.0 + full sampler) — faithful to run_followup_sync.
const SYSTEM: &str =
    "You are a helpful data analyst. Read the user's tool output and reply with \
     1-2 plain-English sentences naming the single most important finding. Be \
     concrete: give the actual date/time, category, or value and the magnitude \
     of any peak, spike, or anomaly — e.g. 'X peaked on <date> at about N× the \
     baseline' — not vague phrasing like 'a significant peak around a certain \
     time'. State the headline finding; do not reproduce the table or list \
     every row.";
const DECODE_CAP: usize = 768;

fn result(answer: &str) -> RuneResult {
    RuneResult { answer: answer.into(), details: None, success: true, timing_us: 0, structured: false }
}

fn narrate(engine: &mut Engine, prompt: &str) -> String {
    engine.generate(prompt, SYSTEM, &|_: GenEvent| {}).unwrap_or_default().trim().to_string()
}

#[test]
#[ignore = "Pi-only E2E, needs the q3kffnimpl production model"]
fn gate_suppresses_grid_and_passes_summary() {
    let home = std::env::var("HOME").unwrap();
    let path = Path::new(&home).join(".olorin/models/gemma-4-e2b-it-Q4_K_M-q3kffnimpl.gguf");
    if !path.exists() {
        eprintln!("SKIP: no production model at {}", path.display());
        return;
    }
    let p = path.clone();
    std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(move || run(&p))
        .unwrap()
        .join()
        .unwrap();
}

fn run(path: &Path) {
    let mut engine = Box::new(Engine::load(path, 2048).expect("load"));
    engine.max_tokens = DECODE_CAP;

    // A: 24-row grid — must be suppressed on every draw.
    let mut grid = String::new();
    for h in 0..24 {
        grid.push_str(&format!("{h:02}:00  {} files  {:.1} MB  {:.1}%\n",
            h % 7, h as f32 * 0.3, h as f32 * 0.5));
    }
    let prompt_a = build_narration_prompt("eatime", OutputSafety::Trusted, result(&grid))
        .expect("long grid must still build a prompt");
    for i in 0..3 {
        let out = narrate(&mut engine, &prompt_a);
        let suppressed = out.is_empty() || is_grid_continuation(&prompt_a, &out);
        println!("[grid {i}] suppressed={suppressed}  out={out:?}");
        assert!(suppressed, "24-row grid narration must be suppressed; got: {out:?}");
    }

    // B: 5-row grid — same row shape, but the model summarizes, so the gate
    // must let it through (non-empty, not flagged).
    let mut small = String::new();
    for h in 8..13 {
        small.push_str(&format!("{h:02}:00  {} files  {:.1} MB  {:.1}%\n",
            h, h as f32 * 0.5, h as f32 * 1.1));
    }
    let prompt_b = build_narration_prompt("eatime", OutputSafety::Trusted, result(&small))
        .expect("small grid must build a prompt");
    let out_b = narrate(&mut engine, &prompt_b);
    println!("[summary] out={out_b:?}");
    assert!(
        !out_b.is_empty() && !is_grid_continuation(&prompt_b, &out_b),
        "5-row grid must narrate cleanly (not suppressed); got: {out_b:?}"
    );
}

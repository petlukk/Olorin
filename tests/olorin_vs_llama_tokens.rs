//! Direct greedy-token comparison: Olorin vs llama.cpp on the same prompt.
//!
//! Purpose: confirm whether the supertools branch actually matches llama.cpp
//! logits on a long structured prompt. At temp=0.0 with no sampler tricks,
//! greedy argmax produces one canonical sequence per logit distribution.
//! If Olorin's first N greedy tokens match llama.cpp's first N greedy tokens,
//! logits agree at least through position N.
//!
//! Run: cargo test --release --test olorin_vs_llama_tokens -- --nocapture --ignored

use olorin::inference::generate::Engine;
use olorin::inference::gguf::GgufFile;
use olorin::inference::tokenizer::Tokenizer;
use std::cell::RefCell;
use std::path::Path;

const PROMPT: &str = "Hi";

const SYSTEM: &str = "";

#[test]
#[ignore = "run explicitly with --ignored"]
fn dump_greedy_tokens() {
    let home = std::env::var("HOME").unwrap();
    let path: std::path::PathBuf =
        Path::new(&home).join(".olorin/models/gemma-4-e2b-it-Q4_K_M.gguf");
    if !path.exists() {
        eprintln!("SKIP: no model at {}", path.display());
        return;
    }

    let path2 = path.clone();
    let handle = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || run_comparison(&path2))
        .unwrap();
    handle.join().unwrap();
}

fn run_comparison(path: &Path) {
    // ---------- 1. Show Olorin's exact tokenization of the prompt ----------
    let gguf = GgufFile::open(path).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");

    let formatted = format_gemma_chat(PROMPT, SYSTEM);
    let mut token_ids = vec![tok.bos_id];
    token_ids.extend(tok.encode(&formatted));

    println!("\n========== OLORIN TOKENIZATION ==========");
    println!("formatted prompt bytes: {}", formatted.len());
    println!("token count: {}", token_ids.len());
    println!("first 20 token ids: {:?}", &token_ids[..token_ids.len().min(20)]);
    println!("last 10 token ids: {:?}", &token_ids[token_ids.len().saturating_sub(10)..]);
    println!("formatted prompt (first 400 chars):");
    println!("{}", &formatted.chars().take(400).collect::<String>());

    // ---------- 2. Generate 20 greedy tokens with Olorin ----------
    let mut engine = Box::new(Engine::load(path, 1024).expect("load"));
    engine.temperature = 0.0;
    engine.top_k = 1;
    engine.top_p = 1.0;
    engine.min_p = 0.0;
    engine.repetition_penalty = 1.0;
    engine.max_tokens = 20;

    let got = RefCell::new(String::new());
    let tokens_seen: RefCell<Vec<String>> = RefCell::new(Vec::new());
    let on_token = |t: &str| {
        got.borrow_mut().push_str(t);
        tokens_seen.borrow_mut().push(t.to_string());
    };
    engine.generate(PROMPT, SYSTEM, &on_token).expect("generate");

    println!("\n========== OLORIN FIRST 20 GREEDY TOKENS ==========");
    println!("concatenated output: {:?}", got.into_inner());
    println!("per-token: {:?}", tokens_seen.into_inner());
}

fn format_gemma_chat(user: &str, system: &str) -> String {
    let mut out = String::with_capacity(system.len() + user.len() + 96);
    let sys_trim = system.trim();
    out.push_str("<|turn>system\n");
    if !sys_trim.is_empty() {
        out.push_str(sys_trim);
    }
    out.push_str("<turn|>\n");
    out.push_str("<|turn>user\n");
    out.push_str(user.trim());
    out.push_str("<turn|>\n");
    out.push_str("<|turn>model\n");
    out
}

//! Probe Gemma 4 to learn the actual thinking-mode markers.
//!
//! 1. Does the vocab contain `<|think|>` / `<|/think|>` as single tokens?
//! 2. What does the model emit right after a `<|think|>`-enabled prompt?
//!
//! Run: cargo test --release --test probe_think_marker -- --ignored --nocapture

use olorin::inference::generate::{Engine, GenEvent};
use olorin::inference::gguf::GgufFile;
use olorin::inference::tokenizer::Tokenizer;
use std::cell::RefCell;
use std::path::Path;

#[test]
#[ignore = "needs GGUF, run with --ignored"]
fn dump_vocab_markers() {
    let home = std::env::var("HOME").unwrap();
    let path = Path::new(&home).join(".olorin/models/gemma-4-e2b-it-Q4_K_M.gguf");
    if !path.exists() {
        eprintln!("SKIP: no model");
        return;
    }
    let gguf = GgufFile::open(&path).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");

    for candidate in [
        "<|think|>", "<|/think|>",
        "<|thinking|>", "<|/thinking|>",
        "<think>", "</think>",
        "<start_of_thinking>", "<end_of_thinking>",
        "<|turn>", "<turn|>", "<|/turn|>",
    ] {
        let ids = tok.encode(candidate);
        println!("vocab probe {:25} -> {:?}", candidate, ids);
    }

    // Dump the Jinja chat template directly from the GGUF — this is the
    // authoritative source of what markers Gemma 4 expects.
    if let Some(tmpl) = gguf.get_str("tokenizer.chat_template") {
        println!("\n========== tokenizer.chat_template ==========");
        println!("{tmpl}");
        println!("========== end chat_template ==========");
    } else {
        println!("\n(no tokenizer.chat_template in GGUF metadata)");
    }

    // Decode individual suspected token ids to text.
    println!("\n--- decoding suspect ids ---");
    for id in [98u32, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 120, 125] {
        let text = tok.decode(&[id]);
        println!("  id {:6} -> {:?}", id, text);
    }

    // Enumerate ALL special tokens (low IDs typically).
    println!("\n--- all tokens with '<|' prefix or ending '|>' ---");
    for id in 0..256u32 {
        let text = tok.decode(&[id]);
        if text.starts_with("<|") || text.ends_with("|>") || text.starts_with("<") && text.ends_with(">") {
            println!("  id {:6} -> {:?}", id, text);
        }
    }
}

#[test]
#[ignore = "runs full model, run with --ignored"]
fn observe_thinking_output() {
    let home = std::env::var("HOME").unwrap();
    let path: std::path::PathBuf =
        Path::new(&home).join(".olorin/models/gemma-4-e2b-it-Q4_K_M.gguf");
    if !path.exists() {
        eprintln!("SKIP: no model");
        return;
    }
    let path2 = path.clone();
    let handle = std::thread::Builder::new()
        .stack_size(64 * 1024 * 1024)
        .spawn(move || run_observation(&path2))
        .unwrap();
    handle.join().unwrap();
}

fn run_observation(path: &Path) {
    // Also reload the tokenizer so we can print token IDs as they arrive.
    let gguf = GgufFile::open(path).expect("gguf");
    let tok = Tokenizer::from_gguf(&gguf).expect("tokenizer");
    eprintln!("[probe] <|think|> id = {:?}", tok.token_to_id("<|think|>"));
    eprintln!("[probe] <|channel> id = {:?}", tok.token_to_id("<|channel>"));
    eprintln!("[probe] <channel|> id = {:?}", tok.token_to_id("<channel|>"));

    let mut engine = Box::new(Engine::load(path, 1024).expect("load"));
    engine.temperature = 0.0;
    engine.max_tokens = 120;

    // Rebuild the chat template by hand with <|think|> enabled, to see what
    // the model emits and where it closes the thinking block.
    // Normal engine.generate() uses format_chat which strips <|think|>.
    // This test side-steps that so we can observe the raw output.
    let user = "What is 2 + 2?";
    let system = "";

    let prompt_with_think = format!(
        "<|turn>system\n<|think|>{}<turn|>\n<|turn>user\n{}<turn|>\n<|turn>model\n",
        system.trim(), user
    );

    // We need Engine::generate_from_prompt but Engine only has generate(user, system, cb).
    // Simplest: call engine.generate with the user+system munged to include a <|think|>
    // at the start of the system field — format_chat will put it inside the system turn.
    // Note: format_chat trims the system so a leading <|think|> survives trimming.
    let system_with_think = "<|think|>";

    let got = RefCell::new(String::new());
    let on_event = |ev: GenEvent| if let GenEvent::Token(t) = ev { got.borrow_mut().push_str(t); };
    engine.generate(user, system_with_think, &on_event).expect("generate");

    println!("\n========== RAW OUTPUT (thinking enabled) ==========");
    let full = got.into_inner();
    println!("bytes: {}", full.len());
    println!("full text:");
    println!("{full}");
    println!("---");
    // Hex dump the first 400 bytes so we can see exact markers
    let b = full.as_bytes();
    for (i, chunk) in b.chunks(32).take(12).enumerate() {
        print!("{:04x}: ", i * 32);
        for &x in chunk { print!("{:02x} ", x); }
        print!("  |");
        for &x in chunk {
            let c = if (32..127).contains(&x) { x as char } else { '.' };
            print!("{c}");
        }
        println!("|");
    }

    // Search for likely markers
    for marker in ["<|think|>", "<|/think|>", "</think>", "<|endthink|>", "<end_of_thought>"] {
        if let Some(pos) = full.find(marker) {
            println!("found {:20} at byte {pos}", marker);
        }
    }

    // Note the prompt we actually used
    println!("\n(observed prompt reconstruction: {:?})", prompt_with_think);
}

//! Brainstorm experiment: see how Gemma 4 E2B narrates fake rune outputs.
//!
//! Loads the model once, runs each candidate rune × output-format combination,
//! prints the results so we can pick the format the 2B handles best.
//!
//! Run: cargo test --release --test rune_narration_eval -- --nocapture --ignored

use olorin::inference::generate::{Engine, GenEvent};
use std::cell::RefCell;
use std::path::Path;

#[test]
#[ignore = "interactive eval, run explicitly with --ignored"]
fn narrate_rune_outputs() {
    let home = std::env::var("HOME").unwrap();
    let path: std::path::PathBuf =
        Path::new(&home).join(".olorin/models/gemma-4-e2b-it-Q4_K_M.gguf");
    if !path.exists() {
        eprintln!("SKIP: no model at {}", path.display());
        return;
    }

    let path2 = path.clone();
    let handle = std::thread::Builder::new()
        .stack_size(32 * 1024 * 1024)
        .spawn(move || drive_narration(&path2))
        .unwrap();
    handle.join().unwrap();
}

fn drive_narration(path: &Path) {
    // max_tokens needs headroom for the thinking block + the answer —
    // thinking alone can run ~200 tokens on a structured prompt.
    let mut engine = Box::new(Engine::load(path, 2048).expect("load"));
    engine.temperature = 0.0;
    engine.max_tokens = 400;

    let system = "You are Olorin, a helpful assistant on a Raspberry Pi. \
                  You have access to fast SIMD tools that scan files and text. \
                  When a tool returns a result, answer the user's original question \
                  in 1-2 short, plain-language sentences. Do not show JSON or \
                  technical fields back to the user.";

    let cases = build_cases();

    for case in &cases {
        println!("\n\n========================================================");
        println!("RUNE: {}    USER ASK: {:?}", case.rune, case.user_ask);
        println!("========================================================");
        for (label, fmt) in &case.formats {
            let user = format!(
                "{}\n\nI ran the SIMD tool `{}` and it returned:\n{}\n\n\
                 Answer my question in 1-2 short sentences.",
                case.user_ask, case.rune, fmt
            );
            let out = run_one(&mut engine, system, &user);
            println!("\n--- format: {label} ---");
            println!("RAW INPUT TO MODEL:\n  question = {:?}\n  tool_result = {:?}",
                     case.user_ask, fmt);
            println!("MODEL SAID:\n{out}");
        }
    }

    println!("\n\n========== END OF EXPERIMENT ==========");
}

fn run_one(engine: &mut Engine, system: &str, user: &str) -> String {
    let got = RefCell::new(String::new());
    let on_event = |ev: GenEvent| if let GenEvent::Token(t) = ev { got.borrow_mut().push_str(t); };
    engine.generate(user, system, &on_event).expect("generate");
    got.into_inner().trim().to_string()
}

struct Case {
    rune: &'static str,
    user_ask: &'static str,
    formats: Vec<(&'static str, &'static str)>,
}

fn build_cases() -> Vec<Case> {
    vec![
        Case {
            rune: "eascam",
            user_ask: "Is this a scam? \"URGENT: your account is locked, click http://accounts-paypal-secure.tk/verify to fix\"",
            formats: vec![
                ("json",     r#"{"score": 8, "max": 10, "reasons": ["brand-impersonation", "urgency", "url-obfuscation"]}"#),
                ("kv",       "score: 8/10\nreasons: brand-impersonation, urgency, url-obfuscation"),
                ("verdict",  "SCAM (score 8/10): brand impersonation, urgent language, obfuscated URL"),
                ("sentence", "Phishing score 8 of 10. Detected: brand impersonation, urgent language, obfuscated URL."),
            ],
        },
        Case {
            rune: "easecret",
            user_ask: "Anything sensitive in this before I share it? \"Hey here is my number 555-123-4567 and email bob@example.com, also my card 4111-1111-1111-1111\"",
            formats: vec![
                ("json",     r#"{"phone": 1, "email": 1, "credit_card": 1, "api_key": 0, "ssn": 0}"#),
                ("kv",       "phone: 1\nemail: 1\ncredit_card: 1\napi_key: 0\nssn: 0"),
                ("sentence", "Found 1 phone number, 1 email address, and 1 credit card number."),
            ],
        },
        Case {
            rune: "easafe",
            user_ask: "Is this file safe? `~/Downloads/invoice.zip`",
            formats: vec![
                ("json",     r#"{"type": "zip", "size_bytes": 4194304, "entropy": 7.91, "hash": "a3f...e21", "verdict": "unknown"}"#),
                ("kv",       "type: zip\nsize: 4 MB\nentropy: 7.91\nhash: a3f...e21\nverdict: unknown"),
                ("verdict",  "UNKNOWN: 4 MB ZIP archive, entropy 7.91 (typical for compressed), not in known-bad list"),
            ],
        },
        Case {
            rune: "easpot",
            user_ask: "Check these logs for anything weird: `~/var/log/auth.log`",
            formats: vec![
                ("json",     r#"{"anomalies": [{"ts": "03:14", "count": 247, "z": 5.2, "label": "auth-fail spike"}, {"ts": "11:02", "count": 89, "z": 3.1, "label": "unusual user-agent"}]}"#),
                ("kv-list",  "03:14 - auth-fail spike (count=247, z=5.2)\n11:02 - unusual user-agent (count=89, z=3.1)"),
                ("ranked",   "Top anomalies:\n1. 03:14 - auth-fail spike (247 events, z-score 5.2)\n2. 11:02 - unusual user-agent (89 events, z-score 3.1)"),
            ],
        },
        Case {
            rune: "eadupe",
            user_ask: "Have I seen this file before? `~/Downloads/invoice.zip`",
            formats: vec![
                ("json",     r#"{"match": true, "first_seen": "2026-01-15", "label": "invoice from acme"}"#),
                ("kv",       "match: true\nfirst_seen: 2026-01-15\nlabel: invoice from acme"),
                ("sentence", "Match found. First seen on 2026-01-15, labeled 'invoice from acme'."),
            ],
        },
        Case {
            rune: "eacrunch",
            user_ask: "What's in this spreadsheet - where does my money go? `~/Downloads/statement.csv`",
            formats: vec![
                ("json",     r#"{"rows": 1247, "cols": [{"name": "category", "type": "text", "top": ["food", "rent", "transport"]}, {"name": "amount", "type": "num", "mean": 47.30, "median": 22.00, "max": 1850.0, "sum": 58931.0}]}"#),
                ("kv",       "rows: 1247\ncolumns: 3\ncategory (text): top values are food, rent, transport\namount (number): mean=47.30, median=22.00, max=1850, total=58931"),
                ("summary",  "1247 rows. Top spending categories: food, rent, transport. Amount column: total $58931, mean $47.30, max $1850."),
            ],
        },
    ]
}

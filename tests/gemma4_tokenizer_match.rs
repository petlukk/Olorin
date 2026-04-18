//! Token-for-token equality against llama.cpp's canonical tokenization.
//!
//! These ground-truth token IDs were produced by:
//!   `llama-tokenize --ids -m <gguf> -f <file>` using /home/peter/projects/llama.cpp
//!   on gemma-4-e2b-it-Q4_K_M.gguf.
//!
//! Run: cargo test --release --test gemma4_tokenizer_match -- --nocapture --ignored

use olorin::inference::gguf::GgufFile;
use olorin::inference::tokenizer::Tokenizer;
use std::path::Path;

fn load_tokenizer() -> Option<Tokenizer> {
    let home = std::env::var("HOME").ok()?;
    let path = Path::new(&home).join(".olorin/models/gemma-4-e2b-it-Q4_K_M.gguf");
    if !path.exists() {
        eprintln!("SKIP: no model at {}", path.display());
        return None;
    }
    let gguf = GgufFile::open(&path).ok()?;
    Tokenizer::from_gguf(&gguf).ok()
}

#[test]
#[ignore = "needs GGUF, run with --ignored"]
fn olorin_word_matches_llama_cpp() {
    let Some(tok) = load_tokenizer() else { return };
    // llama-tokenize on " Olorin," produces: 708 ' O', 3159 'lor', 495 'in', 236764 ','
    // (BOS and trailing newline stripped from the comparison)
    let got = tok.encode(" Olorin,");
    assert_eq!(
        got,
        vec![708, 3159, 495, 236764],
        "Gemma 4 SPM-BPE: ' Olorin,' should tokenize as [708, 3159, 495, 236764]"
    );
}

#[test]
#[ignore = "needs GGUF, run with --ignored"]
fn full_rune_prompt_matches_llama_cpp() {
    let Some(tok) = load_tokenizer() else { return };
    // Reference ground truth captured from:
    //   llama-tokenize --ids -m <gguf> -f /tmp/formatted_prompt.txt
    // where formatted_prompt.txt holds the full chat-formatted eadupe prompt.
    // BOS (token 2) is prepended externally by the caller; test the encode()
    // output proper (no BOS).
    let system = "You are Olorin, a helpful assistant on a Raspberry Pi. \
                  You have access to fast SIMD tools that scan files and text. \
                  When a tool returns a result, answer the user's original question \
                  in 1-2 short, plain-language sentences. Do not show JSON or \
                  technical fields back to the user.";
    let user = "Have I seen this file before? `~/Downloads/invoice.zip`\n\n\
                I ran the SIMD tool `eadupe` and it returned:\n\
                {\"match\": true, \"first_seen\": \"2026-01-15\", \"label\": \"invoice from acme\"}\n\n\
                Answer my question in 1-2 short sentences.";
    let formatted = format!(
        "<|turn>system\n{}<turn|>\n<|turn>user\n{}<turn|>\n<|turn>model\n",
        system, user
    );

    let expected: Vec<u32> = vec![
        105, 9731, 107, 3048, 659, 708, 3159, 495, 236764, 496, 11045, 16326, 580,
        496, 80556, 18048, 236761, 1599, 735, 2802, 531, 4592, 28554, 236796, 6436,
        600, 13952, 5734, 532, 1816, 236761, 3026, 496, 5904, 7623, 496, 1354, 236764,
        3890, 506, 2430, 236789, 236751, 3303, 2934, 528, 236743, 236770, 236772,
        236778, 2822, 236764, 14529, 236772, 19859, 23974, 236761, 3574, 711, 1407,
        10434, 653, 8330, 6192, 1063, 531, 506, 2430, 236761, 106, 107, 105, 2364,
        107, 19845, 564, 3472, 672, 2129, 1680, 236881, 2165, 77593, 84682, 236786,
        53209, 236761, 15905, 236929, 108, 236777, 11536, 506, 28554, 236796, 5904,
        2165, 4847, 55926, 236929, 532, 625, 8323, 236787, 107, 14937, 10480, 1083,
        1847, 236764, 623, 6005, 236779, 27531, 1083, 623, 236778, 236771, 236778,
        236825, 236772, 236771, 236770, 236772, 236770, 236810, 827, 623, 2491,
        1083, 623, 53209, 699, 1226, 1336, 25938, 108, 7925, 1041, 2934, 528,
        236743, 236770, 236772, 236778, 2822, 23974, 236761, 106, 107, 105, 4368,
        107,
    ];

    let got = tok.encode(&formatted);
    assert_eq!(
        got.len(),
        expected.len(),
        "token count mismatch: got {} expected {}",
        got.len(),
        expected.len()
    );
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        assert_eq!(g, e, "token diverges at position {i}: got {g}, expected {e}");
    }
}

#[test]
#[ignore = "needs GGUF, run with --ignored"]
fn basic_hello_matches_llama_cpp() {
    let Some(tok) = load_tokenizer() else { return };
    // llama-tokenize on "Hi" with the Gemma 4 model produces token 10979.
    let got = tok.encode("Hi");
    assert_eq!(got, vec![10979], "'Hi' should be token 10979 (matches llama-tokenize)");
}

#[test]
#[ignore = "needs GGUF, run with --ignored"]
fn smoke_prompt_matches_llama_cpp() {
    let Some(tok) = load_tokenizer() else { return };
    // The exact prompt gemma4_smoke feeds the model, without BOS.
    let formatted = "<|turn>system\n<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n";
    let got = tok.encode(formatted);
    // llama-tokenize output (minus leading BOS=2):
    let expected: Vec<u32> = vec![105, 9731, 107, 106, 107, 105, 2364, 107, 10979, 106, 107, 105, 4368, 107];
    assert_eq!(got, expected, "smoke prompt must match llama-tokenize exactly");
}


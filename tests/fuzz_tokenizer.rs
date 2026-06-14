//! Standing adversarial fuzzer for the BPE tokenizer — the last untested
//! untrusted-input surface. Every byte of user text (REPL / Web / WhatsApp)
//! flows through `encode`, and every model output id through `decode`, so a
//! panic, a hang, or an out-of-range token id here is a real robustness hole.
//!
//! Model-gated: the tokenizer's only constructor is `from_gguf`, so the fuzz
//! itself needs the GGUF and is `#[ignore]`d (run with `--ignored` where the
//! model lives — the Pi, or a local box with the model). This is NOT a
//! CI-standing harness, by necessity. BUT the invariant checks are factored
//! into model-free helpers, and `tokenizer_fuzz_self_test_detects_oob_id`
//! exercises them in plain CI — so "the detector can fail" is proven even
//! though the fuzz needs hardware.
//!
//! Invariants:
//!   encode (adversarial valid-UTF-8 text):
//!     - never panics, never blows up super-linearly into a hang
//!     - every emitted id is a valid vocab index (< vocab_size) — else the
//!       embedding lookup in the forward pass indexes out of bounds
//!     - deterministic: same text → same ids
//!   decode (adversarial id streams incl. out-of-range / u32::MAX):
//!     - never panics (must tolerate ids past the vocab, not index-panic)
//!
//! Determinism: a fixed seed makes findings reproducible. Override with
//! `OLORIN_TOKFUZZ_SEED`; soak with `OLORIN_TOKFUZZ_ITERS`.

use olorin::inference::gguf::GgufFile;
use olorin::inference::tokenizer::Tokenizer;
use std::path::Path;
use std::time::{Duration, Instant};

/// A single encode of ≤16 KB of text is near-instant; an order of magnitude
/// over a generous ceiling means a super-linear blow-up (tokenizer DoS), not
/// slowness. Loose enough not to flake on a loaded Pi.
const ENCODE_CEILING: Duration = Duration::from_secs(10);

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

// ─── deterministic RNG: xorshift64* (shared shape across the fuzzers) ─────────

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng((seed ^ 0x2545_F491_4F6C_DD1D) | 1)
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        if n == 0 { 0 } else { (self.next_u64() % n as u64) as usize }
    }
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi - lo + 1)
    }
}

fn iter_rng(seed: u64, iter: u32) -> Rng {
    Rng::new(seed ^ ((iter as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)))
}

fn iters() -> u32 {
    std::env::var("OLORIN_TOKFUZZ_ITERS").ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2000)
}

fn base_seed() -> u64 {
    std::env::var("OLORIN_TOKFUZZ_SEED").ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0xB7E1_5163_8AED_2A6B)
}

// ─── model-free invariant validators (shared with the CI negative control) ────

/// The encode id-validity invariant. Returns the offending id on violation.
fn validate_encoded(ids: &[u32], vocab: usize) -> Result<(), String> {
    for &id in ids {
        if id as usize >= vocab {
            return Err(format!("encode emitted out-of-range id {id} >= vocab_size {vocab}"));
        }
    }
    Ok(())
}

// ─── adversarial input generators ────────────────────────────────────────────

// Special / structural literals the tokenizer treats specially — the merge,
// whitespace (▁), and control-token paths most likely to trip on edge input.
const LITERALS: &[&str] = &[
    "<|turn>", "<turn|>", "<eos>", "<bos>", "<pad>", "<unk>", "<|think|>",
    "\u{2581}", "▁▁▁", "\n\n\n", "    ", "\t\t", "<start_of_turn>", "<end_of_turn>",
];

/// Build one adversarial **valid-UTF-8** string (encode takes `&str`, so bytes
/// must stay valid). Strategies target distinct tokenizer paths.
fn gen_text(rng: &mut Rng) -> String {
    match rng.below(6) {
        // 0: arbitrary scalar values across the whole unicode range.
        0 => {
            let n = rng.range(0, 2000);
            let mut s = String::new();
            while s.chars().count() < n {
                if let Some(c) = char::from_u32(rng.below(0x11_0000) as u32) {
                    s.push(c);
                }
            }
            s
        }
        // 1: ASCII-structured (letters, digits, the whitespace/punct that drive
        //    merges and the ▁ marker).
        1 => {
            const POOL: &[u8] = b"abcABC123 \n\t.,:;!?-_/\\\"'()[]{}<>|=+*&^%$#@~`";
            let n = rng.range(0, 3000);
            (0..n).map(|_| POOL[rng.below(POOL.len())] as char).collect()
        }
        // 2: one char repeated — BPE merge / quadratic-blowup stress.
        2 => {
            let c = char::from_u32(rng.below(0x11_0000) as u32).unwrap_or('a');
            let n = rng.range(1, 4000);
            std::iter::repeat(c).take(n).collect()
        }
        // 3: special-token literals concatenated with noise between them.
        3 => {
            let mut s = String::new();
            for _ in 0..rng.range(1, 12) {
                s.push_str(LITERALS[rng.below(LITERALS.len())]);
                if rng.below(2) == 0 {
                    if let Some(c) = char::from_u32(rng.below(128) as u32) { s.push(c); }
                }
            }
            s
        }
        // 4: whitespace storms (leading/trailing/runs — ▁ prefix handling).
        4 => {
            let ws = [' ', '\n', '\t', '\u{2581}', '\u{00A0}'];
            let n = rng.range(1, 3000);
            (0..n).map(|_| ws[rng.below(ws.len())]).collect()
        }
        // 5: char-level mutation of a seed sentence (insert/dup/drop).
        _ => {
            let mut chars: Vec<char> =
                " Olorin scans files on a Raspberry Pi, 200% faster.\n".chars().collect();
            for _ in 0..rng.range(1, 40) {
                if chars.is_empty() { break; }
                match rng.below(3) {
                    0 => { let i = rng.below(chars.len());
                           chars.insert(i, char::from_u32(rng.below(0x3000) as u32).unwrap_or('?')); }
                    1 => { let i = rng.below(chars.len()); let c = chars[i]; chars.insert(i, c); }
                    _ => { let i = rng.below(chars.len()); chars.remove(i); }
                }
            }
            chars.into_iter().collect()
        }
    }
}

/// Build one adversarial id stream — a mix of in-range, boundary, and
/// out-of-range ids (the values a corrupted/forged token stream could carry).
fn gen_ids(rng: &mut Rng, vocab: usize) -> Vec<u32> {
    let n = rng.range(0, 512);
    (0..n).map(|_| match rng.below(6) {
        0 => rng.below(vocab) as u32,                       // in range
        1 => vocab as u32,                                   // first invalid
        2 => (vocab + rng.below(1_000_000)) as u32,          // far out of range
        3 => u32::MAX,                                        // extreme
        4 => rng.below(300) as u32,                          // low / special ids
        _ => rng.next_u64() as u32,                          // arbitrary
    }).collect()
}

fn dump_repro(kind: &str, seed: u64, iter: u32, bytes: &[u8]) -> String {
    let repro = format!("/tmp/olorin_tokfuzz_repro_{kind}_seed{seed:x}_iter{iter}.bin");
    let _ = std::fs::write(&repro, bytes);
    repro
}

// ─── encode fuzz (model-gated) ───────────────────────────────────────────────

#[test]
#[ignore = "needs GGUF, run with --ignored"]
fn fuzz_tokenizer_encode() {
    let Some(tok) = load_tokenizer() else { return };
    let vocab = tok.vocab_size();
    let seed = base_seed();
    for iter in 0..iters() {
        let mut rng = iter_rng(seed, iter);
        let text = gen_text(&mut rng);

        let started = Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tok.encode(&text)));
        let elapsed = started.elapsed();

        let fail = |msg: String| {
            let repro = dump_repro("encode", seed, iter, text.as_bytes());
            panic!(
                "\n=== TOKENIZER FUZZ FINDING ===\nsurface: encode\nfailure: {msg}\n\
                 seed: {seed:#x}\niter: {iter}\ntext: {} chars / {} bytes -> {repro}\n\
                 replay: OLORIN_TOKFUZZ_SEED={seed} cargo test --release --test fuzz_tokenizer \
                 -- --ignored fuzz_tokenizer_encode\n",
                text.chars().count(), text.len()
            );
        };

        let ids = match result {
            Err(_) => { fail("PANIC in encode".into()); unreachable!() }
            Ok(ids) => ids,
        };
        if elapsed > ENCODE_CEILING {
            fail(format!("encode took {elapsed:?} (super-linear blow-up / DoS)"));
        }
        if let Err(e) = validate_encoded(&ids, vocab) {
            fail(e);
        }
        // Determinism: a tokenizer must be a pure function of its input.
        if tok.encode(&text) != ids {
            fail("encode is non-deterministic (two calls disagree)".into());
        }
    }
}

// ─── decode fuzz (model-gated) ───────────────────────────────────────────────

#[test]
#[ignore = "needs GGUF, run with --ignored"]
fn fuzz_tokenizer_decode() {
    let Some(tok) = load_tokenizer() else { return };
    let vocab = tok.vocab_size();
    let seed = base_seed() ^ 0xDEC0_DE; // distinct stream from encode
    for iter in 0..iters() {
        let mut rng = iter_rng(seed, iter);
        let ids = gen_ids(&mut rng, vocab);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tok.decode(&ids)));
        if result.is_err() {
            let raw: Vec<u8> = ids.iter().flat_map(|id| id.to_le_bytes()).collect();
            let repro = dump_repro("decode", seed, iter, &raw);
            panic!(
                "\n=== TOKENIZER FUZZ FINDING ===\nsurface: decode\nfailure: PANIC in decode\n\
                 seed: {seed:#x}\niter: {iter}\nids: {} ids -> {repro}\n\
                 replay: OLORIN_TOKFUZZ_SEED={} cargo test --release --test fuzz_tokenizer \
                 -- --ignored fuzz_tokenizer_decode\n",
                ids.len(), base_seed()
            );
        }
    }
}

// ─── negative control (model-free → runs in plain CI) ────────────────────────

/// Proves the encode id-validity detector can actually fail — a fuzzer whose
/// checks never fire is worthless. Needs no model, so it runs in normal CI.
#[test]
fn tokenizer_fuzz_self_test_detects_oob_id() {
    // The exact validator the encode fuzz relies on must reject an id at and
    // past the vocab boundary, and accept ids inside it.
    assert!(validate_encoded(&[5, 99, 100], 100).is_err(), "must flag id 100 == vocab_size");
    assert!(validate_encoded(&[0, 50, 99], 100).is_ok(), "must accept all in-range ids");
    assert!(validate_encoded(&[u32::MAX], 100).is_err(), "must flag u32::MAX");
    assert!(validate_encoded(&[], 100).is_ok(), "empty is valid");
}

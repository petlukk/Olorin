//! Standing mutation fuzzer for the file-parsing runes.
//!
//! Sibling of `rune_adversarial_inputs.rs`: that file pins the v1 contract
//! against a handful of hand-picked pathological inputs; this one throws an
//! unbounded stream of *mutated* inputs at the same contract and hunts for
//! the three failure modes the honest-state assessment named — **panic /
//! crash, hang, and silent-wrong-shape** (missing or empty structured
//! output).
//!
//! Model (inherited from the adversarial test, for good reason): each input
//! is fed to a fresh `olorin --strict` subprocess via `/rune <name> --json`.
//! A subprocess isolates the failure — a segfault inside a `.ea` SIMD kernel
//! (which `catch_unwind` cannot catch) shows up as a non-zero child exit, and
//! a wedged parser is a killable child, not a wedged test runner.
//!
//! Per-input invariants (a violation of any is a finding):
//!   1. the child exits cleanly (no signal / segfault / non-zero status)
//!   2. stdout carries a parseable `RuneOutput` JSON line
//!   3. `success:false` ⇒ `error` is present and non-empty
//!   4. the `rune` field echoes the rune that was asked for
//!   5. the child finishes inside the watchdog deadline (else: hang)
//!
//! Determinism: a fixed xorshift seed makes every run byte-reproducible.
//! Override `OLORIN_FUZZ_SEED` to explore a different stream and
//! `OLORIN_FUZZ_ITERS` to soak (CI default is small; soak with e.g. 5000).
//! On any finding the offending bytes are dumped to `/tmp/olorin_fuzz_repro_*`
//! and the panic message carries seed + iteration + strategy for replay.

use olorin::runes::output::RuneOutput;
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

const OLORIN: &str = env!("CARGO_BIN_EXE_olorin");

/// Wall-clock budget per single rune invocation. A real summary of these
/// tiny seeds finishes in well under a second even on a loaded CI box; an
/// order of magnitude over that is a hang, not slowness.
const PER_RUN_TIMEOUT: Duration = Duration::from_secs(20);

// ─── deterministic RNG: xorshift64* ─────────────────────────────────────────
// Hand-rolled (no `rand` dep — fits the zero-dep ethos and keeps the stream
// stable across toolchains so a finding replays identically).

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // xorshift requires nonzero state; the multiplier scrambles weak seeds.
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
    /// Uniform in `[0, n)`. `n == 0` yields 0 (callers guard empty ranges).
    fn below(&mut self, n: usize) -> usize {
        if n == 0 { 0 } else { (self.next_u64() % n as u64) as usize }
    }
    fn byte(&mut self) -> u8 {
        (self.next_u64() >> 33) as u8
    }
    /// Inclusive range `[lo, hi]`.
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi - lo + 1)
    }
}

// ─── mutators ───────────────────────────────────────────────────────────────
// Each strategy stacks on a clone of a base seed. Seeds are small and valid so
// mutations land *near* the valid grammar — far more likely to reach deep
// parser / kernel code than uniform random noise would.

#[derive(Clone, Copy, Debug)]
enum Strategy {
    BitFlip,
    SetBytes,
    Truncate,
    Grow,
    InsertRun,
    DupRegion,
    ZeroRegion,
    Splice,
}

const STRATEGIES: [Strategy; 8] = [
    Strategy::BitFlip,
    Strategy::SetBytes,
    Strategy::Truncate,
    Strategy::Grow,
    Strategy::InsertRun,
    Strategy::DupRegion,
    Strategy::ZeroRegion,
    Strategy::Splice,
];

/// Apply one strategy in place (or return a fresh buffer for cross-seed
/// splice). Returns the strategy applied, for the repro message.
fn apply(strategy: Strategy, buf: &mut Vec<u8>, corpus: &[&[u8]], rng: &mut Rng) {
    match strategy {
        Strategy::BitFlip => {
            if buf.is_empty() { return; }
            for _ in 0..rng.range(1, 8) {
                let i = rng.below(buf.len());
                buf[i] ^= 1 << rng.below(8);
            }
        }
        Strategy::SetBytes => {
            if buf.is_empty() { return; }
            for _ in 0..rng.range(1, 8) {
                let i = rng.below(buf.len());
                buf[i] = rng.byte();
            }
        }
        Strategy::Truncate => {
            if buf.is_empty() { return; }
            let keep = rng.below(buf.len());
            buf.truncate(keep);
        }
        Strategy::Grow => {
            let n = rng.range(1, 256);
            for _ in 0..n { buf.push(rng.byte()); }
        }
        Strategy::InsertRun => {
            // A run of NULs or one repeated byte at a random offset — exercises
            // length math and the NUL handling each scanner has to survive.
            let at = rng.below(buf.len() + 1);
            let n = rng.range(1, 64);
            let fill = if rng.below(2) == 0 { 0u8 } else { rng.byte() };
            let tail = buf.split_off(at);
            buf.extend(std::iter::repeat(fill).take(n));
            buf.extend_from_slice(&tail);
        }
        Strategy::DupRegion => {
            if buf.is_empty() { return; }
            let a = rng.below(buf.len());
            let b = rng.range(a, buf.len() - 1);
            let slice = buf[a..=b].to_vec();
            let at = rng.below(buf.len() + 1);
            let tail = buf.split_off(at);
            buf.extend_from_slice(&slice);
            buf.extend_from_slice(&tail);
        }
        Strategy::ZeroRegion => {
            if buf.is_empty() { return; }
            let a = rng.below(buf.len());
            let b = rng.range(a, buf.len() - 1);
            for byte in &mut buf[a..=b] { *byte = 0; }
        }
        Strategy::Splice => {
            // Prefix of this buffer + suffix of another seed. Mixes grammars
            // (e.g. a JSON tail onto a CSV head) to probe format confusion.
            let other = corpus[rng.below(corpus.len())];
            let cut_a = rng.below(buf.len() + 1);
            let cut_b = rng.below(other.len() + 1);
            buf.truncate(cut_a);
            buf.extend_from_slice(&other[cut_b..]);
        }
    }
}

/// Build one mutated input: pick a base seed, stack 1–3 strategies.
fn mutate(corpus: &[&[u8]], rng: &mut Rng) -> (Vec<u8>, Vec<Strategy>) {
    let mut buf = corpus[rng.below(corpus.len())].to_vec();
    let rounds = rng.range(1, 3);
    let mut applied = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let s = STRATEGIES[rng.below(STRATEGIES.len())];
        apply(s, &mut buf, corpus, rng);
        applied.push(s);
    }
    (buf, applied)
}

// ─── subprocess harness with watchdog ───────────────────────────────────────

fn write_tmp(tag: &str, bytes: &[u8]) -> String {
    // Unique per test process + tag so the 6 rune tests run in parallel
    // without clobbering each other's input files.
    let path = format!("/tmp/olorin_fuzz_{}_{tag}.bin", std::process::id());
    let mut f = std::fs::File::create(&path).expect("tmp create");
    f.write_all(bytes).expect("tmp write");
    path
}

fn extract_rune_json(stdout: &str) -> Option<&str> {
    let start = stdout.find("{\"schema_version\":")?;
    let rest = &stdout[start..];
    let end = rest.find('\n').unwrap_or(rest.len());
    Some(&rest[..end])
}

/// Outcome of one invocation. `Ok(())` = all invariants held. `Err(msg)` =
/// a finding; `msg` already describes the failure mode.
enum Outcome {
    Ok,
    Hang,
    Crash(String),
    BadOutput(String),
}

/// Spawn `olorin --strict`, feed the rune script, drain stdout on a reader
/// thread (so a chatty rune can't deadlock on a full pipe), and enforce the
/// watchdog deadline. `child` stays in this thread so the deadline path can
/// actually `kill()` a wedged parser instead of orphaning it. stderr is
/// discarded — only the structured stdout JSON matters to the contract.
fn run_once(rune: &str, path: &str) -> Outcome {
    let script = format!("/rune {rune} --json {path}\n/quit\n");
    let mut child = Command::new(OLORIN)
        .arg("--strict")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawn olorin");

    // Dropping stdin after the write sends EOF, so the REPL exits even if the
    // `/quit` line is somehow not reached.
    child.stdin.take().expect("stdin").write_all(script.as_bytes()).expect("write stdin");

    let mut stdout = child.stdout.take().expect("stdout");
    let (tx, rx) = mpsc::channel();
    let reader = std::thread::spawn(move || {
        let mut out = String::new();
        let _ = stdout.read_to_string(&mut out);
        let _ = tx.send(out);
    });

    let deadline = Instant::now() + PER_RUN_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let out = rx.recv().unwrap_or_default();
                let _ = reader.join();
                if !status.success() {
                    return Outcome::Crash(format!(
                        "non-zero/abnormal exit: {status} (signal/segfault in parser or kernel)"
                    ));
                }
                return check_contract(rune, &out);
            }
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = reader.join();
                    return Outcome::Hang;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Err(e) => return Outcome::Crash(format!("try_wait failed: {e}")),
        }
    }
}

/// Invariants 2–4: parseable RuneOutput, matching rune name, error present on
/// failure. The exit-status invariant (1) is checked by the caller.
fn check_contract(rune: &str, stdout: &str) -> Outcome {
    let json = match extract_rune_json(stdout) {
        Some(j) => j,
        None => return Outcome::BadOutput(format!(
            "no RuneOutput JSON in stdout (len={})", stdout.len()
        )),
    };
    let out = match RuneOutput::from_json(json.as_bytes()) {
        Ok(o) => o,
        Err(e) => return Outcome::BadOutput(format!("RuneOutput unparseable: {e}\njson: {json}")),
    };
    if out.rune != rune {
        return Outcome::BadOutput(format!("rune field mismatch: got {}", out.rune));
    }
    if !out.success && !out.error.as_ref().is_some_and(|e| !e.is_empty()) {
        return Outcome::BadOutput(format!("success=false but error missing/empty: {json}"));
    }
    Outcome::Ok
}

// ─── env knobs ───────────────────────────────────────────────────────────────

fn iters() -> u32 {
    std::env::var("OLORIN_FUZZ_ITERS").ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64)
}

fn base_seed() -> u64 {
    std::env::var("OLORIN_FUZZ_SEED").ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0x9E37_79B9_7F4A_7C15)
}

/// Drive `iters` mutated inputs through one rune. On the first finding, dump
/// the input and panic with a fully reproducible description.
fn fuzz_rune(rune: &str, corpus: &[&[u8]]) {
    let seed = base_seed();
    let iter_count = iters();
    for iter in 0..iter_count {
        // Derive an independent, reproducible sub-stream per iteration so a
        // single (seed, iter) pair pins the exact bytes.
        let mut rng = Rng::new(seed ^ ((iter as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)));
        let (input, strategies) = mutate(corpus, &mut rng);
        let path = write_tmp(&format!("{rune}_{iter}"), &input);
        let outcome = run_once(rune, &path);
        let _ = std::fs::remove_file(&path);

        let failure = match outcome {
            Outcome::Ok => continue,
            Outcome::Hang => format!("HANG (exceeded {:?})", PER_RUN_TIMEOUT),
            Outcome::Crash(m) => format!("CRASH: {m}"),
            Outcome::BadOutput(m) => format!("BAD OUTPUT: {m}"),
        };
        let repro = format!("/tmp/olorin_fuzz_repro_{rune}_seed{seed:x}_iter{iter}.bin");
        let _ = std::fs::write(&repro, &input);
        panic!(
            "\n=== FUZZ FINDING ===\nrune:       {rune}\nfailure:    {failure}\n\
             seed:       {seed:#x}\niter:       {iter}\nstrategies: {strategies:?}\n\
             input:      {} bytes, dumped to {repro}\nreplay:     \
             OLORIN_FUZZ_SEED={seed} OLORIN_FUZZ_ITERS={} cargo test --release \
             --test fuzz_runes fuzz_{rune}\n",
            input.len(), iter + 1
        );
    }
}

// ─── seed corpora (small, valid, CI-hermetic) ───────────────────────────────

const CSV: &[u8] = b"city,fare,ts\nNYC,10.5,2026-01-01\nLA,7.25,2026-01-02\nNYC,3.0,2026-01-03\n\"q,x\",1,2026-01-04\n";
const JSONL: &[u8] = b"{\"a\":1,\"b\":{\"c\":2},\"t\":\"x\"}\n{\"a\":3,\"b\":{\"c\":4},\"t\":\"y\"}\n{\"a\":-5,\"b\":{\"c\":0}}\n";
const LOG: &[u8] = b"INFO starting up\nERROR disk full\nWARNING retrying\nDEBUG trace 42\n127.0.0.1 - - [01/Jan/2026:06:00:00 +0000] \"GET / HTTP/1.1\" 200 12\n";
const TIMELOG: &[u8] = b"2026-05-11T06:00:00 a\n2026-05-11T06:30:00 b\n2026-05-11T07:00:00 c\n2026-05-11T07:00:01 d\n";
const SQL: &[u8] = b"CREATE TABLE t (id INT, name TEXT);\nINSERT INTO t VALUES (1,'a');\nINSERT INTO t VALUES (2,'b');\nINSERT INTO t VALUES (3,'c');\n";
// Parquet: magic + a small body + 4-byte little-endian footer length + magic.
// Mutating the footer-length field is the point — it drives the bounds math in
// the footer/thrift decoder, a classic out-of-range-read crash site.
const PARQUET: &[u8] = b"PAR1\x15\x00\x15\x10\x15\x10\x2c\x15\x04\x00\x00\x00\x08\x00\x00\x00PAR1";

/// The other seeds are visible to the splice strategy so it can graft one
/// grammar's tail onto another — cross-format confusion is a real bug class.
fn all_seeds() -> [&'static [u8]; 6] {
    [CSV, JSONL, LOG, TIMELOG, SQL, PARQUET]
}

fn corpus_for(primary: &'static [u8]) -> Vec<&'static [u8]> {
    // Primary seed weighted first; all seeds available for splice variety.
    let mut v = vec![primary];
    v.extend(all_seeds());
    v
}

// ─── per-rune tests (parallel, self-naming on failure) ───────────────────────

#[test]
fn fuzz_eacrunch() { fuzz_rune("eacrunch", &corpus_for(CSV)); }

#[test]
fn fuzz_eajson() { fuzz_rune("eajson", &corpus_for(JSONL)); }

#[test]
fn fuzz_ealog() { fuzz_rune("ealog", &corpus_for(LOG)); }

#[test]
fn fuzz_eatime() { fuzz_rune("eatime", &corpus_for(TIMELOG)); }

#[test]
fn fuzz_easql() { fuzz_rune("easql", &corpus_for(SQL)); }

#[test]
fn fuzz_eaparquet() { fuzz_rune("eaparquet", &corpus_for(PARQUET)); }

/// Negative control — proves the harness can actually *fail*. A green fuzzer
/// that cannot detect a violation is worthless. Pointing it at a non-existent
/// rune yields no `RuneOutput` JSON, which must trip the `BadOutput` path and
/// panic with a `FUZZ FINDING`. `#[ignore]`d so the real suite stays green;
/// run on demand with `cargo test --test fuzz_runes -- --ignored`.
#[test]
#[ignore]
#[should_panic(expected = "FUZZ FINDING")]
fn fuzz_self_test_detects_violation() {
    fuzz_rune("nonexistent_rune", &corpus_for(CSV));
}

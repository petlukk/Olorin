//! Standing randomized crash-state injection for vault format v4.
//!
//! Sibling of `vault_crash_consistency.rs`: that pins the durability invariant
//! against a handful of hand-placed faults; this throws a randomized stream of
//! on-disk crash states — the byte patterns a power-loss actually leaves behind
//! — at the same invariant. It needs **no power-off and no hardware**: a crash
//! does nothing magic to a file, it just leaves a truncated tail, an
//! uncommitted trailing record, or a corrupted region. We generate those states
//! directly, on a throwaway vault, so this runs in CI and on the live Pi
//! without ever touching the prod vault.
//!
//! What it does NOT cover: live write/fsync *ordering* at runtime. That rests on
//! the v4 commit protocol (record fsync'd before the header commit) and the
//! `f1_torn_*` case in the sibling test — not on file-state injection.
//!
//! Durability invariant (spec-free — a property, not a re-statement of the
//! implementation), checked for every injected fault:
//!   (U) Universal: reopen never panics; block_count never exceeds what was
//!       committed; and every block that decrypts returns *exactly* the bytes
//!       originally written at that index — never silently different plaintext.
//!   (R) Recovery: when the fault touches only the uncommitted tail (a trailing
//!       garbage record == a crash between the record fsync and the header
//!       commit), reopen recovers the full committed prefix, all intact.
//!
//! Determinism: a fixed seed makes findings byte-reproducible. Override with
//! `OLORIN_VAULTFUZZ_SEED`; soak deeper with `OLORIN_VAULTFUZZ_ITERS`.

use olorin::kernels::ffi;
use olorin::storage::argon2id::Params;
use olorin::storage::vault::Vault;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

const PASS: &[u8] = b"vault-crash-fuzz-passphrase";

// ─── deterministic RNG: xorshift64* (shared shape with the rune fuzzer) ──────

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
    fn byte(&mut self) -> u8 {
        (self.next_u64() >> 33) as u8
    }
}

fn iter_rng(seed: u64, iter: u32) -> Rng {
    Rng::new(seed ^ ((iter as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)))
}

// ─── env knobs ───────────────────────────────────────────────────────────────

fn iters() -> u32 {
    std::env::var("OLORIN_VAULTFUZZ_ITERS").ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(120)
}

fn base_seed() -> u64 {
    std::env::var("OLORIN_VAULTFUZZ_SEED").ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0x5DEE_CE66_D2C7_91A3)
}

// ─── vault construction ──────────────────────────────────────────────────────

fn unique_dir(iter: u32) -> PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir()
        .join(format!("olorin_vaultfuzz_{}_{iter}_{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// Build a vault with `n` committed blocks and return the exact decrypted bytes
/// of each — the ground truth a later reopen must reproduce or refuse. The
/// handle is dropped here, releasing the advisory lock before injection.
fn build_vault(dir: &std::path::Path, n: usize) -> Vec<Vec<u8>> {
    let mut v = Vault::open_with(dir, PASS, Params::TEST_FAST).unwrap();
    for i in 0..n {
        v.append(b"user", format!("committed message {i} :: payload-{i}{i}{i}").as_bytes())
            .unwrap();
    }
    (0..n).map(|i| v.decrypt_block(i).unwrap()).collect()
}

// ─── fault model: the on-disk states a crash leaves behind ───────────────────

enum Fault {
    /// A trailing garbage record — a crash between the record fsync and the
    /// header commit. Touches only the uncommitted tail → recovery must be full.
    Garbage(Vec<u8>),
    /// A torn / partial write that shortened the file.
    Truncate(usize),
    /// A zeroed run inside the committed region (a partially-flushed page).
    Zero(usize, usize),
    /// Bit-rot in the committed region.
    Flip(usize, usize),
}

/// Pick a fault and report whether it leaves the committed region intact (`R`).
fn pick_fault(clean: &[u8], rng: &mut Rng) -> (Fault, bool) {
    let len = clean.len();
    match rng.below(4) {
        0 => {
            let g = (0..rng.range(1, 256)).map(|_| rng.byte()).collect();
            (Fault::Garbage(g), true) // only the uncommitted tail is touched
        }
        1 => (Fault::Truncate(rng.below(len)), false),
        2 => {
            let start = rng.below(len);
            let run = rng.range(1, len - start);
            (Fault::Zero(start, run), false)
        }
        _ => {
            let start = rng.below(len);
            let run = rng.range(1, len - start);
            (Fault::Flip(start, run), false)
        }
    }
}

fn apply(fault: &Fault, clean: &[u8], rng: &mut Rng) -> Vec<u8> {
    let mut b = clean.to_vec();
    match fault {
        Fault::Garbage(g) => b.extend_from_slice(g),
        Fault::Truncate(l) => b.truncate(*l),
        Fault::Zero(start, run) => {
            for x in &mut b[*start..*start + *run] { *x = 0; }
        }
        Fault::Flip(start, run) => {
            for x in &mut b[*start..*start + *run] { *x ^= 1 << rng.below(8); }
        }
    }
    b
}

fn describe(fault: &Fault) -> String {
    match fault {
        Fault::Garbage(g) => format!("Garbage(+{} bytes)", g.len()),
        Fault::Truncate(l) => format!("Truncate(to {l})"),
        Fault::Zero(s, r) => format!("Zero(at {s}, {r} bytes)"),
        Fault::Flip(s, r) => format!("Flip(at {s}, {r} bytes)"),
    }
}

// ─── the harness ─────────────────────────────────────────────────────────────

#[test]
fn vault_crash_fuzz() {
    ffi::init().unwrap();
    let seed = base_seed();
    for iter in 0..iters() {
        let mut rng = iter_rng(seed, iter);
        let n = rng.range(1, 8);
        let dir = unique_dir(iter);
        let originals = build_vault(&dir, n);
        let path = dir.join("vault.bin");
        let clean = std::fs::read(&path).unwrap();

        let (fault, intact_tail) = pick_fault(&clean, &mut rng);
        let corrupted = apply(&fault, &clean, &mut rng);
        std::fs::write(&path, &corrupted).unwrap();

        let ctx = || format!(
            "seed={seed:#x} iter={iter} n={n} fault={} (replay: \
             OLORIN_VAULTFUZZ_SEED={seed} cargo test --release --test vault_crash_fuzz)",
            describe(&fault)
        );

        // (U) reopen must never panic.
        let opened = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Vault::open_with(&dir, PASS, Params::TEST_FAST)
        }));
        assert!(opened.is_ok(), "PANIC on reopen — {}", ctx());

        match opened.unwrap() {
            Err(_) => {
                // A clean rejection is allowed for faults that damage the header
                // or record region — but a fault touching only the uncommitted
                // tail must still open and recover.
                assert!(!intact_tail,
                    "intact-tail fault made the vault unopenable (lost committed data) — {}", ctx());
            }
            Ok(mut v) => {
                let m = v.block_count() as usize;
                // (U) block_count is MAC-protected → corruption can shrink or
                // reject it, never forge it upward.
                assert!(m <= n, "block_count grew from {n} to {m} — {}", ctx());

                // (U) every decodable block is byte-exact or cleanly refused —
                // never silently different plaintext.
                for i in 0..m {
                    if let Ok(pt) = v.decrypt_block(i) {
                        assert!(pt == originals[i],
                            "SILENT CORRUPTION: block {i} decrypted to different bytes — {}", ctx());
                    }
                }

                // (R) intact-tail faults recover the full committed prefix.
                if intact_tail {
                    assert!(m == n,
                        "intact-tail fault lost committed blocks: {m} of {n} — {}", ctx());
                    for i in 0..n {
                        let pt = v.decrypt_block(i).unwrap_or_else(|_| panic!(
                            "intact-tail fault corrupted committed block {i} — {}", ctx()));
                        assert!(pt == originals[i],
                            "intact-tail recovery mismatch at block {i} — {}", ctx());
                    }
                }
            }
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}

// ─── negative control: the harness must be able to FAIL ──────────────────────

/// Proves the invariant checks actually fire — a crash-consistency test that
/// can't detect a regression is worthless. Forces an intact-tail fault, then
/// deletes a committed block's bytes (a defect v4 must not exhibit) and asserts
/// the recovery check trips. `#[ignore]`d so the real suite stays green; run
/// with `cargo test --release --test vault_crash_fuzz -- --ignored`.
#[test]
#[ignore]
#[should_panic(expected = "intact-tail")]
fn vault_crash_fuzz_self_test_detects_lost_block() {
    ffi::init().unwrap();
    let dir = unique_dir(99_999);
    let originals = build_vault(&dir, 3);
    let path = dir.join("vault.bin");
    let clean = std::fs::read(&path).unwrap();

    // Append a trailing garbage record (intact-tail / R fault), then chop the
    // file back to BEFORE the last committed block — a fabricated total-loss
    // defect. The (R) check must catch the missing committed data.
    let mut bytes = clean.clone();
    bytes.extend_from_slice(&[0xAB; 64]);
    bytes.truncate(clean.len() / 2); // destroy committed records
    std::fs::write(&path, &bytes).unwrap();

    if let Ok(Ok(mut v)) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Vault::open_with(&dir, PASS, Params::TEST_FAST)
    })) {
        let m = v.block_count() as usize;
        // This is the (R) assertion from the harness, with intact_tail forced.
        assert!(m == 3, "intact-tail fault lost committed blocks: {m} of 3");
        for i in 0..3 {
            let pt = v.decrypt_block(i).unwrap_or_else(|_| panic!(
                "intact-tail fault corrupted committed block {i}"));
            assert!(pt == originals[i], "intact-tail recovery mismatch at block {i}");
        }
    } else {
        // Open failed outright on an intact-tail fault — also a detected defect.
        panic!("intact-tail fault made the vault unopenable");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

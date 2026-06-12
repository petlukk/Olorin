//! Robustness wave two — vault crash-consistency.
//!
//! Wave one stressed the runes; wave two stresses the encrypted
//! conversation store under *failure*: a crash mid-append, a power loss /
//! torn write, two processes opening the same vault, and a corrupted file.
//! The oracle is a spec-free durability invariant, not a re-statement of the
//! implementation:
//!
//!   "After any interruption during `append`, reopening the vault must either
//!    recover every block up to the last committed one, or at worst lose only
//!    the in-flight block — never silently lose or corrupt a previously
//!    committed block, and never panic."
//!
//! F3 (concurrent-open nonce reuse) is FIXED — an exclusive advisory file
//! lock now rejects a second opener; `f3_fixed_*` asserts that. F1 (crash
//! mid-append) is still a **characterization test for a KNOWN-UNFIXED
//! finding** (`finding_f1_*`): it asserts the *current* defective behaviour so
//! CI tells us the moment it changes — invert it when atomic append lands. The
//! `passes_*` tests assert correct behaviour that already holds.
//!
//! See benchmarks/robustness/FINDINGS.md (wave two) for the write-up.

use olorin::kernels::ffi;
use olorin::storage::argon2id::Params;
use olorin::storage::vault::Vault;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

const PASS: &[u8] = b"crash-consistency-passphrase";

fn unique_dir(label: &str) -> std::path::PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "olorin_vault_crash_{label}_{}_{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn make_vault_with_blocks(dir: &Path, n: usize) {
    let mut v = Vault::open_with(dir, PASS, Params::TEST_FAST).unwrap();
    for i in 0..n {
        v.append(b"user", format!("committed message {i}").as_bytes()).unwrap();
    }
}

fn header_index_offset(bytes: &[u8]) -> usize {
    u64::from_le_bytes(bytes[10..18].try_into().unwrap()) as usize
}

// ── Finding F1: crash mid-append destroys ALL prior blocks ────────────────────

#[test]
fn finding_f1_crash_mid_append_loses_all_committed_blocks() {
    ffi::init().unwrap();
    let dir = unique_dir("f1");
    make_vault_with_blocks(&dir, 3);
    let path = dir.join("vault.bin");

    // Sanity: the committed vault opens with all 3 blocks.
    {
        let mut v = Vault::open_with(&dir, PASS, Params::TEST_FAST).unwrap();
        assert_eq!(v.block_count(), 3);
        assert_eq!(v.decrypt_block(0).unwrap(), b"user: committed message 0\n");
    }

    // Simulate a crash during the 4th append: `flush_block` writes the new
    // block's ciphertext at `block_offset = old index_offset` — i.e. directly
    // on top of the committed index — and only fsyncs the header at the very
    // end. Reproduce the on-disk state AFTER the block hit disk but BEFORE the
    // header was rewritten: header still says {block_count:3, index_offset:io},
    // but `io` now holds block ciphertext, not the index.
    let mut bytes = std::fs::read(&path).unwrap();
    let io = header_index_offset(&bytes);
    let in_flight_ct = vec![0xABu8; 48]; // stand-in for the 4th block's ct+tag
    let end = (io + in_flight_ct.len()).min(bytes.len());
    bytes[io..end].copy_from_slice(&in_flight_ct[..end - io]);
    std::fs::write(&path, &bytes).unwrap();

    // FINDING F1 (HIGH): the vault is now unopenable. The header MAC covers the
    // index region, which the in-flight block overwrote, so open fails — and
    // there is no recovery path. A single interrupted append loses the ENTIRE
    // conversation history, not just the in-flight message.
    let result = Vault::open_with(&dir, PASS, Params::TEST_FAST);
    assert!(
        result.is_err(),
        "FINDING F1 reproduced: crash mid-append leaves the vault unopenable \
         (all 3 committed blocks lost). Invert this assertion when atomic \
         append / recovery lands."
    );

    let _ = std::fs::remove_dir_all(&dir);
}

// ── Finding F3: no file locking → concurrent append = silent loss + nonce reuse ─

#[test]
fn f3_fixed_concurrent_open_is_rejected_by_the_lock() {
    ffi::init().unwrap();
    let dir = unique_dir("f3");
    make_vault_with_blocks(&dir, 2);

    // First handle takes the exclusive advisory lock and holds it.
    let mut a = Vault::open_with(&dir, PASS, Params::TEST_FAST).unwrap();
    assert_eq!(a.block_count(), 2);

    // F3 FIX: a second open of the same vault dir (a second process, e.g. REPL
    // alongside the server) is now rejected instead of silently racing into
    // nonce reuse + data loss.
    let second = Vault::open_with(&dir, PASS, Params::TEST_FAST);
    assert!(
        second.is_err(),
        "F3 fix: a concurrent open must be rejected while the vault is held"
    );

    // A keeps working under its lock.
    a.append(b"user", b"AAAA-from-the-only-writer").unwrap();
    assert_eq!(a.block_count(), 3);

    // The lock releases on Drop, so a fresh open afterwards succeeds and sees
    // the committed append — no stale lock left behind.
    drop(a);
    let mut c = Vault::open_with(&dir, PASS, Params::TEST_FAST)
        .expect("lock released on drop — vault reopens cleanly");
    assert_eq!(c.block_count(), 3);
    assert_eq!(c.decrypt_block(2).unwrap(), b"user: AAAA-from-the-only-writer\n");

    let _ = std::fs::remove_dir_all(&dir);
}

// ── Passing invariant: truncation / torn tail never panics ────────────────────

#[test]
fn passes_truncation_never_panics() {
    ffi::init().unwrap();
    let dir = unique_dir("trunc");
    make_vault_with_blocks(&dir, 3);
    let path = dir.join("vault.bin");
    let full = std::fs::read(&path).unwrap();

    // Truncate at a spread of strategic boundaries (mid-header, header end,
    // mid-block, mid-index, one short of full). Every prefix must yield a
    // clean Result — Err or Ok — never a panic or a hang.
    let io = header_index_offset(&full);
    let cuts = [
        0usize, 1, 10, 18, 33, 63, 64, 64 + 8, io.saturating_sub(8),
        io, io + 1, io + 144, full.len().saturating_sub(1),
    ];
    for &cut in &cuts {
        let cut = cut.min(full.len());
        std::fs::write(&path, &full[..cut]).unwrap();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Vault::open_with(&dir, PASS, Params::TEST_FAST)
        }));
        assert!(
            outcome.is_ok(),
            "open panicked on a vault truncated to {cut} bytes (must fail gracefully)"
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

// ── Passing invariant: a bit-flip in a committed block body is caught ─────────

#[test]
fn passes_bitflip_in_block_body_is_rejected_on_decrypt() {
    ffi::init().unwrap();
    let dir = unique_dir("bitflip");
    make_vault_with_blocks(&dir, 3);
    let path = dir.join("vault.bin");

    // Flip a byte inside block 1's ciphertext (read its offset from index
    // entry 1). The header+index are untouched, so the vault still opens —
    // but decrypting the damaged block must fail the AEAD tag, never return
    // corrupted plaintext.
    let mut bytes = std::fs::read(&path).unwrap();
    let io = header_index_offset(&bytes);
    let entry1 = io + 288; // second index entry
    let blk1_off = u64::from_le_bytes(bytes[entry1..entry1 + 8].try_into().unwrap()) as usize;
    bytes[blk1_off] ^= 0x01;
    std::fs::write(&path, &bytes).unwrap();

    let mut v = Vault::open_with(&dir, PASS, Params::TEST_FAST)
        .expect("header+index intact, vault still opens");
    assert_eq!(v.block_count(), 3);
    assert!(
        v.decrypt_block(1).is_err(),
        "a bit-flip in the block body must fail AEAD verification, not decrypt"
    );
    // Undamaged blocks remain readable.
    assert_eq!(v.decrypt_block(0).unwrap(), b"user: committed message 0\n");

    let _ = std::fs::remove_dir_all(&dir);
}

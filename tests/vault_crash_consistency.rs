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
//! All findings are now FIXED (vault format v4): F1/F2 by the append-only
//! record log + double-buffered header (`f1_fixed_*`, `f1_torn_*`), and F3 by
//! the exclusive advisory file lock (`f3_fixed_*`). The `passes_*` tests cover
//! the already-holding invariants (no panic on truncation, AEAD catches a
//! block-body bit-flip).
//!
//! See benchmarks/robustness/FINDINGS.md (wave two) for the write-up.

use olorin::kernels::ffi;
use olorin::storage::argon2id::Params;
use olorin::storage::vault::{Vault, HEADER_SIZE_V2};
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

// ── F1/F2: crash mid-append is recoverable (v4 atomic append) ─────────────────

#[test]
fn f1_fixed_crash_mid_append_recovers_all_committed_blocks() {
    ffi::init().unwrap();
    let dir = unique_dir("f1");
    make_vault_with_blocks(&dir, 3);
    let path = dir.join("vault.bin");

    // v4 append writes the new record at the data-end (== EOF after a clean
    // append) and only commits the header AFTER an fsync. Simulate a crash
    // *during* the 4th append: a partial in-flight record has hit the disk past
    // the committed data-end, but the header was never committed. Append
    // garbage at EOF, leaving both header slots untouched.
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.extend_from_slice(&[0xABu8; 200]); // partial in-flight record
    std::fs::write(&path, &bytes).unwrap();

    // F1 FIXED: the vault still opens and recovers all 3 committed blocks — the
    // header still says block_count 3 with the data-end before the garbage, so
    // the in-flight record is ignored — and every block decrypts.
    let mut v = Vault::open_with(&dir, PASS, Params::TEST_FAST)
        .expect("crash mid-append must remain recoverable");
    assert_eq!(v.block_count(), 3, "all committed blocks recovered");
    for i in 0..3 {
        assert_eq!(
            v.decrypt_block(i).unwrap(),
            format!("user: committed message {i}\n").into_bytes()
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn f1_torn_header_commit_falls_back_to_previous_generation() {
    ffi::init().unwrap();
    let dir = unique_dir("f1torn");
    make_vault_with_blocks(&dir, 3);
    let path = dir.join("vault.bin");

    // The two 64-byte header slots alternate by generation. Simulate a torn
    // final header commit by corrupting the tag of the higher-generation slot;
    // open must fall back to the other (valid, previous-generation) slot rather
    // than fail. `header_rewrites` is bytes [42..46]; the tag starts at byte 46.
    let mut bytes = std::fs::read(&path).unwrap();
    let gen0 = u32::from_le_bytes(bytes[42..46].try_into().unwrap());
    let gen1 = u32::from_le_bytes(bytes[64 + 42..64 + 46].try_into().unwrap());
    let active = if gen0 >= gen1 { 0 } else { 64 };
    bytes[active + 46] ^= 0x01; // corrupt the active slot's tag → torn write
    std::fs::write(&path, &bytes).unwrap();

    // Falls back to the previous generation: opens with the state before the
    // last append (2 blocks), all decryptable — never a total loss.
    let mut v = Vault::open_with(&dir, PASS, Params::TEST_FAST)
        .expect("a torn header commit must recover from the other slot");
    assert_eq!(v.block_count(), 2, "recovered the previous committed generation");
    assert_eq!(v.decrypt_block(1).unwrap(), b"user: committed message 1\n");

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

    // v4 record 0 sits at RECORDS_START (2 × 64-byte header slots); its
    // ciphertext starts after the 288-byte index entry. Flip a byte there. The
    // headers + index entry are untouched, so the vault still opens — but
    // decrypting the damaged block must fail the AEAD tag, never return
    // corrupted plaintext.
    let records_start = HEADER_SIZE_V2 * 2;
    let ct0_off = records_start + 288; // skip record 0's index entry
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[ct0_off] ^= 0x01;
    std::fs::write(&path, &bytes).unwrap();

    let mut v = Vault::open_with(&dir, PASS, Params::TEST_FAST)
        .expect("headers + index intact, vault still opens");
    assert_eq!(v.block_count(), 3);
    assert!(
        v.decrypt_block(0).is_err(),
        "a bit-flip in the block body must fail AEAD verification, not decrypt"
    );
    // Undamaged blocks remain readable.
    assert_eq!(v.decrypt_block(1).unwrap(), b"user: committed message 1\n");

    let _ = std::fs::remove_dir_all(&dir);
}

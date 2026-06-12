//! Header tamper coverage for the v4 vault (double-buffered header).
//!
//! v4 keeps two 64-byte header slots (offsets 0 and 64); open takes the
//! MAC-valid slot with the highest generation. So the security property is:
//! a forged/tampered slot is never *accepted* — at worst it's ignored and the
//! other slot wins (resilience), and if BOTH slots are tampered in a
//! MAC-covered field, open fails closed. The MAC covers `header[0..46]`
//! (magic, version, block_count, data_end, key_id, nonce_seed, header_rewrites)
//! plus the tag at 46..62; per-block index fields are authenticated by each
//! block's own AEAD AAD, so a record-body flip is caught at decrypt, not open
//! (see `vault_crash_consistency.rs`).

use olorin::kernels::ffi;
use olorin::storage::argon2id::Params;
use olorin::storage::vault::Vault;
use std::sync::atomic::{AtomicU64, Ordering};

const SLOT: usize = 64; // second header slot offset

fn unique_dir(label: &str) -> std::path::PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "olorin_vault_tamper_{label}_{}_{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn make_vault_with_blocks(dir: &std::path::Path, n: usize) {
    let mut v = Vault::open_with(dir, b"test-passphrase", Params::TEST_FAST).unwrap();
    for i in 0..n {
        v.append(b"user", format!("message {i}").as_bytes()).unwrap();
    }
}

fn open(dir: &std::path::Path) -> olorin::error::Result<Vault> {
    Vault::open_with(dir, b"test-passphrase", Params::TEST_FAST)
}

/// Flip byte `off` in BOTH header slots, then attempt open.
fn flip_both_and_open(label: &str, off: usize) -> bool {
    let dir = unique_dir(label);
    make_vault_with_blocks(&dir, 3);
    let path = dir.join("vault.bin");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[off] ^= 0x01;
    bytes[SLOT + off] ^= 0x01;
    std::fs::write(&path, &bytes).unwrap();
    let ok = open(&dir).is_ok();
    let _ = std::fs::remove_dir_all(&dir);
    ok
}

#[test]
fn both_slots_block_count_tamper_rejected() {
    ffi::init().unwrap();
    assert!(!flip_both_and_open("block_count", 6), "tampering block_count in both slots must reject");
}

#[test]
fn both_slots_key_id_tamper_rejected() {
    ffi::init().unwrap();
    assert!(!flip_both_and_open("key_id", 20), "key_id tamper in both slots must reject");
}

#[test]
fn both_slots_nonce_seed_tamper_rejected() {
    ffi::init().unwrap();
    assert!(!flip_both_and_open("nonce_seed", 34), "nonce_seed tamper in both slots must reject");
}

#[test]
fn both_slots_tag_tamper_rejected() {
    ffi::init().unwrap();
    assert!(!flip_both_and_open("tag", 46), "header_tag tamper in both slots must reject");
}

#[test]
fn single_slot_tamper_is_recovered() {
    ffi::init().unwrap();
    // Corrupt only the *lower-generation* (inactive) slot's block_count. Open
    // must fall back to the still-valid active slot and recover all 3 blocks —
    // tampering one header copy can't deny service.
    let dir = unique_dir("single");
    make_vault_with_blocks(&dir, 3);
    let path = dir.join("vault.bin");
    let mut bytes = std::fs::read(&path).unwrap();
    let gen0 = u32::from_le_bytes(bytes[42..46].try_into().unwrap());
    let gen1 = u32::from_le_bytes(bytes[SLOT + 42..SLOT + 46].try_into().unwrap());
    let inactive = if gen0 < gen1 { 0 } else { SLOT }; // the older slot
    bytes[inactive + 6] ^= 0x01; // corrupt its block_count
    std::fs::write(&path, &bytes).unwrap();

    let mut v = open(&dir).expect("one tampered slot must be recovered from the other");
    assert_eq!(v.block_count(), 3);
    assert_eq!(v.decrypt_block(0).unwrap(), b"user: message 0\n");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn legitimate_open_still_works() {
    ffi::init().unwrap();
    let dir = unique_dir("legit");
    make_vault_with_blocks(&dir, 3);
    let mut v = open(&dir).expect("untampered vault must open");
    assert_eq!(v.block_count(), 3);
    assert_eq!(v.decrypt_block(0).unwrap(), b"user: message 0\n");
    let _ = std::fs::remove_dir_all(&dir);
}

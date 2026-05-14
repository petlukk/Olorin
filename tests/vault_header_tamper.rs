//! Header-tag tamper coverage for v2 vaults.
//!
//! After Task 10, any byte flipped inside the MAC region
//! (header[0..46] || serialized index) must cause `Vault::open` to fail.
//! The MAC excludes the tag itself (46..62) and the 2 reserved bytes
//! (62..64); reserved-byte tampering is out of scope.

use olorin::kernels::ffi;
use olorin::storage::vault::Vault;
use std::sync::atomic::{AtomicU64, Ordering};

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
    let mut v = Vault::open(dir).unwrap();
    for i in 0..n {
        v.append(b"user", format!("message {i}").as_bytes()).unwrap();
    }
}

#[test]
fn block_count_flip_detected() {
    ffi::init().unwrap();
    let dir = unique_dir("block_count");
    make_vault_with_blocks(&dir, 3);

    let path = dir.join("vault.bin");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[6] ^= 0x01;
    std::fs::write(&path, &bytes).unwrap();

    let result = Vault::open(&dir);
    assert!(result.is_err(), "block_count tamper must be rejected");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn key_id_flip_detected() {
    ffi::init().unwrap();
    let dir = unique_dir("key_id");
    make_vault_with_blocks(&dir, 2);

    let path = dir.join("vault.bin");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[20] ^= 0x80; // mid of key_id
    std::fs::write(&path, &bytes).unwrap();

    let result = Vault::open(&dir);
    assert!(result.is_err(), "key_id tamper must be rejected");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn nonce_seed_flip_detected() {
    ffi::init().unwrap();
    let dir = unique_dir("nonce_seed");
    make_vault_with_blocks(&dir, 2);

    let path = dir.join("vault.bin");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[34] ^= 0x01; // first byte of nonce_seed_8
    std::fs::write(&path, &bytes).unwrap();

    let result = Vault::open(&dir);
    assert!(result.is_err(), "nonce_seed_8 tamper must be rejected");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn header_rewrites_flip_detected() {
    ffi::init().unwrap();
    let dir = unique_dir("rewrites");
    make_vault_with_blocks(&dir, 1);

    let path = dir.join("vault.bin");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[42] ^= 0x01; // low byte of header_rewrites
    std::fs::write(&path, &bytes).unwrap();

    let result = Vault::open(&dir);
    assert!(result.is_err(), "header_rewrites tamper must be rejected");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn header_tag_flip_detected() {
    ffi::init().unwrap();
    let dir = unique_dir("tag");
    make_vault_with_blocks(&dir, 2);

    let path = dir.join("vault.bin");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[46] ^= 0x01; // first byte of header_tag
    std::fs::write(&path, &bytes).unwrap();

    let result = Vault::open(&dir);
    assert!(result.is_err(), "header_tag tamper must be rejected");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn index_entry_flip_detected_at_open() {
    ffi::init().unwrap();
    let dir = unique_dir("index");
    make_vault_with_blocks(&dir, 3);

    let path = dir.join("vault.bin");
    let mut bytes = std::fs::read(&path).unwrap();
    let index_offset = u64::from_le_bytes(bytes[10..18].try_into().unwrap()) as usize;
    bytes[index_offset + 12] ^= 0x01; // timestamp byte of first entry
    std::fs::write(&path, &bytes).unwrap();

    let result = Vault::open(&dir);
    assert!(result.is_err(), "index entry tamper must be rejected at open");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn legitimate_open_still_works() {
    ffi::init().unwrap();
    let dir = unique_dir("legit");
    make_vault_with_blocks(&dir, 3);

    let mut v = Vault::open(&dir).expect("untampered vault must open");
    assert_eq!(v.block_count(), 3);
    let pt = v.decrypt_block(0).unwrap();
    assert_eq!(&pt, b"user: message 0\n");
    let _ = std::fs::remove_dir_all(&dir);
}

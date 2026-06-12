//! v3 vault round-trip + header-tag mutation tests.
//!
//! Filename kept as `vault_v2_write_read` because the byte-layout
//! tested here is identical to v2 — the v3 bump is a key-derivation
//! change, not a header-format change.
//!
//! After Task 9 the write path is ChaCha20-Poly1305 AEAD per block and
//! the header is MACed with a domain-separated nonce.  This file proves
//! the basic round-trip works and that the header_tag and header_rewrites
//! advance on each append (preconditions for Task 10's tamper check).

use olorin::kernels::ffi;
use olorin::storage::vault::Vault;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicU64, Ordering};

fn unique_dir(label: &str) -> std::path::PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "olorin_vault_v2_{label}_{}_{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

#[test]
fn fresh_vault_is_v3_and_round_trips() {
    ffi::init().unwrap();
    let dir = unique_dir("rtrip");
    {
        let mut v = Vault::open_with(&dir, b"test-passphrase", olorin::storage::argon2id::Params::TEST_FAST).expect("create");
        v.append(b"user", b"hello").unwrap();
        v.append(b"assistant", b"world").unwrap();
        assert_eq!(v.block_count(), 2);
    }

    let mut v = Vault::open_with(&dir, b"test-passphrase", olorin::storage::argon2id::Params::TEST_FAST).expect("reopen");
    assert_eq!(v.block_count(), 2);
    let pt0 = v.decrypt_block(0).unwrap();
    assert_eq!(&pt0, b"user: hello\n");
    let pt1 = v.decrypt_block(1).unwrap();
    assert_eq!(&pt1, b"assistant: world\n");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn fresh_vault_writes_version_4_byte() {
    ffi::init().unwrap();
    let dir = unique_dir("version4");
    let _v = Vault::open_with(&dir, b"test-passphrase", olorin::storage::argon2id::Params::TEST_FAST).expect("create");
    let bytes = std::fs::read(dir.join("vault.bin")).unwrap();
    let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
    assert_eq!(version, 4, "fresh vault should be v4 (record log + double header)");
    let _ = std::fs::remove_dir_all(&dir);
}

/// The current commit (highest generation across the two header slots) and its
/// tag.
fn active_header(bytes: &[u8]) -> (u32, Vec<u8>) {
    let g0 = u32::from_le_bytes(bytes[42..46].try_into().unwrap());
    let g1 = u32::from_le_bytes(bytes[64 + 42..64 + 46].try_into().unwrap());
    if g0 >= g1 {
        (g0, bytes[46..62].to_vec())
    } else {
        (g1, bytes[64 + 46..64 + 62].to_vec())
    }
}

#[test]
fn header_generation_advances_on_each_append() {
    ffi::init().unwrap();
    let dir = unique_dir("rewrites");
    let path = dir.join("vault.bin");

    let mut v = Vault::open_with(&dir, b"test-passphrase", olorin::storage::argon2id::Params::TEST_FAST).unwrap();
    v.append(b"u", b"a").unwrap();
    let (gen1, tag1) = active_header(&std::fs::read(&path).unwrap());

    v.append(b"u", b"b").unwrap();
    let (gen2, tag2) = active_header(&std::fs::read(&path).unwrap());

    // v4 alternates slots, so compare the *active* (highest-gen) header each
    // time: the generation increments by one and its tag changes per append.
    assert_eq!(gen2, gen1 + 1, "the committed generation must increment per append");
    assert_ne!(tag1, tag2, "the active header tag must change after an append");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn aead_tag_rejects_block_tampering() {
    ffi::init().unwrap();
    let dir = unique_dir("tamper_block");
    let path = dir.join("vault.bin");

    {
        let mut v = Vault::open_with(&dir, b"test-passphrase", olorin::storage::argon2id::Params::TEST_FAST).unwrap();
        v.append(b"u", b"secret payload").unwrap();
    }

    // Flip one byte inside block 0's ciphertext. v4 record 0 starts at offset
    // 128 (two 64-byte header slots); its 288-byte index entry is followed by
    // the ciphertext at 416, so 420 is 4 bytes into the ct.
    {
        let mut f = std::fs::OpenOptions::new().read(true).write(true).open(&path).unwrap();
        f.seek(SeekFrom::Start(420)).unwrap();
        let mut b = [0u8; 1];
        f.read_exact(&mut b).unwrap();
        b[0] ^= 0x01;
        f.seek(SeekFrom::Start(420)).unwrap();
        f.write_all(&b).unwrap();
    }

    let mut v = Vault::open_with(&dir, b"test-passphrase", olorin::storage::argon2id::Params::TEST_FAST).unwrap();
    let result = v.decrypt_block(0);
    assert!(result.is_err(), "tampered block must fail AEAD verification");
    let _ = std::fs::remove_dir_all(&dir);
}

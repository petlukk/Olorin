//! v2 vault round-trip + header-tag mutation tests.
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
fn fresh_vault_is_v2_and_round_trips() {
    ffi::init().unwrap();
    let dir = unique_dir("rtrip");
    {
        let mut v = Vault::open(&dir).expect("create");
        v.append(b"user", b"hello").unwrap();
        v.append(b"assistant", b"world").unwrap();
        assert_eq!(v.block_count(), 2);
    }

    let mut v = Vault::open(&dir).expect("reopen");
    assert_eq!(v.block_count(), 2);
    let pt0 = v.decrypt_block(0).unwrap();
    assert_eq!(&pt0, b"user: hello\n");
    let pt1 = v.decrypt_block(1).unwrap();
    assert_eq!(&pt1, b"assistant: world\n");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn fresh_vault_writes_version_2_byte() {
    ffi::init().unwrap();
    let dir = unique_dir("version2");
    let _v = Vault::open(&dir).expect("create");
    let bytes = std::fs::read(dir.join("vault.bin")).unwrap();
    let version = u16::from_le_bytes(bytes[4..6].try_into().unwrap());
    assert_eq!(version, 2, "fresh vault should be v2");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn header_tag_and_rewrites_advance_on_each_append() {
    ffi::init().unwrap();
    let dir = unique_dir("rewrites");
    let path = dir.join("vault.bin");

    let mut v = Vault::open(&dir).unwrap();
    v.append(b"u", b"a").unwrap();
    let bytes1 = std::fs::read(&path).unwrap();
    let tag1 = bytes1[46..62].to_vec();
    let rewrites1 = u32::from_le_bytes(bytes1[42..46].try_into().unwrap());

    v.append(b"u", b"b").unwrap();
    let bytes2 = std::fs::read(&path).unwrap();
    let tag2 = bytes2[46..62].to_vec();
    let rewrites2 = u32::from_le_bytes(bytes2[42..46].try_into().unwrap());

    assert_ne!(tag1, tag2, "header tag must change after an append");
    assert_eq!(
        rewrites2,
        rewrites1 + 1,
        "header_rewrites must increment by exactly 1 per append"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn aead_tag_rejects_block_tampering() {
    ffi::init().unwrap();
    let dir = unique_dir("tamper_block");
    let path = dir.join("vault.bin");

    {
        let mut v = Vault::open(&dir).unwrap();
        v.append(b"u", b"secret payload").unwrap();
    }

    // Flip one byte inside the first block (offset 70 = 6 bytes into block 0,
    // which starts right after the 64-byte v2 header).
    {
        let mut f = std::fs::OpenOptions::new().read(true).write(true).open(&path).unwrap();
        f.seek(SeekFrom::Start(70)).unwrap();
        let mut b = [0u8; 1];
        f.read_exact(&mut b).unwrap();
        b[0] ^= 0x01;
        f.seek(SeekFrom::Start(70)).unwrap();
        f.write_all(&b).unwrap();
    }

    let mut v = Vault::open(&dir).unwrap();
    let result = v.decrypt_block(0);
    assert!(result.is_err(), "tampered block must fail AEAD verification");
    let _ = std::fs::remove_dir_all(&dir);
}

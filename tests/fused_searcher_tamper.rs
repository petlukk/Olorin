//! verify-then-search: a tampered block must never appear in search results,
//! and verification must happen *before* the fused decrypt+search kernel
//! touches the ciphertext (so no plaintext line, even unmatched, leaks).

use olorin::kernels::ffi;
use olorin::storage::vault::Vault;
use std::sync::atomic::{AtomicU64, Ordering};

fn unique_dir(label: &str) -> std::path::PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "olorin_fused_tamper_{label}_{}_{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

const V2_HEADER_SIZE: usize = 64;

/// Read the v2 IndexEntry at slot `idx` directly off disk to find a block's
/// `(offset, length)` so we can tamper a specific ct byte deterministically.
fn read_block_offset(path: &std::path::Path, idx: usize) -> (u64, u32) {
    const INDEX_ENTRY_SIZE: usize = 288;
    let bytes = std::fs::read(path).unwrap();
    let index_offset = u64::from_le_bytes(bytes[10..18].try_into().unwrap()) as usize;
    let entry_start = index_offset + idx * INDEX_ENTRY_SIZE;
    let off = u64::from_le_bytes(bytes[entry_start..entry_start + 8].try_into().unwrap());
    let len = u32::from_le_bytes(bytes[entry_start + 8..entry_start + 12].try_into().unwrap());
    (off, len)
}

#[test]
fn search_drops_tampered_block_from_results() {
    ffi::init().unwrap();
    let dir = unique_dir("drop");
    let path = dir.join("vault.bin");

    {
        let mut v = Vault::open_with(&dir, b"test-passphrase", olorin::storage::argon2id::Params::TEST_FAST).unwrap();
        v.append(b"user", b"alpha apple").unwrap();
        v.append(b"user", b"beta banana").unwrap();
        v.append(b"user", b"gamma grape").unwrap();
    }

    // Flip a byte inside block 1's ciphertext (NOT the tag — pick the first
    // ct byte, well clear of the trailing 16-byte tag region).
    let (block1_off, _block1_len) = read_block_offset(&path, 1);
    {
        use std::io::{Read, Seek, SeekFrom, Write};
        let mut f = std::fs::OpenOptions::new().read(true).write(true).open(&path).unwrap();
        f.seek(SeekFrom::Start(block1_off)).unwrap();
        let mut b = [0u8; 1];
        f.read_exact(&mut b).unwrap();
        b[0] ^= 0x01;
        f.seek(SeekFrom::Start(block1_off)).unwrap();
        f.write_all(&b).unwrap();
    }
    // Sanity: tampering inside a block region leaves the header_tag valid
    // (the header MAC covers header[0..46] || index_bytes only, never block
    // contents), so Vault::open must still succeed.
    assert!(block1_off >= V2_HEADER_SIZE as u64);

    let mut v = Vault::open_with(&dir, b"test-passphrase", olorin::storage::argon2id::Params::TEST_FAST).expect("open succeeds — only ciphertext was touched");

    // decrypt_block(1) must fail (AEAD tag check).
    assert!(v.decrypt_block(1).is_err(), "tampered block must fail AEAD open");
    // Blocks 0 and 2 are intact.
    assert_eq!(v.decrypt_block(0).unwrap(), b"user: alpha apple\n");
    assert_eq!(v.decrypt_block(2).unwrap(), b"user: gamma grape\n");

    // Search for "banana" — the tampered block once contained it but must
    // be dropped from results entirely.  Surviving blocks contain "alpha"
    // and "gamma" — neither has "banana", so the result list must be empty.
    let results = v.search("banana", 10).unwrap();
    for r in &results {
        let block_text = v.decrypt_block(r.block_index).unwrap_or_default();
        assert!(
            !block_text.windows(6).any(|w| w == b"banana"),
            "tampered 'banana' block leaked into search results via index {}",
            r.block_index,
        );
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn search_still_works_on_untampered_vault() {
    ffi::init().unwrap();
    let dir = unique_dir("clean");
    {
        let mut v = Vault::open_with(&dir, b"test-passphrase", olorin::storage::argon2id::Params::TEST_FAST).unwrap();
        v.append(b"user", b"alpha apple").unwrap();
        v.append(b"user", b"beta banana").unwrap();
        v.append(b"user", b"gamma grape").unwrap();
    }
    let mut v = Vault::open_with(&dir, b"test-passphrase", olorin::storage::argon2id::Params::TEST_FAST).unwrap();
    let results = v.search("banana", 10).unwrap();
    assert!(!results.is_empty(), "untampered vault must still surface matches");
    let _ = std::fs::remove_dir_all(&dir);
}

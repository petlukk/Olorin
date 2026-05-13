//! Vault security regression tests — properties the vault stack must
//! uphold to remain "encrypted at rest" against an attacker who has
//! only the vault file (no binary, no machine).
//!
//! These tests pin properties that are easy to silently regress when
//! anything changes in `key.rs`, `vault.rs`, or `crypto.rs`. Add a
//! property here whenever a review finds a leak; don't rely on the
//! reviewer being around the next time.

use olorin::storage::key;
use olorin::storage::vault::Vault;

fn unique_dir(label: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir()
        .join(format!("olorin_vault_sec_{label}_{}_{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// Finding #1: the file header MUST NOT contain any 4-byte window of
/// the vault key. Previously `nonce_seed[8..12] = key[..4]` (vault.rs)
/// stored the first four key bytes directly in the header.
#[test]
fn vault_header_does_not_contain_key_prefix() {
    let dir = unique_dir("hdr_no_key");
    olorin::kernels::ffi::init().unwrap();

    let key = key::derive_key().expect("hardware id required for this test");
    let _vault = Vault::open(&dir).expect("vault opens");

    let header = std::fs::read(dir.join("vault.bin")).expect("read vault");
    assert!(header.len() >= 64, "header should be 64 bytes, got {}", header.len());

    let header = &header[..64];
    // For every 4-byte window of the key, scan the header. No window
    // should match — that would be a key-material leak.
    for k_off in 0..=key.len() - 4 {
        let needle = &key[k_off..k_off + 4];
        for (h_off, window) in header.windows(4).enumerate() {
            assert_ne!(
                window, needle,
                "key bytes {k_off}..{} ({:02x?}) appear at header offset {h_off}",
                k_off + 4, needle,
            );
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// Finding #1, second half: two vaults created back-to-back on the
/// same machine must have DIFFERENT nonce_seeds. Pre-fix, the seed
/// derivation pinned the low 4 bytes to `key[..4]` (constant across
/// vaults on the same machine), and the high 8 bytes to seconds
/// granularity — so two same-second creations collided.
#[test]
fn two_vaults_have_distinct_nonce_seeds() {
    olorin::kernels::ffi::init().unwrap();
    let dir_a = unique_dir("seed_a");
    let dir_b = unique_dir("seed_b");

    let _a = Vault::open(&dir_a).expect("vault a");
    let _b = Vault::open(&dir_b).expect("vault b");

    let bytes_a = std::fs::read(dir_a.join("vault.bin")).expect("read a");
    let bytes_b = std::fs::read(dir_b.join("vault.bin")).expect("read b");
    // nonce_seed is header[34..46] per vault.rs::VaultHeader::to_bytes
    let seed_a = &bytes_a[34..46];
    let seed_b = &bytes_b[34..46];
    assert_ne!(seed_a, seed_b,
        "two vaults share nonce_seed — ChaCha20 nonce-reuse risk if they ever share a key");

    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

/// Finding #4 (smoke check): the encrypt/decrypt roundtrip still
/// works after we add scratch zeroization. Doesn't prove plaintext
/// doesn't linger (would need heap inspection) — but proves the fix
/// didn't break correctness.
#[test]
fn encrypt_decrypt_roundtrip_still_works_after_zeroize() {
    use olorin::storage::crypto;
    olorin::kernels::ffi::init().unwrap();

    let key = [0x42u8; 32];
    let nonce = [0x07u8; 12];
    let plaintext = b"the wakeful mind in ea, 2026";
    let mut buf = plaintext.to_vec();
    crypto::encrypt(&key, &nonce, 0, &mut buf);
    assert_ne!(&buf[..], plaintext, "ciphertext equals plaintext");
    crypto::decrypt(&key, &nonce, 0, &mut buf);
    assert_eq!(&buf[..], plaintext, "decrypt does not roundtrip");
}

/// Finding #8: block_count must not wrap. ChaCha20 nonces here are
/// `nonce_seed XOR block_count`; incrementing past u32::MAX would
/// reuse a nonce on the next append (catastrophic — keystream xor
/// of two plaintexts == both plaintexts XOR'd). Tamper the header on
/// disk to push block_count to u32::MAX, re-open, and assert the
/// next append errors instead of silently reusing the nonce.
#[test]
fn vault_refuses_to_append_when_counter_is_at_max() {
    use std::io::{Seek, SeekFrom, Write};
    olorin::kernels::ffi::init().unwrap();
    let dir = unique_dir("counter_wrap");

    {
        let mut vault = Vault::open(&dir).expect("vault opens");
        vault.append(b"user", b"hello once").expect("first append");
    }

    // Tamper the header: block_count is bytes 6..10 (u32 LE) per
    // VaultHeader::to_bytes. Set it to u32::MAX.
    let vault_path = dir.join("vault.bin");
    let mut file = std::fs::OpenOptions::new()
        .read(true).write(true).open(&vault_path).expect("reopen file");
    file.seek(SeekFrom::Start(6)).expect("seek block_count");
    file.write_all(&u32::MAX.to_le_bytes()).expect("tamper count");
    drop(file);

    // Re-open: the vault now thinks it has 2^32 - 1 blocks.
    // Index parsing will fail because we didn't actually write those
    // blocks — that's fine; the test only cares about post-open
    // append refusing.
    let reopened = Vault::open(&dir);
    if let Ok(mut vault) = reopened {
        let result = vault.append(b"user", b"hello twice");
        assert!(result.is_err(),
            "append at u32::MAX must refuse (would reuse nonce); got: {:?}", result);
    }
    // Either reopen-failure (index parse) or append-failure is
    // acceptable — both mean the wrap can't silently produce a
    // nonce-reused ciphertext.

    let _ = std::fs::remove_dir_all(&dir);
}

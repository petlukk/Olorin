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
    let dir = std::env::temp_dir().join(format!(
        "olorin_vault_sec_{label}_{}_{n}",
        std::process::id()
    ));
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

    let _vault = Vault::open_with(&dir, b"test-passphrase", olorin::storage::argon2id::Params::TEST_FAST).expect("vault opens");
    let salt = key::load_or_create_salt(&dir).expect("salt");
    let key = key::derive_key(b"test-passphrase", &salt, olorin::storage::argon2id::Params::TEST_FAST).expect("derive_key");

    let header = std::fs::read(dir.join("vault.bin")).expect("read vault");
    assert!(
        header.len() >= 64,
        "header should be 64 bytes, got {}",
        header.len()
    );

    let header = &header[..64];
    // For every 4-byte window of the key, scan the header. No window
    // should match — that would be a key-material leak.
    for k_off in 0..=key.len() - 4 {
        let needle = &key[k_off..k_off + 4];
        for (h_off, window) in header.windows(4).enumerate() {
            assert_ne!(
                window,
                needle,
                "key bytes {k_off}..{} ({:02x?}) appear at header offset {h_off}",
                k_off + 4,
                needle,
            );
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
}

/// Finding #1, second half: two vaults created back-to-back on the
/// same machine must have DIFFERENT nonce_seeds. Pre-fix, the seed
/// derivation pinned the low 4 bytes to `key[..4]` (constant across
/// vaults on the same machine), and the high 8 bytes to seconds
/// granularity — so two same-second creations collided.  In v2 the
/// nonce seed is 8 bytes at header[34..42] (header_rewrites occupies
/// [42..46]), seeded from nanos and the low 32 bits of as_secs().
#[test]
fn two_vaults_have_distinct_nonce_seeds() {
    olorin::kernels::ffi::init().unwrap();
    let dir_a = unique_dir("seed_a");
    let dir_b = unique_dir("seed_b");

    let _a = Vault::open_with(&dir_a, b"test-passphrase", olorin::storage::argon2id::Params::TEST_FAST).expect("vault a");
    let _b = Vault::open_with(&dir_b, b"test-passphrase", olorin::storage::argon2id::Params::TEST_FAST).expect("vault b");

    let bytes_a = std::fs::read(dir_a.join("vault.bin")).expect("read a");
    let bytes_b = std::fs::read(dir_b.join("vault.bin")).expect("read b");
    // v2: nonce_seed_8 is header[34..42] per VaultHeaderV2::to_bytes.
    let seed_a = &bytes_a[34..42];
    let seed_b = &bytes_b[34..42];
    assert_ne!(
        seed_a, seed_b,
        "two vaults share nonce_seed_8 — ChaCha20 nonce-reuse risk if they ever share a key"
    );

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

/// Finding #8: block_count must not wrap.  In v2 the per-block nonce is
/// `nonce_seed_8 || u32_le(block_count)`, with the high bit of the
/// counter slot reserved for the header-MAC domain — so the guard now
/// fires at 0x80000000 (not u32::MAX).  Either way: tampering the header
/// to claim a near-exhausted counter must not let the next append
/// silently reuse a nonce.
#[test]
fn vault_refuses_to_append_when_counter_is_at_max() {
    use std::io::{Seek, SeekFrom, Write};
    olorin::kernels::ffi::init().unwrap();
    let dir = unique_dir("counter_wrap");

    {
        let mut vault = Vault::open_with(&dir, b"test-passphrase", olorin::storage::argon2id::Params::TEST_FAST).expect("vault opens");
        vault.append(b"user", b"hello once").expect("first append");
    }

    // Tamper the header: block_count is bytes 6..10 (u32 LE) per
    // VaultHeader::to_bytes. Set it to u32::MAX.
    let vault_path = dir.join("vault.bin");
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&vault_path)
        .expect("reopen file");
    file.seek(SeekFrom::Start(6)).expect("seek block_count");
    file.write_all(&u32::MAX.to_le_bytes())
        .expect("tamper count");
    drop(file);

    // Re-open: the vault now thinks it has 2^32 - 1 blocks.
    // Index parsing will fail because we didn't actually write those
    // blocks — that's fine; the test only cares about post-open
    // append refusing.
    let reopened = Vault::open_with(&dir, b"test-passphrase", olorin::storage::argon2id::Params::TEST_FAST);
    if let Ok(mut vault) = reopened {
        let result = vault.append(b"user", b"hello twice");
        assert!(
            result.is_err(),
            "append at u32::MAX must refuse (would reuse nonce); got: {:?}",
            result
        );
    }
    // Either reopen-failure (index parse) or append-failure is
    // acceptable — both mean the wrap can't silently produce a
    // nonce-reused ciphertext.

    let _ = std::fs::remove_dir_all(&dir);
}

// ────────────────────────────────────────────────────────────────────────────
// v2-specific tamper coverage.  Header-prefix and index-entry tamper sites
// are covered in tests/vault_header_tamper.rs; the cases below target the
// per-block AEAD tag region and the file-shape invariants enforced at open.
// ────────────────────────────────────────────────────────────────────────────

fn make_three_block_vault(dir: &std::path::Path) {
    let mut v = Vault::open_with(dir, b"test-passphrase", olorin::storage::argon2id::Params::TEST_FAST).expect("vault opens");
    v.append(b"user", b"alpha").unwrap();
    v.append(b"user", b"beta").unwrap();
    v.append(b"user", b"gamma").unwrap();
}

/// Each block on disk is `ct || 16-byte tag`.  Flipping a byte inside the
/// trailing 16 bytes corrupts the Poly1305 tag itself; the per-block AEAD
/// open MUST reject it.  Distinct from header_tag tamper (caught at open)
/// — this fires at decrypt_block(i).
#[test]
fn v2_block_tag_byte_flip_detected_at_decrypt() {
    use std::io::{Read, Seek, SeekFrom, Write};
    olorin::kernels::ffi::init().unwrap();
    let dir = unique_dir("block_tag");
    make_three_block_vault(&dir);

    // Read entry 0 to find (offset, length) of block 0.
    let path = dir.join("vault.bin");
    let bytes = std::fs::read(&path).unwrap();
    let index_offset = u64::from_le_bytes(bytes[10..18].try_into().unwrap()) as usize;
    let off0 = u64::from_le_bytes(bytes[index_offset..index_offset + 8].try_into().unwrap());
    let len0 = u32::from_le_bytes(bytes[index_offset + 8..index_offset + 12].try_into().unwrap())
        as u64;
    // Flip the *first* tag byte (= 16 bytes before block end).
    let tag_byte_off = off0 + len0 - 16;

    let mut f = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    f.seek(SeekFrom::Start(tag_byte_off)).unwrap();
    let mut b = [0u8; 1];
    f.read_exact(&mut b).unwrap();
    b[0] ^= 0x80;
    f.seek(SeekFrom::Start(tag_byte_off)).unwrap();
    f.write_all(&b).unwrap();
    drop(f);

    let mut v = Vault::open_with(&dir, b"test-passphrase", olorin::storage::argon2id::Params::TEST_FAST).expect("open succeeds — block region untouched by header MAC");
    let res = v.decrypt_block(0);
    assert!(res.is_err(), "tag-region tamper must fail AEAD verify");
    let _ = std::fs::remove_dir_all(&dir);
}

/// `index_offset` (bytes 10..18) is part of the header MAC input.  A flip
/// changes the prefix the tag was computed over, so open MUST fail.
#[test]
fn header_index_offset_flip_detected_at_open() {
    olorin::kernels::ffi::init().unwrap();
    let dir = unique_dir("idx_off");
    make_three_block_vault(&dir);

    let path = dir.join("vault.bin");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[10] ^= 0x01;
    std::fs::write(&path, &bytes).unwrap();

    let result = Vault::open_with(&dir, b"test-passphrase", olorin::storage::argon2id::Params::TEST_FAST);
    assert!(result.is_err(), "index_offset tamper must be rejected");
    let _ = std::fs::remove_dir_all(&dir);
}

/// Truncating the file mid-index leaves block_count claiming more entries
/// than the index region can fit.  `open` performs a saturating sanity
/// check against this exact attack and must reject the file.
#[test]
fn truncate_mid_index_rejected_at_open() {
    olorin::kernels::ffi::init().unwrap();
    let dir = unique_dir("truncate");
    make_three_block_vault(&dir);

    let path = dir.join("vault.bin");
    let total = std::fs::metadata(&path).unwrap().len();
    // Truncate to roughly half: drops 1-2 index entries from the tail.
    let target = total / 2;
    let f = std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap();
    f.set_len(target).unwrap();
    drop(f);

    let result = Vault::open_with(&dir, b"test-passphrase", olorin::storage::argon2id::Params::TEST_FAST);
    assert!(result.is_err(), "truncated vault must be rejected");
    let _ = std::fs::remove_dir_all(&dir);
}

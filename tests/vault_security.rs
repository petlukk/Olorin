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

/// Finding #8 (v4): a tampered block_count must never let the next append
/// silently reuse a nonce. In v4 the header is MAC'd in two slots, so setting
/// block_count to a near-exhausted value in BOTH slots invalidates both MACs
/// and the vault is rejected at open — it can never reach a tampered-max state
/// from which it would append. (The `>= 0x8000_0000` wrap guard in flush_block
/// remains as a backstop for the legitimately-exhausted case.)
#[test]
fn tampered_block_count_cannot_force_nonce_reuse() {
    olorin::kernels::ffi::init().unwrap();
    let dir = unique_dir("counter_wrap");

    {
        let mut vault = Vault::open_with(&dir, b"test-passphrase", olorin::storage::argon2id::Params::TEST_FAST).expect("vault opens");
        vault.append(b"user", b"hello once").expect("first append");
    }

    // block_count is bytes 6..10 in each 64-byte slot. Set it to u32::MAX in
    // both slots, invalidating both header MACs.
    let path = dir.join("vault.bin");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[6..10].copy_from_slice(&u32::MAX.to_le_bytes());
    bytes[64 + 6..64 + 10].copy_from_slice(&u32::MAX.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();

    let reopened = Vault::open_with(&dir, b"test-passphrase", olorin::storage::argon2id::Params::TEST_FAST);
    assert!(
        reopened.is_err(),
        "a block_count tampered to u32::MAX in both slots must be rejected at open"
    );

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

    // v4: record 0 is at RECORDS_START (2×64-byte header slots). Its index
    // entry occupies the first 288 bytes; `length` (ct+16) is entry bytes 8..12.
    // The ciphertext follows the entry; the 16-byte tag is the last of it.
    let path = dir.join("vault.bin");
    let bytes = std::fs::read(&path).unwrap();
    let entry0 = 128usize; // RECORDS_START
    let len0 = u32::from_le_bytes(bytes[entry0 + 8..entry0 + 12].try_into().unwrap()) as u64;
    let ct0_off = (entry0 + 288) as u64;
    // Flip the *first* tag byte (= 16 bytes before the record's ciphertext end).
    let tag_byte_off = ct0_off + len0 - 16;

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

/// `index_offset` / data-end (bytes 10..18) is in the header MAC input. v4 has
/// two slots, so flipping it in BOTH invalidates both MACs → open MUST fail.
/// (Flipping just one is recovered from the other — see vault_header_tamper.)
#[test]
fn header_data_end_flip_in_both_slots_rejected_at_open() {
    olorin::kernels::ffi::init().unwrap();
    let dir = unique_dir("idx_off");
    make_three_block_vault(&dir);

    let path = dir.join("vault.bin");
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[10] ^= 0x01;
    bytes[64 + 10] ^= 0x01;
    std::fs::write(&path, &bytes).unwrap();

    let result = Vault::open_with(&dir, b"test-passphrase", olorin::storage::argon2id::Params::TEST_FAST);
    assert!(result.is_err(), "data-end tamper in both slots must be rejected");
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

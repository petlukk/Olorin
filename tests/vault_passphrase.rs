//! Vault behaviour around the new passphrase + Argon2id key derivation.
//!
//! Covers the four properties that distinguish the v3 (passphrase) vault
//! from v2 (hwid-derived):
//!
//!   1. Salt file is created on first open and reused on subsequent opens.
//!   2. Same passphrase + same salt + same params → vault opens.
//!   3. Wrong passphrase against an existing vault is rejected.
//!   4. A synthetic v2 vault is rejected with "unsupported vault version".
//!
//! All tests use `Params::TEST_FAST` so the suite doesn't pay for full
//! Argon2id passes — the cryptographic strength is unit-tested in
//! `argon2id_kat.rs` against the RFC 9106 §5.2 vector.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use olorin::kernels::ffi;
use olorin::storage::argon2id::Params;
use olorin::storage::key::{self, SALT_BYTES};
use olorin::storage::vault::Vault;

fn unique_dir(label: &str) -> PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "olorin_vault_pw_{label}_{}_{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn setup() {
    ffi::init().expect("kernel init");
}

#[test]
fn salt_file_is_created_on_first_open_and_persisted() {
    setup();
    let dir = unique_dir("salt_persist");

    let _ = Vault::open_with(&dir, b"pw", Params::TEST_FAST).expect("first open");
    let salt_path = dir.join("vault.salt");
    assert!(salt_path.exists(), "vault.salt must be created on first open");

    let salt_a = std::fs::read(&salt_path).expect("read salt");
    assert_eq!(salt_a.len(), SALT_BYTES, "salt must be {SALT_BYTES} bytes");

    // Reopen — salt must be reused, not regenerated.
    let _ = Vault::open_with(&dir, b"pw", Params::TEST_FAST).expect("reopen");
    let salt_b = std::fs::read(&salt_path).expect("read salt again");
    assert_eq!(salt_a, salt_b, "salt must be reused across opens");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn two_fresh_vaults_get_distinct_salts() {
    // Same passphrase, different vault dirs → different salts → different
    // derived keys.  Protects against pre-computed dictionary attacks
    // across vaults.
    setup();
    let dir_a = unique_dir("distinct_a");
    let dir_b = unique_dir("distinct_b");

    let _ = Vault::open_with(&dir_a, b"pw", Params::TEST_FAST).expect("a");
    let _ = Vault::open_with(&dir_b, b"pw", Params::TEST_FAST).expect("b");

    let salt_a = std::fs::read(dir_a.join("vault.salt")).unwrap();
    let salt_b = std::fs::read(dir_b.join("vault.salt")).unwrap();
    assert_ne!(salt_a, salt_b, "fresh vaults must get distinct salts");

    let _ = std::fs::remove_dir_all(&dir_a);
    let _ = std::fs::remove_dir_all(&dir_b);
}

#[test]
fn wrong_passphrase_rejects_existing_vault() {
    setup();
    let dir = unique_dir("wrong_pw");

    {
        let mut v = Vault::open_with(&dir, b"correct", Params::TEST_FAST).expect("create");
        v.append(b"user", b"hello").expect("append");
    }

    // Re-open with the wrong passphrase: salt file is reused, Argon2id
    // produces a different key, header MAC verify fails.
    let err = Vault::open_with(&dir, b"wrong", Params::TEST_FAST);
    assert!(err.is_err(), "wrong passphrase must fail to open the vault");

    // Correct passphrase still works.
    let _ = Vault::open_with(&dir, b"correct", Params::TEST_FAST).expect("re-open with correct pw");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn v2_vault_is_rejected_with_unsupported_version() {
    // Synthesise a v2 vault.bin header — same byte layout as v3 except
    // version=2 — and try to open it.  v3 binaries must refuse v2.
    setup();
    let dir = unique_dir("v2_rejected");
    let vault_path = dir.join("vault.bin");

    let mut header = vec![0u8; 64];
    header[0..4].copy_from_slice(b"OLRN");
    header[4..6].copy_from_slice(&2u16.to_le_bytes()); // version = 2
    std::fs::write(&vault_path, &header).expect("write synth header");
    std::fs::write(dir.join("vault.salt"), [0u8; SALT_BYTES]).expect("write salt");

    let err = Vault::open_with(&dir, b"pw", Params::TEST_FAST);
    assert!(err.is_err(), "v2 vault must be rejected");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn derive_key_is_deterministic_for_same_inputs() {
    setup();
    let salt = [0x42u8; SALT_BYTES];
    let k_a = key::derive_key(b"pw", &salt, Params::TEST_FAST).unwrap();
    let k_b = key::derive_key(b"pw", &salt, Params::TEST_FAST).unwrap();
    assert_eq!(k_a, k_b);

    // Different salt → different key.
    let salt2 = [0x43u8; SALT_BYTES];
    let k_c = key::derive_key(b"pw", &salt2, Params::TEST_FAST).unwrap();
    assert_ne!(k_a, k_c, "different salt must produce different key");
}

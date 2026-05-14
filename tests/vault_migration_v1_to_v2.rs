//! v1 → v2 auto-migration test.
//!
//! Synthesizes a v1 vault at the byte level (deliberately not using
//! `Vault::append`, which now emits v2), then opens it with the current
//! binary and asserts:
//!   1. Plaintexts round-trip exactly.
//!   2. `vault.bin.v1.bak` exists after a successful migration.
//!   3. A second open is a plain v2 open (no further migration).
//!   4. A corrupted v1 ciphertext refuses to migrate AND does not leave a
//!      backup behind — the integrity check fires before any file mutation.

use olorin::kernels::ffi;
use olorin::storage::{crypto, key};
use olorin::storage::vault::Vault;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const V1_HEADER_SIZE: usize = 64;
const V1_INDEX_ENTRY_SIZE: usize = 288;
const VAULT_MAGIC: [u8; 4] = *b"OLRN";

fn unique_dir(label: &str) -> std::path::PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "olorin_vault_migrate_{label}_{}_{n}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Build a v1 vault.bin matching the pre-Task-9 binary layout exactly:
///   64-byte v1 header (magic|version=1|block_count|index_offset|key_id|nonce_seed(12)|reserved(18))
///   ChaCha20-XOR'd plaintexts (no tag) at sequential offsets
///   288-byte v1 IndexEntries with the xxhash field populated
fn synthesize_v1_vault(
    path: &std::path::Path,
    key_bytes: &[u8; 32],
    messages: &[(&[u8], &[u8])],
) {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let mut nonce_seed = [0u8; 12];
    nonce_seed[..8].copy_from_slice(&now.as_secs().to_le_bytes());
    nonce_seed[8..12].copy_from_slice(&now.subsec_nanos().to_le_bytes());

    let h1 = key::xxhash64(key_bytes, 0);
    let h2 = key::xxhash64(key_bytes, h1);
    let mut key_id = [0u8; 16];
    key_id[..8].copy_from_slice(&h1.to_le_bytes());
    key_id[8..16].copy_from_slice(&h2.to_le_bytes());

    let mut entries: Vec<[u8; V1_INDEX_ENTRY_SIZE]> = Vec::new();
    let mut ct_blocks: Vec<Vec<u8>> = Vec::new();
    let mut cursor: u64 = V1_HEADER_SIZE as u64;
    for (i, (role, content)) in messages.iter().enumerate() {
        let mut plaintext = Vec::with_capacity(role.len() + 2 + content.len() + 1);
        plaintext.extend_from_slice(role);
        plaintext.extend_from_slice(b": ");
        plaintext.extend_from_slice(content);
        plaintext.push(b'\n');

        let xxh = key::xxhash64(&plaintext, 0);
        let histogram = key::compute_histogram(&plaintext);

        let nonce_counter = i as u32;
        let mut nonce = nonce_seed;
        let cb = nonce_counter.to_le_bytes();
        for j in 0..4 {
            nonce[j] ^= cb[j];
        }

        let mut ct = plaintext.clone();
        crypto::encrypt(key_bytes, &nonce, 0, &mut ct);

        let mut entry = [0u8; V1_INDEX_ENTRY_SIZE];
        entry[0..8].copy_from_slice(&cursor.to_le_bytes());
        entry[8..12].copy_from_slice(&(ct.len() as u32).to_le_bytes());
        entry[12..20].copy_from_slice(&now.as_secs().to_le_bytes());
        entry[20..28].copy_from_slice(&xxh.to_le_bytes());
        entry[28..32].copy_from_slice(&nonce_counter.to_le_bytes());
        entry[32..288].copy_from_slice(&histogram);

        cursor += ct.len() as u64;
        ct_blocks.push(ct);
        entries.push(entry);
    }
    let index_offset = cursor;

    let mut header = [0u8; V1_HEADER_SIZE];
    header[0..4].copy_from_slice(&VAULT_MAGIC);
    header[4..6].copy_from_slice(&1u16.to_le_bytes());
    header[6..10].copy_from_slice(&(messages.len() as u32).to_le_bytes());
    header[10..18].copy_from_slice(&index_offset.to_le_bytes());
    header[18..34].copy_from_slice(&key_id);
    header[34..46].copy_from_slice(&nonce_seed);
    // header[46..64] = zero (reserved)

    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&header).unwrap();
    for ct in &ct_blocks {
        f.write_all(ct).unwrap();
    }
    for e in &entries {
        f.write_all(e).unwrap();
    }
    f.sync_all().unwrap();
}

fn derive_test_key() -> [u8; 32] {
    key::derive_key().expect("hardware id required for this test")
}

#[test]
fn migration_round_trips_plaintexts() {
    ffi::init().unwrap();
    let dir = unique_dir("rtrip");
    let path = dir.join("vault.bin");
    let key = derive_test_key();
    synthesize_v1_vault(
        &path,
        &key,
        &[
            (b"user", b"first"),
            (b"assistant", b"second"),
            (b"user", b"third"),
        ],
    );

    let mut v = Vault::open(&dir).expect("migrate");
    assert_eq!(v.block_count(), 3);
    assert_eq!(v.decrypt_block(0).unwrap(), b"user: first\n");
    assert_eq!(v.decrypt_block(1).unwrap(), b"assistant: second\n");
    assert_eq!(v.decrypt_block(2).unwrap(), b"user: third\n");
    drop(v);

    let bak = path.with_extension("bin.v1.bak");
    assert!(bak.exists(), "v1 backup must exist after successful migration");

    // Backup bytes should still be the v1 file.
    let bak_bytes = std::fs::read(&bak).unwrap();
    let bak_version = u16::from_le_bytes(bak_bytes[4..6].try_into().unwrap());
    assert_eq!(bak_version, 1);

    // New vault.bin is v2.
    let new_bytes = std::fs::read(&path).unwrap();
    let new_version = u16::from_le_bytes(new_bytes[4..6].try_into().unwrap());
    assert_eq!(new_version, 2);

    // Second open is a plain v2 open — must succeed and still round-trip.
    let mut v2 = Vault::open(&dir).expect("reopen v2");
    assert_eq!(v2.block_count(), 3);
    assert_eq!(v2.decrypt_block(1).unwrap(), b"assistant: second\n");

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn migration_refuses_corrupt_v1() {
    ffi::init().unwrap();
    let dir = unique_dir("corrupt");
    let path = dir.join("vault.bin");
    let key = derive_test_key();
    synthesize_v1_vault(&path, &key, &[(b"user", b"hello")]);

    // Flip the first ciphertext byte (offset 64 = right after the v1 header).
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[V1_HEADER_SIZE] ^= 0xff;
    std::fs::write(&path, &bytes).unwrap();

    let result = Vault::open(&dir);
    assert!(result.is_err(), "corrupt v1 must refuse to migrate");

    let bak = path.with_extension("bin.v1.bak");
    assert!(
        !bak.exists(),
        "backup must NOT be created when migration aborts on integrity failure",
    );

    // Original file untouched — still v1.
    let still_v1 = std::fs::read(&path).unwrap();
    assert_eq!(
        u16::from_le_bytes(still_v1[4..6].try_into().unwrap()),
        1,
        "vault.bin must remain v1 on failed migration",
    );

    let _ = std::fs::remove_dir_all(&dir);
}

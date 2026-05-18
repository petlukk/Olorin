//! Vault on-disk format — types, constants, and the v1 plaintext reader.
//!
//! Extracted from `vault.rs` so the `Vault` struct's runtime logic stays
//! readable.  Everything here knows the *byte layout* of `vault.bin`:
//!
//! - [`VaultHeaderV2`] — the 64-byte file header
//! - [`IndexEntry`] — the 288-byte per-block index entry
//! - [`derive_block_nonce`] / [`derive_header_nonce`] — the v2 nonce schedule
//! - [`build_block_aad`] — the per-block AEAD associated data
//! - [`read_all_v1_plaintexts`] — one-shot v1 reader used only by migration
//!
//! `VaultHeaderV2` and `HEADER_SIZE_V2` are re-exported from `vault.rs`
//! so external callers (and the `tests/vault_header_v2.rs` round-trip test)
//! keep their import path `olorin::storage::vault::{VaultHeaderV2, ...}`
//! working.

use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{Error, Result};
use crate::storage::crypto;
use crate::storage::key;

pub(super) const VAULT_MAGIC: [u8; 4] = *b"OLRN";
pub(super) const INDEX_ENTRY_SIZE: usize = 288;
pub(super) const VAULT_VERSION_V2: u16 = 2;

// The v1 VaultHeader / VAULT_VERSION / HEADER_SIZE were dropped after the
// migration helper switched to raw byte parsing.  v1 vaults are still
// recognised + auto-migrated on first open by a v2 binary.

// ── VaultHeaderV2 ─────────────────────────────────────────────────────────────

pub const HEADER_SIZE_V2: usize = 64;

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct VaultHeaderV2 {
    pub magic: [u8; 4],
    pub version: u16,
    pub block_count: u32,
    pub index_offset: u64,
    pub key_id: [u8; 16],
    pub nonce_seed_8: [u8; 8],
    pub header_rewrites: u32,
    pub header_tag: [u8; 16],
    pub reserved: [u8; 2],
}

impl VaultHeaderV2 {
    pub(crate) fn new(key_id: [u8; 16], nonce_seed_8: [u8; 8]) -> Self {
        Self {
            magic: VAULT_MAGIC,
            version: VAULT_VERSION_V2,
            block_count: 0,
            index_offset: HEADER_SIZE_V2 as u64,
            key_id,
            nonce_seed_8,
            header_rewrites: 0,
            header_tag: [0; 16],
            reserved: [0; 2],
        }
    }

    pub fn to_bytes(self) -> [u8; HEADER_SIZE_V2] {
        let mut buf = [0u8; HEADER_SIZE_V2];
        buf[0..4].copy_from_slice(&self.magic);
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());
        buf[6..10].copy_from_slice(&self.block_count.to_le_bytes());
        buf[10..18].copy_from_slice(&self.index_offset.to_le_bytes());
        buf[18..34].copy_from_slice(&self.key_id);
        buf[34..42].copy_from_slice(&self.nonce_seed_8);
        buf[42..46].copy_from_slice(&self.header_rewrites.to_le_bytes());
        buf[46..62].copy_from_slice(&self.header_tag);
        buf[62..64].copy_from_slice(&self.reserved);
        buf
    }

    pub fn from_bytes(buf: &[u8; HEADER_SIZE_V2]) -> Result<Self> {
        let magic: [u8; 4] = buf[0..4].try_into().unwrap();
        if magic != VAULT_MAGIC {
            return Err(Error::Vault("bad magic"));
        }
        let version = u16::from_le_bytes(buf[4..6].try_into().unwrap());
        if version != VAULT_VERSION_V2 {
            return Err(Error::Vault("unsupported vault version"));
        }
        let block_count = u32::from_le_bytes(buf[6..10].try_into().unwrap());
        let index_offset = u64::from_le_bytes(buf[10..18].try_into().unwrap());
        let mut key_id = [0u8; 16];
        key_id.copy_from_slice(&buf[18..34]);
        let mut nonce_seed_8 = [0u8; 8];
        nonce_seed_8.copy_from_slice(&buf[34..42]);
        let header_rewrites = u32::from_le_bytes(buf[42..46].try_into().unwrap());
        let mut header_tag = [0u8; 16];
        header_tag.copy_from_slice(&buf[46..62]);
        let mut reserved = [0u8; 2];
        reserved.copy_from_slice(&buf[62..64]);
        Ok(Self {
            magic,
            version,
            block_count,
            index_offset,
            key_id,
            nonce_seed_8,
            header_rewrites,
            header_tag,
            reserved,
        })
    }
}

// ── IndexEntry ────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub(super) struct IndexEntry {
    pub(super) offset: u64,
    pub(super) length: u32,
    pub(super) timestamp: u64,
    /// Was xxhash(plaintext) in v1; integrity is now carried by the per-block
    /// AEAD tag.  Zeroed on write; ignored on read.  Layout preserved so a
    /// v1→v2 migration can stream entries without shifting offsets.
    pub(super) _reserved: u64,
    pub(super) nonce_counter: u32,
    pub(super) histogram: [u8; 256],
}

impl IndexEntry {
    pub(super) fn to_bytes(&self) -> [u8; INDEX_ENTRY_SIZE] {
        let mut buf = [0u8; INDEX_ENTRY_SIZE];
        buf[0..8].copy_from_slice(&self.offset.to_le_bytes());
        buf[8..12].copy_from_slice(&self.length.to_le_bytes());
        buf[12..20].copy_from_slice(&self.timestamp.to_le_bytes());
        buf[20..28].copy_from_slice(&0u64.to_le_bytes());
        buf[28..32].copy_from_slice(&self.nonce_counter.to_le_bytes());
        buf[32..288].copy_from_slice(&self.histogram);
        buf
    }

    pub(super) fn from_bytes(buf: &[u8; INDEX_ENTRY_SIZE]) -> Self {
        let offset = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let length = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let timestamp = u64::from_le_bytes(buf[12..20].try_into().unwrap());
        let nonce_counter = u32::from_le_bytes(buf[28..32].try_into().unwrap());
        let mut histogram = [0u8; 256];
        histogram.copy_from_slice(&buf[32..288]);
        Self { offset, length, timestamp, _reserved: 0, nonce_counter, histogram }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

pub(super) fn now_epoch() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

/// v2 per-block nonce: 8 bytes of nonce_seed_8 || u32_le(counter).
/// Counter values are bounded to [0, 0x80000000) — the high bit is
/// reserved for the header-tag domain (see [`derive_header_nonce`]).
pub(super) fn derive_block_nonce(seed8: &[u8; 8], counter: u32) -> [u8; 12] {
    let mut nonce = [0u8; 12];
    nonce[0..8].copy_from_slice(seed8);
    nonce[8..12].copy_from_slice(&counter.to_le_bytes());
    nonce
}

/// v2 header nonce: 8 bytes of nonce_seed_8 || u32_le(0x80000000 | rewrites).
/// High-bit-set 4-byte tail can never collide with a block nonce because
/// `flush_block` refuses counter ≥ 0x80000000.
pub(super) fn derive_header_nonce(seed8: &[u8; 8], rewrites: u32) -> [u8; 12] {
    let mut nonce = [0u8; 12];
    nonce[0..8].copy_from_slice(seed8);
    let domain = 0x8000_0000u32 | rewrites;
    nonce[8..12].copy_from_slice(&domain.to_le_bytes());
    nonce
}

/// Per-block AAD: key_id || version_le || counter_le || timestamp_le || histogram.
/// Binds the ciphertext to this exact (vault, version, position, time, content
/// shape) tuple so a swapped block fails AEAD verification.
pub(super) fn build_block_aad(
    key_id: &[u8; 16],
    version: u16,
    nonce_counter: u32,
    timestamp: u64,
    histogram: &[u8; 256],
) -> Vec<u8> {
    let mut aad = Vec::with_capacity(286);
    aad.extend_from_slice(key_id);
    aad.extend_from_slice(&version.to_le_bytes());
    aad.extend_from_slice(&nonce_counter.to_le_bytes());
    aad.extend_from_slice(&timestamp.to_le_bytes());
    aad.extend_from_slice(histogram);
    aad
}

/// Decrypt every v1 block off disk and verify each block's xxhash tag.
/// Used only by `Vault::migrate_v1_to_v2`.  Returns the plaintexts in
/// on-disk order — the caller re-emits them as v2 AEAD blocks.
pub(super) fn read_all_v1_plaintexts(path: &Path, key_bytes: &[u8; 32]) -> Result<Vec<Vec<u8>>> {
    const V1_HEADER_SIZE: usize = 64;
    const V1_INDEX_ENTRY_SIZE: usize = 288;

    let mut file = OpenOptions::new().read(true).open(path)?;
    let file_size = file.metadata()?.len();

    let mut hdr = [0u8; V1_HEADER_SIZE];
    file.read_exact(&mut hdr)?;
    if hdr[0..4] != VAULT_MAGIC {
        return Err(Error::Vault("bad magic"));
    }
    let v1_version = u16::from_le_bytes(hdr[4..6].try_into().unwrap());
    if v1_version != 1 {
        return Err(Error::Vault("expected v1 vault"));
    }
    let block_count = u32::from_le_bytes(hdr[6..10].try_into().unwrap()) as usize;
    let index_offset = u64::from_le_bytes(hdr[10..18].try_into().unwrap());
    let mut nonce_seed_v1 = [0u8; 12];
    nonce_seed_v1.copy_from_slice(&hdr[34..46]);

    let entries_region = file_size.saturating_sub(index_offset);
    let max_entries = (entries_region / V1_INDEX_ENTRY_SIZE as u64) as usize;
    if block_count > max_entries {
        return Err(Error::Vault("v1 header block_count exceeds file size"));
    }

    file.seek(SeekFrom::Start(index_offset))?;
    let mut entries: Vec<(u64, u32, u64, u32)> = Vec::with_capacity(block_count);
    for _ in 0..block_count {
        let mut e = [0u8; V1_INDEX_ENTRY_SIZE];
        file.read_exact(&mut e)?;
        let off = u64::from_le_bytes(e[0..8].try_into().unwrap());
        let len = u32::from_le_bytes(e[8..12].try_into().unwrap());
        let xxh = u64::from_le_bytes(e[20..28].try_into().unwrap());
        let nc = u32::from_le_bytes(e[28..32].try_into().unwrap());
        entries.push((off, len, xxh, nc));
    }

    let mut plaintexts: Vec<Vec<u8>> = Vec::with_capacity(block_count);
    for &(off, len, expected_xxh, nc) in &entries {
        let mut ct = vec![0u8; len as usize];
        file.seek(SeekFrom::Start(off))?;
        file.read_exact(&mut ct)?;

        // v1 nonce derivation: XOR counter into the low 4 bytes of seed.
        let mut nonce = nonce_seed_v1;
        let cb = nc.to_le_bytes();
        for j in 0..4 {
            nonce[j] ^= cb[j];
        }
        crypto::decrypt(key_bytes, &nonce, 0, &mut ct);

        if key::xxhash64(&ct, 0) != expected_xxh {
            return Err(Error::Vault("v1 block integrity check failed"));
        }
        plaintexts.push(ct);
    }
    Ok(plaintexts)
}

//! Encrypted vault — append-only conversation storage.
//!
//! Key derivation: XOR-obfuscated seed ^ hardware_id (from /etc/machine-id).
//! Format: binary header + encrypted blocks + index at tail.
//! Plaintext never lingers — blocks are encrypted immediately on append.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{Error, Result};
use crate::kernels::ffi;
use crate::storage::crypto;
use crate::storage::search::FusedSearcher;

// ── Format constants ──────────────────────────────────────────────────────────

const VAULT_MAGIC: [u8; 4] = *b"OLRN";
const VAULT_VERSION: u16 = 1;
const HEADER_SIZE: usize = 64;
const INDEX_ENTRY_SIZE: usize = 288;

// ── VaultHeader ───────────────────────────────────────────────────────────────

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct VaultHeader {
    magic: [u8; 4],
    version: u16,
    block_count: u32,
    index_offset: u64,
    key_id: [u8; 16],
    nonce_seed: [u8; 12],
    reserved: [u8; 18],
}

impl VaultHeader {
    fn new(key_id: [u8; 16], nonce_seed: [u8; 12]) -> Self {
        Self {
            magic: VAULT_MAGIC,
            version: VAULT_VERSION,
            block_count: 0,
            index_offset: HEADER_SIZE as u64,
            key_id,
            nonce_seed,
            reserved: [0u8; 18],
        }
    }

    fn to_bytes(self) -> [u8; HEADER_SIZE] {
        let mut buf = [0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(&self.magic);
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());
        buf[6..10].copy_from_slice(&self.block_count.to_le_bytes());
        buf[10..18].copy_from_slice(&self.index_offset.to_le_bytes());
        buf[18..34].copy_from_slice(&self.key_id);
        buf[34..46].copy_from_slice(&self.nonce_seed);
        buf[46..64].copy_from_slice(&self.reserved);
        buf
    }

    fn from_bytes(buf: &[u8; HEADER_SIZE]) -> Result<Self> {
        let magic: [u8; 4] = buf[0..4].try_into().unwrap();
        if magic != VAULT_MAGIC {
            return Err(Error::Vault("bad magic"));
        }
        let version = u16::from_le_bytes(buf[4..6].try_into().unwrap());
        if version != VAULT_VERSION {
            return Err(Error::Vault("unsupported version"));
        }
        let block_count = u32::from_le_bytes(buf[6..10].try_into().unwrap());
        let index_offset = u64::from_le_bytes(buf[10..18].try_into().unwrap());
        let mut key_id = [0u8; 16];
        key_id.copy_from_slice(&buf[18..34]);
        let mut nonce_seed = [0u8; 12];
        nonce_seed.copy_from_slice(&buf[34..46]);
        let mut reserved = [0u8; 18];
        reserved.copy_from_slice(&buf[46..64]);
        Ok(Self { magic, version, block_count, index_offset, key_id, nonce_seed, reserved })
    }
}

// ── IndexEntry ────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct IndexEntry {
    offset: u64,
    length: u32,
    timestamp: u64,
    xxhash: u64,
    nonce_counter: u32,
    histogram: [u8; 256],
}

impl IndexEntry {
    fn to_bytes(&self) -> [u8; INDEX_ENTRY_SIZE] {
        let mut buf = [0u8; INDEX_ENTRY_SIZE];
        buf[0..8].copy_from_slice(&self.offset.to_le_bytes());
        buf[8..12].copy_from_slice(&self.length.to_le_bytes());
        buf[12..20].copy_from_slice(&self.timestamp.to_le_bytes());
        buf[20..28].copy_from_slice(&self.xxhash.to_le_bytes());
        buf[28..32].copy_from_slice(&self.nonce_counter.to_le_bytes());
        buf[32..288].copy_from_slice(&self.histogram);
        buf
    }

    fn from_bytes(buf: &[u8; INDEX_ENTRY_SIZE]) -> Self {
        let offset = u64::from_le_bytes(buf[0..8].try_into().unwrap());
        let length = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let timestamp = u64::from_le_bytes(buf[12..20].try_into().unwrap());
        let xxhash = u64::from_le_bytes(buf[20..28].try_into().unwrap());
        let nonce_counter = u32::from_le_bytes(buf[28..32].try_into().unwrap());
        let mut histogram = [0u8; 256];
        histogram.copy_from_slice(&buf[32..288]);
        Self { offset, length, timestamp, xxhash, nonce_counter, histogram }
    }
}

// ── Key derivation ────────────────────────────────────────────────────────────

/// Compile-time mask — obfuscates the seed in .rodata.
const COMPILE_MASK: [u8; 32] = [
    0xA7, 0x3B, 0xC9, 0x14, 0xE6, 0x58, 0xF2, 0x0D,
    0x8B, 0x61, 0xD4, 0x37, 0x9E, 0xAC, 0x55, 0x73,
    0xC2, 0x1F, 0xB8, 0x46, 0x7A, 0xE3, 0x09, 0xD1,
    0x5C, 0x84, 0xF7, 0x2E, 0x63, 0xA0, 0x4B, 0x19,
];

/// seed ^ COMPILE_MASK — raw seed never appears in .rodata.
const OBFUSCATED_SEED: [u8; 32] = {
    let seed = b"olorin-vault-seed-v0.5-default!!";
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        out[i] = seed[i] ^ COMPILE_MASK[i];
        i += 1;
    }
    out
};

/// Read hardware identifier for key binding.
fn hardware_id() -> [u8; 32] {
    let raw = std::fs::read_to_string("/sys/class/dmi/id/product_uuid")
        .or_else(|_| std::fs::read_to_string("/etc/machine-id"))
        .or_else(|_| std::fs::read_to_string("/etc/hostname"))
        .unwrap_or_else(|_| "olorin-fallback-id".to_string());
    let mut id = [0u8; 32];
    let bytes = raw.trim().as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        id[i % 32] ^= b;
    }
    for i in 1..32 {
        id[i] ^= id[i - 1].wrapping_mul(31);
    }
    id
}

/// Derive the vault key: `OBFUSCATED_SEED ^ COMPILE_MASK ^ hardware_id()` = `seed ^ hw`.
fn derive_key() -> [u8; 32] {
    let hw = hardware_id();
    let mut key = [0u8; 32];
    for i in 0..32 {
        key[i] = OBFUSCATED_SEED[i] ^ COMPILE_MASK[i] ^ hw[i];
    }
    key
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn compute_histogram(data: &[u8]) -> [u8; 256] {
    let mut hist = [0u8; 256];
    for &b in data {
        hist[b as usize] = hist[b as usize].saturating_add(1);
    }
    hist
}

fn normalize_histogram(hist: &[u8; 256]) -> [f32; 256] {
    let mut norm = [0.0f32; 256];
    let mut sum_sq = 0.0f32;
    for i in 0..256 {
        let v = hist[i] as f32;
        norm[i] = v;
        sum_sq += v * v;
    }
    let mag = sum_sq.sqrt();
    if mag > 0.0 {
        for n in &mut norm { *n /= mag; }
    }
    norm
}

fn cosine_similarity(a: &[f32; 256], b: &[f32; 256]) -> f32 {
    let mut dot = 0.0f32;
    for i in 0..256 { dot += a[i] * b[i]; }
    dot
}

fn xxhash64(data: &[u8], seed: u64) -> u64 {
    const P1: u64 = 0x9E3779B185EBCA87;
    const P2: u64 = 0xC2B2AE3D27D4EB4F;
    const P3: u64 = 0x165667B19E3779F9;
    const P4: u64 = 0x85EBCA77C2B2AE63;
    const P5: u64 = 0x27D4EB2F165667C5;

    let len = data.len();
    let mut h: u64;

    let round = |acc: u64, inp: u64| -> u64 {
        acc.wrapping_add(inp.wrapping_mul(P2)).rotate_left(31).wrapping_mul(P1)
    };
    let merge = |acc: u64, val: u64| -> u64 {
        (acc ^ round(0, val)).wrapping_mul(P1).wrapping_add(P4)
    };
    let av = |mut x: u64| -> u64 {
        x ^= x >> 33; x = x.wrapping_mul(P2);
        x ^= x >> 29; x = x.wrapping_mul(P3);
        x ^= x >> 32; x
    };
    let r64 = |s: &[u8]| u64::from_le_bytes(s[..8].try_into().unwrap());
    let r32 = |s: &[u8]| u32::from_le_bytes(s[..4].try_into().unwrap()) as u64;

    if len >= 32 {
        let mut v1 = seed.wrapping_add(P1).wrapping_add(P2);
        let mut v2 = seed.wrapping_add(P2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(P1);
        let mut i = 0;
        while i + 32 <= len {
            v1 = round(v1, r64(&data[i..]));
            v2 = round(v2, r64(&data[i+8..]));
            v3 = round(v3, r64(&data[i+16..]));
            v4 = round(v4, r64(&data[i+24..]));
            i += 32;
        }
        h = v1.rotate_left(1).wrapping_add(v2.rotate_left(7))
              .wrapping_add(v3.rotate_left(12)).wrapping_add(v4.rotate_left(18));
        h = merge(h, v1); h = merge(h, v2); h = merge(h, v3); h = merge(h, v4);
    } else {
        h = seed.wrapping_add(P5);
    }

    h = h.wrapping_add(len as u64);
    let rem = &data[len & !31..];
    let mut i = 0;
    while i + 8 <= rem.len() {
        h ^= round(0, r64(&rem[i..]));
        h = h.rotate_left(27).wrapping_mul(P1).wrapping_add(P4);
        i += 8;
    }
    if i + 4 <= rem.len() {
        h ^= r32(&rem[i..]).wrapping_mul(P1);
        h = h.rotate_left(23).wrapping_mul(P2).wrapping_add(P3);
        i += 4;
    }
    while i < rem.len() {
        h ^= (rem[i] as u64).wrapping_mul(P5);
        h = h.rotate_left(11).wrapping_mul(P1);
        i += 1;
    }
    av(h)
}

fn now_epoch() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

fn derive_nonce(seed: &[u8; 12], counter: u32) -> [u8; 12] {
    let mut nonce = *seed;
    let cb = counter.to_le_bytes();
    for i in 0..4 { nonce[i] ^= cb[i]; }
    nonce
}

// ── Vault ─────────────────────────────────────────────────────────────────────

pub struct Vault {
    #[allow(dead_code)]
    path: PathBuf,
    file: File,
    header: VaultHeader,
    index: Vec<IndexEntry>,
    key: [u8; 32],
    nonce_seed: [u8; 12],
    searcher: FusedSearcher,
}

impl Vault {
    /// Open or create a vault in `dir`. Key auto-derived from hardware ID.
    pub fn open(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("vault.bin");
        let key = derive_key();

        if path.exists() {
            Self::open_existing(&path, key)
        } else {
            Self::create_new(&path, key)
        }
    }

    fn create_new(path: &Path, key: [u8; 32]) -> Result<Self> {
        let key_id = {
            let h = xxhash64(&key, 0);
            let mut id = [0u8; 16];
            id[..8].copy_from_slice(&h.to_le_bytes());
            id[8..16].copy_from_slice(&xxhash64(&key, h).to_le_bytes());
            id
        };
        let nonce_seed = {
            let t = now_epoch();
            let mut seed = [0u8; 12];
            seed[..8].copy_from_slice(&t.to_le_bytes());
            seed[8..12].copy_from_slice(&key[..4]);
            seed
        };
        let header = VaultHeader::new(key_id, nonce_seed);
        let mut file = OpenOptions::new()
            .read(true).write(true).create(true).truncate(true)
            .open(path)?;
        file.write_all(&header.to_bytes())?;
        file.flush()?;
        Ok(Self { path: path.to_path_buf(), file, header, index: Vec::new(), key, nonce_seed, searcher: FusedSearcher::new() })
    }

    fn open_existing(path: &Path, key: [u8; 32]) -> Result<Self> {
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;
        let mut hdr_buf = [0u8; HEADER_SIZE];
        file.read_exact(&mut hdr_buf)?;
        let header = VaultHeader::from_bytes(&hdr_buf)?;
        let nonce_seed = header.nonce_seed;
        let mut index = Vec::with_capacity(header.block_count as usize);
        file.seek(SeekFrom::Start(header.index_offset))?;
        for _ in 0..header.block_count {
            let mut entry_buf = [0u8; INDEX_ENTRY_SIZE];
            file.read_exact(&mut entry_buf)?;
            index.push(IndexEntry::from_bytes(&entry_buf));
        }
        Ok(Self { path: path.to_path_buf(), file, header, index, key, nonce_seed, searcher: FusedSearcher::new() })
    }

    /// Append a message (role + content), encrypt, and write immediately.
    /// Format: `role: content\n`
    pub fn append(&mut self, role: &[u8], content: &[u8]) -> Result<()> {
        let mut plaintext = Vec::with_capacity(role.len() + 2 + content.len() + 1);
        plaintext.extend_from_slice(role);
        plaintext.extend_from_slice(b": ");
        plaintext.extend_from_slice(content);
        plaintext.push(b'\n');
        self.flush_block(&plaintext)
    }

    fn flush_block(&mut self, plaintext: &[u8]) -> Result<()> {
        let histogram = compute_histogram(plaintext);
        let hash = xxhash64(plaintext, 0);
        let nonce_counter = self.header.block_count;
        let nonce = derive_nonce(&self.nonce_seed, nonce_counter);

        let mut ciphertext = plaintext.to_vec();
        crypto::encrypt(&self.key, &nonce, 0, &mut ciphertext);

        let block_offset = self.header.index_offset;
        self.file.seek(SeekFrom::Start(block_offset))?;
        self.file.write_all(&ciphertext)?;

        let entry = IndexEntry {
            offset: block_offset,
            length: ciphertext.len() as u32,
            timestamp: now_epoch(),
            xxhash: hash,
            nonce_counter,
            histogram,
        };
        self.index.push(entry);
        self.header.block_count += 1;
        self.header.index_offset = block_offset + ciphertext.len() as u64;

        self.write_index()?;
        self.write_header()
    }

    /// Decrypt a specific block by index.
    pub fn decrypt_block(&mut self, block_index: usize) -> Result<Vec<u8>> {
        if block_index >= self.index.len() {
            return Err(Error::Vault("block index out of range"));
        }
        let entry = &self.index[block_index];
        let offset = entry.offset;
        let length = entry.length as usize;
        let nonce_counter = entry.nonce_counter;
        let expected_hash = entry.xxhash;

        let mut ciphertext = vec![0u8; length];
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read_exact(&mut ciphertext)?;

        let nonce = derive_nonce(&self.nonce_seed, nonce_counter);
        crypto::decrypt(&self.key, &nonce, 0, &mut ciphertext);

        let actual_hash = xxhash64(&ciphertext, 0);
        if actual_hash != expected_hash {
            return Err(Error::Vault("integrity check failed"));
        }
        Ok(ciphertext)
    }

    /// Read raw encrypted block + its nonce. Used by fused search.
    pub(crate) fn read_encrypted_block(&mut self, block_index: usize)
        -> Result<(Vec<u8>, [u8; 12])>
    {
        if block_index >= self.index.len() {
            return Err(Error::Vault("block index out of range"));
        }
        let entry = &self.index[block_index];
        let offset = entry.offset;
        let length = entry.length as usize;
        let nonce_counter = entry.nonce_counter;

        let mut ciphertext = vec![0u8; length];
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read_exact(&mut ciphertext)?;

        let nonce = derive_nonce(&self.nonce_seed, nonce_counter);
        Ok((ciphertext, nonce))
    }

    /// Get the vault encryption key (for fused search).
    pub(crate) fn key(&self) -> &[u8; 32] { &self.key }

    /// How many blocks are in the vault.
    pub fn block_count(&self) -> u32 { self.header.block_count }

    /// Hash of the last block (for session token integrity).
    pub fn last_block_hash(&self) -> Option<u64> {
        self.index.last().map(|e| e.xxhash)
    }

    /// Search vault blocks for the query using histogram cosine similarity + fused decrypt+search.
    pub fn search(&mut self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        if self.index.is_empty() {
            return Ok(vec![]);
        }
        let n = self.index.len();
        let query_hist = compute_histogram(query.as_bytes());
        let query_norm = normalize_histogram(&query_hist);
        let qmag: f32 = query_norm.iter().map(|x| x * x).sum::<f32>().sqrt();
        if qmag < 1e-9 {
            return Ok(vec![]);
        }

        // Score blocks by cosine similarity
        let mut scored: Vec<(usize, f32)> = (0..n).map(|i| {
            let bnorm = normalize_histogram(&self.index[i].histogram);
            let recency = if n <= 1 { 1.0 } else { i as f32 / (n - 1) as f32 };
            let cos = cosine_similarity(&query_norm, &bnorm);
            (i, cos * (0.85 + 0.15 * recency))
        }).collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored.retain(|(_, s)| *s > 0.01);

        let needles: Vec<&[u8]> = query.split_whitespace().map(|w| w.as_bytes()).collect();
        let key_copy = *self.key();
        let mut results = Vec::with_capacity(scored.len());

        for (block_idx, score) in scored {
            let (ciphertext, nonce) = self.read_encrypted_block(block_idx)?;
            let fused = self.searcher.search(&key_copy, &nonce, &ciphertext, &needles);
            let lines: Vec<String> = fused.context_lines
                .into_iter()
                .map(|l| String::from_utf8_lossy(&l).to_string())
                .collect();
            results.push(SearchResult { block_index: block_idx, score, lines });
        }
        Ok(results)
    }

    /// Decrypt last `n` blocks (for /teleport — explicit user action).
    pub fn decrypt_last_n(&mut self, n: usize) -> Result<Vec<Vec<u8>>> {
        let start = self.index.len().saturating_sub(n);
        let mut blocks = Vec::with_capacity(n);
        for i in start..self.index.len() {
            blocks.push(self.decrypt_block(i)?);
        }
        Ok(blocks)
    }

    fn write_index(&mut self) -> Result<()> {
        self.file.seek(SeekFrom::Start(self.header.index_offset))?;
        for entry in &self.index {
            self.file.write_all(&entry.to_bytes())?;
        }
        self.file.flush()?;
        Ok(())
    }

    fn write_header(&mut self) -> Result<()> {
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&self.header.to_bytes())?;
        self.file.flush()?;
        Ok(())
    }
}

/// A search result with score and matched context lines.
pub struct SearchResult {
    pub block_index: usize,
    pub score: f32,
    pub lines: Vec<String>,
}

impl Drop for Vault {
    fn drop(&mut self) {
        unsafe {
            ffi::zeroize(self.key.as_mut_ptr(), 32);
        }
    }
}

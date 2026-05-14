//! Encrypted vault — append-only conversation storage.
//!
//! Key derivation: XOR-obfuscated seed ^ hardware_id (from /etc/machine-id).
//! Format: binary header + encrypted blocks + index at tail.
//! Plaintext never lingers — blocks are encrypted immediately on append.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{Error, Result};
use crate::kernels::ffi;
use crate::storage::aead;
use crate::storage::crypto;
use crate::storage::key;
use crate::storage::search::FusedSearcher;
use crate::storage::secure::SecureBuffer;

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

// ── VaultHeaderV2 (AEAD layout, on-disk wiring lands in Task 9) ───────────────

pub const HEADER_SIZE_V2: usize = 64;
const VAULT_VERSION_V2: u16 = 2;

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
struct IndexEntry {
    offset: u64,
    length: u32,
    timestamp: u64,
    /// Was xxhash(plaintext) in v1; integrity is now carried by the per-block
    /// AEAD tag.  Zeroed on write; ignored on read.  Layout preserved so a
    /// v1→v2 migration can stream entries without shifting offsets.
    _reserved: u64,
    nonce_counter: u32,
    histogram: [u8; 256],
}

impl IndexEntry {
    fn to_bytes(&self) -> [u8; INDEX_ENTRY_SIZE] {
        let mut buf = [0u8; INDEX_ENTRY_SIZE];
        buf[0..8].copy_from_slice(&self.offset.to_le_bytes());
        buf[8..12].copy_from_slice(&self.length.to_le_bytes());
        buf[12..20].copy_from_slice(&self.timestamp.to_le_bytes());
        buf[20..28].copy_from_slice(&0u64.to_le_bytes());
        buf[28..32].copy_from_slice(&self.nonce_counter.to_le_bytes());
        buf[32..288].copy_from_slice(&self.histogram);
        buf
    }

    fn from_bytes(buf: &[u8; INDEX_ENTRY_SIZE]) -> Self {
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

fn now_epoch() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}

/// v2 per-block nonce: 8 bytes of nonce_seed_8 || u32_le(counter).
/// Counter values are bounded to [0, 0x80000000) — the high bit is
/// reserved for the header-tag domain (see [`derive_header_nonce`]).
fn derive_block_nonce(seed8: &[u8; 8], counter: u32) -> [u8; 12] {
    let mut nonce = [0u8; 12];
    nonce[0..8].copy_from_slice(seed8);
    nonce[8..12].copy_from_slice(&counter.to_le_bytes());
    nonce
}

/// v2 header nonce: 8 bytes of nonce_seed_8 || u32_le(0x80000000 | rewrites).
/// High-bit-set 4-byte tail can never collide with a block nonce because
/// `flush_block` refuses counter ≥ 0x80000000.
fn derive_header_nonce(seed8: &[u8; 8], rewrites: u32) -> [u8; 12] {
    let mut nonce = [0u8; 12];
    nonce[0..8].copy_from_slice(seed8);
    let domain = 0x8000_0000u32 | rewrites;
    nonce[8..12].copy_from_slice(&domain.to_le_bytes());
    nonce
}

/// Per-block AAD: key_id || version_le || counter_le || timestamp_le || histogram.
/// Binds the ciphertext to this exact (vault, version, position, time, content
/// shape) tuple so a swapped block fails AEAD verification.
fn build_block_aad(
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

// ── Vault ─────────────────────────────────────────────────────────────────────

pub struct Vault {
    file: File,
    header: VaultHeaderV2,
    index: Vec<IndexEntry>,
    /// Encryption key in mlock'd + SIMD-zeroize-on-Drop memory. Backed
    /// by `SecureBuffer` so the key never reaches the swap file and is
    /// wiped on clean shutdown. Size is always 32 bytes.
    key: SecureBuffer,
    searcher: FusedSearcher,
}

/// Pull the 32-byte key array out of the SecureBuffer for the crypto
/// API. Safe — the buffer is initialized to 32 bytes at construction.
fn key_array(buf: &SecureBuffer) -> [u8; 32] {
    let mut out = [0u8; 32];
    out.copy_from_slice(buf.as_slice());
    out
}

impl Vault {
    /// Open or create a vault in `dir`. Key auto-derived from hardware ID.
    pub fn open(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("vault.bin");
        let key = key::derive_key().map_err(Error::Vault)?;

        if path.exists() {
            Self::open_existing(&path, key)
        } else {
            Self::create_new(&path, key)
        }
    }

    fn create_new(path: &Path, key_bytes: [u8; 32]) -> Result<Self> {
        let key_id = {
            let h = key::xxhash64(&key_bytes, 0);
            let mut id = [0u8; 16];
            id[..8].copy_from_slice(&h.to_le_bytes());
            id[8..16].copy_from_slice(&key::xxhash64(&key_bytes, h).to_le_bytes());
            id
        };
        // nonce_seed_8 uniqueness: two vaults created on the same machine
        // must not share it (would risk nonce reuse if they shared a key).
        // Nanosecond precision is good enough — `unique_dir`-style back-to-back
        // creations land in different nanos buckets.  Never mix key bytes in:
        // doing so would write a key prefix into the cleartext file header.
        let nonce_seed_8 = {
            let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
            let mut seed = [0u8; 8];
            seed[..4].copy_from_slice(&now.subsec_nanos().to_le_bytes());
            seed[4..8].copy_from_slice(&(now.as_secs() as u32).to_le_bytes());
            seed
        };
        let header = VaultHeaderV2::new(key_id, nonce_seed_8);
        let file = OpenOptions::new()
            .read(true).write(true).create(true).truncate(true)
            .open(path)?;
        let mut key = SecureBuffer::new(32);
        key.write(&key_bytes);
        let mut vault = Self { file, header, index: Vec::new(), key, searcher: FusedSearcher::new() };
        // First write: a valid v2 header with header_rewrites = 1 and a tag
        // covering the empty index.  Subsequent appends bump rewrites.
        vault.recompute_and_write_header_tag()?;
        Ok(vault)
    }

    fn open_existing(path: &Path, key_bytes: [u8; 32]) -> Result<Self> {
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;
        let file_size = file.metadata()?.len();

        // Peek the version byte (header bytes 4..6) to route v1 vs v2.
        // v1 layout was 64 bytes too but with version=1; we defer auto-migration
        // to a later commit and refuse here so callers get a clear error.
        let mut version_buf = [0u8; 6];
        file.seek(SeekFrom::Start(0))?;
        file.read_exact(&mut version_buf)?;
        let version = u16::from_le_bytes(version_buf[4..6].try_into().unwrap());
        if version == 1 {
            return Err(Error::Vault(
                "v1 vault detected — auto-migration not yet wired (Task 11)",
            ));
        }

        let mut hdr_buf = [0u8; HEADER_SIZE_V2];
        file.seek(SeekFrom::Start(0))?;
        file.read_exact(&mut hdr_buf)?;
        let header = VaultHeaderV2::from_bytes(&hdr_buf)?;

        // Sanity-check block_count against actual file size. A tampered
        // or corrupted header claiming u32::MAX blocks would otherwise
        // make Vec::with_capacity attempt a multi-terabyte allocation
        // and abort the process — a DoS vector via vault.bin tampering.
        let entries_region = file_size.saturating_sub(header.index_offset);
        let max_entries = (entries_region / INDEX_ENTRY_SIZE as u64) as usize;
        if header.block_count as usize > max_entries {
            return Err(Error::Vault("vault header block_count exceeds file size"));
        }

        let mut index = Vec::with_capacity(header.block_count as usize);
        file.seek(SeekFrom::Start(header.index_offset))?;
        for _ in 0..header.block_count {
            let mut entry_buf = [0u8; INDEX_ENTRY_SIZE];
            file.read_exact(&mut entry_buf)?;
            index.push(IndexEntry::from_bytes(&entry_buf));
        }
        let mut key = SecureBuffer::new(32);
        key.write(&key_bytes);
        Ok(Self { file, header, index, key, searcher: FusedSearcher::new() })
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
        // Counter-wrap guard.  Tightened from u32::MAX to 0x80000000 because
        // the high bit of the nonce-counter slot is reserved for the
        // header-tag domain (see `derive_header_nonce`).  At 1 append/s this
        // still gives ~68 years before exhaustion.
        if self.header.block_count >= 0x8000_0000 {
            return Err(Error::Vault(
                "vault block counter exhausted — refuse to reuse nonce",
            ));
        }

        let histogram = key::compute_histogram(plaintext);
        let timestamp = now_epoch();
        let nonce_counter = self.header.block_count;
        let nonce = derive_block_nonce(&self.header.nonce_seed_8, nonce_counter);
        let aad = build_block_aad(
            &self.header.key_id,
            VAULT_VERSION_V2,
            nonce_counter,
            timestamp,
            &histogram,
        );

        let key_bytes = key_array(&self.key);
        let mut ct = plaintext.to_vec();
        let mut tag = [0u8; 16];
        aead::seal(&key_bytes, &nonce, &aad, &mut ct, &mut tag);

        let block_offset = self.header.index_offset;
        self.file.seek(SeekFrom::Start(block_offset))?;
        self.file.write_all(&ct)?;
        self.file.write_all(&tag)?;

        let entry = IndexEntry {
            offset: block_offset,
            length: (ct.len() + 16) as u32,
            timestamp,
            _reserved: 0,
            nonce_counter,
            histogram,
        };
        self.index.push(entry);
        self.header.block_count += 1;
        self.header.index_offset = block_offset + ct.len() as u64 + 16;

        self.write_index()?;
        self.recompute_and_write_header_tag()
    }

    /// Decrypt a specific block by index.
    pub fn decrypt_block(&mut self, block_index: usize) -> Result<Vec<u8>> {
        if block_index >= self.index.len() {
            return Err(Error::Vault("block index out of range"));
        }
        let entry = &self.index[block_index];
        let offset = entry.offset;
        let length = entry.length as usize;
        if length < 16 {
            return Err(Error::Vault("block length less than AEAD tag size"));
        }
        let nonce_counter = entry.nonce_counter;
        let timestamp = entry.timestamp;
        let histogram = entry.histogram;

        let mut ct_and_tag = vec![0u8; length];
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read_exact(&mut ct_and_tag)?;

        let ct_len = length - 16;
        let mut tag = [0u8; 16];
        tag.copy_from_slice(&ct_and_tag[ct_len..]);
        ct_and_tag.truncate(ct_len);

        let nonce = derive_block_nonce(&self.header.nonce_seed_8, nonce_counter);
        let key_bytes = key_array(&self.key);
        let aad = build_block_aad(
            &self.header.key_id,
            VAULT_VERSION_V2,
            nonce_counter,
            timestamp,
            &histogram,
        );
        aead::open(&key_bytes, &nonce, &aad, &mut ct_and_tag, &tag)?;
        Ok(ct_and_tag)
    }

    /// Read raw encrypted block (ct only, AEAD tag stripped) + its nonce.
    /// Used by fused search; tag verification is the caller's responsibility
    /// (Task 12 will switch search to a verify-then-search flow).
    pub(crate) fn read_encrypted_block(&mut self, block_index: usize)
        -> Result<(Vec<u8>, [u8; 12])>
    {
        if block_index >= self.index.len() {
            return Err(Error::Vault("block index out of range"));
        }
        let entry = &self.index[block_index];
        let offset = entry.offset;
        let length = entry.length as usize;
        if length < 16 {
            return Err(Error::Vault("block length less than AEAD tag size"));
        }
        let nonce_counter = entry.nonce_counter;

        let mut ct_and_tag = vec![0u8; length];
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read_exact(&mut ct_and_tag)?;
        ct_and_tag.truncate(length - 16);

        let nonce = derive_block_nonce(&self.header.nonce_seed_8, nonce_counter);
        Ok((ct_and_tag, nonce))
    }

    /// Get a copy of the vault encryption key (for fused search).
    /// Returns by value rather than by reference because the underlying
    /// storage is a SecureBuffer; callers were already copying out via
    /// `let key_copy = *self.key()`.
    pub(crate) fn key(&self) -> [u8; 32] { key_array(&self.key) }

    /// How many blocks are in the vault.
    pub fn block_count(&self) -> u32 { self.header.block_count }

    /// Search vault blocks for the query using histogram cosine similarity + fused decrypt+search.
    pub fn search(&mut self, query: &str, top_k: usize) -> Result<Vec<SearchResult>> {
        if self.index.is_empty() {
            return Ok(vec![]);
        }
        let n = self.index.len();
        let query_hist = key::compute_histogram(query.as_bytes());
        let query_norm = key::normalize_histogram(&query_hist);
        let qmag: f32 = query_norm.iter().map(|x| x * x).sum::<f32>().sqrt();
        if qmag < 1e-9 {
            return Ok(vec![]);
        }

        // Score blocks by cosine similarity
        let mut scored: Vec<(usize, f32)> = (0..n).map(|i| {
            let bnorm = key::normalize_histogram(&self.index[i].histogram);
            let recency = if n <= 1 { 1.0 } else { i as f32 / (n - 1) as f32 };
            let cos = key::cosine_similarity(&query_norm, &bnorm);
            (i, cos * (0.85 + 0.15 * recency))
        }).collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored.retain(|(_, s)| *s > 0.01);

        let needles: Vec<&[u8]> = query.split_whitespace().map(|w| w.as_bytes()).collect();
        let key_copy = self.key();
        let mut results = Vec::with_capacity(scored.len());

        for (block_idx, score) in scored {
            let (ciphertext, nonce) = self.read_encrypted_block(block_idx)?;
            // v2 encrypts blocks at counter=1 (counter=0 is reserved for the
            // Poly1305 OTK).  Tag verification will move into search itself
            // in Task 12 (verify-then-search).
            let fused = self.searcher.search(&key_copy, &nonce, 1, &ciphertext, &needles);
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

    /// Bump `header_rewrites`, Poly1305-MAC the new header[0..46] || index
    /// using a domain-separated nonce, write the header back to the file,
    /// and fsync.  Called at the end of every operation that mutates the
    /// header (create, append, future migration writes).
    fn recompute_and_write_header_tag(&mut self) -> Result<()> {
        self.header.header_rewrites = self.header.header_rewrites.wrapping_add(1);

        let header_bytes = self.header.to_bytes();
        let prefix = &header_bytes[0..46];
        let mut mac_input =
            Vec::with_capacity(46 + self.index.len() * INDEX_ENTRY_SIZE);
        mac_input.extend_from_slice(prefix);
        for e in &self.index {
            mac_input.extend_from_slice(&e.to_bytes());
        }

        let key_bytes = key_array(&self.key);
        let header_nonce =
            derive_header_nonce(&self.header.nonce_seed_8, self.header.header_rewrites);

        // Same OTK derivation as aead::seal, but no encryption — just MAC.
        let mut otk = [0u8; 32];
        crypto::keystream(&key_bytes, &header_nonce, 0, &mut otk);
        let mut tag = [0u8; 16];
        unsafe {
            ffi::poly1305_mac(
                otk.as_ptr(),
                mac_input.as_ptr(),
                mac_input.len() as i32,
                tag.as_mut_ptr(),
            );
            ffi::zeroize(otk.as_mut_ptr(), 32);
        }
        self.header.header_tag = tag;

        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&self.header.to_bytes())?;
        self.file.sync_data()?;
        Ok(())
    }
}

/// A search result with score and matched context lines.
pub struct SearchResult {
    pub block_index: usize,
    pub score: f32,
    pub lines: Vec<String>,
}

// Drop impl is no longer needed: SecureBuffer's own Drop handles
// SIMD-zeroization and page-unlock for `self.key`. Removing it
// avoids a double-zero pass (harmless but wasteful).


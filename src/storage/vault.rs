//! Encrypted vault — append-only conversation storage.
//!
//! Key derivation: XOR-obfuscated seed ^ hardware_id (from /etc/machine-id).
//! Format: binary header + encrypted blocks + index at tail (see
//! [`vault_format`] for the byte-layout details).
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
use crate::storage::vault_format::{
    build_block_aad, derive_block_nonce, derive_header_nonce, now_epoch,
    IndexEntry, INDEX_ENTRY_SIZE, VAULT_VERSION_V2,
};

pub use crate::storage::vault_format::{VaultHeaderV2, HEADER_SIZE_V2};

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

        let mut hdr_buf = [0u8; HEADER_SIZE_V2];
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

        // Verify header_tag — Poly1305 MAC over header[0..46] || serialized
        // index, with OTK derived from a domain-separated header nonce.
        // Catches tampering of block_count, index_offset, key_id, nonce_seed_8,
        // header_rewrites, or any index entry byte.  The reserved bytes
        // (62..64) and the tag itself (46..62) are not in the MAC input.
        let header_bytes = header.to_bytes();
        let mut mac_input =
            Vec::with_capacity(46 + index.len() * INDEX_ENTRY_SIZE);
        mac_input.extend_from_slice(&header_bytes[0..46]);
        for e in &index {
            mac_input.extend_from_slice(&e.to_bytes());
        }
        let header_nonce =
            derive_header_nonce(&header.nonce_seed_8, header.header_rewrites);
        let mut otk = [0u8; 32];
        crypto::keystream(&key_bytes, &header_nonce, 0, &mut otk);
        let ok = unsafe {
            ffi::poly1305_verify(
                otk.as_ptr(),
                mac_input.as_ptr(),
                mac_input.len() as i32,
                header.header_tag.as_ptr(),
            )
        };
        unsafe { ffi::zeroize(otk.as_mut_ptr(), 32); }
        if ok == 0 {
            return Err(Error::Vault("vault header or index has been tampered"));
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

    /// Read a block from disk and split it into `(ct, tag, nonce)`.  Used by
    /// `Vault::search`'s verify-then-search path: the caller MAC-verifies
    /// `tag` against `ct`+AAD before passing `ct` to the fused decrypt+search
    /// kernel, so a tampered block never reaches search results.
    pub(crate) fn read_encrypted_block(&mut self, block_index: usize)
        -> Result<(Vec<u8>, [u8; 16], [u8; 12])>
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
        let ct_len = length - 16;
        let mut tag = [0u8; 16];
        tag.copy_from_slice(&ct_and_tag[ct_len..]);
        ct_and_tag.truncate(ct_len);

        let nonce = derive_block_nonce(&self.header.nonce_seed_8, nonce_counter);
        Ok((ct_and_tag, tag, nonce))
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
            let (ciphertext, tag, nonce) = self.read_encrypted_block(block_idx)?;

            // MAC-verify the candidate block before letting the fused
            // decrypt+search kernel touch it.  A tampered block silently
            // drops out of the result set — no plaintext, partial line,
            // or even block index ever leaks through.
            let entry = &self.index[block_idx];
            let aad = build_block_aad(
                &self.header.key_id,
                VAULT_VERSION_V2,
                entry.nonce_counter,
                entry.timestamp,
                &entry.histogram,
            );
            if aead::verify(&key_copy, &nonce, &aad, &ciphertext, &tag).is_err() {
                continue;
            }

            // v2 encrypts blocks at counter=1 (counter=0 is reserved for the
            // Poly1305 OTK).
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

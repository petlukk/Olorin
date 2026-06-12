//! Encrypted vault — append-only conversation storage.
//!
//! Key derivation: Argon2id (RFC 9106) over passphrase + per-vault salt
//! (see [`crate::storage::key`] and [`crate::storage::argon2id`]).
//!
//! Format v4 — `[header slot A][header slot B][record0][record1]…`, where each
//! record is `index_entry ‖ ciphertext ‖ tag`. Append writes the record at the
//! data-end (fresh space, never over a committed record), fsyncs, then commits
//! the header to the *other* slot and fsyncs. A crash before the commit leaves
//! the in-flight record beyond `block_count`, so reopen ignores it — committed
//! history is never clobbered or made unopenable (robustness wave-two F1/F2).
//! On open, the MAC-valid header slot with the highest generation wins, so a
//! torn header commit falls back to the previous generation.
//! Plaintext never lingers — blocks are encrypted immediately on append.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{Error, Result};
use crate::kernels::ffi;
use crate::storage::aead;
use crate::storage::argon2id::Params as KdfParams;
use crate::storage::crypto;
use crate::storage::key;
use crate::storage::search::FusedSearcher;
use crate::storage::secure::SecureBuffer;
use crate::storage::vault_format::{
    build_block_aad, derive_block_nonce, derive_header_nonce, now_epoch,
    IndexEntry, INDEX_ENTRY_SIZE, N_HEADER_SLOTS, RECORDS_START, VAULT_VERSION,
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

/// File offset of the header slot for a given generation (`header_rewrites`).
/// The two slots alternate by parity, so committing generation G to one slot
/// always leaves G−1 intact in the other — a torn header write can never
/// destroy the only valid commit point (v4 atomic-commit, F1/F2).
fn header_slot_offset(header_rewrites: u32) -> u64 {
    (header_rewrites as u64 % N_HEADER_SLOTS) * HEADER_SIZE_V2 as u64
}

/// Poly1305 tag over `header[0..46]` (magic, version, block_count, data_end,
/// key_id, nonce_seed, header_rewrites — everything load-bearing; the tag
/// itself and reserved bytes are excluded), OTK from the generation-derived
/// header nonce. The per-block index fields (timestamp/counter/histogram) are
/// authenticated by each block's own AEAD AAD, so the header MAC need not cover
/// the records.
fn compute_header_mac(key_bytes: &[u8; 32], header: &VaultHeaderV2) -> [u8; 16] {
    let header_bytes = header.to_bytes();
    let header_nonce = derive_header_nonce(&header.nonce_seed_8, header.header_rewrites);
    let mut otk = [0u8; 32];
    crypto::keystream(key_bytes, &header_nonce, 0, &mut otk);
    let mut tag = [0u8; 16];
    unsafe {
        ffi::poly1305_mac(otk.as_ptr(), header_bytes.as_ptr(), 46, tag.as_mut_ptr());
        ffi::zeroize(otk.as_mut_ptr(), 32);
    }
    tag
}

/// Constant-time verify of a header slot's tag against `header[0..46]`.
fn verify_header_mac(key_bytes: &[u8; 32], header: &VaultHeaderV2) -> bool {
    let header_bytes = header.to_bytes();
    let header_nonce = derive_header_nonce(&header.nonce_seed_8, header.header_rewrites);
    let mut otk = [0u8; 32];
    crypto::keystream(key_bytes, &header_nonce, 0, &mut otk);
    let ok = unsafe {
        ffi::poly1305_verify(otk.as_ptr(), header_bytes.as_ptr(), 46, header.header_tag.as_ptr())
    };
    unsafe { ffi::zeroize(otk.as_mut_ptr(), 32); }
    ok != 0
}

impl Vault {
    /// Open or create a vault in `dir` using the production KDF
    /// profile (`KdfParams::VAULT_DEFAULT`).  Generates `vault.salt`
    /// on first call; subsequent calls reuse it.  Wrong passphrase →
    /// `Error::Vault("vault header or index has been tampered")` (the
    /// key_id+MAC mismatch surfaces as a tamper error since both
    /// derivations produce 32 random-looking bytes).
    pub fn open(dir: &Path, passphrase: &[u8]) -> Result<Self> {
        Self::open_with(dir, passphrase, KdfParams::VAULT_DEFAULT)
    }

    /// Open or create a vault with explicit KDF parameters.  Same as
    /// [`Vault::open`] but lets tests use a low-cost Argon2id profile
    /// so the suite isn't dominated by KDF work.  Production callers
    /// should use `open()` — the default profile is the threat-model
    /// commitment.
    pub fn open_with(dir: &Path, passphrase: &[u8], kdf: KdfParams) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("vault.bin");
        let salt = key::load_or_create_salt(dir)?;
        let key = key::derive_key(passphrase, &salt, kdf)?;

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
        // Single-writer guard: an exclusive advisory lock held for the Vault's
        // lifetime. Without it, two processes opening the same vault would
        // append at the same offset and reuse the same per-block nonce (a
        // two-time-pad). Released automatically on Drop / process death.
        if !crate::platform::lock::try_lock_file_exclusive(&file) {
            return Err(Error::Vault("vault is already open by another Olorin process"));
        }
        let mut key = SecureBuffer::new(32);
        key.write(&key_bytes);
        let mut vault = Self { file, header, index: Vec::new(), key, searcher: FusedSearcher::new() };
        // Initialize BOTH header slots with a valid generation-1 header (empty
        // vault), so `open` always finds a valid slot no matter which one the
        // first append's commit lands on. Subsequent commits bump the generation
        // and alternate slots.
        vault.header.header_rewrites = 1;
        vault.header.header_tag = compute_header_mac(&key_bytes, &vault.header);
        let bytes = vault.header.to_bytes();
        for slot in 0..N_HEADER_SLOTS {
            vault.file.seek(SeekFrom::Start(slot * HEADER_SIZE_V2 as u64))?;
            vault.file.write_all(&bytes)?;
        }
        vault.file.sync_data()?;
        Ok(vault)
    }

    fn open_existing(path: &Path, key_bytes: [u8; 32]) -> Result<Self> {
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;
        // Single-writer guard (see create_new): reject a concurrent open so two
        // processes can't append at the same offset and reuse a block nonce.
        if !crate::platform::lock::try_lock_file_exclusive(&file) {
            return Err(Error::Vault("vault is already open by another Olorin process"));
        }
        let file_size = file.metadata()?.len();
        if file_size < RECORDS_START {
            return Err(Error::Vault("vault file too small for a v4 header"));
        }

        // Recover the commit point: read both header slots, keep the MAC-valid
        // one with the highest generation. A torn/corrupt slot is skipped, so a
        // crash during a header commit falls back to the previous generation.
        let mut header: Option<VaultHeaderV2> = None;
        for slot in 0..N_HEADER_SLOTS {
            let mut buf = [0u8; HEADER_SIZE_V2];
            file.seek(SeekFrom::Start(slot * HEADER_SIZE_V2 as u64))?;
            if file.read_exact(&mut buf).is_err() { continue; }
            let Ok(h) = VaultHeaderV2::from_bytes(&buf) else { continue; };
            if !verify_header_mac(&key_bytes, &h) { continue; }
            if header.as_ref().map_or(true, |b| h.header_rewrites > b.header_rewrites) {
                header = Some(h);
            }
        }
        let header = header.ok_or(Error::Vault("vault header or index has been tampered"))?;

        // `index_offset` is the data-end. Clamp to the real file size, then
        // bound block_count against the smallest possible record (entry + empty
        // ct + tag) so a corrupt count can't drive a huge allocation (DoS guard).
        let data_end = header.index_offset.min(file_size);
        let min_record = INDEX_ENTRY_SIZE as u64 + 16;
        let max_blocks = data_end.saturating_sub(RECORDS_START) / min_record;
        if header.block_count as u64 > max_blocks {
            return Err(Error::Vault("vault header block_count exceeds file size"));
        }

        // Scan committed records to rebuild the in-memory index. Each record is
        // `entry(288) ‖ ct ‖ tag`; the ct offset is recomputed from the cursor
        // (not trusted from the stored entry). A short/oversized record fails
        // closed. Per-block content integrity is checked later by AEAD.
        let mut index = Vec::with_capacity(header.block_count as usize);
        let mut cursor = RECORDS_START;
        for _ in 0..header.block_count {
            if cursor + INDEX_ENTRY_SIZE as u64 > data_end {
                return Err(Error::Vault("vault record truncated"));
            }
            let mut entry_buf = [0u8; INDEX_ENTRY_SIZE];
            file.seek(SeekFrom::Start(cursor))?;
            file.read_exact(&mut entry_buf)?;
            let mut entry = IndexEntry::from_bytes(&entry_buf);
            let ct_offset = cursor + INDEX_ENTRY_SIZE as u64;
            let length = entry.length as u64;
            if length < 16 || ct_offset + length > data_end {
                return Err(Error::Vault("vault record length invalid"));
            }
            entry.offset = ct_offset;
            index.push(entry);
            cursor = ct_offset + length;
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
            VAULT_VERSION,
            nonce_counter,
            timestamp,
            &histogram,
        );

        let key_bytes = key_array(&self.key);
        let mut ct = plaintext.to_vec();
        let mut tag = [0u8; 16];
        aead::seal(&key_bytes, &nonce, &aad, &mut ct, &mut tag);

        // v4 append-only record: write `entry ‖ ct ‖ tag` at the data-end — in
        // fresh space, never over a committed record — then fsync, THEN commit
        // the header. A crash before the commit leaves this record beyond
        // block_count, so reopen ignores it; committed history is never
        // clobbered and is never made unopenable (F1/F2).
        let entry_offset = self.header.index_offset;
        let ct_offset = entry_offset + INDEX_ENTRY_SIZE as u64;
        let entry = IndexEntry {
            offset: ct_offset,
            length: (ct.len() + 16) as u32,
            timestamp,
            _reserved: 0,
            nonce_counter,
            histogram,
        };
        self.file.seek(SeekFrom::Start(entry_offset))?;
        self.file.write_all(&entry.to_bytes())?;
        self.file.write_all(&ct)?;
        self.file.write_all(&tag)?;
        self.file.sync_data()?; // barrier: record durable before the header commit

        self.index.push(entry);
        self.header.block_count += 1;
        self.header.index_offset = ct_offset + ct.len() as u64 + 16;

        self.commit_header()
    }

    /// Commit the in-memory header: bump the generation, MAC `header[0..46]`,
    /// write it to the generation's slot, and fsync. The alternating slot means
    /// the previous generation survives a torn write (v4 atomic-commit).
    fn commit_header(&mut self) -> Result<()> {
        self.header.header_rewrites = self.header.header_rewrites.wrapping_add(1);
        let key_bytes = key_array(&self.key);
        self.header.header_tag = compute_header_mac(&key_bytes, &self.header);
        let bytes = self.header.to_bytes();
        let off = header_slot_offset(self.header.header_rewrites);
        self.file.seek(SeekFrom::Start(off))?;
        self.file.write_all(&bytes)?;
        self.file.sync_data()?;
        Ok(())
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
            VAULT_VERSION,
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
                VAULT_VERSION,
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

}

/// A search result with score and matched context lines.
pub struct SearchResult {
    pub block_index: usize,
    pub score: f32,
    pub lines: Vec<String>,
}

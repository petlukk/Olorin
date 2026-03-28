//! Encrypted vault — conversation storage with append-only write path.

pub mod format;
pub mod index;
pub mod search;
pub mod fused_search;

use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use format::{VaultHeader, IndexEntry, HEADER_SIZE, INDEX_ENTRY_SIZE, BLOCK_SIZE};
use index::{compute_histogram, xxhash64};
pub use search::SearchResult;

#[derive(thiserror::Error, Debug)]
pub enum VaultError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid vault file: {0}")]
    InvalidFormat(String),
    #[error("encryption error: {0}")]
    Crypto(String),
    #[error("integrity check failed for block {0}")]
    IntegrityFailed(usize),
}

/// Trait for the crypto layer, allowing real eachacha or test implementations.
pub trait VaultCrypto: Send {
    fn encrypt(&self, plaintext: &[u8], key: &[u8; 32], nonce: &[u8; 12]) -> Result<Vec<u8>, String>;
    fn decrypt(&self, ciphertext: &[u8], key: &[u8; 32], nonce: &[u8; 12]) -> Result<Vec<u8>, String>;
}

/// Real eachacha-backed crypto. Requires a compiled .so kernel.
pub struct EachachaCrypto {
    lib_path: PathBuf,
}

impl EachachaCrypto {
    pub fn new(lib_path: PathBuf) -> Self {
        Self { lib_path }
    }
}

impl VaultCrypto for EachachaCrypto {
    fn encrypt(&self, plaintext: &[u8], key: &[u8; 32], nonce: &[u8; 12]) -> Result<Vec<u8>, String> {
        eachacha::encrypt(plaintext, key, nonce, &self.lib_path).map_err(|e| e.to_string())
    }

    fn decrypt(&self, ciphertext: &[u8], key: &[u8; 32], nonce: &[u8; 12]) -> Result<Vec<u8>, String> {
        eachacha::decrypt(ciphertext, key, nonce, &self.lib_path).map_err(|e| e.to_string())
    }
}

/// Find `libchacha20.so` — checks build-time path first, then `~/.olorin/lib/`.
pub fn find_chacha_lib() -> Option<PathBuf> {
    // Build-time path set by olorin-core build.rs
    if let Some(p) = option_env!("CHACHA_LIB_PATH") {
        let path = PathBuf::from(p);
        if path.is_file() {
            return Some(path);
        }
    }
    // Runtime: search extracted kernels in ~/.olorin/lib/
    let home = home::home_dir()?;
    let lib_base = home.join(".olorin/lib");
    if !lib_base.is_dir() {
        return None;
    }
    let mut best: Option<(PathBuf, std::time::SystemTime)> = None;
    for entry in std::fs::read_dir(&lib_base).ok()? {
        let entry = entry.ok()?;
        let so = entry.path().join("libchacha20.so");
        if so.is_file() {
            let mtime = so.metadata().ok()?.modified().ok()?;
            if best.as_ref().map_or(true, |(_, t)| mtime > *t) {
                best = Some((so, mtime));
            }
        }
    }
    best.map(|(p, _)| p)
}

fn derive_nonce(seed: &[u8; 12], counter: u32) -> [u8; 12] {
    let mut nonce = *seed;
    let counter_bytes = counter.to_le_bytes();
    for i in 0..4 {
        nonce[i] ^= counter_bytes[i];
    }
    nonce
}

fn now_epoch() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub struct Vault {
    #[allow(dead_code)]
    path: PathBuf,
    file: File,
    header: VaultHeader,
    index: Vec<IndexEntry>,
    buffer: Vec<u8>,
    key: [u8; 32],
    nonce_seed: [u8; 12],
    crypto: Box<dyn VaultCrypto>,
}

impl Vault {
    /// Create a new vault file.
    pub fn create(path: &Path, key: &[u8; 32], crypto: Box<dyn VaultCrypto>) -> Result<Self, VaultError> {
        let key_id = {
            let h = xxhash64(key, 0);
            let mut id = [0u8; 16];
            id[..8].copy_from_slice(&h.to_le_bytes());
            id[8..16].copy_from_slice(&xxhash64(key, h).to_le_bytes());
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
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        file.write_all(&header.to_bytes())?;
        file.flush()?;

        Ok(Self {
            path: path.to_path_buf(),
            file,
            header,
            index: Vec::new(),
            buffer: Vec::new(),
            key: *key,
            nonce_seed,
            crypto,
        })
    }

    /// Open an existing vault file. Reads header + index into memory.
    pub fn open(path: &Path, key: &[u8; 32], crypto: Box<dyn VaultCrypto>) -> Result<Self, VaultError> {
        let mut file = OpenOptions::new().read(true).write(true).open(path)?;

        let mut hdr_buf = [0u8; HEADER_SIZE];
        file.read_exact(&mut hdr_buf)?;
        let header = VaultHeader::from_bytes(&hdr_buf)
            .map_err(|e| VaultError::InvalidFormat(e.to_string()))?;

        let nonce_seed = header.nonce_seed;

        let mut index = Vec::with_capacity(header.block_count as usize);
        file.seek(SeekFrom::Start(header.index_offset))?;
        for _ in 0..header.block_count {
            let mut entry_buf = [0u8; INDEX_ENTRY_SIZE];
            file.read_exact(&mut entry_buf)?;
            index.push(IndexEntry::from_bytes(&entry_buf));
        }

        Ok(Self {
            path: path.to_path_buf(),
            file,
            header,
            index,
            buffer: Vec::new(),
            key: *key,
            nonce_seed,
            crypto,
        })
    }

    /// Append a message to the internal buffer.
    /// Automatically flushes when buffer exceeds BLOCK_SIZE.
    pub fn append_message(&mut self, text: &str) -> Result<(), VaultError> {
        self.buffer.extend_from_slice(text.as_bytes());
        if self.buffer.len() >= BLOCK_SIZE {
            self.flush()?;
        }
        Ok(())
    }

    /// Encrypt the buffer and write as a new block. Updates index and header.
    pub fn flush(&mut self) -> Result<(), VaultError> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        let plaintext = &self.buffer;
        let histogram = compute_histogram(plaintext);
        let hash = xxhash64(plaintext, 0);
        let nonce_counter = self.header.block_count;

        let nonce = derive_nonce(&self.nonce_seed, nonce_counter);
        let ciphertext = self.crypto.encrypt(plaintext, &self.key, &nonce)
            .map_err(VaultError::Crypto)?;

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
        self.write_header()?;

        self.buffer.clear();
        Ok(())
    }

    /// Decrypt a specific block by index.
    pub fn decrypt_block(&mut self, block_index: usize) -> Result<Vec<u8>, VaultError> {
        if block_index >= self.index.len() {
            return Err(VaultError::InvalidFormat(
                format!("block index {} out of range (have {})", block_index, self.index.len()),
            ));
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
        let plaintext = self.crypto.decrypt(&ciphertext, &self.key, &nonce)
            .map_err(VaultError::Crypto)?;

        let actual_hash = xxhash64(&plaintext, 0);
        if actual_hash != expected_hash {
            return Err(VaultError::IntegrityFailed(block_index));
        }

        Ok(plaintext)
    }

    /// Read raw encrypted block bytes and derive its nonce.
    /// Used by fused search — no decryption happens here.
    pub(crate) fn read_encrypted_block(&mut self, block_index: usize)
        -> Result<(Vec<u8>, [u8; 12]), VaultError>
    {
        if block_index >= self.index.len() {
            return Err(VaultError::InvalidFormat(
                format!("block index {} out of range (have {})", block_index, self.index.len()),
            ));
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
    pub(crate) fn key(&self) -> &[u8; 32] {
        &self.key
    }

    /// Get the hash of the last block (for session token integrity).
    pub fn last_block_hash(&self) -> Option<u64> {
        self.index.last().map(|e| e.xxhash)
    }

    /// How many blocks in the vault.
    pub fn block_count(&self) -> u32 {
        self.header.block_count
    }

    fn write_index(&mut self) -> Result<(), VaultError> {
        self.file.seek(SeekFrom::Start(self.header.index_offset))?;
        for entry in &self.index {
            self.file.write_all(&entry.to_bytes())?;
        }
        self.file.flush()?;
        Ok(())
    }

    fn write_header(&mut self) -> Result<(), VaultError> {
        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&self.header.to_bytes())?;
        self.file.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tmp_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("olorin_vault_tests");
        fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn test_key() -> [u8; 32] {
        [0x42u8; 32]
    }

    fn test_crypto() -> Box<dyn VaultCrypto> {
        let lib = find_chacha_lib().expect("libchacha20.so not found — build with ea compiler");
        Box::new(EachachaCrypto::new(lib))
    }

    #[test]
    fn test_vault_create_and_append() {
        let path = tmp_path("create_and_append.vault");
        let _ = fs::remove_file(&path);

        let mut vault = Vault::create(&path, &test_key(), test_crypto()).unwrap();
        assert_eq!(vault.block_count(), 0);

        vault.append_message("hello world").unwrap();
        vault.flush().unwrap();
        assert_eq!(vault.block_count(), 1);

        vault.append_message("second message").unwrap();
        vault.flush().unwrap();
        assert_eq!(vault.block_count(), 2);

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_vault_roundtrip() {
        let path = tmp_path("roundtrip.vault");
        let _ = fs::remove_file(&path);

        let mut vault = Vault::create(&path, &test_key(), test_crypto()).unwrap();
        let msg = "the ring goes south";
        vault.append_message(msg).unwrap();
        vault.flush().unwrap();

        let decrypted = vault.decrypt_block(0).unwrap();
        assert_eq!(decrypted, msg.as_bytes());

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_vault_reopen() {
        let path = tmp_path("reopen.vault");
        let _ = fs::remove_file(&path);

        {
            let mut vault = Vault::create(&path, &test_key(), test_crypto()).unwrap();
            vault.append_message("block one").unwrap();
            vault.flush().unwrap();
            vault.append_message("block two").unwrap();
            vault.flush().unwrap();
            assert_eq!(vault.block_count(), 2);
        }

        {
            let mut vault = Vault::open(&path, &test_key(), test_crypto()).unwrap();
            assert_eq!(vault.block_count(), 2);

            let b0 = vault.decrypt_block(0).unwrap();
            assert_eq!(b0, b"block one");

            let b1 = vault.decrypt_block(1).unwrap();
            assert_eq!(b1, b"block two");
        }

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_vault_auto_flush() {
        let path = tmp_path("auto_flush.vault");
        let _ = fs::remove_file(&path);

        let mut vault = Vault::create(&path, &test_key(), test_crypto()).unwrap();
        let big_msg = "x".repeat(BLOCK_SIZE + 100);
        vault.append_message(&big_msg).unwrap();
        assert_eq!(vault.block_count(), 1);

        let decrypted = vault.decrypt_block(0).unwrap();
        assert_eq!(decrypted.len(), BLOCK_SIZE + 100);

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_vault_last_block_hash() {
        let path = tmp_path("last_hash.vault");
        let _ = fs::remove_file(&path);

        let mut vault = Vault::create(&path, &test_key(), test_crypto()).unwrap();
        assert!(vault.last_block_hash().is_none());

        vault.append_message("data").unwrap();
        vault.flush().unwrap();

        let expected = xxhash64(b"data", 0);
        assert_eq!(vault.last_block_hash(), Some(expected));

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_vault_reopen_and_append() {
        let path = tmp_path("reopen_append.vault");
        let _ = fs::remove_file(&path);

        {
            let mut vault = Vault::create(&path, &test_key(), test_crypto()).unwrap();
            vault.append_message("first").unwrap();
            vault.flush().unwrap();
        }

        {
            let mut vault = Vault::open(&path, &test_key(), test_crypto()).unwrap();
            assert_eq!(vault.block_count(), 1);
            vault.append_message("second").unwrap();
            vault.flush().unwrap();
            assert_eq!(vault.block_count(), 2);
        }

        {
            let mut vault = Vault::open(&path, &test_key(), test_crypto()).unwrap();
            assert_eq!(vault.block_count(), 2);
            let b0 = vault.decrypt_block(0).unwrap();
            assert_eq!(b0, b"first");
            let b1 = vault.decrypt_block(1).unwrap();
            assert_eq!(b1, b"second");
        }

        fs::remove_file(&path).unwrap();
    }
}

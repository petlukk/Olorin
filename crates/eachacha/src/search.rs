//! Decrypt-then-search: find needles in ChaCha20 encrypted data.

use crate::{decrypt, ChachaError};
use std::path::Path;

/// A match result from searching encrypted data.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Total number of matches found.
    pub match_count: usize,
    /// Byte offset of each match in the plaintext.
    pub offsets: Vec<usize>,
    /// Index into the `needles` slice for each match.
    pub needle_ids: Vec<usize>,
}

/// Search encrypted data for one or more needles without writing plaintext to disk.
///
/// Decrypts `ciphertext` in memory, then scans the plaintext for each needle.
/// Returns all match offsets and the needle index that matched.
///
/// # Arguments
/// * `ciphertext` — encrypted bytes
/// * `needles` — patterns to search for
/// * `key` — 32-byte ChaCha20 key
/// * `nonce` — 12-byte nonce
/// * `lib_path` — path to `libchacha20.so`
pub fn search(
    ciphertext: &[u8],
    needles: &[&[u8]],
    key: &[u8; 32],
    nonce: &[u8; 12],
    lib_path: &Path,
) -> Result<SearchResult, ChachaError> {
    if ciphertext.is_empty() || needles.is_empty() {
        return Ok(SearchResult {
            match_count: 0,
            offsets: Vec::new(),
            needle_ids: Vec::new(),
        });
    }

    let plaintext = decrypt(ciphertext, key, nonce, lib_path)?;

    let mut offsets = Vec::new();
    let mut needle_ids = Vec::new();

    for (idx, needle) in needles.iter().enumerate() {
        if needle.is_empty() {
            continue;
        }
        let mut start = 0;
        while start + needle.len() <= plaintext.len() {
            if let Some(pos) = plaintext[start..]
                .windows(needle.len())
                .position(|w| w == *needle)
            {
                offsets.push(start + pos);
                needle_ids.push(idx);
                start += pos + 1;
            } else {
                break;
            }
        }
    }

    let match_count = offsets.len();
    Ok(SearchResult {
        match_count,
        offsets,
        needle_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encrypt;
    use std::path::PathBuf;

    fn so_path() -> PathBuf {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let root = PathBuf::from(manifest).join("..").join("..");
        root.join("kernels/prebuilt/x86/libchacha20.so")
    }

    #[test]
    fn test_search_single_needle() {
        let path = so_path();
        if !path.exists() {
            eprintln!("skipping: {} not found", path.display());
            return;
        }
        let key = [0x42u8; 32];
        let nonce = [0x24u8; 12];
        let plaintext = b"INFO: starting\nERROR: disk full\nINFO: done";

        let ct = encrypt(plaintext.as_slice(), &key, &nonce, &path).unwrap();
        let result = search(&ct, &[b"ERROR"], &key, &nonce, &path).unwrap();

        assert_eq!(result.match_count, 1);
        assert_eq!(result.offsets[0], 15); // "ERROR" starts at byte 15
        assert_eq!(result.needle_ids[0], 0);
    }

    #[test]
    fn test_search_multi_needle() {
        let path = so_path();
        if !path.exists() {
            eprintln!("skipping: {} not found", path.display());
            return;
        }
        let key = [0xAAu8; 32];
        let nonce = [0xBBu8; 12];
        let plaintext = b"apple banana cherry apple banana";

        let ct = encrypt(plaintext.as_slice(), &key, &nonce, &path).unwrap();
        let result = search(&ct, &[b"apple", b"banana"], &key, &nonce, &path).unwrap();

        // "apple" at 0, 20; "banana" at 6, 26
        assert_eq!(result.match_count, 4);
        // apple matches come first (needle 0), then banana (needle 1)
        assert_eq!(result.offsets, vec![0, 20, 6, 26]);
        assert_eq!(result.needle_ids, vec![0, 0, 1, 1]);
    }

    #[test]
    fn test_search_no_match() {
        let path = so_path();
        if !path.exists() {
            eprintln!("skipping: {} not found", path.display());
            return;
        }
        let key = [0xCCu8; 32];
        let nonce = [0xDDu8; 12];
        let plaintext = b"nothing interesting here";

        let ct = encrypt(plaintext.as_slice(), &key, &nonce, &path).unwrap();
        let result = search(&ct, &[b"MISSING"], &key, &nonce, &path).unwrap();

        assert_eq!(result.match_count, 0);
        assert!(result.offsets.is_empty());
        assert!(result.needle_ids.is_empty());
    }

    #[test]
    fn test_search_empty_ciphertext() {
        let path = so_path();
        let key = [0u8; 32];
        let nonce = [0u8; 12];
        let result = search(&[], &[b"test"], &key, &nonce, &path).unwrap();
        assert_eq!(result.match_count, 0);
    }

    #[test]
    fn test_search_empty_needles() {
        let path = so_path();
        let key = [0u8; 32];
        let nonce = [0u8; 12];
        let result = search(b"data", &[], &key, &nonce, &path).unwrap();
        assert_eq!(result.match_count, 0);
    }
}

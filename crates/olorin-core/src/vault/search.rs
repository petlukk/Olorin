//! Vault search — SIMD cosine similarity over byte histograms with recency boost.
//! Uses fused ChaCha20 decrypt+search: plaintext never exists in memory.

use super::{Vault, VaultError};
use super::index::{compute_histogram, normalize_histogram};
use crate::kernels::search;

const DIM: usize = 256;

/// A single search result with score and matched context lines.
/// Only lines matching the query are returned — the full block
/// is never decrypted to memory.
pub struct SearchResult {
    pub block_index: usize,
    pub score: f32,
    pub lines: Vec<String>,
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

impl Vault {
    /// Search vault for blocks most similar to the query.
    /// Returns top-k results sorted by score (descending).
    /// Uses fused decrypt+search: only matched context lines are returned.
    pub fn search(&mut self, query: &str, top_k: usize) -> Result<Vec<SearchResult>, VaultError> {
        if self.index.is_empty() {
            return Ok(vec![]);
        }

        let n = self.index.len();

        // Compute and normalize query histogram
        let query_hist = compute_histogram(query.as_bytes());
        let mut query_norm = normalize_histogram(&query_hist);
        search::normalize_vectors(&mut query_norm, DIM, 1);
        let qnorm = l2_norm(&query_norm);

        if qnorm < 1e-9 {
            return Ok(vec![]);
        }

        // Build flat buffer of normalized block histograms for SIMD batch search
        let mut vecs = vec![0.0f32; n * DIM];
        for (i, entry) in self.index.iter().enumerate() {
            let norm = normalize_histogram(&entry.histogram);
            vecs[i * DIM..(i + 1) * DIM].copy_from_slice(&norm);
        }
        search::normalize_vectors(&mut vecs, DIM, n);

        // SIMD batch cosine similarity
        let mut scores = search::batch_cosine(&query_norm, qnorm, &vecs, DIM, n);

        // Apply recency boost
        for (i, score) in scores.iter_mut().enumerate() {
            let recency = if n <= 1 {
                1.0
            } else {
                i as f32 / (n - 1) as f32
            };
            *score *= 0.85 + 0.15 * recency;
        }

        // SIMD top-k
        let (indices, top_scores) = search::top_k(&scores, top_k);

        // Collect and sort candidates
        let mut scored: Vec<(usize, f32)> = indices
            .into_iter()
            .zip(top_scores)
            .filter(|(_, s)| *s > 0.01)
            .map(|(idx, s)| (idx as usize, s))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Tokenize query into needles
        let needle_strs: Vec<&[u8]> = query.split_whitespace()
            .map(|w| w.as_bytes())
            .collect();

        // Copy key before the loop to avoid simultaneous mutable/immutable borrows of self
        let key_copy = *self.key();

        // Fused decrypt+search per block
        let mut results = Vec::with_capacity(scored.len());
        for (block_idx, score) in scored {
            let (ciphertext, nonce) = self.read_encrypted_block(block_idx)?;

            let fused = self.searcher.search(
                &ciphertext,
                &needle_strs,
                &key_copy,
                &nonce,
            ).map_err(|e| VaultError::Crypto(e))?;

            let lines: Vec<String> = fused.context_lines
                .into_iter()
                .map(|l| String::from_utf8_lossy(&l).to_string())
                .collect();

            results.push(SearchResult { block_index: block_idx, score, lines });
        }

        Ok(results)
    }

    /// Decrypt the last N blocks (for /teleport greeting generation).
    /// This is an explicit user action — full decrypt is intentional.
    pub fn decrypt_last_n(&mut self, n: usize) -> Result<Vec<Vec<u8>>, VaultError> {
        let start = self.index.len().saturating_sub(n);
        let mut blocks = Vec::with_capacity(n);
        for i in start..self.index.len() {
            blocks.push(self.decrypt_block(i)?);
        }
        Ok(blocks)
    }
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use std::fs;
    use std::path::PathBuf;

    fn tmp_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("olorin_vault_search_tests");
        fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn test_key() -> [u8; 32] { [0x42u8; 32] }

    fn test_crypto() -> Box<dyn VaultCrypto> {
        let lib = find_chacha_lib().expect("libchacha20.so not found");
        Box::new(EachachaCrypto::new(lib))
    }

    #[test]
    fn test_vault_search_empty() {
        crate::kernels::ffi::init().unwrap();
        let path = tmp_path("search_empty.vault");
        let _ = fs::remove_file(&path);

        let mut vault = Vault::create(&path, &test_key(), test_crypto()).unwrap();
        let results = vault.search("anything", 5).unwrap();
        assert!(results.is_empty());

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_vault_search_finds_relevant() {
        crate::kernels::ffi::init().unwrap();
        let path = tmp_path("search_relevant.vault");
        let _ = fs::remove_file(&path);

        let mut vault = Vault::create(&path, &test_key(), test_crypto()).unwrap();

        vault.append_message("stars planets galaxies nebula cosmos astronomy telescope").unwrap();
        vault.flush().unwrap();

        vault.append_message("recipe flour sugar butter eggs bake oven kitchen cooking").unwrap();
        vault.flush().unwrap();

        vault.append_message("star constellation orbit planet astronomy celestial moon").unwrap();
        vault.flush().unwrap();

        let results = vault.search("stars planets astronomy cosmos", 3).unwrap();
        assert!(!results.is_empty());
        assert!(!results[0].lines.is_empty(), "should have context lines");

        let top = &results[0];
        assert!(top.block_index == 0 || top.block_index == 2,
            "expected astronomy block, got block {}", top.block_index);

        if results.len() >= 2 {
            assert!(results[0].score >= results[1].score);
        }

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_vault_search_recency_boost() {
        crate::kernels::ffi::init().unwrap();
        let path = tmp_path("search_recency.vault");
        let _ = fs::remove_file(&path);

        let mut vault = Vault::create(&path, &test_key(), test_crypto()).unwrap();

        let content = "identical content for recency test abcdefg";
        vault.append_message(content).unwrap();
        vault.flush().unwrap();
        vault.append_message(content).unwrap();
        vault.flush().unwrap();

        let results = vault.search(content, 2).unwrap();
        assert_eq!(results.len(), 2);

        assert_eq!(results[0].block_index, 1);
        assert_eq!(results[1].block_index, 0);
        assert!(results[0].score > results[1].score);

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_vault_decrypt_last_n() {
        let path = tmp_path("decrypt_last_n.vault");
        let _ = fs::remove_file(&path);

        let mut vault = Vault::create(&path, &test_key(), test_crypto()).unwrap();
        for i in 0..5 {
            vault.append_message(&format!("block number {}", i)).unwrap();
            vault.flush().unwrap();
        }

        let last2 = vault.decrypt_last_n(2).unwrap();
        assert_eq!(last2.len(), 2);
        assert_eq!(last2[0], b"block number 3");
        assert_eq!(last2[1], b"block number 4");

        let all = vault.decrypt_last_n(100).unwrap();
        assert_eq!(all.len(), 5);

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_vault_decrypt_last_n_empty() {
        let path = tmp_path("decrypt_last_n_empty.vault");
        let _ = fs::remove_file(&path);

        let mut vault = Vault::create(&path, &test_key(), test_crypto()).unwrap();
        let blocks = vault.decrypt_last_n(5).unwrap();
        assert!(blocks.is_empty());

        fs::remove_file(&path).unwrap();
    }
}

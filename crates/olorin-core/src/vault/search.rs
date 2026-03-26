//! Vault search — cosine similarity over byte histograms with recency boost.

use super::{Vault, VaultError};
use super::index::{compute_histogram, normalize_histogram, cosine_similarity, xxhash64};

/// A single search result with score and decrypted plaintext.
pub struct SearchResult {
    pub block_index: usize,
    pub score: f32,
    pub text: Vec<u8>,
}

impl Vault {
    /// Search vault for blocks most similar to the query.
    /// Returns top-k results sorted by score (descending).
    pub fn search(&mut self, query: &str, top_k: usize) -> Result<Vec<SearchResult>, VaultError> {
        if self.index.is_empty() {
            return Ok(vec![]);
        }

        // 1. Compute query histogram and normalize
        let query_hist = compute_histogram(query.as_bytes());
        let query_norm = normalize_histogram(&query_hist);

        // 2. Score each block
        let mut scored: Vec<(usize, f32)> = Vec::with_capacity(self.index.len());
        for (i, entry) in self.index.iter().enumerate() {
            let block_norm = normalize_histogram(&entry.histogram);
            let similarity = cosine_similarity(&query_norm, &block_norm);

            // Apply recency boost: score = similarity * (0.85 + 0.15 * recency)
            // recency is 0.0 for oldest, 1.0 for newest
            let recency = if self.index.len() <= 1 {
                1.0
            } else {
                i as f32 / (self.index.len() - 1) as f32
            };
            let boosted = similarity * (0.85 + 0.15 * recency);

            if boosted > 0.01 {
                scored.push((i, boosted));
            }
        }

        // 3. Sort by score descending, take top-k
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        // 4. Decrypt matched blocks
        let mut results = Vec::with_capacity(scored.len());
        for (block_idx, score) in scored {
            let plaintext = self.decrypt_block(block_idx)?;

            // Integrity already verified inside decrypt_block, but double-check hash
            let actual_hash = xxhash64(&plaintext, 0);
            if actual_hash != self.index[block_idx].xxhash {
                return Err(VaultError::IntegrityFailed(block_idx));
            }

            results.push(SearchResult {
                block_index: block_idx,
                score,
                text: plaintext,
            });
        }

        Ok(results)
    }

    /// Decrypt the last N blocks (for /teleport greeting generation).
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

    fn test_key() -> [u8; 32] {
        [0x42u8; 32]
    }

    #[test]
    fn test_vault_search_empty() {
        let path = tmp_path("search_empty.vault");
        let _ = fs::remove_file(&path);

        let mut vault = Vault::create(&path, &test_key(), Box::new(XorCrypto)).unwrap();
        let results = vault.search("anything", 5).unwrap();
        assert!(results.is_empty());

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_vault_search_finds_relevant() {
        let path = tmp_path("search_relevant.vault");
        let _ = fs::remove_file(&path);

        let mut vault = Vault::create(&path, &test_key(), Box::new(XorCrypto)).unwrap();

        // Block 0: astronomy topic
        vault.append_message("stars planets galaxies nebula cosmos astronomy telescope").unwrap();
        vault.flush().unwrap();

        // Block 1: cooking topic
        vault.append_message("recipe flour sugar butter eggs bake oven kitchen cooking").unwrap();
        vault.flush().unwrap();

        // Block 2: astronomy again
        vault.append_message("star constellation orbit planet astronomy celestial moon").unwrap();
        vault.flush().unwrap();

        let results = vault.search("stars planets astronomy cosmos", 3).unwrap();
        assert!(!results.is_empty());

        // The astronomy blocks (0 or 2) should rank higher than cooking (1)
        // Block 2 has recency boost so it should be first
        let top = &results[0];
        assert!(top.block_index == 0 || top.block_index == 2,
            "expected astronomy block, got block {}", top.block_index);

        // Cooking block should not be the top result
        if results.len() >= 2 {
            assert!(results[0].score >= results[1].score);
        }

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_vault_search_recency_boost() {
        let path = tmp_path("search_recency.vault");
        let _ = fs::remove_file(&path);

        let mut vault = Vault::create(&path, &test_key(), Box::new(XorCrypto)).unwrap();

        // Write identical content in two blocks
        let content = "identical content for recency test abcdefg";
        vault.append_message(content).unwrap();
        vault.flush().unwrap();
        vault.append_message(content).unwrap();
        vault.flush().unwrap();

        let results = vault.search(content, 2).unwrap();
        assert_eq!(results.len(), 2);

        // Newer block (index 1) should score higher due to recency boost
        assert_eq!(results[0].block_index, 1);
        assert_eq!(results[1].block_index, 0);
        assert!(results[0].score > results[1].score);

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_vault_decrypt_last_n() {
        let path = tmp_path("decrypt_last_n.vault");
        let _ = fs::remove_file(&path);

        let mut vault = Vault::create(&path, &test_key(), Box::new(XorCrypto)).unwrap();
        for i in 0..5 {
            vault.append_message(&format!("block number {}", i)).unwrap();
            vault.flush().unwrap();
        }

        let last2 = vault.decrypt_last_n(2).unwrap();
        assert_eq!(last2.len(), 2);
        assert_eq!(last2[0], b"block number 3");
        assert_eq!(last2[1], b"block number 4");

        // Request more than available
        let all = vault.decrypt_last_n(100).unwrap();
        assert_eq!(all.len(), 5);

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_vault_decrypt_last_n_empty() {
        let path = tmp_path("decrypt_last_n_empty.vault");
        let _ = fs::remove_file(&path);

        let mut vault = Vault::create(&path, &test_key(), Box::new(XorCrypto)).unwrap();
        let blocks = vault.decrypt_last_n(5).unwrap();
        assert!(blocks.is_empty());

        fs::remove_file(&path).unwrap();
    }
}

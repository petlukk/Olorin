//! Vault search — SIMD cosine similarity over byte histograms with recency boost.

use super::{Vault, VaultError};
use super::index::{compute_histogram, normalize_histogram, xxhash64};
use crate::kernels::search;

const DIM: usize = 256;

/// A single search result with score and decrypted plaintext.
pub struct SearchResult {
    pub block_index: usize,
    pub score: f32,
    pub text: Vec<u8>,
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

impl Vault {
    /// Search vault for blocks most similar to the query.
    /// Returns top-k results sorted by score (descending).
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

        // Collect, sort, and decrypt
        let mut scored: Vec<(usize, f32)> = indices
            .into_iter()
            .zip(top_scores)
            .filter(|(_, s)| *s > 0.01)
            .map(|(idx, s)| (idx as usize, s))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut results = Vec::with_capacity(scored.len());
        for (block_idx, score) in scored {
            let plaintext = self.decrypt_block(block_idx)?;
            let actual_hash = xxhash64(&plaintext, 0);
            if actual_hash != self.index[block_idx].xxhash {
                return Err(VaultError::IntegrityFailed(block_idx));
            }
            results.push(SearchResult { block_index: block_idx, score, text: plaintext });
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

        vault.append_message("stars planets galaxies nebula cosmos astronomy telescope").unwrap();
        vault.flush().unwrap();

        vault.append_message("recipe flour sugar butter eggs bake oven kitchen cooking").unwrap();
        vault.flush().unwrap();

        vault.append_message("star constellation orbit planet astronomy celestial moon").unwrap();
        vault.flush().unwrap();

        let results = vault.search("stars planets astronomy cosmos", 3).unwrap();
        assert!(!results.is_empty());

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
        let path = tmp_path("search_recency.vault");
        let _ = fs::remove_file(&path);

        let mut vault = Vault::create(&path, &test_key(), Box::new(XorCrypto)).unwrap();

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

        let mut vault = Vault::create(&path, &test_key(), Box::new(XorCrypto)).unwrap();
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

        let mut vault = Vault::create(&path, &test_key(), Box::new(XorCrypto)).unwrap();
        let blocks = vault.decrypt_last_n(5).unwrap();
        assert!(blocks.is_empty());

        fs::remove_file(&path).unwrap();
    }
}

//! SIMD-accelerated conversation recall using JL-projected byte-histogram embeddings.
//!
//! Each text is embedded as a 256-dim byte-histogram, then projected to 64-dim
//! via Johnson-Lindenstrauss random rotation (sign-flip + FWHT + truncate).
//! This preserves cosine similarity while using 4x less memory and faster search.
//!
//! The store uses a ring buffer with configurable capacity (default 1024).
//! Recency boost gives recent entries a slight score advantage.

use crate::kernels::search;

/// Raw embedding dimension — one per byte value.
const RAW_DIM: usize = 256;
/// Projected dimension after JL transform.
const PROJ_DIM: usize = search::JL_DIM;
/// Default ring buffer capacity.
const DEFAULT_CAPACITY: usize = 1024;
/// Recency boost: final_score = similarity * (RECENCY_BASE + RECENCY_WEIGHT * recency)
/// where recency ∈ [0.0, 1.0] (oldest → newest).
const RECENCY_BASE: f32 = 0.85;
const RECENCY_WEIGHT: f32 = 0.15;

/// A recalled conversation entry.
#[derive(Debug, Clone)]
pub struct RecallResult {
    pub index: usize,
    pub score: f32,
    pub text: String,
}

/// Generate deterministic sign mask for JL projection.
/// Uses a simple xorshift seeded with a fixed value so the same
/// projection is used across restarts.
fn gen_jl_signs() -> Vec<f32> {
    let mut signs = vec![0.0f32; RAW_DIM];
    let mut rng: u64 = 0x4F6C6F72696E4A4C; // "OlorinJL" as bytes
    for s in signs.iter_mut() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        *s = if rng % 2 == 0 { 1.0 } else { -1.0 };
    }
    signs
}

/// Vector store for conversation recall.
/// Uses a ring buffer of JL-projected byte-histogram embeddings
/// with SIMD cosine similarity and recency-boosted scoring.
pub struct VectorStore {
    /// Flat embedding buffer: vecs[i*PROJ_DIM .. (i+1)*PROJ_DIM]
    vecs: Vec<f32>,
    /// Original text for each slot
    texts: Vec<Option<String>>,
    /// Scratch buffer for 256-dim raw embedding
    raw_buf: Vec<f32>,
    /// Scratch buffer for 64-dim projected query
    proj_buf: Vec<f32>,
    /// JL sign mask (256 floats, each +1.0 or -1.0)
    jl_signs: Vec<f32>,
    /// Ring buffer capacity
    capacity: usize,
    /// Total number of inserts (monotonically increasing)
    total_inserts: usize,
    /// Next write position in the ring buffer
    write_pos: usize,
    /// Number of occupied slots (≤ capacity)
    count: usize,
}

impl VectorStore {
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            vecs: vec![0.0f32; cap * PROJ_DIM],
            texts: (0..cap).map(|_| None).collect(),
            raw_buf: vec![0.0f32; RAW_DIM],
            proj_buf: vec![0.0f32; PROJ_DIM],
            jl_signs: gen_jl_signs(),
            capacity: cap,
            total_inserts: 0,
            write_pos: 0,
            count: 0,
        }
    }

    /// Number of stored entries.
    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Index a text entry. Computes byte-histogram, projects via JL, normalizes.
    /// When the ring buffer is full, overwrites the oldest entry.
    pub fn insert(&mut self, text: &str) {
        embed_bytes(text.as_bytes(), &mut self.raw_buf);
        search::normalize_vectors(&mut self.raw_buf, RAW_DIM, 1);

        self.proj_buf = search::jl_project(&self.raw_buf, &self.jl_signs, PROJ_DIM);
        search::normalize_vectors(&mut self.proj_buf, PROJ_DIM, 1);

        let offset = self.write_pos * PROJ_DIM;
        self.vecs[offset..offset + PROJ_DIM].copy_from_slice(&self.proj_buf);
        self.texts[self.write_pos] = Some(text.to_string());

        self.write_pos = (self.write_pos + 1) % self.capacity;
        self.total_inserts += 1;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    /// Find the top-k most similar entries to the query.
    /// Scores are boosted by recency (newer entries score slightly higher).
    /// Returns results sorted by descending boosted score.
    pub fn recall(&mut self, query: &str, k: usize) -> Vec<RecallResult> {
        if self.count == 0 || k == 0 {
            return Vec::new();
        }

        embed_bytes(query.as_bytes(), &mut self.raw_buf);
        search::normalize_vectors(&mut self.raw_buf, RAW_DIM, 1);

        self.proj_buf = search::jl_project(&self.raw_buf, &self.jl_signs, PROJ_DIM);
        search::normalize_vectors(&mut self.proj_buf, PROJ_DIM, 1);

        let query_norm = l2_norm(&self.proj_buf);
        if query_norm < 1e-9 {
            return Vec::new();
        }

        let scan_n = if self.count == self.capacity {
            self.capacity
        } else {
            self.count
        };

        let scores = search::batch_cosine(
            &self.proj_buf, query_norm,
            &self.vecs[..scan_n * PROJ_DIM], PROJ_DIM, scan_n,
        );

        let mut boosted = scores;
        for (i, score) in boosted.iter_mut().enumerate() {
            if self.texts[i].is_none() {
                *score = 0.0;
                continue;
            }
            let recency = self.slot_recency(i);
            *score *= RECENCY_BASE + RECENCY_WEIGHT * recency;
        }

        let (indices, top_scores) = search::top_k(&boosted, k);

        let mut results: Vec<RecallResult> = indices
            .into_iter()
            .zip(top_scores)
            .filter(|(idx, score)| *score > 0.01 && self.texts[*idx as usize].is_some())
            .map(|(idx, score)| RecallResult {
                index: idx as usize,
                score,
                text: self.texts[idx as usize].as_ref().unwrap().clone(),
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Search and format results as a display string.
    pub fn recall_formatted(&mut self, query: &str, k: usize) -> String {
        if query.is_empty() {
            return format!("usage: /recall <query> ({} entries indexed)", self.len());
        }
        let results = self.recall(query, k);
        if results.is_empty() {
            return "No matching entries found.".to_string();
        }
        let mut out = format!("Recall ({} results):\n", results.len());
        for (i, r) in results.iter().enumerate() {
            let preview: String = r.text.chars().take(120).collect();
            out.push_str(&format!(
                "  {}. [{:.2}] {}{}\n",
                i + 1,
                r.score,
                preview,
                if r.text.len() > 120 { "..." } else { "" }
            ));
        }
        out
    }

    /// Clear all stored entries.
    pub fn clear(&mut self) {
        for t in self.texts.iter_mut() {
            *t = None;
        }
        self.write_pos = 0;
        self.total_inserts = 0;
        self.count = 0;
    }

    /// Recency of a slot: 0.0 = oldest occupied, 1.0 = most recent insert.
    fn slot_recency(&self, slot: usize) -> f32 {
        if self.count <= 1 {
            return 1.0;
        }
        let newest = (self.write_pos + self.capacity - 1) % self.capacity;
        let age = (newest + self.capacity - slot) % self.capacity;
        let age = age.min(self.count - 1);
        1.0 - (age as f32 / (self.count - 1) as f32)
    }
}

impl Default for VectorStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute byte-histogram embedding: count frequency of each byte value.
fn embed_bytes(input: &[u8], out: &mut [f32]) {
    assert!(out.len() >= RAW_DIM);
    for v in out.iter_mut().take(RAW_DIM) {
        *v = 0.0;
    }
    for &b in input {
        out[b as usize] += 1.0;
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_bytes_basic() {
        let mut buf = vec![0.0f32; RAW_DIM];
        embed_bytes(b"aab", &mut buf);
        assert_eq!(buf[b'a' as usize], 2.0);
        assert_eq!(buf[b'b' as usize], 1.0);
        assert_eq!(buf[b'c' as usize], 0.0);
    }

    #[test]
    fn test_insert_and_recall() {
        let mut store = VectorStore::new();
        store.insert("hello world");
        store.insert("cargo build test");
        store.insert("hello there friend");

        let results = store.recall("hello", 2);
        assert!(!results.is_empty());
        assert!(results[0].text.contains("hello"), "top result: {}", results[0].text);
    }

    #[test]
    fn test_recall_empty() {
        let mut store = VectorStore::new();
        let results = store.recall("anything", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_recall_exact_match() {
        let mut store = VectorStore::new();
        store.insert("the quick brown fox");
        store.insert("jumps over the lazy dog");
        store.insert("the quick brown fox jumps");

        let results = store.recall("the quick brown fox", 1);
        assert_eq!(results.len(), 1);
        assert!(results[0].score > 0.5, "score: {}", results[0].score);
    }

    #[test]
    fn test_recall_different_content() {
        let mut store = VectorStore::new();
        store.insert("rust programming language");
        store.insert("12345 numbers only");
        store.insert("rust cargo build system");

        let results = store.recall("rust compiler", 2);
        for r in &results {
            assert!(
                r.text.contains("rust"),
                "expected rust entries, got: {}",
                r.text
            );
        }
    }

    #[test]
    fn test_clear() {
        let mut store = VectorStore::new();
        store.insert("test");
        assert_eq!(store.len(), 1);
        store.clear();
        assert_eq!(store.len(), 0);
        assert!(store.recall("test", 1).is_empty());
    }

    #[test]
    fn test_many_entries() {
        let mut store = VectorStore::with_capacity(100);
        for i in 0..100 {
            store.insert(&format!("entry number {i} with some text"));
        }
        assert_eq!(store.len(), 100);
        let results = store.recall("entry number 42", 3);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_ring_buffer_wraps() {
        let mut store = VectorStore::with_capacity(3);
        store.insert("first message");
        store.insert("second message");
        store.insert("third message");
        assert_eq!(store.len(), 3);

        store.insert("fourth message");
        assert_eq!(store.len(), 3);

        let results = store.recall("first", 3);
        for r in &results {
            assert!(!r.text.contains("first message"),
                "first message should have been evicted, got: {}", r.text);
        }

        let results = store.recall("fourth", 1);
        assert!(!results.is_empty());
        assert!(results[0].text.contains("fourth"), "got: {}", results[0].text);
    }

    #[test]
    fn test_ring_buffer_capacity_one() {
        let mut store = VectorStore::with_capacity(1);
        store.insert("only one");
        assert_eq!(store.len(), 1);
        store.insert("replaced");
        assert_eq!(store.len(), 1);
        let results = store.recall("replaced", 1);
        assert_eq!(results[0].text, "replaced");
    }

    #[test]
    fn test_recency_boost() {
        let mut store = VectorStore::with_capacity(100);
        store.insert("rust programming");
        for _ in 0..50 {
            store.insert("filler text padding content");
        }
        store.insert("rust programming");

        let results = store.recall("rust programming", 2);
        assert!(results.len() >= 2);
        let newer_idx = results[0].index;
        let older_idx = results[1].index;
        assert!(newer_idx > older_idx,
            "newer entry (idx {}) should rank above older (idx {})", newer_idx, older_idx);
    }

    #[test]
    fn test_jl_projection_preserves_ranking() {
        let mut store = VectorStore::with_capacity(200);
        // Realistic scenario: known entries mixed with diverse filler
        store.insert("SIMD vector optimization for ARM NEON and AVX-512 processing units");
        store.insert("SIMD kernel acceleration and vector math operations for modern x86 CPUs");
        // Filler with very different byte distributions
        for i in 0..50 {
            store.insert(&format!("{i}{i}{i}{i}{i}{i}{i}{i}{i}{i} {i}{i}{i}{i}{i}{i}{i}{i}{i}{i}"));
        }

        // SIMD entries must appear in results (filler is purely numeric)
        let results = store.recall("SIMD vector processing", 3);
        let simd_count = results.iter().filter(|r| r.text.contains("SIMD")).count();
        assert!(simd_count >= 1,
            "expected at least 1 SIMD entry in top-3, got: {:?}",
            results.iter().map(|r| &r.text).collect::<Vec<_>>());
    }
}

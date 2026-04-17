//! SIMD-accelerated conversation recall using JL-projected byte-histogram embeddings.
//!
//! Each text is embedded as a 256-dim byte-histogram, projected to 64-dim via
//! Johnson-Lindenstrauss (sign-flip + FWHT + truncate). Cosine similarity search
//! via SIMD kernels. Ring buffer with recency-boosted scoring.

use crate::kernels::ffi;
use crate::storage::secure::SecureBuffer;

/// Raw embedding dimension — one per byte value.
const RAW_DIM:  usize = 256;
/// Projected dimension after JL transform.
const PROJ_DIM: usize = 64;
/// Recency boost: score *= (BASE + WEIGHT * recency), recency ∈ [0, 1].
const RECENCY_BASE:   f32 = 0.85;
const RECENCY_WEIGHT: f32 = 0.15;
/// Dedup threshold: pairwise cosine > this → duplicate.
const DEDUP_THRESHOLD: f32 = 0.85;

/// A recalled conversation entry.
#[derive(Debug, Clone)]
pub struct RecallResult {
    pub index: usize,
    pub score: f32,
    pub text:  String,
}

/// Generate deterministic JL sign mask (seeded, reproducible across restarts).
fn gen_jl_signs() -> Vec<f32> {
    let mut signs = vec![0.0f32; RAW_DIM];
    let mut rng: u64 = 0x4F6C6F72696E4A4C; // "OlorinJL"
    for s in signs.iter_mut() {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        *s = if rng % 2 == 0 { 1.0 } else { -1.0 };
    }
    signs
}

/// Vector store for conversation recall.
/// Pre-allocates all buffers at construction — zero heap on hot path.
pub struct VectorStore {
    /// Flat embedding buffer: vecs[i*PROJ_DIM .. (i+1)*PROJ_DIM]
    vecs:          Vec<f32>,
    /// Original text for each slot
    texts:         Vec<Option<String>>,
    /// Scratch: 256-dim raw histogram (SecureBuffer — zeroed on Drop)
    raw_scratch:   SecureBuffer,
    /// Scratch: 64-dim projected query (SecureBuffer — zeroed on Drop)
    proj_scratch:  SecureBuffer,
    /// JL sign mask (256 f32, each +1.0 or -1.0)
    jl_signs:      Vec<f32>,
    capacity:      usize,
    total_inserts: usize,
    write_pos:     usize,
    count:         usize,
}

impl VectorStore {
    pub fn new(capacity: usize) -> Self {
        let cap = capacity.max(1);
        Self {
            vecs:         vec![0.0f32; cap * PROJ_DIM],
            texts:        (0..cap).map(|_| None).collect(),
            raw_scratch:  SecureBuffer::new(RAW_DIM * 4),   // f32 = 4 bytes
            proj_scratch: SecureBuffer::new(PROJ_DIM * 4),
            jl_signs:     gen_jl_signs(),
            capacity:     cap,
            total_inserts: 0,
            write_pos:    0,
            count:        0,
        }
    }

    pub fn len(&self)      -> usize { self.count }
    pub fn is_empty(&self) -> bool  { self.count == 0 }

    /// Index a text entry. Overwrites oldest slot when ring buffer is full.
    pub fn add(&mut self, text: &str) {
        embed_bytes(text.as_bytes(), self.raw_scratch.as_mut_slice());
        jl_project_inplace(
            self.raw_scratch.as_mut_slice(),
            &self.jl_signs,
            self.proj_scratch.as_mut_slice(),
        );

        let offset = self.write_pos * PROJ_DIM;
        let proj = f32_slice(self.proj_scratch.as_slice());
        self.vecs[offset..offset + PROJ_DIM].copy_from_slice(proj);
        self.texts[self.write_pos] = Some(text.to_string());

        self.write_pos = (self.write_pos + 1) % self.capacity;
        self.total_inserts += 1;
        if self.count < self.capacity { self.count += 1; }
    }

    /// Find top-k most similar entries to the query, boosted by recency.
    pub fn search(&mut self, query: &str, k: usize) -> Vec<RecallResult> {
        if self.count == 0 || k == 0 { return Vec::new(); }

        embed_bytes(query.as_bytes(), self.raw_scratch.as_mut_slice());
        jl_project_inplace(
            self.raw_scratch.as_mut_slice(),
            &self.jl_signs,
            self.proj_scratch.as_mut_slice(),
        );

        let proj = f32_slice(self.proj_scratch.as_slice());
        let query_norm = l2_norm(proj);
        if query_norm < 1e-9 { return Vec::new(); }

        let scan_n = self.count;
        let mut scores = vec![0.0f32; scan_n];

        unsafe {
            ffi::batch_cosine(
                proj.as_ptr(),
                query_norm,
                self.vecs.as_ptr(),
                PROJ_DIM as i32,
                scan_n as i32,
                scores.as_mut_ptr(),
            );
        }

        for (i, score) in scores.iter_mut().enumerate() {
            if self.texts[i].is_none() { *score = 0.0; continue; }
            let recency = self.slot_recency(i);
            *score *= RECENCY_BASE + RECENCY_WEIGHT * recency;
        }

        let top_k = k.min(scan_n);
        let mut indices = vec![0i32; top_k];
        let mut top_scores = vec![0.0f32; top_k];

        unsafe {
            ffi::top_k(
                scores.as_ptr(),
                scan_n as i32,
                top_k as i32,
                indices.as_mut_ptr(),
                top_scores.as_mut_ptr(),
            );
        }

        let mut results: Vec<RecallResult> = indices.iter()
            .zip(top_scores.iter())
            .filter(|(&idx, &s)| s > 0.01 && self.texts[idx as usize].is_some())
            .map(|(&idx, &s)| RecallResult {
                index: idx as usize,
                score: s,
                text:  self.texts[idx as usize].as_ref().unwrap().clone(),
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Search with deduplication of near-identical results.
    pub fn search_dedup(&mut self, query: &str, k: usize) -> Vec<RecallResult> {
        let results = self.search(query, k);
        if results.len() <= 1 { return results; }

        let n = results.len();
        let mut result_vecs = vec![0.0f32; n * PROJ_DIM];
        for (i, r) in results.iter().enumerate() {
            let offset = r.index * PROJ_DIM;
            result_vecs[i * PROJ_DIM..(i + 1) * PROJ_DIM]
                .copy_from_slice(&self.vecs[offset..offset + PROJ_DIM]);
        }

        let mut keep = vec![true; n];
        for i in 0..n {
            if !keep[i] { continue; }
            let qi = &result_vecs[i * PROJ_DIM..(i + 1) * PROJ_DIM];
            let qi_norm = l2_norm(qi);
            if qi_norm < 1e-9 { continue; }

            let mut sims = vec![0.0f32; n];
            unsafe {
                ffi::batch_cosine(
                    qi.as_ptr(), qi_norm,
                    result_vecs.as_ptr(),
                    PROJ_DIM as i32, n as i32,
                    sims.as_mut_ptr(),
                );
            }
            for j in (i + 1)..n {
                if keep[j] && sims[j] > DEDUP_THRESHOLD { keep[j] = false; }
            }
        }

        results.into_iter().enumerate()
            .filter(|(i, _)| keep[*i])
            .map(|(_, r)| r)
            .collect()
    }

    /// Recall and format as display string.
    pub fn recall_formatted(&mut self, query: &str, k: usize) -> String {
        if query.is_empty() {
            return format!("usage: /recall <query> ({} entries indexed)", self.len());
        }
        let results = self.search(query, k);
        if results.is_empty() { return "No matching entries found.".to_string(); }
        let mut out = format!("Recall ({} results):\n", results.len());
        for (i, r) in results.iter().enumerate() {
            let preview: String = r.text.chars().take(120).collect();
            out.push_str(&format!(
                "  {}. [{:.2}] {}{}\n", i + 1, r.score, preview,
                if r.text.len() > 120 { "..." } else { "" }
            ));
        }
        out
    }

    /// Synthesize a compact context block for LLM injection. Skips entries
    /// that are near-duplicates of the query itself (prior copies of the
    /// same question would self-match at score 1.0 and crowd out real facts).
    pub fn synthesize_context(&mut self, query: &str, k: usize) -> Option<String> {
        // Oversearch so filtering still leaves k usable entries.
        let raw = self.search_dedup(query, k.saturating_mul(2).saturating_add(1));
        let query_norm = normalize_context_line(query);
        let results: Vec<_> = raw.into_iter()
            .filter(|r| normalize_context_line(&r.text) != query_norm)
            .take(k)
            .collect();
        if results.is_empty() { return None; }
        let mut ctx = String::from("Earlier in this conversation:\n");
        for r in &results {
            let preview: String = r.text.chars().take(100).collect();
            ctx.push_str(&preview);
            if r.text.len() > 100 { ctx.push_str("..."); }
            ctx.push('\n');
        }
        Some(ctx)
    }

    /// Clear all stored entries.
    pub fn clear(&mut self) {
        for t in self.texts.iter_mut() { *t = None; }
        self.write_pos    = 0;
        self.total_inserts = 0;
        self.count        = 0;
    }

    fn slot_recency(&self, slot: usize) -> f32 {
        if self.count <= 1 { return 1.0; }
        let newest = (self.write_pos + self.capacity - 1) % self.capacity;
        let age = (newest + self.capacity - slot) % self.capacity;
        let age = age.min(self.count - 1);
        1.0 - (age as f32 / (self.count - 1) as f32)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Count byte frequencies into f32 histogram (256 bins).
fn embed_bytes(input: &[u8], out: &mut [u8]) {
    let floats = out.as_mut_ptr() as *mut f32;
    let n = out.len() / 4;
    for i in 0..n {
        unsafe { *floats.add(i) = 0.0; }
    }
    for &b in input {
        let idx = b as usize;
        if idx < n {
            unsafe { *floats.add(idx) += 1.0; }
        }
    }
}

/// JL projection: sign-flip → FWHT → truncate to PROJ_DIM via Eä kernel.
fn jl_project_inplace(raw: &mut [u8], signs: &[f32], proj: &mut [u8]) {
    let in_dim = raw.len() / 4;
    let out_dim = proj.len() / 4;
    let raw_f32 = raw.as_mut_ptr() as *mut f32;
    let proj_f32 = proj.as_mut_ptr() as *mut f32;

    // jl_project does sign-flip + FWHT + truncation in one kernel call
    unsafe {
        ffi::jl_project(
            raw_f32 as *const f32,
            signs.as_ptr(),
            in_dim as i32,
            out_dim as i32,
            proj_f32,
            raw_f32, // scratch buffer (reuse raw)
        );
    }

    // Normalize projected vector
    unsafe {
        ffi::normalize_vectors(proj_f32, out_dim as i32, 1);
    }
}

fn f32_slice(bytes: &[u8]) -> &[f32] {
    unsafe {
        std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4)
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Normalize a line for self-match detection: lowercase + strip trailing
/// punctuation/whitespace so "What is my name?" == "what is my name".
fn normalize_context_line(s: &str) -> String {
    s.trim()
        .trim_end_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace())
        .to_ascii_lowercase()
}

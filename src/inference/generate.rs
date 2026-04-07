//! Public inference API — prompt in, text out.
//!
//! Engine loads a Gemma 4 GGUF, owns all state, and drives the
//! prefill + decode loop with streaming token output.

use std::path::{Path, PathBuf};
use crate::error::{Error, Result};
use crate::inference::gguf::GgufFile;
use crate::inference::engine::Gemma4Model;
use crate::inference::tokenizer::Tokenizer;
use crate::inference::forward::Gemma4State;

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

pub struct Engine {
    _gguf: GgufFile,           // owns the mmap — keeps weight pointers valid
    model: Gemma4Model,
    tokenizer: Tokenizer,
    state: Gemma4State,
    pool: crate::inference::threadpool::ThreadPool,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub draft_k: usize,
}

impl Engine {
    /// Load a GGUF model from disk.
    pub fn load(path: &Path, max_seq_len: usize) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        let model = Gemma4Model::from_gguf(&gguf)
            .map_err(|e| Error::Inference(e))?;
        crate::kernels::ffi::init()
            .map_err(|e| Error::Inference(e))?;
        let state = Gemma4State::new(&model, max_seq_len);
        let pool = crate::inference::threadpool::ThreadPool::new();
        eprintln!("[Olorin] Thread pool: {} threads", pool.thread_count());

        Ok(Self {
            _gguf: gguf,
            model,
            tokenizer,
            state,
            pool,
            max_tokens: 2048,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repetition_penalty: 1.0,
            draft_k: 0,
        })
    }

    /// Load a draft model for speculative decoding.
    pub fn load_draft(&mut self, _path: &Path) -> Result<()> {
        Err(Error::Inference(
            "Gemma 4 speculative decoding not yet implemented".into(),
        ))
    }

    /// Detect quantization type of the loaded model.
    pub fn quant_type_str(&self) -> &str {
        "gemma4-q4k"
    }

    /// Generate text from a prompt. Calls `on_token` for each generated token.
    /// Returns the complete generated text.
    pub fn generate(
        &mut self,
        prompt: &str,
        system: &str,
        on_token: &dyn Fn(&str),
    ) -> Result<String> {
        // 1. Format as Gemma chat template
        let formatted = format_chat(prompt, system);

        // 2. Tokenize. The Gemma 4 jinja chat_template emits {{ bos_token }}
        //    at the start, so we prepend BOS as a token id (matches what the
        //    template would produce when rendered + tokenized).
        let mut tokens = vec![self.tokenizer.bos_id];
        tokens.extend(self.tokenizer.encode(&formatted));
        if tokens.len() <= 1 {
            return Err(Error::Inference("empty prompt after tokenization".into()));
        }

        // 3. Reset state for new sequence
        self.state.reset();

        // 4. Prefill: forward each prompt token (discard logits except last)
        let n_prompt = tokens.len();
        for &tok in &tokens[..n_prompt - 1] {
            self.state.forward_one(&self.model, tok, &self.pool);
        }

        // Last prompt token gives us the first set of logits
        let mut logits_snapshot = {
            let logits = self.state.forward_one(&self.model, tokens[n_prompt - 1], &self.pool);
            logits.to_vec()
        };

        // 5. Decode loop
        let mut rng = xorshift_seed();
        let mut output = String::new();
        let eos = self.tokenizer.eos_id;
        let stop_ids = &self.tokenizer.stop_ids;

        for _ in 0..self.max_tokens {
            // Sample
            let token_id = sample(
                &mut logits_snapshot,
                self.temperature,
                self.top_k,
                self.top_p,
                self.min_p,
                &mut rng,
            );

            // Check EOS / stop tokens
            if token_id == eos || stop_ids.contains(&token_id) {
                break;
            }

            // Decode to text
            let text = self.tokenizer.decode(&[token_id]);
            on_token(&text);
            output.push_str(&text);

            // Forward to get next logits
            let logits = self.state.forward_one(&self.model, token_id, &self.pool);
            logits_snapshot.clear();
            logits_snapshot.extend_from_slice(logits);
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Chat template
// ---------------------------------------------------------------------------

fn format_chat(user: &str, system: &str) -> String {
    // Gemma 4 chat format (verbatim from GGUF jinja chat_template):
    //   <bos><|turn>system\n{system_trimmed}<turn|>\n<|turn>user\n{user_trimmed}<turn|>\n<|turn>model\n
    // BOS is added by the caller via tokenizer.bos_id (matches {{ bos_token }}).
    let mut out = String::with_capacity(system.len() + user.len() + 96);
    let sys_trim = system.trim();
    if !sys_trim.is_empty() {
        out.push_str("<|turn>system\n");
        out.push_str(sys_trim);
        out.push_str("<turn|>\n");
    }
    out.push_str("<|turn>user\n");
    out.push_str(user.trim());
    out.push_str("<turn|>\n");
    out.push_str("<|turn>model\n");
    out
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

fn sample(
    logits: &mut [f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    rng: &mut u64,
) -> u32 {
    let n = logits.len();

    // Greedy
    if temperature < 1e-6 {
        return argmax(logits);
    }

    // Apply temperature
    for l in logits.iter_mut() {
        *l /= temperature;
    }

    // Find max for numerical stability + min_p threshold
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_p_thresh = max_logit + min_p.max(1e-10).ln();

    // Build (index, logit) pairs, filter by min_p
    let mut candidates: Vec<(u32, f32)> = Vec::with_capacity(top_k.min(n));
    for (i, &l) in logits.iter().enumerate() {
        if l >= min_p_thresh {
            candidates.push((i as u32, l));
        }
    }

    if candidates.is_empty() {
        return argmax(logits);
    }

    // Top-k: sort descending, truncate
    candidates.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    if candidates.len() > top_k {
        candidates.truncate(top_k);
    }

    // Softmax over candidates
    let cmax = candidates[0].1;
    let mut sum = 0.0f32;
    for c in candidates.iter_mut() {
        c.1 = (c.1 - cmax).exp();
        sum += c.1;
    }
    let inv_sum = 1.0 / sum;
    for c in candidates.iter_mut() {
        c.1 *= inv_sum;
    }

    // Top-p: accumulate, cut
    let mut cumulative = 0.0f32;
    let mut cutoff = candidates.len();
    for (i, c) in candidates.iter().enumerate() {
        cumulative += c.1;
        if cumulative > top_p {
            cutoff = i + 1;
            break;
        }
    }
    candidates.truncate(cutoff);

    // Re-normalize after top-p
    let sum2: f32 = candidates.iter().map(|c| c.1).sum();
    let inv2 = 1.0 / sum2;
    for c in candidates.iter_mut() {
        c.1 *= inv2;
    }

    // Sample with xorshift64
    let r = xorshift64(rng);
    let threshold = (r as f64) / (u64::MAX as f64);
    let mut acc = 0.0f64;
    for c in &candidates {
        acc += c.1 as f64;
        if acc >= threshold {
            return c.0;
        }
    }

    // Fallback: last candidate
    candidates.last().unwrap().0
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
}

// ---------------------------------------------------------------------------
// RNG
// ---------------------------------------------------------------------------

fn xorshift_seed() -> u64 {
    let mut s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    if s == 0 { s = 0xDEAD_BEEF_CAFE_BABE; }
    s
}

fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ---------------------------------------------------------------------------
// Model discovery (unchanged)
// ---------------------------------------------------------------------------

/// Find a GGUF model in standard locations. Prefers gemma4 if present.
pub fn find_model() -> Option<PathBuf> {
    let dir = models_dir()?;
    let gemma4 = dir.join("gemma-4-e2b-it-Q4_K_M.gguf");
    if gemma4.exists() {
        return Some(gemma4);
    }
    std::fs::read_dir(&dir).ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.extension().map(|e| e == "gguf").unwrap_or(false))
}

fn models_dir() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let dir = Path::new(&home).join(".olorin/models");
    dir.is_dir().then_some(dir)
}

/// List all .gguf files in ~/.olorin/models/ — stem names only.
pub fn available_models() -> Vec<String> {
    let dir = match models_dir() {
        Some(d) => d,
        None => return Vec::new(),
    };
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "gguf").unwrap_or(false))
        .filter_map(|e| e.path().file_stem().map(|s| s.to_string_lossy().into_owned()))
        .collect();
    names.sort();
    names
}

/// Resolve model path from CLI argument or auto-detect.
pub fn resolve_model(arg: Option<&str>) -> Option<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_default();
    let olorin_models = Path::new(&home).join(".olorin/models");
    let aliases: &[(&str, &str)] = &[
        ("bitnet",  "ggml-model-i2_s.gguf"),
        ("llama",   "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
        ("llama8b", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
        ("qwen",    "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"),
        ("gemma4",  "gemma-4-e2b-it-Q4_K_M.gguf"),
    ];
    match arg {
        Some(name) => {
            for &(alias, filename) in aliases {
                if name == alias {
                    let p = olorin_models.join(filename);
                    return p.exists().then_some(p);
                }
            }
            let stem_path = olorin_models.join(format!("{name}.gguf"));
            if stem_path.exists() { return Some(stem_path); }
            let p = PathBuf::from(name);
            p.exists().then_some(p)
        }
        None => find_model(),
    }
}

//! Public inference API — prompt in, text out.
//!
//! Engine loads a Gemma 4 GGUF, owns all state, and drives the
//! prefill + decode loop with streaming token output.

use std::path::{Path, PathBuf};
use std::time::Instant;
use crate::error::{Error, Result};
use crate::inference::gguf::GgufFile;
use crate::inference::engine::Gemma4Model;
use crate::inference::forward::{timing_enabled, Gemma4State};
use crate::inference::tokenizer::Tokenizer;

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

/// Streaming event emitted by `Engine::generate`.
///
/// `Token` carries user-visible text (thinking content is already filtered
/// out). `Thinking(true|false)` fires when the model opens / closes a
/// `<|channel>...<channel|>` block — no token text is emitted while the
/// block is active.
pub enum GenEvent<'a> {
    Token(&'a str),
    Thinking(bool),
}

pub struct Engine {
    _gguf: GgufFile,           // owns the mmap — keeps weight pointers valid
    model: Gemma4Model,
    tokenizer: Tokenizer,
    state: Gemma4State,
    graph_pool: crate::inference::threadpool::GraphPool,
    /// `<|channel>` token id — opens a thinking block. None if not in vocab.
    channel_open_id: Option<u32>,
    /// `<channel|>` token id — closes the thinking block.
    channel_close_id: Option<u32>,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
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
        let graph_pool = crate::inference::threadpool::GraphPool::new();
        eprintln!("[Olorin] Thread pool: {} threads", graph_pool.thread_count());
        let state = Gemma4State::new(&model, max_seq_len, &graph_pool);

        // Gemma 4 brackets chain-of-thought in `<|channel>...<channel|>`.
        // Look up the token ids once so the decode loop can compare by id.
        let channel_open_id  = tokenizer.token_to_id("<|channel>");
        let channel_close_id = tokenizer.token_to_id("<channel|>");

        Ok(Self {
            _gguf: gguf,
            model,
            tokenizer,
            state,
            graph_pool,
            channel_open_id,
            channel_close_id,
            // Defaults match llama.cpp + GGUF metadata for this model:
            //   general.sampling.top_k = 64
            //   general.sampling.top_p = 0.95
            //   general.sampling.temp  = 1.0
            // min_p / repetition_penalty come from llama.cpp CLI defaults.
            max_tokens: 2048,
            temperature: 1.0,
            top_k: 64,
            top_p: 0.95,
            min_p: 0.05,
            repetition_penalty: 1.0,
        })
    }

    /// Detect quantization type of the loaded model.
    pub fn quant_type_str(&self) -> &str {
        "gemma4-q4k"
    }

    /// Read-only accessor for the loaded model — used by offline telemetry
    /// (exit_probe, weight_stats). Not part of the public inference API.
    pub fn model(&self) -> &Gemma4Model {
        &self.model
    }

    /// Read-only accessor for the KV cache — used by offline telemetry
    /// (kv_stats). Not part of the public inference API.
    pub fn kv_cache(&self) -> &crate::inference::cache::KvCache {
        &self.state.cache
    }

    /// Generate text from a prompt.
    ///
    /// `on_event` is called with every user-visible token and with state
    /// transitions when the model opens or closes a `<|channel>` thinking
    /// block. Returned `String` is the user-visible text (thinking content
    /// is excluded).
    pub fn generate(
        &mut self,
        prompt: &str,
        system: &str,
        on_event: &dyn Fn(GenEvent),
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

        let timing = timing_enabled();

        // 4. Prefill: batched forward (all prompt tokens at once)
        let t_prefill_start = Instant::now();
        let mut logits_snapshot = {
            let logits = self.state.forward_batch(&self.model, &tokens, &self.graph_pool);
            logits.to_vec()
        };
        let t_prefill = t_prefill_start.elapsed();
        let n_prompt = tokens.len();

        // 5. Decode loop
        let t_decode_start = Instant::now();
        let mut rng = xorshift_seed();
        let mut output = String::new();
        let eos = self.tokenizer.eos_id;
        let stop_ids = &self.tokenizer.stop_ids;
        let mut n_decode = 0usize;

        let mut t_sample_total: u64 = 0;
        let mut t_forward_total: u64 = 0;
        let mut t_copy_total: u64 = 0;
        let mut t_other_total: u64 = 0;

        // Track whether we're inside a Gemma 4 thinking block.
        let mut in_thinking = false;

        for _ in 0..self.max_tokens {
            let t0 = Instant::now();
            let token_id = sample(
                &mut logits_snapshot,
                self.temperature,
                self.top_k,
                self.top_p,
                self.min_p,
                &mut rng,
            );
            t_sample_total += t0.elapsed().as_micros() as u64;

            if token_id == eos || stop_ids.contains(&token_id) {
                break;
            }

            n_decode += 1;

            let t0 = Instant::now();
            if std::env::var("OLORIN_DEBUG_TOKENS").is_ok() {
                let text = self.tokenizer.decode(&[token_id]);
                let skipped = self.tokenizer.is_control_or_user_defined(token_id);
                eprintln!(
                    "[token] id={token_id:6} skipped={skipped} in_think={in_thinking} text={text:?}"
                );
            }
            if Some(token_id) == self.channel_open_id {
                if !in_thinking {
                    in_thinking = true;
                    on_event(GenEvent::Thinking(true));
                }
            } else if Some(token_id) == self.channel_close_id {
                if in_thinking {
                    in_thinking = false;
                    on_event(GenEvent::Thinking(false));
                }
            } else if !self.tokenizer.is_control_or_user_defined(token_id) {
                let text = self.tokenizer.decode(&[token_id]);
                if !in_thinking {
                    on_event(GenEvent::Token(&text));
                    output.push_str(&text);
                }
            }
            t_other_total += t0.elapsed().as_micros() as u64;

            let t0 = Instant::now();
            let logits = self.state.forward_one_graph(&self.model, token_id, &self.graph_pool);
            t_forward_total += t0.elapsed().as_micros() as u64;

            // Telemetry tap — softmax entropy per decoded-token logit vector.
            // No-op unless OLORIN_LOGIT_ENTROPY=1.
            crate::inference::activation_track::record_logit_entropy(logits);

            let t0 = Instant::now();
            logits_snapshot.clear();
            logits_snapshot.extend_from_slice(logits);
            t_copy_total += t0.elapsed().as_micros() as u64;
        }
        let t_decode = t_decode_start.elapsed();

        if timing {
            let pp_ms = t_prefill.as_secs_f64() * 1000.0;
            let tg_ms = t_decode.as_secs_f64() * 1000.0;
            let pp_tps = if pp_ms > 0.0 { n_prompt as f64 / (pp_ms / 1000.0) } else { 0.0 };
            let tg_tps = if tg_ms > 0.0 { n_decode as f64 / (tg_ms / 1000.0) } else { 0.0 };
            eprintln!("[timing] prefill: {n_prompt} tokens in {pp_ms:.1}ms ({pp_tps:.1} t/s)");
            eprintln!("[timing] decode:  {n_decode} tokens in {tg_ms:.1}ms ({tg_tps:.1} t/s)");
            if n_decode > 0 {
                let ms = |us: u64| us as f64 / 1000.0;
                let per = |us: u64| ms(us) / n_decode as f64;
                eprintln!("[decode-breakdown] per token: forward={:.1}ms sample={:.1}ms copy={:.1}ms other={:.1}ms",
                    per(t_forward_total), per(t_sample_total), per(t_copy_total), per(t_other_total));
                eprintln!("[decode-breakdown] total: forward={:.1}ms sample={:.1}ms copy={:.1}ms other={:.1}ms",
                    ms(t_forward_total), ms(t_sample_total), ms(t_copy_total), ms(t_other_total));
            }
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Chat template
// ---------------------------------------------------------------------------

fn format_chat(user: &str, system: &str) -> String {
    // Gemma 4 chat format with enable_thinking=1 (the default for this
    // instruction-tuned model). Jinja chat_template emits:
    //   {{ bos_token }}                           <- caller adds bos_id
    //   if enable_thinking or system or tools:
    //     <|turn>system\n
    //     if enable_thinking: <|think|>
    //     if system: {system_trimmed}
    //     <turn|>\n
    //   for each msg:
    //     <|turn>{role}\n{content}<turn|>\n
    //   <|turn>model\n
    //
    // The `<|think|>` token signals "chain-of-thought enabled" — the model
    // then wraps its reasoning in `<|channel>...<channel|>` during decode.
    // The decode loop hides that block from user-visible output while still
    // letting the model benefit from the reasoning.
    let mut out = String::with_capacity(system.len() + user.len() + 96);
    let sys_trim = system.trim();
    out.push_str("<|turn>system\n");
    out.push_str("<|think|>");
    if !sys_trim.is_empty() {
        out.push_str(sys_trim);
    }
    out.push_str("<turn|>\n");
    out.push_str("<|turn>user\n");
    out.push_str(user.trim());
    out.push_str("<turn|>\n");
    out.push_str("<|turn>model\n");
    out
}

use super::sample::{sample, xorshift_seed};

// ---------------------------------------------------------------------------
// Model discovery
// ---------------------------------------------------------------------------

/// Find a GGUF model in standard locations. Prefers gemma4 if present.
pub fn find_model() -> Option<PathBuf> {
    let dir = models_dir()?;
    for candidate in GEMMA4_CANDIDATES {
        let p = dir.join(candidate);
        if p.exists() {
            return Some(p);
        }
    }
    std::fs::read_dir(&dir).ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.extension().map(|e| e == "gguf").unwrap_or(false))
}

/// gemma4 gguf preference order:
/// 1. q3kffnimpl — current ship (Q3K on both FFN arms, +13.4% prefill /
///    +14.1% decode / -24.7% RSS vs Q4K_M baseline; requires Q3Kx8 GEMM
///    kernel from commit 9eaebb2).
/// 2. adaptive-imatrix — earlier RSS-leaning variant.
/// 3. plain Q4_K_M baseline.
const GEMMA4_CANDIDATES: &[&str] = &[
    "gemma-4-e2b-it-Q4_K_M-q3kffnimpl.gguf",
    "gemma-4-e2b-it-Q4_K_M-adaptive-imatrix.gguf",
    "gemma-4-e2b-it-Q4_K_M.gguf",
];

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
    match arg {
        Some("gemma4") | None => find_model(),
        Some(name) => {
            let stem_path = olorin_models.join(format!("{name}.gguf"));
            if stem_path.exists() { return Some(stem_path); }
            let p = PathBuf::from(name);
            p.exists().then_some(p)
        }
    }
}

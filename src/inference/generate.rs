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
use crate::inference::sample::{sample, xorshift_seed};

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

pub struct Engine {
    _gguf: GgufFile,           // owns the mmap — keeps weight pointers valid
    model: Gemma4Model,
    tokenizer: Tokenizer,
    state: Gemma4State,
    pool: crate::inference::threadpool::ThreadPool,
    graph_pool: crate::inference::threadpool::GraphPool,
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
        let pool = crate::inference::threadpool::ThreadPool::new();
        let graph_pool = crate::inference::threadpool::GraphPool::new();
        eprintln!("[Olorin] Thread pool: {} threads", pool.thread_count());
        let state = Gemma4State::new(&model, max_seq_len, &pool);

        Ok(Self {
            _gguf: gguf,
            model,
            tokenizer,
            state,
            pool,
            graph_pool,
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
        let mut n_decode = 0usize;

        let mut t_sample_total: u64 = 0;
        let mut t_forward_total: u64 = 0;
        let mut t_copy_total: u64 = 0;
        let mut t_other_total: u64 = 0;

        let mut timing_acc = DecodeTiming::default();

        let mut n_spec_steps: usize = 0;
        let mut n_spec_accepted: usize = 0;

        // Live context (prompt + every emitted token) for prompt-lookup drafts.
        let mut context_tokens: Vec<u32> = tokens.clone();
        // Scratch for speculative decoding, sized for max draft_k.
        let draft_k = self.draft_k;
        let mut draft_buf: Vec<u32> = if draft_k >= 2 { vec![0u32; draft_k - 1] } else { Vec::new() };
        let vocab_size = self.model.vocab_size;
        let mut out_argmax: Vec<u32> = if draft_k >= 2 { vec![0u32; draft_k] } else { Vec::new() };

        'outer: for _ in 0..self.max_tokens {
            let t0 = Instant::now();
            let a0 = sample(
                &mut logits_snapshot,
                self.temperature,
                self.top_k,
                self.top_p,
                self.min_p,
                &mut rng,
            );
            t_sample_total += t0.elapsed().as_micros() as u64;

            // Plain single-token path — speculation disabled or context too short.
            if draft_k <= 1 || context_tokens.len() < 3 {
                if self.emit_and_advance(
                    a0,
                    &mut output,
                    on_token,
                    &mut logits_snapshot,
                    &mut n_decode,
                    &mut timing_acc,
                    eos,
                ) {
                    break;
                }
                context_tokens.push(a0);
                continue;
            }

            // ── Speculative path ──────────────────────────────────────
            // Build n-gram key from the last 3 tokens *including* A_0, so the
            // lookup sees the freshly-sampled token as part of the key.
            context_tokens.push(a0);
            let ctx_len = context_tokens.len();
            let key: [u32; 3] = [
                context_tokens[ctx_len - 3],
                context_tokens[ctx_len - 2],
                context_tokens[ctx_len - 1],
            ];
            let n = crate::kernels::ffi_inference::ngram_lookup(
                &context_tokens,
                &key,
                draft_k - 1,
                &mut draft_buf,
            );

            if n == 0 {
                // No draft match. Pop A_0 so emit_and_advance owns the push.
                context_tokens.pop();
                if self.emit_and_advance(
                    a0,
                    &mut output,
                    on_token,
                    &mut logits_snapshot,
                    &mut n_decode,
                    &mut timing_acc,
                    eos,
                ) {
                    break;
                }
                context_tokens.push(a0);
                continue;
            }

            // Build batch [A_0, draft_buf[0..n]] of size n+1.
            let k_rows = n + 1;
            let mut batch: Vec<u32> = Vec::with_capacity(k_rows);
            batch.push(a0);
            batch.extend_from_slice(&draft_buf[..n]);

            let s_anchor = self.state.cache.seq_len();
            let t_fwd = Instant::now();
            let logits_all = self
                .state
                .forward_batch_all_logits(&self.model, &batch, &self.graph_pool);
            timing_acc.forward_us += t_fwd.elapsed().as_micros() as u64;
            n_spec_steps += 1;

            // verify_draft expects out_argmax.len() == k_rows.
            let verify_slice = &mut out_argmax[..k_rows];
            let j = crate::kernels::ffi_inference::verify_draft(
                &logits_all[..k_rows * vocab_size],
                vocab_size,
                &draft_buf[..n],
                k_rows,
                verify_slice,
            );

            let (accepted, correction) = if j == k_rows {
                (n, verify_slice[n])
            } else {
                (j, verify_slice[j])
            };

            n_spec_accepted += accepted;

            // Rewind KV to keep A_0 + `accepted` drafts' slots.
            self.state.rewind_to(s_anchor + 1 + accepted);

            // Write correction's KV at the rewound position and get fresh logits.
            let t_fwd = Instant::now();
            let corr_logits = self
                .state
                .forward_one_graph(&self.model, correction, &self.graph_pool);
            timing_acc.forward_us += t_fwd.elapsed().as_micros() as u64;
            let t_cp = Instant::now();
            logits_snapshot.clear();
            logits_snapshot.extend_from_slice(corr_logits);
            timing_acc.copy_us += t_cp.elapsed().as_micros() as u64;

            // Emit in order: A_0 (already pushed), accepted drafts, correction.
            // Pop A_0 so emit_token can account for it cleanly? No — emit_token
            // doesn't touch context_tokens. A_0 is already in context_tokens.
            if self.emit_token(a0, &mut output, on_token, &mut n_decode, &mut timing_acc, eos) {
                break 'outer;
            }
            let mut stopped = false;
            for i in 0..accepted {
                let tok = draft_buf[i];
                if self.emit_token(tok, &mut output, on_token, &mut n_decode, &mut timing_acc, eos) {
                    stopped = true;
                    break;
                }
                context_tokens.push(tok);
            }
            if stopped {
                break 'outer;
            }
            if self.emit_token(correction, &mut output, on_token, &mut n_decode, &mut timing_acc, eos) {
                break 'outer;
            }
            context_tokens.push(correction);
        }
        t_forward_total += timing_acc.forward_us;
        t_copy_total += timing_acc.copy_us;
        t_other_total += timing_acc.other_us;
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
            if self.draft_k > 0 && n_spec_steps > 0 {
                let attempted = n_spec_steps * (self.draft_k.saturating_sub(1));
                let rate = if attempted > 0 { n_spec_accepted as f64 / attempted as f64 } else { 0.0 };
                eprintln!(
                    "[timing] speculative: K={} steps={} accepted={} accept_rate={:.2}",
                    self.draft_k, n_spec_steps, n_spec_accepted, rate
                );
            }
        }

        Ok(output)
    }

    /// Emit a token: stop check, optional on_token streaming, output push.
    /// Does NOT run the forward pass and does NOT touch context_tokens.
    /// Returns `true` when decoding should stop (EOS or stop_id).
    #[inline]
    fn emit_token(
        &mut self,
        token_id: u32,
        output: &mut String,
        on_token: &dyn Fn(&str),
        n_decode: &mut usize,
        timing: &mut DecodeTiming,
        eos: u32,
    ) -> bool {
        if token_id == eos || self.tokenizer.stop_ids.contains(&token_id) {
            return true;
        }

        *n_decode += 1;

        let t0 = Instant::now();
        if !self.tokenizer.is_control_or_user_defined(token_id) {
            let text = self.tokenizer.decode(&[token_id]);
            on_token(&text);
            output.push_str(&text);
        }
        timing.other_us += t0.elapsed().as_micros() as u64;

        false
    }

    /// Emit a single sampled token and advance the KV state by one step.
    ///
    /// Composes `emit_token` + `forward_one_graph` + logits copy. Used by the
    /// non-speculative path.
    #[inline]
    fn emit_and_advance(
        &mut self,
        token_id: u32,
        output: &mut String,
        on_token: &dyn Fn(&str),
        logits_snapshot: &mut Vec<f32>,
        n_decode: &mut usize,
        timing: &mut DecodeTiming,
        eos: u32,
    ) -> bool {
        if self.emit_token(token_id, output, on_token, n_decode, timing, eos) {
            return true;
        }

        let t0 = Instant::now();
        let logits = self.state.forward_one_graph(&self.model, token_id, &self.graph_pool);
        timing.forward_us += t0.elapsed().as_micros() as u64;

        let t0 = Instant::now();
        logits_snapshot.clear();
        logits_snapshot.extend_from_slice(logits);
        timing.copy_us += t0.elapsed().as_micros() as u64;

        false
    }
}

#[derive(Default)]
struct DecodeTiming {
    forward_us: u64,
    copy_us: u64,
    other_us: u64,
}

// ---------------------------------------------------------------------------
// Chat template
// ---------------------------------------------------------------------------

fn format_chat(user: &str, system: &str) -> String {
    // Gemma 4 chat format — exact match for llama.cpp default behavior with
    // enable_thinking=1 (the default for this instruction-tuned model).
    //
    // Jinja chat_template emits:
    //   {{ bos_token }}                           <- caller adds bos_id
    //   if enable_thinking or system or tools:
    //     <|turn>system\n
    //     if enable_thinking: <|think|>
    //     if system: {system_trimmed}
    //     <turn|>\n
    //   for each msg:
    //     <|turn>{role}\n{content}<turn|>\n
    //   <|turn>model\n
    let mut out = String::with_capacity(system.len() + user.len() + 96);
    let sys_trim = system.trim();
    // Always emit a system turn — llama.cpp defaults enable_thinking=1.
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
        ("gemma4", "gemma-4-e2b-it-Q4_K_M.gguf"),
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

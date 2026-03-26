//! Raw FFI bindings to llama.cpp.
//!
//! Minimal surface: model loading, context creation, tokenization,
//! decode (prefill/generate), sampling, and state export.

#![allow(non_camel_case_types)]

use std::ffi::{c_char, c_int, c_float, c_void};
use std::os::raw::c_uchar;

// Opaque types
pub enum llama_model {}
pub enum llama_context {}
pub enum llama_sampler {}
pub enum llama_memory_i {}
pub enum llama_vocab {}

// Token type
pub type llama_token = c_int;

/// Model parameters — matches the C struct layout on x86_64 exactly.
///
/// C layout (from llama.h):
///   ptr devices                        offset  0, size 8
///   ptr tensor_buft_overrides          offset  8, size 8
///   i32 n_gpu_layers                   offset 16, size 4
///   i32 split_mode (enum)              offset 20, size 4
///   i32 main_gpu                       offset 24, size 4
///   4 bytes padding                    offset 28, size 4
///   ptr tensor_split                   offset 32, size 8
///   ptr progress_callback              offset 40, size 8
///   ptr progress_callback_user_data    offset 48, size 8
///   ptr kv_overrides                   offset 56, size 8
///   bool vocab_only                    offset 64, size 1
///   bool use_mmap                      offset 65, size 1
///   bool use_direct_io                 offset 66, size 1
///   bool use_mlock                     offset 67, size 1
///   bool check_tensors                 offset 68, size 1
///   bool use_extra_bufts               offset 69, size 1
///   bool no_host                       offset 70, size 1
///   bool no_alloc                      offset 71, size 1
///   Total: 72 bytes (aligned to 8)
#[repr(C)]
pub struct llama_model_params {
    pub devices: *mut c_void,
    pub tensor_buft_overrides: *const c_void,
    pub n_gpu_layers: c_int,
    pub split_mode: c_int,
    pub main_gpu: c_int,
    _pad0: u32,
    pub tensor_split: *const c_float,
    pub progress_callback: *const c_void,
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides: *const c_void,
    pub vocab_only: bool,
    pub use_mmap: bool,
    pub use_direct_io: bool,
    pub use_mlock: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub no_host: bool,
    pub no_alloc: bool,
}

/// Context parameters — matches the C struct layout on x86_64 exactly.
///
/// C layout (from llama.h):
///   u32 n_ctx                          offset  0, size 4
///   u32 n_batch                        offset  4, size 4
///   u32 n_ubatch                       offset  8, size 4
///   u32 n_seq_max                      offset 12, size 4
///   i32 n_threads                      offset 16, size 4
///   i32 n_threads_batch                offset 20, size 4
///   i32 rope_scaling_type (enum)       offset 24, size 4
///   i32 pooling_type (enum)            offset 28, size 4
///   i32 attention_type (enum)          offset 32, size 4
///   i32 flash_attn_type (enum)         offset 36, size 4
///   f32 rope_freq_base                 offset 40, size 4
///   f32 rope_freq_scale                offset 44, size 4
///   f32 yarn_ext_factor                offset 48, size 4
///   f32 yarn_attn_factor               offset 52, size 4
///   f32 yarn_beta_fast                 offset 56, size 4
///   f32 yarn_beta_slow                 offset 60, size 4
///   u32 yarn_orig_ctx                  offset 64, size 4
///   f32 defrag_thold                   offset 68, size 4
///   ptr cb_eval                        offset 72, size 8
///   ptr cb_eval_user_data              offset 80, size 8
///   i32 type_k (enum ggml_type)        offset 88, size 4
///   i32 type_v (enum ggml_type)        offset 92, size 4
///   ptr abort_callback                 offset 96, size 8
///   ptr abort_callback_data            offset 104, size 8
///   bool embeddings                    offset 112, size 1
///   bool offload_kqv                   offset 113, size 1
///   bool no_perf                       offset 114, size 1
///   bool op_offload                    offset 115, size 1
///   bool swa_full                      offset 116, size 1
///   bool kv_unified                    offset 117, size 1
///   2 bytes padding                    offset 118, size 2
///   ptr samplers                       offset 120, size 8
///   usize n_samplers                   offset 128, size 8
///   Total: 136 bytes (aligned to 8)
#[repr(C)]
pub struct llama_context_params {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_threads: c_int,
    pub n_threads_batch: c_int,
    pub rope_scaling_type: c_int,
    pub pooling_type: c_int,
    pub attention_type: c_int,
    pub flash_attn_type: c_int,
    pub rope_freq_base: c_float,
    pub rope_freq_scale: c_float,
    pub yarn_ext_factor: c_float,
    pub yarn_attn_factor: c_float,
    pub yarn_beta_fast: c_float,
    pub yarn_beta_slow: c_float,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: c_float,
    pub cb_eval: *const c_void,
    pub cb_eval_user_data: *mut c_void,
    pub type_k: c_int,
    pub type_v: c_int,
    pub abort_callback: *const c_void,
    pub abort_callback_data: *mut c_void,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    _pad0: [u8; 2],
    pub samplers: *mut c_void,
    pub n_samplers: usize,
}

/// Batch for decode
#[repr(C)]
#[derive(Copy, Clone)]
pub struct llama_batch {
    pub n_tokens: i32,
    pub token: *mut llama_token,
    pub embd: *mut c_float,
    pub pos: *mut i32,
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut i32,
    pub logits: *mut i8,
}

#[repr(C)]
pub struct llama_sampler_chain_params {
    pub no_perf: bool,
}

extern "C" {
    // Default params
    pub fn llama_model_default_params() -> llama_model_params;
    pub fn llama_context_default_params() -> llama_context_params;

    // Model
    pub fn llama_model_load_from_file(
        path_model: *const c_char,
        params: llama_model_params,
    ) -> *mut llama_model;
    pub fn llama_model_free(model: *mut llama_model);

    // Vocab
    pub fn llama_model_get_vocab(model: *const llama_model) -> *const llama_vocab;

    // Context
    pub fn llama_init_from_model(
        model: *mut llama_model,
        params: llama_context_params,
    ) -> *mut llama_context;
    pub fn llama_free(ctx: *mut llama_context);

    // Tokenization (takes vocab, not model)
    pub fn llama_tokenize(
        vocab: *const llama_vocab,
        text: *const c_char,
        text_len: i32,
        tokens: *mut llama_token,
        n_tokens_max: i32,
        add_special: bool,
        parse_special: bool,
    ) -> i32;

    pub fn llama_token_to_piece(
        vocab: *const llama_vocab,
        token: llama_token,
        buf: *mut c_char,
        length: i32,
        lstrip: i32,
        special: bool,
    ) -> i32;

    // Special tokens (takes vocab, not model)
    pub fn llama_vocab_eos(vocab: *const llama_vocab) -> llama_token;
    pub fn llama_vocab_bos(vocab: *const llama_vocab) -> llama_token;

    // Vocab size
    pub fn llama_vocab_n_tokens(vocab: *const llama_vocab) -> i32;

    // Decode
    pub fn llama_decode(ctx: *mut llama_context, batch: llama_batch) -> c_int;

    // Batch helpers
    pub fn llama_batch_init(n_tokens: i32, embd: i32, n_seq_max: i32) -> llama_batch;
    pub fn llama_batch_free(batch: llama_batch);

    // Logits
    pub fn llama_get_logits_ith(ctx: *mut llama_context, i: i32) -> *mut c_float;

    // Memory (KV cache)
    pub fn llama_get_memory(ctx: *const llama_context) -> *mut llama_memory_i;
    pub fn llama_memory_clear(mem: *mut llama_memory_i, data: bool);
    pub fn llama_memory_seq_rm(mem: *mut llama_memory_i, seq_id: i32, p0: i32, p1: i32) -> bool;

    // State (for Approach A — eakv bridge)
    pub fn llama_state_seq_get_size(ctx: *mut llama_context, seq_id: i32) -> usize;
    pub fn llama_state_seq_get_data(
        ctx: *mut llama_context,
        dst: *mut c_uchar,
        size: usize,
        seq_id: i32,
    ) -> usize;

    // Sampler
    pub fn llama_sampler_chain_init(params: llama_sampler_chain_params) -> *mut llama_sampler;
    pub fn llama_sampler_chain_add(chain: *mut llama_sampler, sampler: *mut llama_sampler);
    pub fn llama_sampler_free(sampler: *mut llama_sampler);
    pub fn llama_sampler_sample(sampler: *mut llama_sampler, ctx: *mut llama_context, idx: i32) -> llama_token;
    pub fn llama_sampler_reset(sampler: *mut llama_sampler);

    // Built-in samplers
    pub fn llama_sampler_init_greedy() -> *mut llama_sampler;
    pub fn llama_sampler_init_dist(seed: u32) -> *mut llama_sampler;
    pub fn llama_sampler_init_temp(temp: c_float) -> *mut llama_sampler;
    pub fn llama_sampler_init_top_p(p: c_float, min_keep: usize) -> *mut llama_sampler;
    pub fn llama_sampler_init_top_k(k: i32) -> *mut llama_sampler;

    pub fn llama_sampler_chain_default_params() -> llama_sampler_chain_params;

    // Logging
    pub fn llama_log_set(
        log_callback: Option<unsafe extern "C" fn(level: c_int, text: *const c_char, user_data: *mut c_void)>,
        user_data: *mut c_void,
    );

    // eaclaw C++ generation loop (csrc/eaclaw_generate.cpp)
    pub fn eaclaw_generate_loop(
        ctx: *mut llama_context,
        smpl: *mut llama_sampler,
        vocab: *const llama_vocab,
        start_pos: i32,
        max_tokens: i32,
        cb: Option<unsafe extern "C" fn(token: i32, user_data: *mut c_void) -> c_int>,
        user_data: *mut c_void,
    ) -> i32;
}

/// Suppress llama.cpp's verbose logging to stderr.
/// Only errors (4) are forwarded. Warnings are suppressed because llama.cpp
/// emits noisy but harmless warnings for Qwen2.5 GGUF metadata quirks and
/// context size being smaller than training context.
unsafe extern "C" fn quiet_log_callback(level: c_int, text: *const c_char, _user_data: *mut c_void) {
    // GGML_LOG_LEVEL: NONE=0, DEBUG=1, INFO=2, WARN=3, ERROR=4, CONT=5
    if level == 4 {
        let msg = std::ffi::CStr::from_ptr(text).to_string_lossy();
        let msg = msg.trim_end();
        if !msg.is_empty() {
            eprintln!("llama.cpp error: {msg}");
        }
    }
}

/// Safe wrapper around llama.cpp model + context.
pub struct LlamaEngine {
    model: *mut llama_model,
    vocab: *const llama_vocab,
    ctx: *mut llama_context,
    sampler: *mut llama_sampler,
    _n_ctx: u32,
    n_batch: u32,
}

// Safety: LlamaEngine is used behind a Mutex in LocalLlmProvider,
// ensuring single-threaded access to the raw pointers.
unsafe impl Send for LlamaEngine {}

/// Result from `LlamaEngine::generate_stream_timed`, including timing data.
pub struct GenerateResult {
    pub tokens: Vec<llama_token>,
    pub decode_ms: f64,
}

impl LlamaEngine {
    /// Load model and create context.
    pub fn new(model_path: &str, n_ctx: u32, n_batch: u32, n_threads: u32, mlock: bool) -> Result<Self, String> {
        use std::ffi::CString;

        let c_path = CString::new(model_path)
            .map_err(|e| format!("invalid model path: {e}"))?;

        unsafe {
            // Suppress verbose llama.cpp logging (model loader, tensor info, etc.)
            llama_log_set(Some(quiet_log_callback), std::ptr::null_mut());

            let mut model_params = llama_model_default_params();
            model_params.n_gpu_layers = 0; // CPU only
            model_params.use_mlock = mlock;

            let model = llama_model_load_from_file(c_path.as_ptr(), model_params);
            if model.is_null() {
                return Err(format!("failed to load model: {model_path}"));
            }
            if mlock {
                eprintln!("eaclaw: mlock enabled — if model load was slow or failed, \
                           check `ulimit -l` or set memlock in /etc/security/limits.conf");
            }

            let vocab = llama_model_get_vocab(model);
            if vocab.is_null() {
                llama_model_free(model);
                return Err("failed to get vocab from model".into());
            }

            let mut ctx_params = llama_context_default_params();
            ctx_params.n_ctx = n_ctx;
            ctx_params.n_batch = n_batch;
            ctx_params.n_ubatch = n_batch;
            ctx_params.n_threads = n_threads as c_int;
            ctx_params.n_threads_batch = n_threads as c_int;

            let ctx = llama_init_from_model(model, ctx_params);
            if ctx.is_null() {
                llama_model_free(model);
                return Err("failed to create llama context".into());
            }

            // Set up sampler chain: top-k -> top-p -> temperature
            let chain_params = llama_sampler_chain_default_params();
            let sampler = llama_sampler_chain_init(chain_params);
            llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
            llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95, 1));
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.7));
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(0xFFFFFFFF));

            Ok(Self { model, vocab, ctx, sampler, _n_ctx: n_ctx, n_batch })
        }
    }

    /// Tokenize text into token IDs.
    pub fn tokenize(&self, text: &str, add_special: bool) -> Vec<llama_token> {
        let c_text = std::ffi::CString::new(text).unwrap_or_default();
        let mut tokens = vec![0i32; text.len() + 16];
        let n = unsafe {
            llama_tokenize(
                self.vocab, c_text.as_ptr(), text.len() as i32,
                tokens.as_mut_ptr(), tokens.len() as i32,
                add_special, true,
            )
        };
        if n < 0 {
            // Buffer too small, resize
            tokens.resize((-n) as usize, 0);
            let n2 = unsafe {
                llama_tokenize(
                    self.vocab, c_text.as_ptr(), text.len() as i32,
                    tokens.as_mut_ptr(), tokens.len() as i32,
                    add_special, true,
                )
            };
            tokens.truncate(n2.max(0) as usize);
        } else {
            tokens.truncate(n as usize);
        }
        tokens
    }

    /// Convert a single token to text.
    pub fn token_to_str(&self, token: llama_token) -> String {
        let mut buf = vec![0u8; 64];
        let n = unsafe {
            llama_token_to_piece(
                self.vocab, token,
                buf.as_mut_ptr() as *mut c_char, buf.len() as i32,
                0, false,
            )
        };
        if n > 0 {
            buf.truncate(n as usize);
            String::from_utf8_lossy(&buf).into_owned()
        } else {
            String::new()
        }
    }

    /// Detokenize a slice of tokens.
    pub fn detokenize(&self, tokens: &[llama_token]) -> String {
        tokens.iter().map(|&t| self.token_to_str(t)).collect()
    }

    /// Build a lookup table mapping every token ID to its string representation.
    /// Used to avoid calling `token_to_str` during generation (which would
    /// require borrowing the engine while C++ holds a mutable reference).
    pub fn build_vocab_table(&self) -> Vec<String> {
        let n = unsafe { llama_vocab_n_tokens(self.vocab) } as usize;
        let mut table = Vec::with_capacity(n);
        for i in 0..n {
            table.push(self.token_to_str(i as llama_token));
        }
        table
    }

    /// EOS token ID.
    pub fn eos_token(&self) -> llama_token {
        unsafe { llama_vocab_eos(self.vocab) }
    }

    /// Decode a batch of tokens (prefill or single-token generate).
    /// Automatically chunks into n_batch-sized pieces for large prefills.
    pub fn decode(&mut self, tokens: &[llama_token], start_pos: i32) -> Result<(), String> {
        let n_batch = self.n_batch as usize;

        for (chunk_idx, chunk) in tokens.chunks(n_batch).enumerate() {
            let chunk_start = start_pos + (chunk_idx * n_batch) as i32;
            let is_last_chunk = chunk_start + chunk.len() as i32 >= start_pos + tokens.len() as i32;
            self.decode_batch(chunk, chunk_start, is_last_chunk)?;
        }
        Ok(())
    }

    fn decode_batch(&mut self, tokens: &[llama_token], start_pos: i32, compute_logits: bool) -> Result<(), String> {
        unsafe {
            let mut batch = llama_batch_init(tokens.len() as i32, 0, 1);
            batch.n_tokens = tokens.len() as i32;

            for (i, &tok) in tokens.iter().enumerate() {
                *batch.token.add(i) = tok;
                *batch.pos.add(i) = start_pos + i as i32;
                *batch.n_seq_id.add(i) = 1;
                let seq_ids = std::slice::from_raw_parts_mut(*batch.seq_id.add(i), 1);
                seq_ids[0] = 0;
                // Only compute logits for the last token of the last chunk
                *batch.logits.add(i) = if compute_logits && i == tokens.len() - 1 { 1 } else { 0 };
            }

            let rc = llama_decode(self.ctx, batch);
            llama_batch_free(batch);

            if rc != 0 {
                return Err(format!("llama_decode failed: {rc}"));
            }
            Ok(())
        }
    }

    /// Sample next token from logits of the last decoded position.
    pub fn sample(&mut self) -> llama_token {
        unsafe {
            let token = llama_sampler_sample(self.sampler, self.ctx, -1);
            llama_sampler_reset(self.sampler);
            token
        }
    }

    /// Clear the KV cache.
    pub fn kv_cache_clear(&mut self) {
        unsafe {
            let mem = llama_get_memory(self.ctx);
            llama_memory_clear(mem, true);
        }
    }

    /// Remove KV cache entries from position p0 to end.
    pub fn kv_cache_truncate(&mut self, p0: i32) {
        unsafe {
            let mem = llama_get_memory(self.ctx);
            llama_memory_seq_rm(mem, 0, p0, -1);
        }
    }

    /// Run generation loop in C++ for maximum throughput.
    ///
    /// Calls the callback for each token. Callback returns `true` to stop.
    /// Returns the list of generated tokens (excluding EOS).
    pub fn generate_stream<F>(&mut self, start_pos: i32, max_tokens: i32, on_token: F) -> Vec<llama_token>
    where
        F: FnMut(llama_token) -> bool,
    {
        self.generate_stream_timed(start_pos, max_tokens, on_token).tokens
    }

    /// Like `generate_stream` but also returns decode wall-clock time.
    pub fn generate_stream_timed<F>(&mut self, start_pos: i32, max_tokens: i32, mut on_token: F) -> GenerateResult
    where
        F: FnMut(llama_token) -> bool,
    {
        struct CallbackState<'a, G: FnMut(llama_token) -> bool> {
            on_token: &'a mut G,
            tokens: Vec<llama_token>,
        }

        unsafe extern "C" fn trampoline<G: FnMut(llama_token) -> bool>(
            token: i32,
            user_data: *mut c_void,
        ) -> c_int {
            let state = &mut *(user_data as *mut CallbackState<G>);
            state.tokens.push(token);
            if (state.on_token)(token) { 1 } else { 0 }
        }

        let mut state = CallbackState {
            on_token: &mut on_token,
            tokens: Vec::new(),
        };

        let t0 = std::time::Instant::now();
        unsafe {
            eaclaw_generate_loop(
                self.ctx,
                self.sampler,
                self.vocab,
                start_pos,
                max_tokens,
                Some(trampoline::<F>),
                &mut state as *mut _ as *mut c_void,
            );
        }
        let decode_ms = t0.elapsed().as_secs_f64() * 1000.0;

        GenerateResult {
            tokens: state.tokens,
            decode_ms,
        }
    }

    /// Export KV state for eakv bridge (Approach A).
    pub fn export_kv_state(&mut self) -> Vec<u8> {
        unsafe {
            let size = llama_state_seq_get_size(self.ctx, 0);
            let mut buf = vec![0u8; size];
            llama_state_seq_get_data(self.ctx, buf.as_mut_ptr(), size, 0);
            buf
        }
    }
}

impl Drop for LlamaEngine {
    fn drop(&mut self) {
        unsafe {
            llama_sampler_free(self.sampler);
            llama_free(self.ctx);
            llama_model_free(self.model);
        }
    }
}

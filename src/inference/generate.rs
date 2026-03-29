//! Public inference API — the sole entry point for text generation.
//!
//! `Engine::load(path)` opens a GGUF model.
//! `Engine::generate(prompt, on_token)` tokenizes, runs forward pass, returns text.
//! Knows nothing about channels, vault, or safety.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use crate::error::Result;
use crate::inference::engine::{BitNetModel, QuantType};
use crate::inference::gguf::GgufFile;
use crate::inference::tokenizer::Tokenizer;

pub struct Engine {
    gguf: GgufFile,
    max_seq_len: usize,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
}

impl Engine {
    /// Load a GGUF model from disk.
    pub fn load(path: &Path, max_seq_len: usize) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        Ok(Engine {
            gguf,
            max_seq_len,
            max_tokens: 128,
            temperature: 0.1,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.3,
        })
    }

    /// Detect quantization type of the loaded model.
    pub fn quant_type_str(&self) -> &str {
        let idx = self.gguf.tensor_map.get("blk.0.attn_q.weight")
            .or_else(|| self.gguf.tensor_map.get("blk.0.attn_qkv.weight"));
        match idx {
            Some(&i) => match self.gguf.tensors[i].dtype {
                36 => "I2S",
                12 | 14 => "Q4K",
                _ => "unknown",
            },
            None => "unknown",
        }
    }

    /// Generate text from a prompt. Calls `on_token` for each generated token.
    /// Returns the complete generated text.
    pub fn generate(&self, prompt: &str, on_token: &dyn Fn(&str)) -> Result<String> {
        let model = BitNetModel::from_gguf(&self.gguf)?;
        let tokenizer = Tokenizer::from_gguf(&self.gguf)?;

        let is_q4k = model.quant_type == QuantType::Q4K;

        // Build prompt tokens with chat template
        let mut tokens = vec![tokenizer.bos_id];
        if is_q4k {
            let has_chatml = tokenizer.token_to_id("<|im_start|>").is_some();
            let chat = if has_chatml {
                format!(
                    "<|im_start|>system\nYou are Olorin, a concise assistant. \
                     Answer in one short sentence.<|im_end|>\n\
                     <|im_start|>user\n{prompt}<|im_end|>\n\
                     <|im_start|>assistant\n"
                )
            } else {
                format!(
                    "<|start_header_id|>system<|end_header_id|>\n\n\
                     You are Olorin, a concise assistant. \
                     Answer in one short sentence.<|eot_id|>\
                     <|start_header_id|>user<|end_header_id|>\n\n\
                     {prompt}<|eot_id|>\
                     <|start_header_id|>assistant<|end_header_id|>\n\n"
                )
            };
            tokens.extend(tokenizer.encode(&chat));
        } else {
            let full = format!(
                "You are Olorin, a concise assistant. \
                 Answer in one short sentence.\n\nQ: {prompt}\nA:"
            );
            tokens.extend(tokenizer.encode(&full));
        }

        let tok_ref = &tokenizer;
        let output = Arc::new(std::sync::Mutex::new(Vec::<u32>::new()));
        let output_ref = output.clone();

        let on_tok = |tok_id: u32| {
            let text = tok_ref.decode(&[tok_id]);
            on_token(&text);
            output_ref.lock().unwrap().push(tok_id);
        };

        let generated = if is_q4k {
            use crate::inference::forward_llama;
            let (gen, _, _) = forward_llama::generate(
                &model, &tokens, self.max_tokens, self.temperature,
                self.top_k, self.top_p, self.repetition_penalty,
                tokenizer.eos_id, self.max_seq_len, on_tok,
            );
            gen
        } else {
            use crate::inference::forward::InferenceState;
            let (gen, _, _) = InferenceState::generate(
                &model, &tokens, self.max_tokens, self.temperature,
                self.top_k, self.top_p, self.repetition_penalty,
                tokenizer.eos_id, self.max_seq_len, on_tok,
            );
            gen
        };

        let gen_tokens: Vec<u32> = generated[tokens.len()..].to_vec();
        let result = tokenizer.decode(&gen_tokens);

        // Wipe token buffers — no plaintext residue
        unsafe {
            std::ptr::write_bytes(tokens.as_mut_ptr(), 0, tokens.len());
            let gen_ptr = generated.as_ptr() as *mut u32;
            std::ptr::write_bytes(gen_ptr, 0, generated.len());
            let gt_ptr = gen_tokens.as_ptr() as *mut u32;
            std::ptr::write_bytes(gt_ptr, 0, gen_tokens.len());
            let mut out = output.lock().unwrap();
            std::ptr::write_bytes(out.as_mut_ptr(), 0, out.len());
        }
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);

        Ok(result)
    }
}

/// Find a GGUF model in standard locations.
pub fn find_model() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let home = Path::new(&home);
    let paths = [
        home.join(".olorin/models/ggml-model-i2_s.gguf"),
        home.join(".cougar/models/ggml-model-i2_s.gguf"),
    ];
    paths.into_iter().find(|p| p.exists())
}

/// Resolve model path from CLI argument or auto-detect.
pub fn resolve_model(arg: Option<&str>) -> Option<PathBuf> {
    let home = std::env::var("HOME").unwrap_or_default();
    let olorin_models = Path::new(&home).join(".olorin/models");
    match arg {
        Some("bitnet") => {
            let p = olorin_models.join("ggml-model-i2_s.gguf");
            p.exists().then_some(p)
        }
        Some("llama") => {
            let p = olorin_models.join("Llama-3.2-3B-Instruct-Q4_K_M.gguf");
            p.exists().then_some(p)
        }
        Some(path) => {
            let p = PathBuf::from(path);
            p.exists().then_some(p)
        }
        None => find_model(),
    }
}

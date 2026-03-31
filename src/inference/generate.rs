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
    /// Backing GGUF data — must outlive model (raw pointers into mmap).
    _gguf: GgufFile,
    model: BitNetModel,
    tokenizer: Tokenizer,
    max_seq_len: usize,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
}

impl Engine {
    /// Load a GGUF model from disk. Parses weights and tokenizer once.
    pub fn load(path: &Path, max_seq_len: usize) -> Result<Self> {
        let gguf = GgufFile::open(path)?;
        let model = BitNetModel::from_gguf(&gguf)?;
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        Ok(Engine {
            _gguf: gguf,
            model,
            tokenizer,
            max_seq_len,
            max_tokens: 64,
            temperature: 0.4,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.05,
        })
    }

    /// Detect quantization type of the loaded model.
    pub fn quant_type_str(&self) -> &str {
        self.model.quant_type_str()
    }

    /// Generate text from a prompt. Calls `on_token` for each generated token.
    /// Returns the complete generated text.
    pub fn generate(&self, prompt: &str, system: &str, on_token: &dyn Fn(&str)) -> Result<String> {
        let model = &self.model;
        let tokenizer = &self.tokenizer;

        let is_q4k = model.quant_type == QuantType::Q4K;

        // Build prompt tokens — match each model's native format
        let skip_bos = model.architecture == "qwen2";
        let mut tokens = if skip_bos { Vec::new() } else { vec![tokenizer.bos_id] };
        if is_q4k {
            let chat = build_chat_prompt(&model.architecture, system, prompt);
            tokens.extend(tokenizer.encode(&chat));
        } else {
            tokens.extend(tokenizer.encode(prompt));
        }

        let output = Arc::new(std::sync::Mutex::new(Vec::<u32>::new()));
        let output_ref = output.clone();

        let on_tok = |tok_id: u32| {
            let text = tokenizer.decode(&[tok_id]);
            on_token(&text);
            output_ref.lock().unwrap().push(tok_id);
        };

        let mut generated = if is_q4k {
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

        let mut gen_tokens: Vec<u32> = generated[tokens.len()..].to_vec();
        let result = tokenizer.decode(&gen_tokens);

        // Wipe token buffers — no plaintext residue
        unsafe {
            std::ptr::write_bytes(tokens.as_mut_ptr(), 0, tokens.len());
            std::ptr::write_bytes(generated.as_mut_ptr(), 0, generated.len());
            std::ptr::write_bytes(gen_tokens.as_mut_ptr(), 0, gen_tokens.len());
            let mut out = output.lock().unwrap();
            std::ptr::write_bytes(out.as_mut_ptr(), 0, out.len());
        }
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);

        Ok(result)
    }
}

/// Build chat prompt in the model's native template format.
fn build_chat_prompt(architecture: &str, system: &str, prompt: &str) -> String {
    let mut chat = String::new();
    match architecture {
        "qwen2" => {
            if !system.is_empty() {
                chat.push_str("<|im_start|>system\n");
                chat.push_str(system);
                chat.push_str("<|im_end|>\n");
            }
            chat.push_str("<|im_start|>user\n");
            chat.push_str(prompt);
            chat.push_str("<|im_end|>\n<|im_start|>assistant\n");
        }
        _ => {
            // Llama 3 format (default)
            if !system.is_empty() {
                chat.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
                chat.push_str(system);
                chat.push_str("<|eot_id|>");
            }
            chat.push_str("<|start_header_id|>user<|end_header_id|>\n\n");
            chat.push_str(prompt);
            chat.push_str("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
        }
    }
    chat
}

/// Find a GGUF model in standard locations.
pub fn find_model() -> Option<PathBuf> {
    let dir = models_dir()?;
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
    ];
    match arg {
        Some(name) => {
            // Try alias
            for &(alias, filename) in aliases {
                if name == alias {
                    let p = olorin_models.join(filename);
                    return p.exists().then_some(p);
                }
            }
            // Try stem name (from available_models)
            let stem_path = olorin_models.join(format!("{name}.gguf"));
            if stem_path.exists() { return Some(stem_path); }
            // Try full path
            let p = PathBuf::from(name);
            p.exists().then_some(p)
        }
        None => find_model(),
    }
}

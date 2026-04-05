//! Public inference API — the sole entry point for text generation.
//!
//! Stubbed for Gemma 4 transition. Model loading and generation will be
//! re-implemented once the forward pass is ready (Task 8).

use std::path::{Path, PathBuf};
use crate::error::Result;

pub struct Engine {
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
    pub fn load(_path: &Path, _max_seq_len: usize) -> Result<Self> {
        Err(crate::error::Error::Inference(
            "Gemma 4 inference not yet implemented".into(),
        ))
    }

    /// Load a draft model for speculative decoding.
    pub fn load_draft(&mut self, _path: &Path) -> Result<()> {
        Err(crate::error::Error::Inference(
            "Gemma 4 speculative decoding not yet implemented".into(),
        ))
    }

    /// Detect quantization type of the loaded model.
    pub fn quant_type_str(&self) -> &str {
        "gemma4"
    }

    /// Generate text from a prompt. Calls `on_token` for each generated token.
    /// Returns the complete generated text.
    pub fn generate(&self, _prompt: &str, _system: &str, _on_token: &dyn Fn(&str)) -> Result<String> {
        Err(crate::error::Error::Inference(
            "Gemma 4 inference not yet implemented".into(),
        ))
    }
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

//! Byte-level BPE tokenizer parsed from GGUF vocab metadata.
//!
//! Handles GPT-2 byte encoding where "bad" bytes (0-32, 127-160, 173) are
//! shifted to unicode 256+. Token strings in the GGUF use this encoding;
//! we reverse it at load time so the vocab stores raw bytes.

use std::collections::HashMap;
use crate::error::{Error, Result};
use crate::inference::gguf::{GgufFile, MetaValue};

pub struct Tokenizer {
    vocab: Vec<Vec<u8>>,
    token_to_id: HashMap<Vec<u8>, u32>,
    scores: Vec<f32>,
    pub bos_id: u32,
    pub eos_id: u32,
}

/// Parse a byte-fallback token like "<0x41>" into the byte value 0x41.
fn parse_byte_token(s: &str) -> Option<u8> {
    if s.len() == 6 && s.starts_with("<0x") && s.ends_with('>') {
        u8::from_str_radix(&s[3..5], 16).ok()
    } else {
        None
    }
}

/// Reverse the GPT-2 byte-to-unicode mapping for a single character.
/// Returns the raw byte value, or None if the character isn't in the GPT-2 mapping.
fn gpt2_unicode_to_byte(cp: u32) -> Option<u8> {
    // "Good" bytes: identity mapping (printable ASCII + Latin-1 supplement ranges)
    if (33..=126).contains(&cp) || (161..=172).contains(&cp) || (174..=255).contains(&cp) {
        return Some(cp as u8);
    }
    // "Bad" bytes: remapped to 256..324 in order: 0-32, 127-160, 173 (68 total)
    if cp >= 256 && cp < 256 + 68 {
        let idx = (cp - 256) as u8;
        if idx <= 32 { return Some(idx); }           // 0-32
        if idx <= 32 + 34 { return Some(idx - 33 + 127); } // 127-160
        if idx == 32 + 34 + 1 { return Some(173); }  // 173
    }
    None
}

/// Decode a GPT-2 encoded token string to raw bytes.
/// Falls back to raw UTF-8 bytes for characters outside the GPT-2 mapping
/// (e.g. special tokens like <|begin_of_text|>).
fn gpt2_decode_str(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    let mut all_mapped = true;
    for c in s.chars() {
        if let Some(b) = gpt2_unicode_to_byte(c as u32) {
            out.push(b);
        } else {
            all_mapped = false;
            break;
        }
    }
    if all_mapped {
        out
    } else {
        // Not a GPT-2 encoded token (e.g. special token) — use raw UTF-8
        s.as_bytes().to_vec()
    }
}

impl Tokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let tokens_arr = gguf.metadata.get("tokenizer.ggml.tokens")
            .ok_or_else(|| Error::Inference("missing tokenizer.ggml.tokens".into()))?;
        let scores_arr = gguf.metadata.get("tokenizer.ggml.scores");

        let token_strs = match tokens_arr {
            MetaValue::Array(arr) => {
                let mut out = Vec::with_capacity(arr.len());
                for v in arr {
                    match v {
                        MetaValue::Str(s) => out.push(s.as_str()),
                        _ => return Err(Error::Inference("tokens array contains non-string".into())),
                    }
                }
                out
            }
            _ => return Err(Error::Inference("tokenizer.ggml.tokens is not an array".into())),
        };

        let scores = match scores_arr {
            Some(MetaValue::Array(arr)) => {
                let mut out = Vec::with_capacity(arr.len());
                for v in arr {
                    match v {
                        MetaValue::F32(f) => out.push(*f),
                        _ => return Err(Error::Inference("scores array contains non-f32".into())),
                    }
                }
                out
            }
            _ => {
                // No scores (tiktoken/BPE models like Llama 3) — use token index as score
                // Lower index = higher priority merge (reverse of sentencepiece convention)
                (0..token_strs.len()).map(|i| -(i as f32)).collect()
            }
        };

        if token_strs.len() != scores.len() {
            return Err(Error::Inference(format!(
                "tokens ({}) and scores ({}) length mismatch",
                token_strs.len(), scores.len()
            )));
        }

        let mut vocab = Vec::with_capacity(token_strs.len());
        let mut token_to_id = HashMap::with_capacity(token_strs.len());

        for (i, tok_str) in token_strs.iter().enumerate() {
            let bytes = if let Some(b) = parse_byte_token(tok_str) {
                vec![b]
            } else {
                gpt2_decode_str(tok_str)
            };
            token_to_id.insert(bytes.clone(), i as u32);
            vocab.push(bytes);
        }

        let bos_id = gguf.get_u32("tokenizer.ggml.bos_token_id").unwrap_or(1);
        let eos_id = gguf.get_u32("tokenizer.ggml.eos_token_id").unwrap_or(2);

        Ok(Tokenizer { vocab, token_to_id, scores, bos_id, eos_id })
    }

    /// Look up a token string in the vocabulary. Returns None if not found.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token.as_bytes()).copied()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Split on special tokens (e.g. <|start_header_id|>) and encode segments
        let mut tokens: Vec<u32> = Vec::with_capacity(text.len());
        let mut remaining = text;
        while !remaining.is_empty() {
            // Find next special token
            if let Some(pos) = remaining.find("<|") {
                // Encode text before the special token
                if pos > 0 {
                    tokens.extend(self.encode_segment(remaining[..pos].as_bytes()));
                }
                // Try to match a known special token
                let after = &remaining[pos..];
                if let Some(end) = after.find("|>") {
                    let candidate = &after[..end + 2];
                    if let Some(&id) = self.token_to_id.get(candidate.as_bytes()) {
                        tokens.push(id);
                        remaining = &after[candidate.len()..];
                        continue;
                    }
                }
                // Not a real special token — encode "<|" as text and continue
                tokens.extend(self.encode_segment(remaining[..pos + 2].as_bytes()));
                remaining = &remaining[pos + 2..];
            } else {
                // No more special tokens — encode the rest
                tokens.extend(self.encode_segment(remaining.as_bytes()));
                break;
            }
        }

        tokens
    }

    /// Encode a text segment (no special tokens) using SIMD pre-tokenizer.
    /// Uses pretokenize kernel for span detection, then direct vocab lookup
    /// per span (matching llama.cpp ignore_merges behavior for tiktoken).
    /// Falls back to BPE merge for spans not in vocab.
    fn encode_segment(&self, bytes: &[u8]) -> Vec<u32> {
        use crate::kernels::ffi;

        let len = bytes.len();
        if len == 0 {
            return Vec::new();
        }

        let mut flags = vec![0u8; len];
        let mut boundaries = vec![0u8; len];

        unsafe {
            ffi::pretokenize(
                bytes.as_ptr(),
                flags.as_mut_ptr(),
                boundaries.as_mut_ptr(),
                len as i32,
            );
        }

        // Collect spans from boundary array
        let mut tokens: Vec<u32> = Vec::with_capacity(len / 3);
        let mut span_start = 0;
        for i in 1..=len {
            if i == len || boundaries[i] == 1 {
                let span = &bytes[span_start..i];
                // Direct vocab lookup (ignore_merges, like llama.cpp for Llama 3)
                if let Some(&id) = self.token_to_id.get(span) {
                    tokens.push(id);
                } else {
                    tokens.extend(self.bpe_encode_span(span));
                }
                span_start = i;
            }
        }

        tokens
    }

    /// BPE merge fallback for spans not found in vocab as a whole token.
    fn bpe_encode_span(&self, span: &[u8]) -> Vec<u32> {
        // Start with one token per byte
        let mut tokens: Vec<u32> = Vec::with_capacity(span.len());
        for &b in span {
            tokens.push(self.token_to_id.get(&vec![b]).copied().unwrap_or(0));
        }

        // BPE merge loop
        loop {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_idx = usize::MAX;

            for i in 0..tokens.len().saturating_sub(1) {
                let mut merged = self.vocab[tokens[i] as usize].clone();
                merged.extend_from_slice(&self.vocab[tokens[i + 1] as usize]);
                if let Some(&merge_id) = self.token_to_id.get(&merged) {
                    let score = self.scores[merge_id as usize];
                    if score > best_score {
                        best_score = score;
                        best_idx = i;
                    }
                }
            }

            if best_idx == usize::MAX {
                break;
            }

            let mut merged = self.vocab[tokens[best_idx] as usize].clone();
            merged.extend_from_slice(&self.vocab[tokens[best_idx + 1] as usize]);
            let merge_id = self.token_to_id[&merged];
            tokens[best_idx] = merge_id;
            tokens.remove(best_idx + 1);
        }

        tokens
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if (id as usize) < self.vocab.len() {
                bytes.extend_from_slice(&self.vocab[id as usize]);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

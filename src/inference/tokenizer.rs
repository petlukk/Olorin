//! Byte-level BPE tokenizer parsed from GGUF vocab metadata.
//!
//! Handles GPT-2 byte encoding where "bad" bytes (0-32, 127-160, 173) are
//! shifted to unicode 256+. Token strings in the GGUF use this encoding;
//! we reverse it at load time so the vocab stores raw bytes.

use std::collections::HashMap;
use crate::error::{Error, Result};
use crate::inference::gguf::{GgufFile, MetaValue};

pub struct Tokenizer {
    pub(crate) vocab: Vec<Vec<u8>>,
    pub(crate) token_to_id: HashMap<Vec<u8>, u32>,
    scores: Vec<f32>,
    /// Per-token type from tokenizer.ggml.token_type. Values:
    ///   1 = NORMAL, 2 = UNKNOWN, 3 = CONTROL, 4 = USER_DEFINED,
    ///   5 = UNUSED, 6 = BYTE.
    /// CONTROL and USER_DEFINED tokens are hidden from user-facing decoded
    /// output (matching llama.cpp behavior).
    token_types: Vec<i32>,
    pub bos_id: u32,
    pub eos_id: u32,
    pub stop_ids: Vec<u32>,
    /// SentencePiece (Unigram) tokenizers do not pre-tokenize on word
    /// boundaries — encoding runs Viterbi over the whole input segment.
    is_sentencepiece: bool,
    /// Gemma 4 uses SPM-style BPE with merges list priority — a different
    /// algorithm from SPM Unigram. When true, encode_segment routes to the
    /// Gemma 4 BPE path in tokenizer_gemma4.
    is_gemma4_bpe: bool,
    /// (left_bytes, right_bytes) -> merge rank for Gemma 4 BPE. Lower rank
    /// wins. Empty for non-BPE tokenizers.
    pub(crate) merges_rank: HashMap<(Vec<u8>, Vec<u8>), u32>,
    /// Byte-fallback table: byte value -> token id, for sentencepiece
    /// fallback when a character isn't in the vocab.
    pub(crate) byte_fallback: [u32; 256],
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

        // Detect tokenizer model: gemma4 uses SentencePiece (raw vocab bytes,
        // ▁ for space prefix). Llama 3 / tiktoken use GPT-2 byte-level encoding.
        let tok_model = gguf.get_str("tokenizer.ggml.model").unwrap_or("").to_string();
        // Gemma 4's GGUF stores both scores (unigram log-probs, used for special
        // token priorities) and merges (the actual BPE merge table). llama.cpp
        // treats vocab type as BPE and uses the merges list. We must do the same:
        // treating Gemma 4 as SentencePiece Unigram produces different token
        // sequences on rare words (see tests/gemma4_tokenizer_match.rs).
        let is_gemma4_bpe = tok_model == "gemma4";
        // Keep the Unigram-Viterbi path for plain SentencePiece ("llama" legacy).
        let is_sentencepiece = tok_model == "llama";

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

        // Per-token type array (NORMAL/UNKNOWN/CONTROL/USER_DEFINED/...).
        let token_types: Vec<i32> = match gguf.metadata.get("tokenizer.ggml.token_type") {
            Some(MetaValue::Array(arr)) => arr.iter().map(|v| match v {
                MetaValue::I32(i) => *i,
                _ => 1,
            }).collect(),
            _ => Vec::new(),
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
            // For sentencepiece, byte-fallback tokens (<0xNN>) are stored with
            // their literal string as the key — they're only used when input
            // contains bytes that aren't in vocab. The raw byte tokens (e.g.
            // token 107 = '\n') must own the [0x0A] hashmap key, not be
            // overwritten by token 248 = '<0x0A>'.
            let bytes = if !is_sentencepiece && !is_gemma4_bpe && parse_byte_token(tok_str).is_some() {
                vec![parse_byte_token(tok_str).unwrap()]
            } else if is_sentencepiece || is_gemma4_bpe {
                // SentencePiece: vocab strings are raw UTF-8. ▁ (U+2581) marks
                // space prefix, but we keep it as-is in the vocab map and let
                // the encoder translate spaces to ▁ before lookup.
                tok_str.as_bytes().to_vec()
            } else {
                gpt2_decode_str(tok_str)
            };
            token_to_id.insert(bytes.clone(), i as u32);
            vocab.push(bytes);
        }

        let bos_id = gguf.get_u32("tokenizer.ggml.bos_token_id").unwrap_or(1);
        let eos_id = gguf.get_u32("tokenizer.ggml.eos_token_id").unwrap_or(2);

        // Collect all stop token IDs: eos + chat-specific end-of-turn markers
        let mut stop_ids = vec![eos_id];
        for special in ["<|eot_id|>", "<|im_end|>", "<turn|>", "<end_of_turn>"] {
            if let Some(id) = token_to_id.get(special.as_bytes()) {
                if *id != eos_id && !stop_ids.contains(id) {
                    stop_ids.push(*id);
                }
            }
        }

        // Build byte-fallback table for sentencepiece. Token strings of the
        // form "<0xNN>" are CONTROL tokens used when a byte isn't covered by
        // any vocab entry. For sentencepiece, lookup by the literal "<0xNN>"
        // string in the vocab map (we stored them that way above).
        let mut byte_fallback = [u32::MAX; 256];
        if is_sentencepiece || is_gemma4_bpe {
            for b in 0u8..=255 {
                let key = format!("<0x{:02X}>", b);
                if let Some(&id) = token_to_id.get(key.as_bytes()) {
                    byte_fallback[b as usize] = id;
                }
            }
        }

        // Parse tokenizer.ggml.merges for Gemma 4 BPE. Each entry is
        // "<left_piece> <right_piece>" with ASCII 0x20 as the separator.
        // Pieces never contain ASCII space (SPM replaces it with ▁ beforehand),
        // so split on first space is unambiguous. Rank = index in the list.
        let mut merges_rank: HashMap<(Vec<u8>, Vec<u8>), u32> = HashMap::new();
        if is_gemma4_bpe {
            if let Some(MetaValue::Array(arr)) = gguf.metadata.get("tokenizer.ggml.merges") {
                merges_rank.reserve(arr.len());
                for (rank, v) in arr.iter().enumerate() {
                    let s = match v {
                        MetaValue::Str(s) => s.as_bytes(),
                        _ => continue,
                    };
                    if let Some(sep) = s.iter().position(|&b| b == b' ') {
                        let left = s[..sep].to_vec();
                        let right = s[sep + 1..].to_vec();
                        merges_rank.insert((left, right), rank as u32);
                    }
                }
            }
        }

        // Pad token_types to match vocab length if missing/short.
        let mut token_types = token_types;
        token_types.resize(vocab.len(), 1);

        Ok(Tokenizer {
            vocab, token_to_id, scores, token_types, bos_id, eos_id, stop_ids,
            is_sentencepiece, is_gemma4_bpe, merges_rank, byte_fallback,
        })
    }

    /// Returns true if a token is CONTROL (3) or USER_DEFINED (4) — these
    /// should not be rendered to user-facing output. Matches llama.cpp's
    /// `special` flag behavior in detokenization.
    pub fn is_control_or_user_defined(&self, id: u32) -> bool {
        let t = self.token_types.get(id as usize).copied().unwrap_or(1);
        t == 3 || t == 4
    }

    /// Look up a token string in the vocabulary. Returns None if not found.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token.as_bytes()).copied()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Special tokens to match: any <...> or <|...|> pattern in vocab.
        // Scan for '<' and try known special tokens at that position.
        let mut tokens: Vec<u32> = Vec::with_capacity(text.len());
        let mut remaining = text;
        while !remaining.is_empty() {
            if let Some(pos) = remaining.find('<') {
                // Encode text before '<'
                if pos > 0 {
                    tokens.extend(self.encode_segment(remaining[..pos].as_bytes()));
                }
                let after = &remaining[pos..];
                // Try to find a closing '>' and match as special token
                let mut matched = false;
                if let Some(end) = after.find('>') {
                    let candidate = &after[..end + 1];
                    if let Some(&id) = self.token_to_id.get(candidate.as_bytes()) {
                        tokens.push(id);
                        remaining = &after[candidate.len()..];
                        matched = true;
                    }
                }
                if !matched {
                    // Not a special token — encode '<' as text
                    tokens.extend(self.encode_segment(remaining[pos..pos + 1].as_bytes()));
                    remaining = &remaining[pos + 1..];
                }
            } else {
                tokens.extend(self.encode_segment(remaining.as_bytes()));
                break;
            }
        }

        tokens
    }

    /// Encode a text segment (no special tokens). Routes to either the
    /// SentencePiece Unigram Viterbi encoder (gemma4, llama) or the SIMD
    /// pretokenize + BPE-merge path (Llama 3 / tiktoken).
    fn encode_segment(&self, bytes: &[u8]) -> Vec<u32> {
        if bytes.is_empty() {
            return Vec::new();
        }
        if self.is_gemma4_bpe {
            return super::tokenizer_gemma4::encode_gemma4_bpe(self, bytes);
        }
        if self.is_sentencepiece {
            return self.encode_sentencepiece(bytes);
        }
        self.encode_bpe_pretokenized(bytes)
    }

    /// SentencePiece Unigram Viterbi encoder.
    /// Replaces ASCII space with U+2581 (▁) then runs DP segmentation that
    /// maximizes total log-score. Falls back to <0xNN> byte tokens for any
    /// position where no vocab entry starts.
    fn encode_sentencepiece(&self, raw: &[u8]) -> Vec<u32> {
        // 1. Replace ' ' with ▁ (\xe2\x96\x81). add_space_prefix=false for
        //    gemma4, so no leading ▁ is prepended — llama.cpp matches this.
        let mut bytes: Vec<u8> = Vec::with_capacity(raw.len() + raw.len() / 4);
        for &b in raw {
            if b == b' ' {
                bytes.extend_from_slice(&[0xE2, 0x96, 0x81]);
            } else {
                bytes.push(b);
            }
        }

        let n = bytes.len();
        // best_score[i] = best total log-score for bytes[0..i]
        // best_id[i]    = token id consumed to reach position i
        // best_prev[i]  = previous position (start of that token)
        let neg_inf = f32::NEG_INFINITY;
        let mut best_score = vec![neg_inf; n + 1];
        let mut best_id = vec![u32::MAX; n + 1];
        let mut best_prev = vec![0usize; n + 1];
        best_score[0] = 0.0;

        // Cap maximum token byte length to bound the inner loop. Gemma vocab
        // has tokens up to ~32 bytes; 64 is safe.
        const MAX_TOK_LEN: usize = 64;

        for i in 0..n {
            if best_score[i] == neg_inf {
                continue;
            }
            let limit = (i + MAX_TOK_LEN).min(n);
            // Try each substring bytes[i..j]
            for j in (i + 1)..=limit {
                if let Some(&id) = self.token_to_id.get(&bytes[i..j]) {
                    let s = best_score[i] + self.scores[id as usize];
                    if s > best_score[j] {
                        best_score[j] = s;
                        best_id[j] = id;
                        best_prev[j] = i;
                    }
                }
            }
            // Single-byte fallback (always available so DP always reaches n)
            let j = i + 1;
            let b = bytes[i];
            let fb = self.byte_fallback[b as usize];
            if fb != u32::MAX {
                // Byte-fallback tokens have score -1000 in gemma; this makes
                // them strictly worse than any normal merge.
                let s = best_score[i] + self.scores[fb as usize];
                if s > best_score[j] {
                    best_score[j] = s;
                    best_id[j] = fb;
                    best_prev[j] = i;
                }
            }
        }

        // 2. Backtrack from n to 0
        let mut out: Vec<u32> = Vec::new();
        let mut p = n;
        while p > 0 {
            let id = best_id[p];
            if id == u32::MAX {
                // Should never happen — byte fallback guarantees coverage.
                break;
            }
            out.push(id);
            p = best_prev[p];
        }
        out.reverse();
        out
    }

    fn encode_bpe_pretokenized(&self, bytes: &[u8]) -> Vec<u32> {
        use crate::kernels::ffi;
        let len = bytes.len();

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
                let tok = &self.vocab[id as usize];
                // Replace sentencepiece space marker U+2581 (▁ = 0xE2 0x96 0x81) with space
                let mut i = 0;
                while i < tok.len() {
                    if i + 2 < tok.len() && tok[i] == 0xE2 && tok[i + 1] == 0x96 && tok[i + 2] == 0x81 {
                        bytes.push(b' ');
                        i += 3;
                    } else {
                        bytes.push(tok[i]);
                        i += 1;
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

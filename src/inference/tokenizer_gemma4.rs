//! Gemma 4 SPM-style BPE tokenization.
//!
//! Mirrors llama.cpp's `LLAMA_VOCAB_PRE_TYPE_GEMMA4` path
//! (`src/llama-vocab.cpp:496-505, 562-672`).
//!
//! Algorithm:
//! 1. Split input on newlines. Newline-only chunks look up directly in vocab
//!    (the merges list contains specific N-newline tokens).
//! 2. For non-newline chunks: replace ASCII 0x20 with ▁ (U+2581 = E2 96 81),
//!    then run priority-queue BPE using `tokenizer.ggml.merges` for rank.
//! 3. Emission: each final symbol becomes a token via direct vocab lookup,
//!    with byte fallback for unrecognized pieces.
//!
//! The initial symbol granularity is UTF-8 codepoints, not raw bytes — merges
//! reference character pieces (including multi-byte pieces like ▁), so starting
//! with bytes would leave most pairs unmerged.

use super::tokenizer::Tokenizer;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub(crate) fn encode_gemma4_bpe(tok: &Tokenizer, input: &[u8]) -> Vec<u32> {
    let mut out = Vec::with_capacity(input.len() / 3 + 1);
    let n = input.len();
    let mut start = 0;
    while start < n {
        let is_nl = input[start] == b'\n';
        let mut end = start;
        while end < n && (input[end] == b'\n') == is_nl {
            end += 1;
        }
        let chunk = &input[start..end];
        if is_nl {
            emit_newline_chunk(tok, chunk, &mut out);
        } else {
            bpe_merge_chunk(tok, chunk, &mut out);
        }
        start = end;
    }
    out
}

fn emit_newline_chunk(tok: &Tokenizer, chunk: &[u8], out: &mut Vec<u32>) {
    // The merges list contains specific multi-newline tokens; try whole-chunk
    // vocab lookup before falling back byte-by-byte.
    if let Some(&id) = tok.token_to_id.get(chunk) {
        out.push(id);
        return;
    }
    for &b in chunk {
        emit_single_byte(tok, b, out);
    }
}

fn emit_single_byte(tok: &Tokenizer, b: u8, out: &mut Vec<u32>) {
    if let Some(&id) = tok.token_to_id.get(&[b][..]) {
        out.push(id);
    } else {
        let fb = tok.byte_fallback[b as usize];
        if fb != u32::MAX {
            out.push(fb);
        }
    }
}

#[derive(Clone)]
struct Sym {
    start: usize,
    len: usize,
    prev: i32,
    next: i32,
}

#[derive(Eq, PartialEq)]
struct Bigram {
    rank: u32,
    left: i32,
    right: i32,
    text: Vec<u8>,
}

impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Lower rank wins; tiebreak on lower left index (stable left-to-right).
        self.rank.cmp(&other.rank).then(self.left.cmp(&other.left))
    }
}

impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn bpe_merge_chunk(tok: &Tokenizer, raw: &[u8], out: &mut Vec<u32>) {
    // Normalize: ASCII space -> ▁ (U+2581 = E2 96 81).
    let mut bytes = Vec::with_capacity(raw.len() + raw.len() / 4);
    for &b in raw {
        if b == b' ' {
            bytes.extend_from_slice(&[0xE2, 0x96, 0x81]);
        } else {
            bytes.push(b);
        }
    }
    if bytes.is_empty() {
        return;
    }

    // Build symbol list: one per UTF-8 codepoint.
    let mut symbols: Vec<Sym> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let cl = utf8_char_len(bytes[i]).min(bytes.len() - i);
        let idx = symbols.len() as i32;
        symbols.push(Sym {
            start: i,
            len: cl,
            prev: if idx == 0 { -1 } else { idx - 1 },
            next: -1,
        });
        i += cl;
    }
    let n_sym = symbols.len();
    for j in 0..n_sym.saturating_sub(1) {
        symbols[j].next = (j + 1) as i32;
    }

    // Seed the priority queue with all initial adjacent bigrams.
    let mut queue: BinaryHeap<Reverse<Bigram>> = BinaryHeap::new();
    for j in 0..n_sym.saturating_sub(1) {
        try_push_bigram(&mut queue, j as i32, (j + 1) as i32, &symbols, &bytes, tok);
    }

    // Drain queue: merge lowest-rank bigram; push new boundary bigrams.
    while let Some(Reverse(bg)) = queue.pop() {
        let li = bg.left as usize;
        let ri = bg.right as usize;
        if symbols[li].len == 0 || symbols[ri].len == 0 {
            continue;
        }
        let current_concat_matches = {
            let l = &bytes[symbols[li].start..symbols[li].start + symbols[li].len];
            let r = &bytes[symbols[ri].start..symbols[ri].start + symbols[ri].len];
            bg.text.len() == l.len() + r.len()
                && &bg.text[..l.len()] == l
                && &bg.text[l.len()..] == r
        };
        if !current_concat_matches {
            continue; // Stale bigram — one side already merged.
        }

        // Merge right into left.
        let new_len = symbols[li].len + symbols[ri].len;
        let new_next = symbols[ri].next;
        symbols[li].len = new_len;
        symbols[li].next = new_next;
        symbols[ri].len = 0;
        if new_next >= 0 {
            symbols[new_next as usize].prev = bg.left;
        }

        // Add new bigrams at the updated boundaries.
        let prev_idx = symbols[li].prev;
        if prev_idx >= 0 {
            try_push_bigram(&mut queue, prev_idx, bg.left, &symbols, &bytes, tok);
        }
        if new_next >= 0 {
            try_push_bigram(&mut queue, bg.left, new_next, &symbols, &bytes, tok);
        }
    }

    // Emit final tokens, walking the linked list from the first non-empty symbol.
    let mut idx = 0i32;
    while idx >= 0 && (idx as usize) < symbols.len() {
        let sym = symbols[idx as usize].clone();
        if sym.len > 0 {
            let text = &bytes[sym.start..sym.start + sym.len];
            if let Some(&id) = tok.token_to_id.get(text) {
                out.push(id);
            } else {
                for &b in text {
                    emit_single_byte(tok, b, out);
                }
            }
        }
        idx = sym.next;
    }
}

fn try_push_bigram(
    queue: &mut BinaryHeap<Reverse<Bigram>>,
    left: i32,
    right: i32,
    symbols: &[Sym],
    bytes: &[u8],
    tok: &Tokenizer,
) {
    let l = &symbols[left as usize];
    let r = &symbols[right as usize];
    let lb = &bytes[l.start..l.start + l.len];
    let rb = &bytes[r.start..r.start + r.len];
    // HashMap key requires owned Vec<u8>. Avoid double allocation by
    // constructing the key pair in place.
    let key = (lb.to_vec(), rb.to_vec());
    if let Some(&rank) = tok.merges_rank.get(&key) {
        let mut text = Vec::with_capacity(lb.len() + rb.len());
        text.extend_from_slice(lb);
        text.extend_from_slice(rb);
        queue.push(Reverse(Bigram { rank, left, right, text }));
    }
}

fn utf8_char_len(first: u8) -> usize {
    match first {
        0x00..=0x7F => 1,
        0xC2..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF4 => 4,
        _ => 1, // invalid or continuation byte — treat as single-byte symbol
    }
}

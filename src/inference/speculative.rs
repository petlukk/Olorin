//! Speculative decoding: draft K tokens from a small model, verify with target.

use crate::inference::engine::BitNetModel;
use crate::inference::forward_llama::{LlamaState, argmax};

/// Speculative decode loop: draft K tokens greedily, verify against target.
///
/// Returns `(all_output_tokens, prefill_ms, decode_ms)`.
pub fn speculative_generate(
    target_model: &BitNetModel,
    draft_model: &BitNetModel,
    prompt_tokens: &[u32],
    max_tokens: usize,
    draft_k: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    repetition_penalty: f32,
    stop_ids: &[u32],
    max_seq_len: usize,
    mut on_token: impl FnMut(u32),
) -> (Vec<u32>, f64, f64) {
    use std::time::Instant;
    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);

    let mut target = LlamaState::new(target_model, max_seq_len);
    let mut draft = LlamaState::new(draft_model, max_seq_len);
    let mut output = Vec::with_capacity(prompt_tokens.len() + max_tokens);

    // --- Prefill both models ---
    let prefill_start = Instant::now();
    target.prefill(target_model, prompt_tokens);
    draft.prefill(draft_model, prompt_tokens);
    output.extend_from_slice(prompt_tokens);
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    // --- Sample first token from target ---
    let first_tok_start = Instant::now();
    target.apply_repetition_penalty(&output, repetition_penalty);
    let first_tok = target.sample_logits(temperature, top_k, top_p, min_p);
    let first_tok_ms = first_tok_start.elapsed().as_secs_f64() * 1000.0;

    if stop_ids.contains(&first_tok) {
        return finish(output, prefill_ms, 0.0, 0, 0, 0, first_tok_ms, n_threads,
                      prompt_tokens.len());
    }

    output.push(first_tok);
    on_token(first_tok);

    let mut pos = prompt_tokens.len();
    target.forward(target_model, first_tok, pos);
    draft.forward(draft_model, first_tok, pos);
    pos += 1;

    let mut n_gen: u32 = 1;
    let mut n_draft_total: u32 = 0;
    let mut n_accepted_total: u32 = 0;
    let mut stopped = false;

    let decode_start = Instant::now();

    while (n_gen as usize) < max_tokens && pos < max_seq_len && !stopped {
        let cp_draft = draft.kv_cache.checkpoint();
        let cp_target = target.kv_cache.checkpoint();

        // --- Draft K tokens greedily ---
        let k = draft_k.min(max_seq_len - pos).min(max_tokens - n_gen as usize);
        let mut draft_tokens = Vec::with_capacity(k);
        for i in 0..k {
            let tok = argmax(draft.logits());
            draft_tokens.push(tok);
            draft.forward(draft_model, tok, pos + i);
        }

        // --- Verify with target ---
        // target_choice_0 = argmax of target's current logits (prediction for pos)
        let target_choice_0 = argmax(target.logits());

        // prefill_verify gives target's per-position predictions:
        // verified[i] = target's prediction for pos+i+1 given draft_tokens[0..=i]
        let verified = target.prefill_verify(target_model, &draft_tokens);

        // Build target_choices: target's greedy choice for each draft position
        // target_choices[0] = target_choice_0 (from existing logits, for pos+0)
        // target_choices[i] = verified[i-1] (target's prediction given tokens through pos+i-1)
        let mut target_choices = Vec::with_capacity(k + 1);
        target_choices.push(target_choice_0);
        for i in 0..k {
            target_choices.push(verified[i]);
        }
        // target_choices[i] = target's choice at position pos+i, for i in 0..k
        // target_choices[k] = verified[k-1] = target's next token after all k accepted

        // --- Count accepted ---
        let mut n_accepted = 0usize;
        for i in 0..k {
            if target_choices[i] == draft_tokens[i] {
                n_accepted += 1;
            } else {
                break;
            }
        }

        n_draft_total += k as u32;
        n_accepted_total += n_accepted as u32;

        // --- Emit accepted tokens ---
        let mut accepted_and_bonus = Vec::with_capacity(n_accepted + 1);
        for i in 0..n_accepted {
            if stop_ids.contains(&draft_tokens[i]) {
                stopped = true;
                break;
            }
            output.push(draft_tokens[i]);
            on_token(draft_tokens[i]);
            accepted_and_bonus.push(draft_tokens[i]);
            n_gen += 1;
        }

        if stopped {
            break;
        }

        // --- Bonus token: target's choice at rejection point ---
        let bonus = target_choices[n_accepted];
        if stop_ids.contains(&bonus) {
            break;
        }
        output.push(bonus);
        on_token(bonus);
        accepted_and_bonus.push(bonus);
        n_gen += 1;

        // --- KV sync: restore both caches, re-forward accepted + bonus ---
        draft.kv_cache.restore(cp_draft).expect("draft cache restore");
        target.kv_cache.restore(cp_target).expect("target cache restore");

        for &tok in &accepted_and_bonus {
            target.forward(target_model, tok, pos);
            draft.forward(draft_model, tok, pos);
            pos += 1;
        }
    }

    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    finish(output, prefill_ms, decode_ms, n_gen, n_draft_total,
           n_accepted_total, first_tok_ms, n_threads, prompt_tokens.len())
}

fn finish(
    output: Vec<u32>,
    prefill_ms: f64,
    decode_ms: f64,
    n_gen: u32,
    n_draft_total: u32,
    n_accepted_total: u32,
    first_tok_ms: f64,
    n_threads: usize,
    n_prompt: usize,
) -> (Vec<u32>, f64, f64) {
    let ptps = n_prompt as f64 / (prefill_ms / 1000.0);
    let dtps = if n_gen > 0 { n_gen as f64 / (decode_ms / 1000.0) } else { 0.0 };
    let avg = if n_gen > 0 { decode_ms / n_gen as f64 } else { 0.0 };
    let accept_rate = if n_draft_total > 0 {
        n_accepted_total as f64 / n_draft_total as f64
    } else {
        0.0
    };
    eprintln!("\n--- speculative perf ({n_threads} threads) ---");
    eprintln!("prefill:    {n_prompt} tokens in {prefill_ms:.0}ms ({ptps:.1} tok/s)");
    eprintln!("first tok:  {first_tok_ms:.0}ms");
    eprintln!("decode:     {n_gen} tokens in {decode_ms:.0}ms ({dtps:.1} tok/s, {avg:.1}ms/tok)");
    eprintln!("accept:     {n_accepted_total}/{n_draft_total} ({:.1}%)", accept_rate * 100.0);
    (output, prefill_ms, decode_ms)
}

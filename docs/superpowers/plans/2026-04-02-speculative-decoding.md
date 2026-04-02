# Speculative Decoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Increase Llama 3B Q4K decode from 4.3 to ~14 tok/s via draft+verify speculative decoding.

**Architecture:** A small draft model (Qwen 0.8B) generates K=5 candidate tokens greedily. The target model (Llama 3B) verifies them in one batched prefill pass. Accepted tokens are emitted; on rejection, the target model's token is emitted. KV caches are synced via checkpoint/restore.

**Tech Stack:** Rust, existing `BitNetModel`/`LlamaState`/`Engine` infrastructure, existing `prefill()` and `forward()` paths.

---

## File Structure

### New files
- `src/inference/speculative.rs` — Draft model wrapper + speculative generate loop (~200 lines)

### Modified files
- `src/inference/prefill_llama.rs` — Add `prefill_verify()` that returns per-token logits
- `src/inference/forward_llama.rs` — Add `argmax()` helper (trivial)
- `src/inference/generate.rs` — Wire `--draft` into Engine, call speculative path
- `src/inference/mod.rs` — Add `pub mod speculative;`

---

### Task 1: prefill_verify — prefill that returns per-token logits

**Files:**
- Modify: `src/inference/prefill_llama.rs`
- Modify: `src/inference/forward_llama.rs`

The existing `prefill()` processes all N tokens but only computes output logits for the last token. Verify needs logits at every position to compare with draft tokens.

Strategy: after all layers complete, run `output_proj` for each token's hidden state. This requires saving all tokens' final hidden states (currently only the last is saved to `self.x`).

- [ ] **Step 1: Add `prefill_verify` method to `LlamaState`**

In `src/inference/prefill_llama.rs`, add a new method after `prefill()`:

```rust
/// Prefill that returns per-token logits for speculative verify.
/// Same as prefill() but computes output_proj for every token.
pub fn prefill_verify(&mut self, model: &BitNetModel, tokens: &[u32]) -> Vec<Vec<f32>> {
    // Run normal prefill (processes all layers)
    // But we need each token's final hidden state.
    // Strategy: run prefill normally, then for verify we actually
    // need to do N separate forward passes to get N sets of logits.
    //
    // Simpler approach: just run forward() for each token sequentially
    // and collect logits. This is what llama.cpp does for small N (K=5).
    let mut all_logits = Vec::with_capacity(tokens.len());
    for (i, &tok) in tokens.iter().enumerate() {
        let pos = self.kv_cache.seq_len() as usize;
        self.forward(model, tok, pos);
        all_logits.push(self.logits.clone());
    }
    all_logits
}
```

Note: This does NOT use the batched prefill path — it runs K separate forward passes. With K=5, this takes 5 × 231ms = 1155ms, not the 240ms we hoped. This is a correctness-first implementation; optimization (batched verify with per-token output_proj) comes in Task 5.

Wait — that defeats the purpose. Let me reconsider.

**Better approach:** Run batched `prefill()` for all K tokens (one weight load, ~240ms). Then we need logits at each position. The trick: after prefill, the KV cache has all K positions. We can run output_proj for each token's hidden state if we saved them.

Modify `prefill()` to optionally save all hidden states:

```rust
/// Prefill that returns per-token argmax token IDs for verify.
/// Runs batched prefill (one weight load), then output_proj per position.
pub fn prefill_verify(&mut self, model: &BitNetModel, tokens: &[u32]) -> Vec<u32> {
    let n = tokens.len();
    let h = model.hidden_dim;

    // Save all tokens' final hidden states during prefill
    let mut hidden_states: Vec<Vec<f32>> = Vec::with_capacity(n);

    // We need to modify the inner loop to save xs[t] after all layers.
    // But xs is consumed by prefill. Solution: clone xs before output_proj.
    // Actually, the simplest correct approach: just re-use the existing
    // prefill code but capture xs at the end.

    // Run the same prefill logic but save each token's final x
    // ... (this requires refactoring prefill internals)

    // SIMPLEST CORRECT APPROACH for v1:
    // Run prefill() to populate KV cache with all K tokens in one batch.
    // Then run output_proj for each token by doing forward() from cache.
    // But forward() recomputes all layers — we can't skip layers.

    // ACTUALLY SIMPLEST: run K separate forward() calls.
    // K=5 × 231ms = 1155ms. Too slow.

    // PRAGMATIC APPROACH: modify prefill to save xs[] before dropping.
    // Then run output_proj for each saved x.
    todo!()
}
```

Actually, let me look at this more carefully. The prefill code saves `xs[n-1]` at the end. We just need to save ALL of them and run output_proj for each.

- [ ] **Step 1: Modify prefill to optionally return all hidden states**

In `src/inference/prefill_llama.rs`, at the end of `prefill()` (line ~304), the code does:
```rust
self.x[..h].copy_from_slice(&xs[n - 1]);
self.output_proj(model);
```

Add a new method `prefill_verify()` that does the same work but returns argmax per token:

```rust
/// Batched prefill + output_proj per token. Returns argmax token ID per position.
pub fn prefill_verify(&mut self, model: &BitNetModel, tokens: &[u32]) -> Vec<u32> {
    let n = tokens.len();
    let h = model.hidden_dim;

    // Run the full prefill (reuse all the existing code by extracting the
    // layer loop into a helper, or just inline the same logic).
    // For now: we know that prefill() stores each token's final hidden
    // state in xs[t]. We copy-paste the prefill body but save xs.

    // ... (same prefill body as prefill()) ...

    // After all layers: output_proj per token
    let mut result = Vec::with_capacity(n);
    for t in 0..n {
        self.x[..h].copy_from_slice(&xs[t]);
        self.output_proj(model);
        result.push(argmax(&self.logits));
    }
    result
}
```

This duplicates the prefill body which violates DRY. Better: extract the layer loop into a shared helper that returns `xs`, then `prefill()` and `prefill_verify()` both call it.

- [ ] **Step 1 (final approach): Extract prefill layer loop, add verify variant**

In `src/inference/prefill_llama.rs`:

1. Rename current `prefill()` body into `fn prefill_layers()` that returns `xs: Vec<Vec<f32>>`.
2. `prefill()` calls `prefill_layers()`, saves `xs[n-1]`, runs `output_proj()` once.
3. `prefill_verify()` calls `prefill_layers()`, runs `output_proj()` per token, returns `Vec<u32>` (argmax per position).

```rust
impl LlamaState {
    /// Internal: run all layers for N tokens, return final hidden states.
    fn prefill_layers(&mut self, model: &BitNetModel, tokens: &[u32]) -> Vec<Vec<f32>> {
        // ... existing prefill body up to but NOT including output_proj ...
        // Returns xs (Vec<Vec<f32>>, one per token)
        xs
    }

    pub fn prefill(&mut self, model: &BitNetModel, tokens: &[u32]) {
        let xs = self.prefill_layers(model, tokens);
        let h = model.hidden_dim;
        self.x[..h].copy_from_slice(&xs[xs.len() - 1]);
        self.output_proj(model);
    }

    pub fn prefill_verify(&mut self, model: &BitNetModel, tokens: &[u32]) -> Vec<u32> {
        let xs = self.prefill_layers(model, tokens);
        let h = model.hidden_dim;
        let mut result = Vec::with_capacity(xs.len());
        for x in &xs {
            self.x[..h].copy_from_slice(x);
            self.output_proj(model);
            result.push(argmax(&self.logits));
        }
        result
    }
}
```

Add `argmax` helper in `forward_llama.rs`:

```rust
pub fn argmax(logits: &[f32]) -> u32 {
    logits.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}
```

- [ ] **Step 2: Verify it compiles**

Run: `PATH="..." cargo build 2>&1 | grep error`
Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add src/inference/prefill_llama.rs src/inference/forward_llama.rs
git commit -m "feat: prefill_verify — batched prefill returning per-token argmax for speculative decode"
```

---

### Task 2: speculative.rs — draft model + spec decode loop

**Files:**
- Create: `src/inference/speculative.rs`
- Modify: `src/inference/mod.rs`

- [ ] **Step 1: Create speculative.rs**

```rust
//! Speculative decoding: draft K tokens with small model, verify with target.

use crate::inference::engine::BitNetModel;
use crate::inference::forward_llama::{LlamaState, argmax};
use crate::inference::gguf::GgufFile;

/// Holds a draft model that can be hot-swapped.
pub struct DraftModel {
    _gguf: GgufFile,
    pub model: BitNetModel,
}

impl DraftModel {
    pub fn load(path: &std::path::Path) -> crate::error::Result<Self> {
        let gguf = GgufFile::open(path)?;
        let model = BitNetModel::from_gguf(&gguf)?;
        Ok(DraftModel { _gguf: gguf, model })
    }
}

/// Run speculative decode: draft K tokens, verify with target, emit accepted.
/// Returns (generated_tokens, prefill_ms, decode_ms).
pub fn speculative_generate(
    target_model: &BitNetModel,
    draft: &DraftModel,
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
    use crate::inference::forward_llama::sample_into;

    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let mut target = LlamaState::new(target_model, max_seq_len);
    let mut draft_state = LlamaState::new(&draft.model, max_seq_len);
    let mut output = Vec::with_capacity(prompt_tokens.len() + max_tokens);

    // Prefill both models with prompt
    let prefill_start = Instant::now();
    target.prefill(target_model, prompt_tokens);
    draft_state.prefill(&draft.model, prompt_tokens);
    output.extend_from_slice(prompt_tokens);
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    let mut pos = prompt_tokens.len();
    let mut n_gen = 0u32;
    let mut n_draft_total = 0u32;
    let mut n_accepted_total = 0u32;

    // Sample first token from target (after prefill, logits are ready)
    target.apply_repetition_penalty(&output, repetition_penalty);
    let first = target.sample_logits(temperature, top_k, top_p, min_p);
    if stop_ids.contains(&first) {
        let decode_ms = 0.0;
        eprintln!("\n--- perf ({n_threads} threads) ---");
        eprintln!("prefill:    {} tokens in {prefill_ms:.0}ms", prompt_tokens.len());
        return (output, prefill_ms, decode_ms);
    }
    output.push(first);
    on_token(first);
    // Forward first token through both models
    target.forward(target_model, first, pos);
    draft_state.forward(&draft.model, first, pos);
    pos += 1;
    n_gen += 1;

    let decode_start = Instant::now();

    while n_gen < max_tokens as u32 && pos < max_seq_len {
        // --- Draft phase: generate K tokens greedily ---
        let draft_checkpoint = draft_state.kv_cache.checkpoint();
        let target_checkpoint = target.kv_cache.checkpoint();
        let mut draft_tokens = Vec::with_capacity(draft_k);
        let mut draft_pos = pos;

        for _ in 0..draft_k {
            if draft_pos >= max_seq_len { break; }
            let tok = argmax(draft_state.logits());
            draft_tokens.push(tok);
            draft_state.forward(&draft.model, tok, draft_pos);
            draft_pos += 1;
        }

        if draft_tokens.is_empty() { break; }
        n_draft_total += draft_tokens.len() as u32;

        // --- Verify phase: run draft tokens through target as batch ---
        let verified = target.prefill_verify(target_model, &draft_tokens);

        // --- Accept/reject ---
        let mut n_accepted = 0;
        for i in 0..draft_tokens.len() {
            // Compare target's greedy choice at position i with draft token
            if verified[i] == draft_tokens[i] {
                n_accepted += 1;
            } else {
                break;
            }
        }
        n_accepted_total += n_accepted as u32;

        // Emit accepted tokens
        let mut stopped = false;
        for i in 0..n_accepted {
            if stop_ids.contains(&draft_tokens[i]) { stopped = true; break; }
            output.push(draft_tokens[i]);
            on_token(draft_tokens[i]);
            n_gen += 1;
        }
        if stopped { break; }

        // Emit target's token at rejection point (or after full accept)
        let bonus_tok = if n_accepted < draft_tokens.len() {
            // Target disagrees at position n_accepted — use target's token
            verified[n_accepted]
        } else {
            // All accepted — target's logits at last position give next token
            // We need to run one more forward on target to get next logits
            // Actually, prefill_verify already advanced the KV cache through
            // all draft_tokens. The last verified[K-1] is the target's choice
            // AFTER all K tokens. We can use it as the bonus token.
            // But we need the target's argmax at position K (one past last draft).
            // prefill_verify returns argmax per input position, not the next token.
            // We need one more forward pass on target.
            target.forward(target_model, draft_tokens[n_accepted - 1], pos + n_accepted - 1);
            argmax(target.logits())
        };

        if !stop_ids.contains(&bonus_tok) {
            output.push(bonus_tok);
            on_token(bonus_tok);
            n_gen += 1;
        } else {
            break;
        }

        // --- Sync KV caches ---
        // Restore draft KV to before drafting, re-run accepted + bonus
        draft_state.kv_cache.restore(draft_checkpoint).unwrap();
        // Target KV: prefill_verify already advanced it through all draft tokens.
        // Restore to checkpoint, then advance by n_accepted + 1 (bonus).
        target.kv_cache.restore(target_checkpoint).unwrap();

        let accepted_plus_bonus: Vec<u32> = output[pos..].to_vec();
        for (i, &tok) in accepted_plus_bonus.iter().enumerate() {
            draft_state.forward(&draft.model, tok, pos + i);
            target.forward(target_model, tok, pos + i);
        }
        pos += accepted_plus_bonus.len();

        if n_gen >= max_tokens as u32 { break; }
    }

    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    let dtps = if n_gen > 1 { (n_gen - 1) as f64 / (decode_ms / 1000.0) } else { 0.0 };
    let avg = if n_gen > 1 { decode_ms / (n_gen - 1) as f64 } else { 0.0 };
    let acc_rate = if n_draft_total > 0 { n_accepted_total as f64 / n_draft_total as f64 * 100.0 } else { 0.0 };
    eprintln!("\n--- perf ({n_threads} threads, spec K={draft_k}) ---");
    eprintln!("prefill:    {} tokens in {prefill_ms:.0}ms", prompt_tokens.len());
    eprintln!("decode:     {n_gen} tokens in {decode_ms:.0}ms ({dtps:.1} tok/s, {avg:.1}ms/tok)");
    eprintln!("draft:      {n_draft_total} drafted, {n_accepted_total} accepted ({acc_rate:.0}%)");
    (output, prefill_ms, decode_ms)
}
```

- [ ] **Step 2: Add `pub mod speculative;` to `src/inference/mod.rs`**

- [ ] **Step 3: Verify it compiles**

Run: `PATH="..." cargo build 2>&1 | grep error`

- [ ] **Step 4: Commit**

```bash
git add src/inference/speculative.rs src/inference/mod.rs
git commit -m "feat: speculative decoding — draft+verify generate loop"
```

---

### Task 3: Wire into Engine + CLI

**Files:**
- Modify: `src/inference/generate.rs`

- [ ] **Step 1: Add draft model to Engine**

Add a field and methods to `Engine`:

```rust
pub struct Engine {
    // ... existing fields ...
    draft: Option<(GgufFile, BitNetModel)>,
    pub draft_k: usize,
}
```

Add `load_draft()` method:

```rust
pub fn load_draft(&mut self, path: &Path) -> Result<()> {
    let gguf = GgufFile::open(path)?;
    let model = BitNetModel::from_gguf(&gguf)?;
    eprintln!("[Olorin] Draft model loaded: {} layers, {} dim",
        model.n_layers, model.hidden_dim);
    self.draft = Some((gguf, model));
    Ok(())
}
```

In `Engine::new()`, add `draft: None, draft_k: 5`.

- [ ] **Step 2: Route to speculative generate when draft is loaded**

In `Engine::generate()`, before the existing Q4K generate call:

```rust
if is_q4k {
    if let Some((_, ref draft_model)) = self.draft {
        use crate::inference::speculative;
        let draft_wrap = speculative::DraftModelRef { model: draft_model };
        let (gen, _, _) = speculative::speculative_generate(
            &model, draft_model, &tokens, self.max_tokens, self.draft_k,
            self.temperature, self.top_k, self.top_p, self.min_p,
            self.repetition_penalty, &tokenizer.stop_ids, self.max_seq_len, on_tok,
        );
        generated = gen;
    } else {
        // existing forward_llama::generate call
    }
}
```

Note: `DraftModel::load` owns GgufFile. For Engine integration, we store `(GgufFile, BitNetModel)` directly — no separate DraftModel struct needed. Adjust speculative_generate to take `&BitNetModel` for draft instead of `&DraftModel`.

- [ ] **Step 3: Parse --draft flag in CLI**

Find where CLI args are parsed (likely `main.rs` or `interface/terminal.rs`). Add:

```rust
if let Some(draft_path) = args.get("--draft") {
    engine.load_draft(Path::new(draft_path))?;
}
if let Some(k) = args.get("--draft-k") {
    engine.draft_k = k.parse().unwrap_or(5);
}
```

- [ ] **Step 4: Build and test**

Run:
```bash
PATH="..." cargo build --release --target aarch64-unknown-linux-gnu
scp ... peter@10.46.0.27:~/
ssh ... 'echo "the capital of France is?" | ./olorin --draft ~/.olorin/models/qwen2.5-1.5b-instruct-q4_k_m.gguf 2>&1'
```

Expected: speculative decode output with acceptance rate stats.

- [ ] **Step 5: Commit**

```bash
git add src/inference/generate.rs
git commit -m "feat: wire speculative decoding into Engine — --draft flag"
```

---

### Task 4: End-to-end test

**Files:**
- Create: `tests/test_speculative.rs`

- [ ] **Step 1: Write test**

```rust
//! Speculative decoding end-to-end test.
//! Verifies that spec decode produces valid output with draft model.

#[test]
fn speculative_decode_produces_output() {
    // Skip if models not available
    let target_path = std::path::Path::new(
        &format!("{}/.olorin/models/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            std::env::var("HOME").unwrap_or_default()));
    let draft_path = std::path::Path::new(
        &format!("{}/.olorin/models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
            std::env::var("HOME").unwrap_or_default()));
    if !target_path.exists() || !draft_path.exists() {
        eprintln!("SKIP: models not found");
        return;
    }

    use olorin::inference::generate::Engine;

    let mut engine = Engine::load(target_path, 512).expect("load target");
    engine.load_draft(draft_path).expect("load draft");
    engine.max_tokens = 10;
    engine.temperature = 0.0;
    engine.draft_k = 3;

    let mut output = String::new();
    engine.generate("Hi", "", &|tok| { output.push_str(tok); }).expect("generate");
    assert!(!output.is_empty(), "speculative decode produced no output");
}
```

- [ ] **Step 2: Commit**

```bash
git add tests/test_speculative.rs
git commit -m "test: speculative decoding end-to-end"
```

---

### Task 5: Optimize verify — batched prefill with per-token output_proj

**Files:**
- Modify: `src/inference/prefill_llama.rs`

This is the critical optimization. v1 from Task 1 may use K separate forward passes for verify (~1155ms). This task makes verify use batched prefill (~240ms) + K output_proj passes (~35ms) = ~275ms total.

- [ ] **Step 1: Implement efficient prefill_verify**

The `prefill_layers()` helper (from Task 1) returns `xs: Vec<Vec<f32>>` — each token's final hidden state. Running `output_proj()` K times on these is cheap (K × ~7ms = 35ms for K=5).

The total verify cost becomes: ~240ms (prefill) + ~35ms (5× output_proj) = ~275ms.

Ensure `prefill_verify()` uses the batched path, not K separate forwards.

- [ ] **Step 2: Benchmark**

```bash
echo "the capital of France is?" | ./olorin --draft ~/.olorin/models/Qwen3.5-0.8B.Q4_K_M.gguf 2>&1
```

Expected: ~13-15 tok/s decode with draft acceptance stats.

- [ ] **Step 3: Commit**

```bash
git add src/inference/prefill_llama.rs
git commit -m "perf: batched verify in speculative decode — 240ms vs 1155ms"
```

---

## Verification Checklist

- [ ] `echo "Hi" | ./olorin` — normal decode still works (no draft = no regression)
- [ ] `echo "Hi" | ./olorin --draft <path>` — speculative decode works, shows acceptance rate
- [ ] BitNet model unaffected by changes
- [ ] `--draft-k 3` and `--draft-k 8` both work
- [ ] All files under 500 lines
- [ ] No fake functions, no silent fallbacks

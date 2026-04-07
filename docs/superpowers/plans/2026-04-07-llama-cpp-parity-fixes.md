# Llama.cpp Parity Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring Olorin's Gemma 4 E2B inference into apples-to-apples parity with llama.cpp by fixing five divergences identified in the parity audit, validating each fix on the Pi 5 before moving to the next.

**Architecture:** Iterative fix-build-deploy-verify loop. Per fix: edit code → `cargo build --release` on WSL → `scp` binary to Pi 5 → run `gemma4_verify` diagnostic to confirm math hasn't regressed → run sample prompts to confirm coherence improves or holds → commit. Only advance when current fix is validated.

**Tech Stack:** Rust + Ea kernels, GGUF Q4K/Q6K weights, ChaCha20 vault, Pi 5 ARM NEON deployment target.

**Pi target:** `peter@10.46.0.27`, SSH key `~/.ssh/id_ed25519_pi`. Binary lives at `~/olorin/olorin` on Pi.

**Build command (WSL):**
```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build --release
```

**Diagnostic command (Pi):**
```bash
cd ~/olorin && ./olorin --verify-layers   # if flag exists; otherwise: cargo test --release --test gemma4_verify -- --nocapture
```

**Sample prompts for coherence check** (run on Pi after each fix, document outputs in commit message):
1. `"Vad är huvudstaden i Frankrike?"` — single factual, expect `Paris.`
2. `"Skriv en haiku om havet."` — short creative, expect 3 lines
3. `"Förklara fotosyntes i två meningar."` — short explanation, expect ≤2 sentences, no babbling
4. `"List three primes."` — list format, expect `2, 3, 5` or similar
5. `"Räkna från 1 till 5."` — simple sequence, expect `1 2 3 4 5`

**Pass criteria per fix:** verify_layers shows no L2 regression vs baseline, AND at least 4/5 prompts produce coherent on-topic responses that terminate cleanly (hit `<end_of_turn>` or EOS, not max-tokens cutoff with garbage).

---

## Task 0: Establish baseline

**Files:** none (read-only)

- [ ] **Step 1: Capture current behavior baseline on Pi**

```bash
ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 'cd ~/olorin && ./olorin --version 2>&1; git -C ~/olorin rev-parse HEAD 2>&1'
```

- [ ] **Step 2: Run diagnostic on Pi, save output**

```bash
ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 'cd ~/olorin && cargo test --release --test gemma4_verify -- --nocapture 2>&1' | tee /tmp/baseline_verify.txt
```

Expected: per-step L2 norms printed. Save these as the regression baseline.

- [ ] **Step 3: Run 5 sample prompts on Pi, save outputs**

```bash
for p in "Vad är huvudstaden i Frankrike?" "Skriv en haiku om havet." "Förklara fotosyntes i två meningar." "List three primes." "Räkna från 1 till 5."; do
  ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 "cd ~/olorin && echo '$p' | ./olorin --once 2>&1"
  echo "---"
done | tee /tmp/baseline_prompts.txt
```

Expected: documented "svamlet" — incoherent, never-stopping outputs.

---

## Task 1: Fix chat template tokens (CRITICAL)

**Files:**
- Modify: `src/inference/generate.rs:147-163`
- Modify: `src/inference/tokenizer.rs:131`
- Modify: `src/core/safety.rs:266`

- [ ] **Step 1: Verify Gemma special tokens exist in vocab**

```bash
strings ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf 2>/dev/null | grep -E "start_of_turn|end_of_turn|<\|turn"
```

Expected: `<start_of_turn>` and `<end_of_turn>` present. `<|turn>` and `<turn|>` absent. If absent, abort and reconsider. (Note: `strings` may or may not surface special tokens; if empty, check via debug logging in tokenizer at runtime.)

- [ ] **Step 2: Edit `src/inference/generate.rs` `format_chat`**

Replace lines 147-163 with:

```rust
fn format_chat(user: &str, system: &str) -> String {
    // Gemma 4 chat format (matches llama.cpp LLM_CHAT_TEMPLATE_GEMMA):
    // <bos><start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n<start_of_turn>model\n
    let mut out = String::with_capacity(system.len() + user.len() + 80);
    out.push_str("<start_of_turn>user\n");
    if !system.is_empty() {
        out.push_str(system);
        out.push_str("\n\n");
    }
    out.push_str(user);
    out.push_str("<end_of_turn>\n");
    out.push_str("<start_of_turn>model\n");
    out
}
```

- [ ] **Step 3: Edit `src/inference/tokenizer.rs:131` stop-token list**

Replace:
```rust
for special in ["<|eot_id|>", "<|im_end|>", "<turn|>", "<end_of_turn>"] {
```
With:
```rust
for special in ["<|eot_id|>", "<|im_end|>", "<end_of_turn>"] {
```

- [ ] **Step 4: Edit `src/core/safety.rs:266` hallucination pattern**

Replace `b"<|turn>"` with `b"<start_of_turn>"`.

- [ ] **Step 5: Build on WSL**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build --release
```

Expected: clean build, no warnings about unused special tokens.

- [ ] **Step 6: First check no llama.cpp/Olorin already running on Pi**

```bash
ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 'pgrep -af "olorin|llama" || echo "clean"'
```

Expected: `clean`. If anything running, ask Peter before killing.

- [ ] **Step 7: Deploy binary to Pi**

```bash
scp -i ~/.ssh/id_ed25519_pi target/release/olorin peter@10.46.0.27:~/olorin/olorin
```

- [ ] **Step 8: Run diagnostic on Pi, diff vs baseline**

```bash
ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 'cd ~/olorin && cargo test --release --test gemma4_verify -- --nocapture 2>&1' | tee /tmp/fix1_verify.txt
diff /tmp/baseline_verify.txt /tmp/fix1_verify.txt
```

Expected: **no diff** in L2 norms (this fix touches template only, not math). Any divergence here means a build environment difference — investigate.

- [ ] **Step 9: Run 5 prompts on Pi**

```bash
for p in "Vad är huvudstaden i Frankrike?" "Skriv en haiku om havet." "Förklara fotosyntes i två meningar." "List three primes." "Räkna från 1 till 5."; do
  echo "=== $p ==="
  ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 "cd ~/olorin && echo '$p' | ./olorin --once 2>&1"
done | tee /tmp/fix1_prompts.txt
```

Expected: ≥4/5 coherent and terminate cleanly. If < 4/5, **STOP**, investigate, do not commit.

- [ ] **Step 10: Commit**

```bash
git add src/inference/generate.rs src/inference/tokenizer.rs src/core/safety.rs
git commit -m "$(cat <<'EOF'
fix(gemma4): chat template uses real <start_of_turn>/<end_of_turn> tokens

Was emitting <|turn>/<turn|> which don't exist in Gemma vocab — tokenizer
byte-fell-back to ~7 garbage tokens per turn boundary, causing the model
to never see clean role transitions and babble incoherently.

Now matches llama.cpp LLM_CHAT_TEMPLATE_GEMMA exactly. Stop-token list
and safety hallucination pattern updated to match.

Coherence: 5/5 sample prompts now terminate cleanly on <end_of_turn>.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Verify BOS handling

**Files:**
- Modify: `src/inference/generate.rs:84-86` (only if divergent)
- Modify: `src/inference/tokenizer.rs` (read `add_bos_token` from GGUF)

- [ ] **Step 1: Check GGUF for `tokenizer.ggml.add_bos_token`**

Add a temporary debug print in `tokenizer.rs::from_gguf`:

```rust
let add_bos = gguf.get_bool("tokenizer.ggml.add_bos_token").unwrap_or(true);
eprintln!("DEBUG: add_bos_token = {}", add_bos);
```

Build, deploy, run once:

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build --release
scp -i ~/.ssh/id_ed25519_pi target/release/olorin peter@10.46.0.27:~/olorin/olorin
ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 "cd ~/olorin && echo 'hi' | ./olorin --once 2>&1 | head -5"
```

Expected: line `DEBUG: add_bos_token = true` (Gemma typically true). If true → no fix needed, skip to Step 4. If false → continue Step 2.

- [ ] **Step 2: Make BOS prepend conditional in `generate.rs:84`**

Replace:
```rust
let mut tokens = vec![self.tokenizer.bos_id];
tokens.extend(self.tokenizer.encode(&formatted));
```
With:
```rust
let mut tokens = Vec::with_capacity(formatted.len() / 4 + 1);
if self.tokenizer.add_bos {
    tokens.push(self.tokenizer.bos_id);
}
tokens.extend(self.tokenizer.encode(&formatted));
```

And add `pub add_bos: bool` to `Tokenizer` struct, populated in `from_gguf`.

- [ ] **Step 3: Remove the debug print**

- [ ] **Step 4: Build, deploy, verify, prompts**

Same commands as Task 1 Steps 5–9. Diagnostic must match baseline (no math change). Prompts must remain ≥4/5.

- [ ] **Step 5: Commit (only if code changed)**

```bash
git add src/inference/generate.rs src/inference/tokenizer.rs
git commit -m "fix(gemma4): respect add_bos_token from GGUF metadata

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

If no code changed, skip commit and document in plan notes.

---

## Task 3: Fix sliding-window ring buffer read order

**Files:**
- Modify: `src/inference/cache.rs` (add `write_pos` accessor or rotate-on-read)
- Modify: `src/inference/forward_attn.rs:383-430` (attention K/V read loop)

- [ ] **Step 1: Confirm bug with a long-prompt test**

Run on Pi:
```bash
ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 'cd ~/olorin && yes "the quick brown fox " | head -c 3000 | ./olorin --once 2>&1 | tail -20'
```

Expected: garbage / incoherent output after ~512 tokens of context. If output is coherent, the bug may already be hidden by attn_len capping — verify by checking forward_attn.rs read pattern manually.

- [ ] **Step 2: Add `write_pos(layer)` to KvCache**

In `src/inference/cache.rs`, add method:

```rust
/// Physical write position in the ring buffer for layer at current seq_len.
/// For sliding-window layers, this is `seq_len % window_size`.
/// For global layers, this is `seq_len`.
#[inline]
pub fn write_pos(&self, layer: usize) -> usize {
    let src = self.shared_source[layer].unwrap_or(layer);
    match self.attn_types[src] {
        AttnType::SlidingWindow => self.seq_len % self.window_size,
        AttnType::Global => self.seq_len,
    }
}

/// True if this layer's storage is a wrap-around ring buffer.
#[inline]
pub fn is_sliding(&self, layer: usize) -> bool {
    let src = self.shared_source[layer].unwrap_or(layer);
    matches!(self.attn_types[src], AttnType::SlidingWindow)
}
```

- [ ] **Step 3: Rotate the read offset in `forward_attn.rs:383`**

Locate the attention read loop. Replace the linear `for p in 0..attn_len` with a wrap-aware index. The oldest valid token in a wrapped sliding window is at `(write_pos + 1) % window_size` (since `write_pos` was just written = newest).

Read order should walk **temporally oldest → newest**:

```rust
let attn_len = self.cache.attn_len(il);
let is_swa = self.cache.is_sliding(il);
let window = /* window_size from cache or model */;
let write_pos = self.cache.write_pos(il);

for p in 0..attn_len {
    let phys_pos = if is_swa && attn_len == window {
        // Wrapped: oldest is at (write_pos + 1) % window
        ((write_pos + 1 + p) % window)
    } else {
        // Not wrapped (or global): linear
        p
    };
    let k_offset = phys_pos * stride + kv_h * head_dim;
    // ... rest of read
}
```

Expose `window_size` via a getter on KvCache if not already accessible.

- [ ] **Step 4: Build on WSL**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build --release
```

- [ ] **Step 5: Check Pi clean, deploy**

```bash
ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 'pgrep -af "olorin|llama" || echo "clean"'
scp -i ~/.ssh/id_ed25519_pi target/release/olorin peter@10.46.0.27:~/olorin/olorin
```

- [ ] **Step 6: Run diagnostic, diff against baseline**

```bash
ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 'cd ~/olorin && cargo test --release --test gemma4_verify -- --nocapture 2>&1' | tee /tmp/fix3_verify.txt
diff /tmp/baseline_verify.txt /tmp/fix3_verify.txt
```

Expected: For seq_len ≤ window_size, identical (no wrap → linear read). For longer sequences, divergence is OK as long as values are sane (not NaN, not exploding). The diagnostic likely runs short sequences so should be unchanged.

- [ ] **Step 7: Re-run long-prompt test**

```bash
ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 'cd ~/olorin && yes "the quick brown fox " | head -c 3000 | ./olorin --once 2>&1 | tail -20'
```

Expected: coherent output even past 512 tokens.

- [ ] **Step 8: Run 5 standard prompts**

Same as Task 1 Step 9. Must hold ≥4/5.

- [ ] **Step 9: Commit**

```bash
git add src/inference/cache.rs src/inference/forward_attn.rs
git commit -m "$(cat <<'EOF'
fix(gemma4): sliding-window ring buffer reads in temporal order

cache.store() wrote SWA layers at seq_len % window_size, but the
attention read loop walked the buffer linearly 0..attn_len, so after
token #512 the temporal order was scrambled.

Now read offset is (write_pos + 1 + p) % window when the ring has
wrapped, walking oldest→newest as the attention math expects.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Fix sampler order (temperature after filtering)

**Files:**
- Modify: `src/inference/generate.rs:169-255` (`sample` function)
- Modify: `src/inference/generate.rs:51-58` (defaults)

- [ ] **Step 1: Reorder operations in `sample()`**

Replace the current body of `sample()` (lines 169-255) with this order matching llama.cpp:

```rust
fn sample(
    logits: &mut [f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    rng: &mut u64,
) -> u32 {
    let n = logits.len();

    // Greedy fast path
    if temperature < 1e-6 {
        return argmax(logits);
    }

    // 1. Build (id, logit) candidates from raw logits
    let mut candidates: Vec<(u32, f32)> = (0..n as u32)
        .map(|i| (i, logits[i as usize]))
        .collect();

    // 2. Top-k: sort desc by raw logit, truncate
    if top_k > 0 && top_k < candidates.len() {
        candidates.select_nth_unstable_by(top_k, |a, b| b.1.partial_cmp(&a.1).unwrap());
        candidates.truncate(top_k);
    }
    candidates.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // 3. Min-p on raw logits
    let max_logit = candidates[0].1;
    let min_p_thresh = max_logit + min_p.max(1e-10).ln();
    candidates.retain(|c| c.1 >= min_p_thresh);

    // 4. Apply temperature
    for c in candidates.iter_mut() {
        c.1 /= temperature;
    }

    // 5. Softmax (numerically stable)
    let max_t = candidates[0].1;
    let mut sum = 0.0f32;
    for c in candidates.iter_mut() {
        c.1 = (c.1 - max_t).exp();
        sum += c.1;
    }
    let inv = 1.0 / sum;
    for c in candidates.iter_mut() { c.1 *= inv; }

    // 6. Top-p (nucleus)
    if top_p < 1.0 {
        let mut cum = 0.0f32;
        let mut cutoff = candidates.len();
        for (i, c) in candidates.iter().enumerate() {
            cum += c.1;
            if cum >= top_p {
                cutoff = i + 1;
                break;
            }
        }
        candidates.truncate(cutoff);
        // re-normalize
        let s: f32 = candidates.iter().map(|c| c.1).sum();
        let r = 1.0 / s;
        for c in candidates.iter_mut() { c.1 *= r; }
    }

    // 7. Sample
    *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
    let r = (*rng as f64 / u64::MAX as f64) as f32;
    let mut acc = 0.0f32;
    for c in &candidates {
        acc += c.1;
        if r <= acc { return c.0; }
    }
    candidates.last().unwrap().0
}
```

- [ ] **Step 2: Update sampler defaults to match llama.cpp**

In the SamplerParams or wherever defaults live (around `generate.rs:51-58`):

```rust
top_p: 0.9,    // was 0.95
min_p: 0.1,    // was 0.05
```

Leave temperature 0.8, top_k 40 unchanged.

- [ ] **Step 3: Build, deploy**

```bash
PATH="/mnt/c/Users/Peter.lukka/Desktop/DEV/eacompute/target/release:$PATH" cargo build --release
ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 'pgrep -af "olorin|llama" || echo "clean"'
scp -i ~/.ssh/id_ed25519_pi target/release/olorin peter@10.46.0.27:~/olorin/olorin
```

- [ ] **Step 4: Diagnostic must match baseline**

```bash
ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 'cd ~/olorin && cargo test --release --test gemma4_verify -- --nocapture 2>&1' | tee /tmp/fix4_verify.txt
diff /tmp/baseline_verify.txt /tmp/fix4_verify.txt
```

Expected: **no diff** — sampler doesn't touch forward pass.

- [ ] **Step 5: Run 5 prompts with seed=42 to get reproducible output**

```bash
for p in "Vad är huvudstaden i Frankrike?" "Skriv en haiku om havet." "Förklara fotosyntes i två meningar." "List three primes." "Räkna från 1 till 5."; do
  echo "=== $p ==="
  ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 "cd ~/olorin && echo '$p' | ./olorin --once --seed 42 2>&1"
done | tee /tmp/fix4_prompts.txt
```

Expected: ≥4/5 coherent. (If `--seed` flag doesn't exist, omit; outputs will differ run-to-run but should still be coherent.)

- [ ] **Step 6: Commit**

```bash
git add src/inference/generate.rs
git commit -m "$(cat <<'EOF'
fix(gemma4): sampler order matches llama.cpp (top-k, min-p, temp, softmax, top-p)

Was applying temperature BEFORE filtering, which shifted min_p threshold
and changed which tokens survived. Now matches llama.cpp pipeline order.

Defaults updated to llama.cpp values: top_p 0.95→0.9, min_p 0.05→0.1.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Plumb or remove `repeat_penalty`

**Files:**
- Inspect: `src/core/router_config.rs:12,28,53`, `src/interface/server.rs:201`
- Modify: `src/inference/generate.rs` (apply penalty if non-1.0)

- [ ] **Step 1: Decide plumb-through vs delete**

Check usage:
```bash
grep -rn "repeat_penalty\|repetition_penalty" src/
```

If `repeat_penalty` is only ever 1.0 in practice (defaults pass through, no UI sets it) → **delete** the field everywhere. Otherwise → **plumb it through** to `sample()`.

Default to plumb-through; deletion is a separate decision that needs Peter's call.

- [ ] **Step 2: Add `repeat_last_n` and `repeat_penalty` params to `sample()`**

Add a `recent_tokens: &[u32]` parameter and apply at the start of `sample()` (before top-k):

```rust
fn sample(
    logits: &mut [f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    repeat_penalty: f32,
    recent_tokens: &[u32],
    rng: &mut u64,
) -> u32 {
    // Apply repetition penalty (matches llama.cpp llama_sample_repetition_penalties)
    if repeat_penalty != 1.0 && !recent_tokens.is_empty() {
        for &tok in recent_tokens {
            let l = &mut logits[tok as usize];
            *l = if *l <= 0.0 { *l * repeat_penalty } else { *l / repeat_penalty };
        }
    }
    // ... rest of sample as in Task 4 ...
}
```

- [ ] **Step 3: Maintain a ring of last 64 tokens in the generation loop**

In `generate.rs` decode loop, push each new `token_id` into a `VecDeque<u32>` capped at 64, pass `&Vec::from_iter(...)` (or slice the deque) into `sample()`.

- [ ] **Step 4: Wire `repeat_penalty` from config through to `sample` call site**

Find where `sample()` is called in `generate.rs` and pass `self.params.repeat_penalty` and the recent-token slice.

- [ ] **Step 5: Build, deploy, diagnostic, prompts**

Same sequence as Task 4 Steps 3–5. Diagnostic unchanged. Prompts ≥4/5.

- [ ] **Step 6: Commit**

```bash
git add src/inference/generate.rs
git commit -m "$(cat <<'EOF'
fix(gemma4): apply repeat_penalty over last 64 tokens (was unused)

Field was wired through config and HTTP API but never reached the
sampler. Now matches llama.cpp default repeat_last_n=64 behavior.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Add `f_attention_scale` field (latent, do last)

**Files:**
- Modify: `src/inference/engine.rs` (add field, hardcode 1.0 for E2B)
- Modify: `src/inference/forward_attn.rs:195` (use field instead of literal)

- [ ] **Step 1: Add field to Gemma4Model**

In `engine.rs` model struct:
```rust
pub attn_scale: f32,
```

In `from_gguf`, set:
```rust
// Gemma 3n / 4 E2B always uses 1.0; larger Gemma 3 variants use 1/sqrt(head_dim)
attn_scale: 1.0,
```

- [ ] **Step 2: Use it in forward_attn.rs**

Replace `let attn_scale = 1.0f32;` (line 195) with:
```rust
let attn_scale = model.attn_scale;
```

- [ ] **Step 3: Build, deploy, diagnostic, prompts**

Diagnostic must be **bit-identical** to baseline (since 1.0 == 1.0).

- [ ] **Step 4: Commit**

```bash
git add src/inference/engine.rs src/inference/forward_attn.rs
git commit -m "$(cat <<'EOF'
refactor(gemma4): add f_attention_scale field for future Gemma 3 variants

E2B uses 1.0 (bit-identical), but larger Gemma 3 models use
1/sqrt(head_dim). Field is now in place for that future case.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final validation

- [ ] **Step 1: Full diagnostic snapshot**

```bash
ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 'cd ~/olorin && cargo test --release --test gemma4_verify -- --nocapture 2>&1' | tee /tmp/final_verify.txt
diff /tmp/baseline_verify.txt /tmp/final_verify.txt
```

Expected: identical or only Task 3 long-sequence improvements visible.

- [ ] **Step 2: Run an extended prompt suite**

Run 5 standard + 1 long (3000 tokens) prompt. Document outputs.

- [ ] **Step 3: Side-by-side vs llama.cpp on Pi**

```bash
ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 '~/llama-cli -m ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf -p "Vad är huvudstaden i Frankrike?" -n 50 --temp 0 2>&1 | tail -10'
ssh -i ~/.ssh/id_ed25519_pi peter@10.46.0.27 'cd ~/olorin && echo "Vad är huvudstaden i Frankrike?" | ./olorin --once --temp 0 2>&1'
```

Expected: with greedy (`temp 0`), both should produce the same first ~10 tokens for the same prompt. Token-level divergence at this point means there's still a math difference somewhere — investigate.

- [ ] **Step 4: Update CLAUDE.md status section if Peter wants**

Note in `CLAUDE.md` that Olorin Gemma 4 E2B is now llama.cpp-parity-validated.

---

## Notes

- **Never** run llama.cpp and Olorin on the Pi simultaneously.
- **Always** check Pi for running processes before deploying a new binary.
- If a fix breaks the diagnostic, **revert that single fix** and investigate before retrying. Don't stack broken fixes.
- Each commit must independently leave the repo in a working state.

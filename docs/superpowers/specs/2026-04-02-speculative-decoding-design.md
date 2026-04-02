# Speculative Decoding — Design Spec

## Goal

Increase Llama 3.2 3B Q4K decode from 4.3 tok/s to ~14 tok/s on Pi 5 via draft+verify speculative decoding.

## Architecture

A small draft model generates K candidate tokens greedily. The target model verifies them in one batched prefill pass (one weight load). Accepted tokens are emitted; on rejection, the target model's token at the rejection point is emitted instead.

### Components

**`DraftModel`** — Wraps `LlamaState` + `BitNetModel` for any GGUF model. Hotswappable: load a new draft model at any time. Owns its own KV-cache. Shares the thread pool with the target model.

**`speculative_decode()`** — Main loop:

```
loop {
    checkpoint = draft.kv_cache.checkpoint()
    draft_tokens = []
    for i in 0..K {
        logits = draft.forward(last_token, pos + i)
        tok = argmax(logits)
        draft_tokens.push(tok)
        last_token = tok
    }

    // Verify: run all draft tokens as batched prefill on target
    target_logits = target.prefill(draft_tokens)

    // Compare: target's argmax vs draft token at each position
    n_accepted = 0
    for i in 0..K {
        if argmax(target_logits[i]) == draft_tokens[i] {
            n_accepted += 1
        } else {
            break
        }
    }

    emit(draft_tokens[..n_accepted])

    if n_accepted < K {
        // Emit target model's token at rejection point
        emit(argmax(target_logits[n_accepted]))
        n_accepted += 1  // total tokens this iteration
    }

    // Sync: truncate draft KV to accepted position
    draft.kv_cache.restore(checkpoint)
    // Re-run accepted tokens through draft to rebuild its KV
    for tok in draft_tokens[..n_accepted] {
        draft.forward(tok, pos)
        pos += 1
    }
}
```

### KV Cache Sync

On rejection, the draft model's KV-cache has K entries that are partially wrong. Strategy: checkpoint before drafting, restore on rejection, re-run only the accepted tokens through the draft model to rebuild correct KV state. `kv_cache.checkpoint()` and `kv_cache.restore()` already exist.

### CLI Interface

- `--draft <path>` — path to draft model GGUF
- `--draft-k <n>` — number of draft tokens per batch (default: 5)
- Without `--draft` — normal decode, no regression

### Hotswap

`DraftModel::load(path)` drops the old model and loads a new one. The speculative loop checks `has_draft()` and falls back to normal decode if no draft is loaded.

## Files

- **Create:** `src/inference/speculative.rs` (~200 lines)
  - `DraftModel` struct (LlamaState + BitNetModel wrapper)
  - `speculative_generate()` function (draft+verify loop with emit callback)
- **Modify:** `src/inference/forward_llama.rs` or `src/inference/generate.rs`
  - Wire speculative path into the generate loop
  - Expose prefill return of per-token logits (currently prefill only returns last token's logits)
- **Modify:** CLI entry point (`main.rs` or `interface/terminal.rs`)
  - Parse `--draft` and `--draft-k` flags
  - Load draft model, pass to generate

## Constraints

- Draft and target share one thread pool — no parallel execution in v1
- Greedy argmax for both draft and verify — no sampling-based acceptance
- No dynamic K adjustment in v1
- Draft model must be any valid GGUF (Q4K or Q6K Llama-architecture)

## Not In v1

- Asynchronous draft (draft batch N+1 while verify runs)
- Dynamic K based on acceptance rate
- Token-level probability matching (speculative sampling)
- Non-Llama draft architectures

## Expected Performance

| Component | Time | Notes |
|-----------|------|-------|
| Draft 5 tokens (0.8B) | 75ms | 5 × 15ms/tok |
| Verify 5 tokens (3B) | 240ms | One prefill batch |
| KV rebuild on reject | ~15ms | Re-run ~1 accepted token |
| **Total per batch** | **~315ms** | |
| Acceptance rate | ~80% | 4 of 5 accepted (typical 0.8B→3B) |
| Tokens per batch | ~4.2 | 4 accepted + 0.2 avg from target |
| **Effective decode** | **~75ms/tok = 13.3 tok/s** | 3.1× improvement |

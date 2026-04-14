# Prompt-Lookup Speculative Decoding — Design Spec

Date: 2026-04-14
Branch context: `q6k-repack-gemm`

## Goal

Speed up Gemma 4 E2B decode on workloads with verbatim repetition (code, edits,
structured output) by speculating K tokens per step via n-gram lookup over the
live context, then verifying them in a single batched forward pass. Target:
≥1.8× on code prompts, ≤5% regression on free-form chat, zero change to
generated output under greedy decoding.

Same model only. No second model, no Medusa heads, no early-exit layers.

## Non-goals

- Draft model (separate weights) — prior spec
  (`2026-04-02-speculative-decoding-design.md`) covers that path and is not
  revived here.
- Early-exit self-speculative decoding — possible follow-up if prompt-lookup
  proves insufficient for chat workloads.
- Adaptive K — tuning left as a v2 once measurement data exists.
- Training-dependent approaches (Medusa).

## Architecture

### Per-decode-step loop

Starting state: `seq_len = S`, last committed token's KV already written,
`logits_snapshot` predicts the token at position `S`.

1. Sample `A_0` from `logits_snapshot` (existing sampler, unchanged).
2. Call `ngram_lookup` with key = last 3 tokens of
   `context_tokens = prompt_ids ++ generated_ids`. Returns up to `K-1` draft
   tokens `D_1..D_{K-1}`, or zero if no match (fall back to plain
   single-token decode with `A_0`).
3. Feed `[A_0, D_1, ..., D_{K-1}]` (K inputs) through `forward_batch` at
   positions `S..S+K-1`. KV is written for all K positions, `seq_len` advances
   to `S+K`. Returns K logits rows.
4. `verify_draft` kernel: compute `A_j = argmax(logits_batch[j-1])` for
   `j=1..K`. Find first `j` in `1..K-1` where `A_j != D_j` (drafts). If none,
   set `j = K` (full accept). Return `j` and all argmaxes up to and including
   `A_j`.
5. Rewind `kv.seq_len = S + j`. Stale KV at `S+j..S+K-1` is not read because
   `attn_len` caps at `seq_len`. For sliding-window layers, future writes
   overwrite the same modular slots — safe.
6. Run `forward_one(A_j)` at position `S+j` to write `A_j`'s KV, advance
   `seq_len` to `S+j+1`, and return a fresh logits snapshot for the next step.
7. Emit, in order: `A_0`, accepted drafts `D_1..D_{j-1}` (empty range if
   `j=1`), correction `A_j`. Append them to `generated_ids`. Total emitted
   per step: `j+1` tokens, where `1 ≤ j+1 ≤ K`.

**Correctness invariant (greedy parity):** every emitted token is the argmax
of the model's own logits at that position. Drafts are never emitted unless
they equal argmax. Therefore, under temperature=0, the output token stream is
bit-identical to the non-speculative path.

**Fall-through to plain decode when:**

- `engine.draft_k == 0` (CLI flag unset)
- `ngram_lookup` returns 0 tokens
- `context_tokens.len() < 3` (prefill just started)

## Kernels

Two new Ea kernels in `kernels/`, auto-discovered by `build.rs`.

### `ngram_lookup.ea`

Signature (FFI):

```rust
ngram_lookup(
    ctx_ptr: *const u32, ctx_len: usize,   // full context so far (prompt + generated)
    key_ptr: *const u32,                    // last 3 tokens of context
    k: usize,                               // max draft length requested
    out_ptr: *mut u32,                      // writes up to k tokens
) -> i32                                    // tokens written, 0 = no match
```

Algorithm:

- Pass 1 (N=3): SIMD-broadcast `key[0]` as `i32x4`/`i32x8`, compare-equal
  against sliding windows of `ctx_tokens`. Iterate right-to-left so we prefer
  recent matches. On any lane match, scalar-check tokens 1 and 2 against
  `key[1]`, `key[2]`. On full 3-gram match, copy up to `k` trailing tokens to
  `out_ptr` and return.
- Pass 2 (N=2): same but matching `key[1]`, `key[2]`. Fallback when N=3 misses.
- Return 0 if neither pass hits.

No allocation. SIMD mirrors the `chacha20_search_v2` broadcast-compare shape.

### `verify_draft.ea`

Signature:

```rust
verify_draft(
    logits_ptr: *const f32,   // K rows × vocab (predictions for positions 1..K)
    vocab: usize,
    drafts_ptr: *const u32,   // K-1 drafts (D_1..D_{K-1}); last row has no draft
    k: usize,                 // number of logits rows
    out_argmax: *mut u32,     // K argmaxes written (A_1..A_K)
) -> i32                      // first j in 1..K-1 where A_j != D_j; k if full accept
```

Algorithm: per row, horizontal argmax via SIMD reduction (vocab split into
SIMD-width chunks, track best-index + best-value, final horizontal reduce).
Write argmax to `out_argmax[i]`. Compare to `drafts[i]`; on mismatch, write
remaining argmaxes (for correction path) then return `i`. Early-exit saves
work on short-accept cases.

Uses the same horizontal-argmax pattern as the sampler's `argmax`.

## KV management

- `forward_batch` is the existing prefill path. Verify is "prefill of K
  speculative tokens" — no new forward path.
- `KvCache::seq_len` is a monotonic cursor; rewind = direct assignment. No
  zeroing needed — `attn_len` caps reads at `seq_len`.
- Sliding-window layers: `store_batch` writes modulo `window_size`. Rewinding
  and re-writing lands in the same slot; stale values are overwritten.
- Shared layers (shared_source set): `store_batch` is a no-op, unaffected.

## Engine API

`Engine.draft_k: usize` is already a field. `--draft-k` CLI flag already
parses into `DispatchContext::new` → `engine.draft_k`.

Changes to `generate.rs::generate`:

- After sampling `A_0`, branch on `self.draft_k > 0`.
- Speculative branch: ngram_lookup → forward_batch → verify_draft → rewind →
  forward_one → emit accepted span + correction.
- Non-speculative branch: current code path, unchanged.

`Engine::load_draft` (currently stubbed "not yet implemented") stays as-is —
prompt-lookup doesn't use a draft model.

## Timing / observability

Extend `GEMMA4_TIMING=1` output with per-run summary:

```
[timing] speculative: K=4, steps=N, accepted=M, accept_rate=M/(N*K), speedup=X.Xx
```

Where speedup is wall-clock ratio against a same-prompt non-speculative run
measured by the test harness (not at runtime).

## Testing

### Greedy parity (strict — gates merge)

E2E test in `tests/`: fixed prompts (one code, one prose, one JSON),
`temperature=0`, compare token streams with `draft_k=0` vs `draft_k=4` vs
`draft_k=8`. Must be bit-identical. Correctness invariant guarantees this;
test asserts it empirically.

### Sampling-mode quality

Manual review: same prompts, `temperature=1.0`, ~10 runs each. Check for
repetition loops, coherence, obvious degradation. No assertion — judgment
call before merge.

### Kernel unit tests (in `tests/`)

`ngram_lookup`:
- Match at end of context
- Match near start (far from key position)
- Multiple matches — recent one preferred
- No match → returns 0
- N=3 miss, N=2 hit
- Context shorter than 3 → returns 0

`verify_draft`:
- All-accept (j=K)
- Immediate reject (j=0)
- Partial (j=2)
- Small vocab (for exhaustive argmax verification)
- Realistic vocab (262144 — Gemma 4)

### Benchmark

Extend existing bench harness with workload triples:
- Code generation prompt — expect ≥1.8× speedup
- Free-form chat — expect ≥0.95× (no worse than 5% regression)
- JSON / structured — expect ≥1.5×

Report accept-rate and speedup.

### Integration

Web UI + `--draft-k 4`, run the six-prompt sequence from the ChatML-hang
debug session (2026-04-14). All prompts must produce sensible output and
timing must show nonzero accept rate on the code prompt.

## Rollout

1. Add kernels, FFI wrappers, unit tests.
2. Add speculative branch in `generate.rs`, parity tests.
3. Extend bench harness, measure on code / chat / JSON.
4. Enable by default (change `draft_k` default in `Engine::load` from 0 → 4)
   only after benchmarks confirm ≤5% chat regression.

## Open questions (deferred to implementation)

- Should `ngram_lookup` be called on every step, or skipped when the last
  lookup had no match and no new tokens changed the key? (Optimization —
  probably unnecessary since lookup is microseconds.)
- Correction token KV is written via an extra `forward_one` call after
  rewind. Could be eliminated by padding `forward_batch` with an extra
  position for the correction — but complicates the kernel contract. Leave
  as follow-up if `forward_one` overhead dominates.

## Rollout outcome (2026-04-14)

Benchmark results at `temperature=0, max_tokens=128` on Gemma 4 E2B Q4K (16-thread pool, x86 SSE2):

| Workload | K=0 (ms) | K=4 (ms) | K=4 speedup | K=8 (ms) | K=8 speedup |
|----------|---------:|---------:|------------:|---------:|------------:|
| code     |   18870  |   17061  |       1.11x |   12445  |       1.52x |
| chat     |   14582  |   15253  |       0.96x |   13070  |       1.12x |
| json     |   12006  |   11400  |       1.05x |   10790  |       1.11x |

Speedups are real but fall short of the spec's thresholds (code ≥1.8×, json ≥1.5×, chat ≥0.95×). Decision: keep the default `draft_k = 0`. Opt in explicitly via `--draft-k 4` or `--draft-k 8`.

Greedy parity is verified by `tests/speculative_parity.rs` across code, prose, and JSON prompts — speculation never changes output under deterministic decoding, so users who opt in see pure speedup, not behavioral drift.

Future work that could close the gap (not in scope for this rollout):
- Adaptive K based on running accept rate
- Longer lookup window or prefix-tree indexing of generated tokens
- Early-exit self-speculative decoding (first-N-layers draft) as a hybrid fallback when prompt-lookup misses

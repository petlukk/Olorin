//! Robustness wave four — inference limits.
//!
//! The forward pass has a fixed context window (`max_seq_len`, 2048 in
//! production). The robustness property: *no input may panic, OOM, or hang
//! the inference path — an over-window prompt must be handled (clamped or
//! refused), never crash.* These tests exercise the KV cache directly (no
//! model load, so they run in CI) to pin the overflow behaviour that the
//! generation path must protect against.
//!
//! Finding W1 (HIGH, DEFERRED): `Engine::generate` does not bound the prompt
//! against `max_seq_len` — only the narration callers budget via
//! `count_prompt_tokens`. The chat path (`router_streaming` / `router.rs`)
//! and the tool-call follow-up (`router_toolcall`, which embeds raw tool
//! output) feed unbounded prompts straight into `forward_batch` →
//! `KvCache::store_batch`, where a Global layer writes at `seq_len + t` with
//! no bound. Past `max_seq_len` that's an out-of-bounds slice → panic (REPL:
//! process crash; server: connection thread dies). A user pasting a long
//! message — not even adversarial — triggers it. See
//! benchmarks/robustness/FINDINGS.md (wave four).

use olorin::inference::cache::KvCache;
use olorin::inference::engine::AttnType;
use olorin::kernels::ffi;

const N_KV_HEADS: usize = 1;
const HEAD_DIM: usize = 2;
const STRIDE: usize = N_KV_HEADS * HEAD_DIM;

fn global_cache(max_seq_len: usize) -> KvCache {
    KvCache::new(
        1,                       // n_layers
        N_KV_HEADS,
        vec![HEAD_DIM],          // head_dim_v
        4,                       // window_size (unused for a Global layer)
        max_seq_len,
        vec![AttnType::Global],
        vec![None],              // shared_source
    )
}

fn zeros(n_tokens: usize) -> Vec<f32> {
    vec![0.0f32; STRIDE * n_tokens]
}

#[test]
fn w1_global_cache_overflow_panics_past_max_seq_len() {
    ffi::init().unwrap();
    // A Global layer sized for 4 positions. Storing 8 tokens' worth in one
    // batch writes position 4 at cache offset 8 into a length-8 buffer → the
    // `kb[cache_off..cache_off + stride]` slice range is out of bounds and
    // panics. This is exactly what an over-window prompt does in prefill.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut cache = global_cache(4);
        let k = zeros(8);
        let v = zeros(8);
        cache.store_batch(0, &k, &v, 8);
    }));
    assert!(
        outcome.is_err(),
        "W1: an over-window store must currently panic (DEFERRED — the fix is a \
         prompt budget in Engine::generate so this is never reached). If this \
         stops panicking because store_batch grew its own bound, update the finding."
    );
}

#[test]
fn w1_within_capacity_does_not_panic() {
    ffi::init().unwrap();
    // Exactly filling the window is fine — the overflow is strictly past it.
    let mut cache = global_cache(4);
    let k = zeros(4);
    let v = zeros(4);
    cache.store_batch(0, &k, &v, 4); // positions 0..=3, exactly the capacity
}

#[test]
fn sliding_window_layer_wraps_and_never_overflows() {
    ffi::init().unwrap();
    // Contrast: a SlidingWindow layer indexes `(seq_len + t) % window_size`,
    // so it ring-wraps and can absorb any token count without overflowing.
    // The overflow risk is specific to Global layers — which is why W1 bites
    // long prompts (Global layers hold the full sequence).
    let mut cache = KvCache::new(
        1, N_KV_HEADS, vec![HEAD_DIM],
        4,                 // window_size
        4,                 // max_seq_len
        vec![AttnType::SlidingWindow],
        vec![None],
    );
    let k = zeros(8);
    let v = zeros(8);
    cache.store_batch(0, &k, &v, 8); // 8 tokens into a 4-slot ring — no panic
}

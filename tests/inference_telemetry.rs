//! Consolidated telemetry measurement for Adaptive-Quant / Spec-Decode go/no-go.
//!
//! Four instruments in one run on a mixed calibration set:
//!   1. Tensor inventory (weight_stats): dtypes + byte totals per tensor role
//!   2. Residual-norm growth per layer (how much "signal" flows per layer)
//!   3. Logit entropy per decoded token (spec-dec acceptance-rate prior)
//!   4. Early-exit logit probe at a sampled subset (spec-dec depth cutoff)
//!
//! Run: cargo test --release --test inference_telemetry -- --ignored --nocapture

use olorin::inference::{activation_track, exit_probe, kv_stats, weight_stats};
use olorin::inference::generate::{Engine, GenEvent};
use olorin::runes;
use std::path::Path;

const PROMPTS: &[&str] = &[
    // Code (7)
    "Here is a Python function that computes Fibonacci numbers:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    ",
    "Write a Rust function that reverses a string:\n\n```rust\nfn reverse_string(s: &str) -> String {\n    ",
    "Example of a binary search in C:\n\n```c\nint binary_search(int* arr, int n, int target) {\n    int lo = 0, hi = n - 1;\n    ",
    "SQL query to find the top 5 customers by total order value:\n\n```sql\nSELECT c.name, SUM(o.total) AS revenue\n",
    "JavaScript async function that fetches JSON from a URL:\n\n```js\nasync function fetchJson(url) {\n    ",
    "Bash script to recursively count .py files in a directory:\n\n```bash\n#!/usr/bin/env bash\n",
    "Go function that reads a file line by line:\n\n```go\nfunc readLines(path string) ([]string, error) {\n    ",
    // Prose / dialogue / creative (5)
    "I love walking in the forest on a spring morning. The birds are singing and ",
    "My grandmother used to bake bread every Sunday. The smell would fill the whole house and ",
    "Hello! How are you doing today? I was wondering if you could help me with ",
    "She sat by the window, watching the rain fall on the cobblestone street. It reminded her of ",
    "The children laughed as they chased each other across the meadow. Their dog bounded along beside them, ",
    // Technical / analytical (4)
    "Quantum entanglement refers to a physical phenomenon where pairs of particles ",
    "The algorithm runs in O(n log n) time because the divide-and-conquer approach ",
    "To prove that the square root of 2 is irrational, assume the opposite: that ",
    "In the context of distributed systems, the CAP theorem states that ",
];

const DECODE_TOKENS: usize = 32;

// Layers to probe — denser near the commit boundary (L24–L33) since the
// first run showed that's where the action is.
const PROBE_LAYERS: &[usize] = &[8, 16, 20, 24, 26, 28, 30, 32, 33];
const PROBE_TOKENS_PER_PROMPT: usize = 3; // first, middle, last of decode

#[test]
#[ignore = "multi-prompt telemetry + early-exit probe, ~3 min; use --ignored --nocapture"]
fn consolidated_telemetry_report() {
    let home = std::env::var("HOME").unwrap();
    let path = Path::new(&home).join(".olorin/models/gemma-4-e2b-it-Q4_K_M.gguf");
    if !path.exists() {
        eprintln!("SKIP: no model at {}", path.display());
        return;
    }
    std::env::set_var("OLORIN_RESIDUAL_TRACK", "1");
    std::env::set_var("OLORIN_RESIDUAL_SNAPSHOT", "1");
    std::env::set_var("OLORIN_LOGIT_ENTROPY", "1");

    let mut engine = Box::new(Engine::load(&path, 1024).expect("load"));
    engine.temperature = 0.0;
    engine.max_tokens = DECODE_TOKENS;

    // Production-realistic: use the real system prompt (rune block + safety
    // guardrails). Running with `""` previously skipped ~150 prefill tokens
    // per prompt that production always sees — activation patterns, attention,
    // and logit entropy all shift meaningfully with the full context.
    let system = runes::runes_prompt_block();
    eprintln!("[telemetry] system prompt: {} chars ({} bytes)",
        system.chars().count(), system.len());

    let on_event = |_: GenEvent| {};

    // ── 1. Tensor inventory ──────────────────────────────────────────
    let model_summary = weight_stats::summarize(engine.model());
    eprintln!();
    eprintln!("╔═══ 1. MODEL TENSOR INVENTORY ═══╗");
    eprintln!();
    eprint!("{}", weight_stats::format_report(&model_summary));
    eprintln!();
    eprintln!("Per-layer dtype fingerprints (first 5 + last 5):");
    let fingerprints = weight_stats::per_layer_dtype_fingerprint(engine.model());
    for fp in fingerprints.iter().take(5) { eprintln!("  {fp}"); }
    if fingerprints.len() > 10 { eprintln!("  ..."); }
    for fp in fingerprints.iter().rev().take(5).rev() { eprintln!("  {fp}"); }

    // ── 2, 3, 4. Run the calibration loop, collecting telemetry ─────
    eprintln!();
    eprintln!("╔═══ Running {} prompts × {} decode tokens ═══╗",
        PROMPTS.len(), DECODE_TOKENS);
    let mut per_prompt_final_logits: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut per_prompt_snapshots: Vec<Vec<Vec<Vec<f32>>>> = Vec::new();

    for (pi, prompt) in PROMPTS.iter().enumerate() {
        // Reset residual-snapshot storage between prompts so we know which
        // snapshots belong to which generation.
        activation_track::reset_snapshots();
        // Capture final logits by running the engine and peeking after.
        // We'll use a callback that records nothing; logits are already captured
        // by the OLORIN_LOGIT_ENTROPY tap via engine internals.
        engine.generate(prompt, system, &on_event).expect("generate");
        let snapshots = activation_track::residual_snapshots();
        eprintln!("  prompt {}: {} snapshotted tokens × {} layers",
            pi, snapshots.len(),
            snapshots.first().map(|t| t.len()).unwrap_or(0));
        per_prompt_snapshots.push(snapshots);
        per_prompt_final_logits.push(Vec::new()); // populated below via reprojection of layer N-1
    }

    // ── 2. Residual-norm growth per layer ────────────────────────────
    let norms = activation_track::residual_norms();
    eprintln!();
    eprintln!("╔═══ 2. RESIDUAL-NORM GROWTH PER LAYER ═══╗");
    eprintln!();
    eprintln!("{:>3}  {:>8}  {:>8}  {:>8}  {:>5}",
        "L", "mean", "std", "max", "n");
    eprintln!("{:-<40}", "");
    for (li, vals) in norms.iter().enumerate() {
        if vals.is_empty() { continue; }
        let n = vals.len() as f32;
        let mean = vals.iter().sum::<f32>() / n;
        let var = vals.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / n;
        let std = var.sqrt();
        let max = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        eprintln!("{:>3}  {:>8.2}  {:>8.2}  {:>8.2}  {:>5}",
            li, mean, std, max, vals.len());
    }
    eprintln!();
    eprintln!("Interpretation: big jumps in mean norm = layers doing heavy work");
    eprintln!("                plateaus = layers making small adjustments (quant-tolerant)");

    // ── 3. Logit-entropy distribution ────────────────────────────────
    let entropies = activation_track::logit_entropies();
    eprintln!();
    eprintln!("╔═══ 3. OUTPUT-LOGIT ENTROPY PER DECODED TOKEN ═══╗");
    eprintln!();
    if entropies.is_empty() {
        eprintln!("(no entropy data — check OLORIN_LOGIT_ENTROPY)");
    } else {
        let n = entropies.len() as f32;
        let mean = entropies.iter().sum::<f32>() / n;
        let mut sorted = entropies.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2];
        let p10 = sorted[sorted.len() / 10];
        let p90 = sorted[(sorted.len() * 9) / 10];
        eprintln!("samples: {}  mean: {:.3}  median: {:.3}  p10: {:.3}  p90: {:.3}",
            entropies.len(), mean, median, p10, p90);
        eprintln!();
        eprintln!("Entropy histogram (nats):");
        let buckets = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0),
                        (2.0, 3.0), (3.0, 4.0), (4.0, 6.0), (6.0, 12.0)];
        for (lo, hi) in buckets {
            let c = entropies.iter().filter(|&&e| e >= lo && e < hi).count();
            let bar = "█".repeat((c * 40 / n as usize).max(if c > 0 { 1 } else { 0 }));
            eprintln!("  [{:>4.1}, {:>4.1})  {:>3}  {}", lo, hi, c, bar);
        }
        eprintln!();
        eprintln!("Spec-dec interpretation: fraction below 0.5 nat = 'easy' tokens");
        let easy = entropies.iter().filter(|&&e| e < 0.5).count();
        eprintln!("  fraction of 'easy' (<0.5 nat) tokens: {:.1}%",
            100.0 * easy as f32 / n);
        eprintln!("  this is a rough *ceiling* on speculative-decode acceptance rate");
    }

    // ── 4. Early-exit logit probe ────────────────────────────────────
    eprintln!();
    eprintln!("╔═══ 4. EARLY-EXIT LOGIT PROBE ═══╗");
    eprintln!();
    eprintln!("Probing layers {:?} on {} prompts × {} tokens",
        PROBE_LAYERS, PROMPTS.len(), PROBE_TOKENS_PER_PROMPT);
    eprintln!("(single-threaded output-head matmul ~200ms/call, total ~{}s)",
        PROBE_LAYERS.len() * PROMPTS.len() * PROBE_TOKENS_PER_PROMPT * 200 / 1000);
    eprintln!();

    let mut per_layer_top1_hits: Vec<(usize, usize)> = PROBE_LAYERS.iter()
        .map(|&l| (l, 0)).collect();
    let mut per_layer_top5_overlap_sum: Vec<(usize, usize)> = PROBE_LAYERS.iter()
        .map(|&l| (l, 0)).collect();
    let mut per_layer_kl_sum: Vec<(usize, f64)> = PROBE_LAYERS.iter()
        .map(|&l| (l, 0.0)).collect();
    let mut total_probes = 0usize;

    let probe_start = std::time::Instant::now();
    for (pi, snapshots) in per_prompt_snapshots.iter().enumerate() {
        if snapshots.is_empty() { continue; }
        // Sample PROBE_TOKENS_PER_PROMPT evenly-spaced tokens (including first + last).
        let token_indices: Vec<usize> = if snapshots.len() >= PROBE_TOKENS_PER_PROMPT {
            if PROBE_TOKENS_PER_PROMPT <= 1 {
                vec![0]
            } else {
                let last = snapshots.len() - 1;
                (0..PROBE_TOKENS_PER_PROMPT)
                    .map(|i| i * last / (PROBE_TOKENS_PER_PROMPT - 1))
                    .collect()
            }
        } else {
            (0..snapshots.len()).collect()
        };

        for &ti in &token_indices {
            let token_snaps = &snapshots[ti];
            if token_snaps.is_empty() { continue; }
            let final_layer = token_snaps.len() - 1;
            if token_snaps[final_layer].is_empty() { continue; }
            let final_logits = exit_probe::reproject_residual(
                &token_snaps[final_layer], engine.model(),
            );
            per_prompt_final_logits[pi].push(final_logits.clone());

            for (idx, &li) in PROBE_LAYERS.iter().enumerate() {
                if li >= token_snaps.len() { continue; }
                if token_snaps[li].is_empty() { continue; }
                let probe_logits = exit_probe::reproject_residual(
                    &token_snaps[li], engine.model(),
                );
                let cmp = exit_probe::compare(&probe_logits, &final_logits);
                if cmp.top1_agreement { per_layer_top1_hits[idx].1 += 1; }
                per_layer_top5_overlap_sum[idx].1 += cmp.top5_overlap;
                per_layer_kl_sum[idx].1 += cmp.kl_final_given_probe as f64;
                total_probes += 1;
            }
        }
    }
    let probe_time = probe_start.elapsed();

    eprintln!("Completed {} probes in {:.1}s ({:.0}ms avg)",
        total_probes, probe_time.as_secs_f64(),
        1000.0 * probe_time.as_secs_f64() / total_probes.max(1) as f64);
    eprintln!();
    eprintln!("{:>5}  {:>10}  {:>12}  {:>12}",
        "layer", "top1_agree", "avg_top5_ovlp", "avg_KL");
    eprintln!("{:-<48}", "");
    let divisor = total_probes / PROBE_LAYERS.len().max(1);
    for (((li, hits), (_, ovlp)), (_, kl)) in per_layer_top1_hits.iter()
        .zip(per_layer_top5_overlap_sum.iter())
        .zip(per_layer_kl_sum.iter())
    {
        let d = divisor.max(1) as f32;
        let top1_pct = 100.0 * *hits as f32 / d;
        let top5_avg = *ovlp as f32 / d;
        let kl_avg = *kl as f32 / d;
        eprintln!("{:>5}  {:>9.1}%  {:>12.2}  {:>12.3}",
            li, top1_pct, top5_avg, kl_avg);
    }
    eprintln!();
    eprintln!("Interpretation for Spec Decode:");
    eprintln!("  top1_agree >= 80%  → that layer depth is a viable draft cutoff");
    eprintln!("  top1_agree  < 50%  → drafts at that depth are unreliable");
    eprintln!();
    eprintln!("Interpretation for Adaptive Quant:");
    eprintln!("  layers after which KL drops sharply are 'commit points' —");
    eprintln!("  pre-commit layers tolerate aggressive quant (small KL budget left)");

    // ── 5. KV-cache entropy + norm distribution ─────────────────────
    eprintln!();
    eprintln!("╔═══ 5. KV-CACHE PER-LAYER STATS (post-final-prompt) ═══╗");
    eprintln!();
    eprintln!("Cache state: after prompt {} (attn_len varies by SWA vs global)", PROMPTS.len() - 1);
    eprintln!();
    let kv_stats_vec = kv_stats::summarize(engine.kv_cache());
    eprint!("{}", kv_stats::format_report(&kv_stats_vec));

    // Aggregates: flag the outlier layers for Adaptive Quant candidates.
    let non_shared: Vec<_> = kv_stats_vec.iter().filter(|s| !s.shared && s.attn_len > 0).collect();
    if !non_shared.is_empty() {
        let mean_k_entropy: f32 = non_shared.iter().map(|s| s.k_norm_entropy).sum::<f32>() / non_shared.len() as f32;
        let mean_v_entropy: f32 = non_shared.iter().map(|s| s.v_norm_entropy).sum::<f32>() / non_shared.len() as f32;
        eprintln!();
        eprintln!("Aggregate: mean K-norm entropy = {:.1}%, mean V-norm entropy = {:.1}%",
            mean_k_entropy * 100.0, mean_v_entropy * 100.0);
        eprintln!();
        let mut k_sorted: Vec<_> = non_shared.iter().collect();
        k_sorted.sort_by(|a, b| a.k_norm_entropy.partial_cmp(&b.k_norm_entropy).unwrap_or(std::cmp::Ordering::Equal));
        eprintln!("Lowest K-entropy layers (focused attention — precision matters):");
        for s in k_sorted.iter().take(5) {
            eprintln!("  L{:<2} K_H={:.1}%  K max/min={:.2}  heads={} hd={}",
                s.layer, s.k_norm_entropy * 100.0,
                s.k_max_norm / s.k_min_norm.max(1e-6),
                s.n_kv_heads, s.head_dim);
        }
        eprintln!();
        eprintln!("Highest K-entropy layers (diffuse attention — quant-tolerant):");
        for s in k_sorted.iter().rev().take(5) {
            eprintln!("  L{:<2} K_H={:.1}%  K max/min={:.2}  heads={} hd={}",
                s.layer, s.k_norm_entropy * 100.0,
                s.k_max_norm / s.k_min_norm.max(1e-6),
                s.n_kv_heads, s.head_dim);
        }
    }
    eprintln!();
    eprintln!("Interpretation:");
    eprintln!("  K-norm entropy ~100% = K magnitudes uniform across positions → diffuse attention");
    eprintln!("  K-norm entropy  <70% = a few K positions dominate → focused, precision-sensitive");
    eprintln!("  high max/min ratio   = outliers present → quant-group headroom should not clip them");

    assert!(total_probes > 0, "no probes completed — check snapshot capture");
}

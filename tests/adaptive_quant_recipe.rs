//! Produce a concrete adaptive-quant recipe from a 16-prompt calibration run.
//!
//! Runs the same calibration loop as `inference_telemetry.rs` but then pipes
//! the tracker snapshots through `adaptive_quant::{compute_profiles,
//! compute_sensitivity, generate_recipe}` to produce:
//!   1. Per-layer sensitivity scores and recommended buckets
//!   2. Per-tensor delta table — what to upgrade / downgrade and by how much
//!   3. Total bandwidth savings estimate
//!
//! Run: cargo test --release --test adaptive_quant_recipe -- --ignored --nocapture

use olorin::inference::{activation_track, adaptive_quant};
use olorin::inference::generate::{Engine, GenEvent};
use olorin::runes;
use std::path::Path;

const PROMPTS: &[&str] = &[
    "Here is a Python function that computes Fibonacci numbers:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    ",
    "Write a Rust function that reverses a string:\n\n```rust\nfn reverse_string(s: &str) -> String {\n    ",
    "Example of a binary search in C:\n\n```c\nint binary_search(int* arr, int n, int target) {\n    int lo = 0, hi = n - 1;\n    ",
    "SQL query to find the top 5 customers by total order value:\n\n```sql\nSELECT c.name, SUM(o.total) AS revenue\n",
    "JavaScript async function that fetches JSON from a URL:\n\n```js\nasync function fetchJson(url) {\n    ",
    "Bash script to recursively count .py files in a directory:\n\n```bash\n#!/usr/bin/env bash\n",
    "Go function that reads a file line by line:\n\n```go\nfunc readLines(path string) ([]string, error) {\n    ",
    "I love walking in the forest on a spring morning. The birds are singing and ",
    "My grandmother used to bake bread every Sunday. The smell would fill the whole house and ",
    "Hello! How are you doing today? I was wondering if you could help me with ",
    "She sat by the window, watching the rain fall on the cobblestone street. It reminded her of ",
    "The children laughed as they chased each other across the meadow. Their dog bounded along beside them, ",
    "Quantum entanglement refers to a physical phenomenon where pairs of particles ",
    "The algorithm runs in O(n log n) time because the divide-and-conquer approach ",
    "To prove that the square root of 2 is irrational, assume the opposite: that ",
    "In the context of distributed systems, the CAP theorem states that ",
];

const DECODE_TOKENS: usize = 32;

#[test]
#[ignore = "16-prompt calibration, ~2 min; use --ignored --nocapture"]
fn generate_adaptive_quant_recipe() {
    let home = std::env::var("HOME").unwrap();
    let path = Path::new(&home).join(".olorin/models/gemma-4-e2b-it-Q4_K_M.gguf");
    if !path.exists() {
        eprintln!("SKIP: no model at {}", path.display());
        return;
    }
    std::env::set_var("OLORIN_ACTIVATION_TRACK", "1");
    std::env::set_var("OLORIN_RESIDUAL_TRACK", "1");
    std::env::set_var("OLORIN_ACTIVATION_DOMAIN", "calibration");

    let mut engine = Box::new(Engine::load(&path, 1024).expect("load"));
    engine.temperature = 0.0;
    engine.max_tokens = DECODE_TOKENS;
    let system = runes::runes_prompt_block();
    let on_event = |_: GenEvent| {};

    eprintln!("[recipe] calibration: {} prompts × {} decode tokens",
        PROMPTS.len(), DECODE_TOKENS);
    for p in PROMPTS {
        engine.generate(p, system, &on_event).expect("gen");
    }

    // ── Pull tracker snapshots ───────────────────────────────────────
    let mean_abs = activation_track::per_layer_mean_abs();
    let max_abs = activation_track::per_layer_max_abs();
    let residual = activation_track::residual_norms();
    let samples = activation_track::per_layer_samples();

    assert!(!mean_abs.is_empty(), "tracker empty — no FFN samples captured");
    assert!(!residual.is_empty(), "tracker empty — no residual samples captured");
    eprintln!("[recipe] captured: {} FFN-layer records ({} samples), {} residual-layer records ({} samples)",
        mean_abs.len(), samples.first().copied().unwrap_or(0),
        residual.len(), residual.first().map(|v| v.len()).unwrap_or(0));

    // ── Profile + sensitivity + recipe ───────────────────────────────
    let profiles = adaptive_quant::compute_profiles(&mean_abs, &max_abs, &residual);
    let sensitivity = adaptive_quant::compute_sensitivity(&profiles);
    let recipe = adaptive_quant::generate_recipe(engine.model(), &sensitivity);

    // ── Section 1: layer profiles ────────────────────────────────────
    eprintln!();
    eprintln!("╔═══ 1. LAYER PROFILES ═══╗");
    eprintln!();
    eprintln!("{:>3}  {:>5}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}",
        "L", "ffn_d", "ffn_mean", "ffn_std", "ffn_max", "res_mean", "res_std", "res_Δ");
    eprintln!("{:-<72}", "");
    for p in &profiles {
        eprintln!("{:>3}  {:>5}  {:>8.3}  {:>8.3}  {:>8.2}  {:>8.2}  {:>8.2}  {:>+8.2}",
            p.layer, p.ffn_dim, p.ffn_mean, p.ffn_std, p.ffn_max,
            p.residual_mean, p.residual_std, p.residual_delta);
    }

    // ── Section 2: sensitivity + bucket ──────────────────────────────
    eprintln!();
    eprintln!("╔═══ 2. SENSITIVITY + RECOMMENDED BUCKETS ═══╗");
    eprintln!();
    eprint!("{}", adaptive_quant::format_recipe(&sensitivity, &recipe));

    // ── Section 3: APPLY THESE — downgrades only (pure bandwidth wins) ──
    let downgrades = adaptive_quant::filter_downgrades(&recipe);
    eprintln!();
    eprintln!("╔═══ 3. APPLY — DOWNGRADE-ONLY RECIPE (safe bandwidth wins) ═══╗");
    eprintln!();
    if downgrades.is_empty() {
        eprintln!("(no tensors flagged for safe downgrade)");
    } else {
        eprint!("{}", adaptive_quant::format_recipe_deltas(&downgrades));
    }

    // ── Section 4: REVIEW — upgrade candidates (FYI only) ───────────
    let upgrades = adaptive_quant::filter_upgrade_candidates(&recipe);
    eprintln!();
    eprintln!("╔═══ 4. REVIEW — POTENTIAL QUALITY-UPGRADE CANDIDATES ═══╗");
    eprintln!();
    eprintln!("These are tensors llama.cpp quantized MORE aggressively than our");
    eprintln!("sensitivity signal suggests. Applying them adds bandwidth for");
    eprintln!("quality — do NOT bundle with downgrades unless you want both.");
    eprintln!();
    if upgrades.is_empty() {
        eprintln!("(none — llama.cpp's aggressive choices all match or exceed our signal)");
    } else {
        eprint!("{}", adaptive_quant::format_recipe_deltas(&upgrades));
    }

    // ── Section 5: summary ──────────────────────────────────────────
    let downgrade_bytes = adaptive_quant::downgrade_savings_bytes(&recipe);
    let upgrade_bytes: i64 = upgrades.iter().map(|r| r.bytes_delta).sum();
    eprintln!();
    eprintln!("╔═══ 5. SUMMARY ═══╗");
    eprintln!();
    let model_bytes = 1423u64 * 1_048_576; // ~1.423 GB Gemma 4 E2B Q4_K_M baseline
    eprintln!("If you apply ONLY downgrades:");
    eprintln!("  Tensors changed:     {}", downgrades.len());
    eprintln!("  Bandwidth saved:     {:.2} MB  ({:.2}% of model)",
        downgrade_bytes as f64 / 1_048_576.0,
        100.0 * downgrade_bytes as f64 / model_bytes as f64);
    eprintln!();
    eprintln!("If you ALSO apply upgrades (quality boost):");
    eprintln!("  Tensors changed:     {}", upgrades.len());
    eprintln!("  Extra bandwidth:     {:.2} MB added",
        upgrade_bytes as f64 / 1_048_576.0);
    eprintln!();
    let bucket_counts = sensitivity.iter().fold([0usize; 3], |mut acc, s| {
        match s.bucket {
            adaptive_quant::QuantBucket::Q4K => acc[0] += 1,
            adaptive_quant::QuantBucket::Q5K => acc[1] += 1,
            adaptive_quant::QuantBucket::Q6K => acc[2] += 1,
        }
        acc
    });
    eprintln!("Sensitivity bucket distribution: Q4K={}  Q5K={}  Q6K={}",
        bucket_counts[0], bucket_counts[1], bucket_counts[2]);

    // ── Section 6: emit llama-quantize recipe file ──────────────────
    let out_path = std::env::temp_dir().join("olorin_adaptive_recipe.txt");
    let body = adaptive_quant::format_llamacpp_tensor_types(&downgrades);
    std::fs::write(&out_path, &body).expect("write recipe file");
    eprintln!();
    eprintln!("╔═══ 6. LLAMA-QUANTIZE RECIPE FILE ═══╗");
    eprintln!();
    eprintln!("Wrote {} downgrade lines to: {}", downgrades.len(), out_path.display());
    eprintln!();
    eprintln!("To apply:");
    eprintln!("  ~/llama.cpp/build/bin/llama-quantize \\");
    eprintln!("    --allow-requantize \\");
    eprintln!("    --tensor-type-file {} \\", out_path.display());
    eprintln!("    <input>.gguf <output>.gguf COPY");
    eprintln!();
    eprintln!("First 5 lines of recipe:");
    for line in body.lines().take(5) {
        eprintln!("  {line}");
    }

    // Sanity asserts
    assert_eq!(sensitivity.len(), profiles.len());
    assert!(recipe.len() >= sensitivity.len() * 3, "recipe too sparse");
}

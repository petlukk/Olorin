//! Produce a concrete adaptive-quant recipe from a 36-prompt calibration run.
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

// Domain-scoped calibration corpus: English + math + code. Deliberately excludes
// other natural languages so tensors specialized for them self-identify as
// low-magnitude and get flagged for downgrade.
const PROMPTS: &[&str] = &[
    // --- Code (14) ---
    "Here is a Python function that computes Fibonacci numbers:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    ",
    "Write a Rust function that reverses a string:\n\n```rust\nfn reverse_string(s: &str) -> String {\n    ",
    "Example of a binary search in C:\n\n```c\nint binary_search(int* arr, int n, int target) {\n    int lo = 0, hi = n - 1;\n    ",
    "SQL query to find the top 5 customers by total order value:\n\n```sql\nSELECT c.name, SUM(o.total) AS revenue\n",
    "JavaScript async function that fetches JSON from a URL:\n\n```js\nasync function fetchJson(url) {\n    ",
    "Bash script to recursively count .py files in a directory:\n\n```bash\n#!/usr/bin/env bash\n",
    "Go function that reads a file line by line:\n\n```go\nfunc readLines(path string) ([]string, error) {\n    ",
    "React custom hook for debounced input:\n\n```ts\nfunction useDebounce<T>(value: T, delay: number): T {\n    ",
    "Dockerfile for a Rust binary with a multi-stage build:\n\n```dockerfile\nFROM rust:1 AS builder\n",
    "Python function to validate an email address with a regex:\n\n```python\nimport re\ndef is_valid_email(addr: str) -> bool:\n    pattern = r\"",
    "CMakeLists.txt for a C++ library with tests enabled:\n\n```cmake\ncmake_minimum_required(VERSION 3.15)\nproject(mylib CXX)\n",
    "Rust implementation of quicksort on a mutable slice:\n\n```rust\nfn quicksort(v: &mut [i32]) {\n    ",
    "Go worker pattern: goroutines reading jobs from a channel:\n\n```go\nfunc worker(id int, jobs <-chan int, results chan<- int) {\n    ",
    "C function wrapping mmap for read-only file mapping:\n\n```c\nvoid* map_readonly(const char* path, size_t* out_len) {\n    int fd = open(path, O_RDONLY);\n    ",
    // --- Math (14) ---
    "Quantum entanglement refers to a physical phenomenon where pairs of particles ",
    "The algorithm runs in O(n log n) time because the divide-and-conquer approach ",
    "To prove that the square root of 2 is irrational, assume the opposite: that ",
    "In the context of distributed systems, the CAP theorem states that ",
    "A train travels 120 km in 1.5 hours, then another 200 km in 2 hours. The average speed over the entire trip is ",
    "To solve the quadratic equation 3x^2 - 7x + 2 = 0 we apply the quadratic formula: x = (7 ± sqrt(",
    "We prove by induction that 1 + 2 + ... + n = n(n+1)/2. Base case n=1: the left side is 1 and the right side is ",
    "The derivative of f(x) = x^3 * cos(x) with respect to x is computed using the product rule: f'(x) = ",
    "To evaluate the integral of 1/(x^2 + 1) dx we recognize this as the derivative of arctan, so the result is ",
    "Multiplying the 2x3 matrix A=[[1,2,3],[4,5,6]] by the 3x2 matrix B=[[7,8],[9,10],[11,12]] gives a 2x2 result. The (0,0) entry is 1*7 + 2*9 + 3*11 = ",
    "To find the eigenvalues of the matrix [[4,1],[2,3]] we solve det(A - λI) = 0, which gives the characteristic polynomial ",
    "A medical test is 95% accurate and the disease affects 1% of the population. Given a positive test result, the probability the patient actually has the disease (by Bayes' theorem) is ",
    "The number of ways to choose 3 items from a set of 10 distinct items, order not mattering, is C(10,3) = 10! / (3! * 7!) = ",
    "To check whether 97 is prime, we test divisibility by primes up to sqrt(97) ≈ 9.85. Checking 2, 3, 5, 7 in turn: ",
    // --- English (8) ---
    "I love walking in the forest on a spring morning. The birds are singing and ",
    "My grandmother used to bake bread every Sunday. The smell would fill the whole house and ",
    "Hello! How are you doing today? I was wondering if you could help me with ",
    "She sat by the window, watching the rain fall on the cobblestone street. It reminded her of ",
    "The children laughed as they chased each other across the meadow. Their dog bounded along beside them, ",
    "Q: What is the largest moon of Saturn and what is unusual about it?\nA: The largest moon of Saturn is Titan, which is unusual because ",
    "How to brew a good cup of French press coffee: first, grind fresh whole beans to a coarse consistency. Then, ",
    "\"I can't believe you forgot my birthday,\" she said, crossing her arms. \"I didn't forget,\" he replied, ",
];

const DECODE_TOKENS: usize = 32;

#[test]
#[ignore = "36-prompt calibration, ~4-5 min; use --ignored --nocapture"]
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
    engine.temperature = std::env::var("OLORIN_CALIBRATION_TEMP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    engine.max_tokens = DECODE_TOKENS;
    eprintln!("[recipe] temperature = {}", engine.temperature);
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

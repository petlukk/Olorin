//! Hardened code-vs-text domain diff experiment.
//!
//! Runs a multi-prompt calibration set per domain (prefill + decode taps),
//! then reports per-layer bottom-30% overlap under two metrics:
//!   * mean_abs — average contribution (bandwidth accounting)
//!   * max_abs  — peak contribution (prune-safety check)
//!
//! Safe-prune candidates are neurons low on BOTH metrics in BOTH domains.
//!
//! Run: cargo test --release --test activation_domain_diff -- --ignored --nocapture

use olorin::inference::activation_track;
use olorin::inference::generate::{Engine, GenEvent};
use std::path::Path;

const BOTTOM_FRAC: f32 = 0.30;
const MAX_TOKENS: usize = 32;

const CODE_PROMPTS: &[&str] = &[
    "Here is a Python function that computes Fibonacci numbers:\n\n```python\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    ",
    "Write a Rust function that reverses a string:\n\n```rust\nfn reverse_string(s: &str) -> String {\n    ",
    "Example of a binary search in C:\n\n```c\nint binary_search(int* arr, int n, int target) {\n    int lo = 0, hi = n - 1;\n    ",
    "JavaScript async function that fetches JSON from a URL:\n\n```js\nasync function fetchJson(url) {\n    ",
    "SQL query to find the top 5 customers by total order value:\n\n```sql\nSELECT c.name, SUM(o.total) AS revenue\n",
    "Python class that implements a simple LRU cache:\n\n```python\nclass LRUCache:\n    def __init__(self, capacity):\n        ",
    "Bash script to recursively count .py files in a directory:\n\n```bash\n#!/usr/bin/env bash\n",
    "Go function that reads a file line by line:\n\n```go\nfunc readLines(path string) ([]string, error) {\n    ",
];

const TEXT_PROMPTS: &[&str] = &[
    "I love walking in the forest on a spring morning. The birds are singing and ",
    "My grandmother used to bake bread every Sunday. The smell would fill the whole house and ",
    "The coffee shop on the corner had the best croissants in the city. Every morning I would ",
    "When winter finally came, the mountains were covered in a thick blanket of snow, and ",
    "She sat by the window, watching the rain fall on the cobblestone street. It reminded her of ",
    "The children laughed as they chased each other across the meadow. Their dog bounded along beside them, ",
    "He opened the old wooden box his father had left him. Inside, wrapped in yellowed paper, was ",
    "Every evening she read a few pages from her favorite novel, savoring each sentence as if ",
];

#[test]
#[ignore = "multi-prompt calibration, ~2 min; use --ignored --nocapture"]
fn code_vs_text_dual_metric_overlap() {
    let home = std::env::var("HOME").unwrap();
    let path = Path::new(&home).join(".olorin/models/gemma-4-e2b-it-Q4_K_M.gguf");
    if !path.exists() {
        eprintln!("SKIP: no model at {}", path.display());
        return;
    }
    std::env::set_var("OLORIN_ACTIVATION_TRACK", "1");
    std::env::set_var("OLORIN_ACTIVATION_DOMAIN", "code");

    let mut engine = Box::new(Engine::load(&path, 1024).expect("load"));
    engine.temperature = 0.0;
    engine.max_tokens = MAX_TOKENS;

    let on_event = |_: GenEvent| {};

    // ── Code domain run ──────────────────────────────────────────────
    eprintln!("[diff] code domain: {} prompts × {} decode-tokens + prefill",
        CODE_PROMPTS.len(), MAX_TOKENS);
    for p in CODE_PROMPTS {
        engine.generate(p, "", &on_event).expect("code gen");
    }
    let code_mean = activation_track::per_layer_mean_abs();
    let code_max = activation_track::per_layer_max_abs();
    let code_samples = activation_track::per_layer_samples();
    assert!(!code_mean.is_empty(), "code snapshot empty");

    // ── Text domain run ──────────────────────────────────────────────
    activation_track::reset("text");
    eprintln!("[diff] text domain: {} prompts × {} decode-tokens + prefill",
        TEXT_PROMPTS.len(), MAX_TOKENS);
    for p in TEXT_PROMPTS {
        engine.generate(p, "", &on_event).expect("text gen");
    }
    let text_mean = activation_track::per_layer_mean_abs();
    let text_max = activation_track::per_layer_max_abs();
    let text_samples = activation_track::per_layer_samples();
    assert_eq!(code_mean.len(), text_mean.len(), "layer count mismatch");

    eprintln!();
    eprintln!("Samples per layer — code: {:?}  text: {:?}",
        &code_samples[..3.min(code_samples.len())],
        &text_samples[..3.min(text_samples.len())]);
    eprintln!();

    // ── Per-layer dual-metric overlap ────────────────────────────────
    eprintln!("Per-layer bottom-{:.0}% overlap, two metrics + safe-prune joint set:",
        BOTTOM_FRAC * 100.0);
    eprintln!("{:>3}  {:>5}  {:>7}  {:>7}  {:>8}",
        "L", "ffn", "mean_%", "max_%", "safe_%");
    eprintln!("{:-<40}", "");

    let mut mean_overlaps = Vec::with_capacity(code_mean.len());
    let mut max_overlaps = Vec::with_capacity(code_mean.len());
    let mut safe_fractions = Vec::with_capacity(code_mean.len());
    for li in 0..code_mean.len() {
        let ffn = code_mean[li].len();
        let k = ((ffn as f32) * BOTTOM_FRAC) as usize;

        let cm_bot = bottom_k_indices(&code_mean[li], k);
        let tm_bot = bottom_k_indices(&text_mean[li], k);
        let mean_overlap = intersect_size(&cm_bot, &tm_bot);

        let cmax_bot = bottom_k_indices(&code_max[li], k);
        let tmax_bot = bottom_k_indices(&text_max[li], k);
        let max_overlap = intersect_size(&cmax_bot, &tmax_bot);

        // Safe-prune set: low on mean_abs AND max_abs in BOTH domains.
        // Intersect the mean-overlap set with the max-overlap set.
        let mean_joint = intersect_sets(&cm_bot, &tm_bot);
        let max_joint = intersect_sets(&cmax_bot, &tmax_bot);
        let safe = intersect_size(&mean_joint, &max_joint);

        let mean_pct = 100.0 * mean_overlap as f32 / k as f32;
        let max_pct = 100.0 * max_overlap as f32 / k as f32;
        let safe_pct = 100.0 * safe as f32 / ffn as f32;
        mean_overlaps.push(mean_pct);
        max_overlaps.push(max_pct);
        safe_fractions.push(safe_pct);
        eprintln!("{:>3}  {:>5}  {:>6.1}%  {:>6.1}%  {:>7.2}%",
            li, ffn, mean_pct, max_pct, safe_pct);
    }

    // ── Summary ──────────────────────────────────────────────────────
    let mean_all = avg(&mean_overlaps);
    let max_all = avg(&max_overlaps);
    let safe_all = avg(&safe_fractions);
    let split = 15;
    let mean_light = avg(&mean_overlaps[..split]);
    let mean_heavy = avg(&mean_overlaps[split..]);
    let max_light = avg(&max_overlaps[..split]);
    let max_heavy = avg(&max_overlaps[split..]);
    let safe_light = avg(&safe_fractions[..split]);
    let safe_heavy = avg(&safe_fractions[split..]);

    eprintln!();
    eprintln!("Summary:");
    eprintln!("  Metric                 All     L00-14 (6144)   L15-34 (12288)");
    eprintln!("  mean bottom-30% ovlp:  {:>5.1}%  {:>5.1}%           {:>5.1}%",
        mean_all, mean_light, mean_heavy);
    eprintln!("  max  bottom-30% ovlp:  {:>5.1}%  {:>5.1}%           {:>5.1}%",
        max_all, max_light, max_heavy);
    eprintln!("  safe-prune %-of-layer: {:>5.2}%  {:>5.2}%           {:>5.2}%",
        safe_all, safe_light, safe_heavy);
    eprintln!();
    eprintln!("Safe-prune = neurons in bottom-30% of BOTH mean AND max, in BOTH domains.");
    eprintln!("This is the honest upper-bound for zero-risk static pruning.");

    // ── L17 / L18 specialist drill-down ──────────────────────────────
    const TARGETS: &[usize] = &[17, 18];
    const TOP_K: usize = 100;
    const BLOCK_SIZE: usize = 256; // Q4K native block
    const EPS: f32 = 1e-6;

    let mut l17_code_top: Vec<usize> = Vec::new();
    let mut l18_code_top: Vec<usize> = Vec::new();
    let mut l17_text_top: Vec<usize> = Vec::new();
    let mut l18_text_top: Vec<usize> = Vec::new();

    for &li in TARGETS {
        let n = code_mean[li].len();
        eprintln!();
        eprintln!("=== L{} drill (ffn={}, code_samples={}, text_samples={}) ===",
            li, n, code_samples[li], text_samples[li]);

        // Per-neuron log2 ratio = log2(code_mean / text_mean). Positive = code-biased.
        let ratios: Vec<f32> = (0..n).map(|i| {
            ((code_mean[li][i] + EPS) / (text_mean[li][i] + EPS)).log2()
        }).collect();

        // Histogram of log2 ratio.
        eprintln!("Ratio distribution (log2 code_mean / text_mean):");
        let buckets: &[(f32, f32, &str)] = &[
            (f32::NEG_INFINITY, -3.0, "< -3.0      "),
            (-3.0, -2.0,               "-3 to -2    "),
            (-2.0, -1.0,               "-2 to -1    "),
            (-1.0, -0.5,               "-1 to -0.5  "),
            (-0.5,  0.5,               "-0.5 to 0.5 "),
            ( 0.5,  1.0,               " 0.5 to  1  "),
            ( 1.0,  2.0,               " 1 to  2    "),
            ( 2.0,  3.0,               " 2 to  3    "),
            ( 3.0, f32::INFINITY,      "> 3.0       "),
        ];
        for &(lo, hi, label) in buckets {
            let c = ratios.iter().filter(|&&r| r >= lo && r < hi).count();
            let bar = "█".repeat((c * 60 / n.max(1)).max(if c > 0 { 1 } else { 0 }));
            eprintln!("  {}  {:>5}   {}", label, c, bar);
        }

        // Separation counts.
        let sep_1 = ratios.iter().filter(|&&r| r.abs() > 1.0).count();
        let sep_2 = ratios.iter().filter(|&&r| r.abs() > 2.0).count();
        eprintln!("High-separation neurons:");
        eprintln!("  |log2 ratio| > 1 (ratio > 2×):  {:>5} / {} ({:.2}%)",
            sep_1, n, 100.0 * sep_1 as f32 / n as f32);
        eprintln!("  |log2 ratio| > 2 (ratio > 4×):  {:>5} / {} ({:.2}%)",
            sep_2, n, 100.0 * sep_2 as f32 / n as f32);

        // Top-K code specialists: highest log2 ratio (positive direction).
        let mut code_top = top_k_by(&ratios, TOP_K, |a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        code_top.sort_unstable();
        // Top-K text specialists: highest -log2 ratio = most negative.
        let mut text_top = top_k_by(&ratios, TOP_K, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        text_top.sort_unstable();

        report_block_coverage("Top-100 code specialists", &code_top, n, BLOCK_SIZE);
        report_block_coverage("Top-100 text specialists", &text_top, n, BLOCK_SIZE);

        if li == 17 { l17_code_top = code_top; l17_text_top = text_top; }
        else        { l18_code_top = code_top; l18_text_top = text_top; }
    }

    // ── Cross-layer stability ────────────────────────────────────────
    if !l17_code_top.is_empty() && !l18_code_top.is_empty() {
        assert_eq!(code_mean[17].len(), code_mean[18].len(),
            "L17 and L18 widths differ — index comparison meaningless");
        let code_stable = intersect_size(&l17_code_top, &l18_code_top);
        let text_stable = intersect_size(&l17_text_top, &l18_text_top);
        eprintln!();
        eprintln!("=== L17 ∩ L18 stability (top-{}) ===", TOP_K);
        eprintln!("Code specialists present in BOTH layers: {:>3} / {}  ({:.1}%)",
            code_stable, TOP_K, 100.0 * code_stable as f32 / TOP_K as f32);
        eprintln!("Text specialists present in BOTH layers: {:>3} / {}  ({:.1}%)",
            text_stable, TOP_K, 100.0 * text_stable as f32 / TOP_K as f32);
        eprintln!();
        eprintln!("Interpretation: high stability (>40) = persistent code/text pathway");
        eprintln!("                low stability (<15)  = layer-local specialization");
    }

    assert!(mean_all > 5.0 && mean_all < 95.0, "mean overlap sanity: {}%", mean_all);
}

fn top_k_by<F>(v: &[f32], k: usize, mut cmp: F) -> Vec<usize>
where F: FnMut(&f32, &f32) -> std::cmp::Ordering
{
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| cmp(&v[a], &v[b]));
    idx.truncate(k);
    idx
}

fn report_block_coverage(label: &str, specialists: &[usize], n: usize, block_size: usize) {
    let n_blocks = (n + block_size - 1) / block_size;
    let mut per_block = vec![0usize; n_blocks];
    for &i in specialists {
        per_block[i / block_size] += 1;
    }
    let pure_skip = per_block.iter().filter(|&&c| c == block_size).count();
    let mixed = per_block.iter().filter(|&&c| c > 0 && c < block_size).count();
    let pure_keep = per_block.iter().filter(|&&c| c == 0).count();
    let max_in_block = *per_block.iter().max().unwrap_or(&0);

    // Largest contiguous run of specialists.
    let mut sorted = specialists.to_vec();
    sorted.sort_unstable();
    let mut longest_run = 0usize;
    let mut cur_run = 0usize;
    let mut prev: Option<usize> = None;
    for i in &sorted {
        match prev {
            Some(p) if p + 1 == *i => cur_run += 1,
            _                      => cur_run = 1,
        }
        longest_run = longest_run.max(cur_run);
        prev = Some(*i);
    }

    // Median gap between consecutive specialists.
    let mut gaps: Vec<usize> = sorted.windows(2).map(|w| w[1] - w[0]).collect();
    gaps.sort_unstable();
    let median_gap = if gaps.is_empty() { 0 } else { gaps[gaps.len() / 2] };

    eprintln!("{} over {}-sized blocks (n_blocks={}):", label, block_size, n_blocks);
    eprintln!("  pure-skip blocks (all {} specialists): {}", block_size, pure_skip);
    eprintln!("  mixed blocks:                          {}", mixed);
    eprintln!("  pure-keep blocks:                      {}", pure_keep);
    eprintln!("  max specialists in any single block:   {}", max_in_block);
    eprintln!("  longest contiguous specialist run:     {}", longest_run);
    eprintln!("  median gap between specialists:        {}", median_gap);
}

fn bottom_k_indices(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    idx.truncate(k);
    idx.sort_unstable();
    idx
}

fn intersect_size(a: &[usize], b: &[usize]) -> usize {
    let (mut i, mut j, mut n) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => { n += 1; i += 1; j += 1; }
            std::cmp::Ordering::Less  => { i += 1; }
            std::cmp::Ordering::Greater => { j += 1; }
        }
    }
    n
}

fn intersect_sets(a: &[usize], b: &[usize]) -> Vec<usize> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Equal => { out.push(a[i]); i += 1; j += 1; }
            std::cmp::Ordering::Less  => { i += 1; }
            std::cmp::Ordering::Greater => { j += 1; }
        }
    }
    out
}

fn avg(v: &[f32]) -> f32 {
    if v.is_empty() { 0.0 } else { v.iter().sum::<f32>() / v.len() as f32 }
}

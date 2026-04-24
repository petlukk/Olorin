//! Throughput, latency, and memory benchmark for olorin's Gemma 4 forward pass.
//!
//! Mirrors llama.cpp's `llama-batched` reporting (load / prompt-eval / eval /
//! total) so the two engines can be compared apples-to-apples on the same gguf
//! file. Adds peak RSS, time-to-first-token, and a per-step latency curve so
//! the impact of growing attn_len on the per-token cost is visible.
//!
//! Run: cargo test --release --test bench_decode_speed -- --nocapture

use std::path::Path;
use std::time::Instant;

const N_DECODE: usize = 64;
const PROMPT: &str = "Write a long story about a robot:";

fn model_path() -> String {
    if let Ok(p) = std::env::var("OLORIN_MODEL_PATH") {
        return p;
    }
    let home = std::env::var("HOME").unwrap();
    format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i as u32
}

/// Read a `Vm*` field from /proc/self/status, returning kilobytes.
/// Linux-only; tests on non-Linux platforms will simply report 0.
fn proc_status_kb(field: &str) -> u64 {
    let Ok(text) = std::fs::read_to_string("/proc/self/status") else { return 0 };
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(field) {
            // Format: "VmRSS:\t   12345 kB"
            let rest = rest.trim_start_matches(':').trim();
            let kb: u64 = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            return kb;
        }
    }
    0
}

fn rss_mb() -> f64 {
    proc_status_kb("VmRSS") as f64 / 1024.0
}

fn peak_rss_mb() -> f64 {
    proc_status_kb("VmHWM") as f64 / 1024.0
}

/// Read /proc/self/stat utime + stime in clock ticks. Field 14 = utime,
/// field 15 = stime. Linux clock tick is 100 Hz (CLK_TCK) on every distro
/// I've ever seen, including Arch.
fn proc_cpu_seconds() -> f64 {
    let Ok(text) = std::fs::read_to_string("/proc/self/stat") else { return 0.0 };
    // The 2nd field is "comm" wrapped in parentheses and may contain spaces;
    // skip past the closing ')' and split the rest.
    let Some(after_comm) = text.rsplitn(2, ')').next() else { return 0.0 };
    let fields: Vec<&str> = after_comm.split_whitespace().collect();
    // After splitting from after `) `, indices shift: original field 14 (utime)
    // becomes index 14 - 3 = 11 (we lost pid, comm, state — three fields).
    let utime: u64 = fields.get(11).and_then(|s| s.parse().ok()).unwrap_or(0);
    let stime: u64 = fields.get(12).and_then(|s| s.parse().ok()).unwrap_or(0);
    (utime + stime) as f64 / 100.0
}

/// Estimate KV-cache resident bytes from public model fields.
/// SWA layers cap at sliding_window; global layers grow up to max_seq_len.
/// Layers with kv_shared_source reuse earlier layers' buffers (no alloc).
fn kv_cache_bytes(
    model: &olorin::inference::engine::Gemma4Model,
    max_seq_len: usize,
) -> usize {
    let mut total = 0usize;
    for il in 0..model.n_layers {
        if model.kv_shared_source[il].is_some() {
            continue; // shared, no own allocation
        }
        let capacity = if model.is_swa[il] {
            model.sliding_window.min(max_seq_len)
        } else {
            max_seq_len
        };
        // K and V buffers, each n_kv_heads * head_dim_v * capacity * sizeof(u16=f16)
        let per_buf = capacity * model.n_kv_heads * model.head_dim_v[il] * 2;
        total += per_buf * 2;
    }
    total
}

#[test]
fn olorin_full_bench() {
    if !Path::new(&model_path()).exists() {
        eprintln!("SKIP: model not present");
        return;
    }

    let rss_before = rss_mb();

    // ── Load ──────────────────────────────────────────────────────────────
    let t_load = Instant::now();
    let gguf = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    let tokenizer = olorin::inference::tokenizer::Tokenizer::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();
    let graph_pool = olorin::inference::threadpool::GraphPool::new();
    let mut state = olorin::inference::forward::Gemma4State::new(&model, 512, &graph_pool);
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;

    let rss_after_load = rss_mb();

    // ── Tokenize prompt ───────────────────────────────────────────────────
    // Olorin's tokenizer doesn't auto-prepend BOS — we add it explicitly
    // to match llama.cpp's `<bos>` framing.
    let mut prompt_ids: Vec<u32> = vec![2]; // BOS
    prompt_ids.extend(tokenizer.encode(PROMPT));
    if let Some(n) = std::env::var("OLORIN_PROMPT_TOKENS").ok().and_then(|s| s.trim().parse::<usize>().ok()) {
        prompt_ids.truncate(n);
        while prompt_ids.len() < n { prompt_ids.push(2); }
    }
    let n_prompt = prompt_ids.len();

    // ── Prefill (prompt eval) — batched forward ────────────────────────
    let t_prefill = Instant::now();
    let logits = state.forward_batch(&model, &prompt_ids, &graph_pool);
    let mut next = argmax(logits);
    let prefill_secs = t_prefill.elapsed().as_secs_f64();
    let prefill_tps = n_prompt as f64 / prefill_secs;
    let prefill_ms_per_tok = prefill_secs * 1000.0 / n_prompt as f64;

    // ── Decode (eval) ────────────────────────────────────────────────────
    let mut step_ms: Vec<f64> = Vec::with_capacity(N_DECODE);

    let cpu_before_decode = proc_cpu_seconds();
    let t_ttft = Instant::now();
    {
        let logits = state.forward_one_graph(&model, next, &graph_pool);
        next = argmax(logits);
    }
    let ttft_ms = t_ttft.elapsed().as_secs_f64() * 1000.0;
    step_ms.push(ttft_ms);

    let t_decode = Instant::now();
    for _ in 1..N_DECODE {
        let t_step = Instant::now();
        let logits = state.forward_one_graph(&model, next, &graph_pool);
        next = argmax(logits);
        step_ms.push(t_step.elapsed().as_secs_f64() * 1000.0);
    }
    let decode_secs = t_ttft.elapsed().as_secs_f64();
    let decode_tps = N_DECODE as f64 / decode_secs;
    let decode_ms_per_tok = decode_secs * 1000.0 / N_DECODE as f64;
    let sustained_secs = t_decode.elapsed().as_secs_f64();
    let sustained_tps = (N_DECODE - 1) as f64 / sustained_secs;
    let cpu_after_decode = proc_cpu_seconds();
    let cpu_decode_secs = cpu_after_decode - cpu_before_decode;
    let cores_busy = if decode_secs > 0.0 { cpu_decode_secs / decode_secs } else { 0.0 };
    let parallel_eff = cores_busy / graph_pool.thread_count() as f64;

    let peak_rss = peak_rss_mb();
    let rss_after = rss_mb();
    let kv_bytes = kv_cache_bytes(&model, 512);
    let kv_mb = kv_bytes as f64 / (1024.0 * 1024.0);

    // ── Report ────────────────────────────────────────────────────────────
    eprintln!();
    eprintln!("=== olorin gemma 4 e2b q4_k_m benchmark ===");
    eprintln!("threads:                {}", graph_pool.thread_count());
    eprintln!("prompt:                 {:?}", PROMPT);
    eprintln!("prompt tokens:          {} (incl. BOS)", n_prompt);
    eprintln!("decode tokens:          {}", N_DECODE);
    eprintln!();
    eprintln!("load time:              {:>10.2} ms", load_ms);
    eprintln!("prompt eval time:       {:>10.2} ms / {} tok  ({:.2} ms/tok, {:.2} t/s)",
        prefill_secs * 1000.0, n_prompt, prefill_ms_per_tok, prefill_tps);
    eprintln!("eval time (incl TTFT):  {:>10.2} ms / {} tok  ({:.2} ms/tok, {:.2} t/s)",
        decode_secs * 1000.0, N_DECODE, decode_ms_per_tok, decode_tps);
    eprintln!("eval time (sustained):  {:>10.2} ms / {} tok  ({:.2} ms/tok, {:.2} t/s)",
        sustained_secs * 1000.0, N_DECODE - 1,
        sustained_secs * 1000.0 / (N_DECODE - 1) as f64, sustained_tps);
    eprintln!("time to first token:    {:>10.2} ms", ttft_ms);
    eprintln!();
    eprintln!("memory:");
    eprintln!("  rss before load:      {:>10.1} MB", rss_before);
    eprintln!("  rss after load:       {:>10.1} MB", rss_after_load);
    eprintln!("  rss after decode:     {:>10.1} MB", rss_after);
    eprintln!("  peak rss (VmHWM):     {:>10.1} MB", peak_rss);
    eprintln!("  model resident:       {:>10.1} MB  (rss after load - before)",
        rss_after_load - rss_before);
    eprintln!("  kv cache (computed):  {:>10.1} MB  (max_seq_len=512, f16, shared layers excluded)",
        kv_mb);
    eprintln!();
    eprintln!("cpu utilization (decode window):");
    eprintln!("  cpu time used:        {:>10.2} s   (utime + stime)", cpu_decode_secs);
    eprintln!("  wall time:            {:>10.2} s", decode_secs);
    eprintln!("  avg cores busy:       {:>10.2}     ({} threads available)",
        cores_busy, graph_pool.thread_count());
    eprintln!("  parallel efficiency:  {:>10.1} %", parallel_eff * 100.0);
    eprintln!();
    eprintln!("per-step latency curve (decode, ms):");
    let buckets = [0, N_DECODE / 8, N_DECODE / 4, N_DECODE / 2, 3 * N_DECODE / 4, N_DECODE - 1];
    for &i in &buckets {
        eprintln!("  step {:>3} (attn_len ~{:>3}): {:>7.2} ms",
            i, n_prompt + i, step_ms[i]);
    }
    let min_step = step_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_step = step_ms.iter().cloned().fold(0.0_f64, f64::max);
    eprintln!("  min / max:                 {:>7.2} / {:>7.2} ms", min_step, max_step);
    eprintln!();
    eprintln!("last sampled token id:  {next}");

    // ── Regression floor (opt-in, non-portable) ─────────────────────
    // Set OLORIN_DECODE_FLOOR_TPS=<N> to make this test fail if sustained
    // decode t/s drops below N. Values are hardware-specific — on Ryzen
    // 7 1700 with 8 physical cores we sustain ~12.9 t/s post-commit 5e3b922
    // (Q6K pre-d + SMT fix + ubatch), so a floor around 12.0 leaves a
    // ~7% headroom for noise before a true regression fires. Intended
    // for local perf gating, not for running unconditionally (would be
    // flaky on slower machines and on first-commit noise).
    if let Some(floor) = std::env::var("OLORIN_DECODE_FLOOR_TPS")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
    {
        assert!(
            sustained_tps >= floor,
            "decode regressed: sustained {sustained_tps:.2} t/s < floor {floor:.2} t/s \
             (prefill {prefill_tps:.2} t/s, parallel eff {:.1}% — check if a recent \
             change hurt decode, or lower the floor if hardware changed)",
            parallel_eff * 100.0,
        );
        eprintln!("decode floor OK: {sustained_tps:.2} ≥ {floor:.2}");
    }
}

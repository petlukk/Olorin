use super::Tool;
use async_trait::async_trait;
use std::time::Instant;

pub struct BenchTool;

#[async_trait]
impl Tool for BenchTool {
    fn name(&self) -> &str {
        "bench"
    }

    fn description(&self) -> &str {
        "Run a quick microbenchmark. Targets: safety, router, recall, vault, search, jl, fused, all."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": ["safety", "router", "recall", "vault", "search", "jl", "fused", "all"],
                    "description": "What to benchmark"
                }
            },
            "required": ["target"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> crate::error::Result<String> {
        let target = params["target"]
            .as_str()
            .ok_or_else(|| crate::error::Error::Tool("missing 'target' parameter".into()))?;

        match target {
            "safety" => bench_safety(),
            "router" => bench_router(),
            "recall" => bench_recall(),
            "vault" => bench_vault(),
            "search" => bench_search(),
            "jl" => bench_jl(),
            "fused" => bench_fused(),
            "all" => bench_all(),
            _ => Err(crate::error::Error::Tool(format!(
                "unknown target '{target}'"
            ))),
        }
    }
}

pub fn bench_safety() -> crate::error::Result<String> {
    use crate::safety::SafetyLayer;

    let input = "Hello, this is a normal user message for benchmarking purposes. ".repeat(16);
    let iterations = 10_000;

    let mut safety = SafetyLayer::with_capacity(input.len());

    // Warmup
    for _ in 0..100 {
        let _ = safety.scan_input(&input);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = safety.scan_input(&input);
    }
    let elapsed = start.elapsed();

    let per_call_ns = elapsed.as_nanos() as f64 / iterations as f64;
    let bytes_per_sec = input.len() as f64 * iterations as f64 / elapsed.as_secs_f64();
    let gb_per_sec = bytes_per_sec / 1e9;

    Ok(format!(
        "─── safety (fused_safety — SIMD byte classifier + leak + injection, single pass) ───\n  {} B input, {} iterations\n  Per call: {:.0} ns\n  Throughput: {:.2} GB/s\n  Total: {:.1} ms",
        input.len(),
        iterations,
        per_call_ns,
        gb_per_sec,
        elapsed.as_secs_f64() * 1000.0,
    ))
}

pub fn bench_recall() -> crate::error::Result<String> {
    use crate::recall::VectorStore;

    let corpus = [
        "SIMD vector optimization for ARM NEON and AVX-512 processing units and pipelines",
        "The Rust programming language guarantees memory safety through ownership",
        "Python machine learning with TensorFlow PyTorch and scikit-learn frameworks",
        "ChaCha20 is a fast stream cipher used in TLS and WireGuard protocols",
        "KV-cache compression reduces memory bandwidth requirements for LLM inference",
        "Johnson-Lindenstrauss lemma preserves pairwise distances under random projection",
        "Walsh-Hadamard transform is a fast orthogonal transformation used in signal processing",
        "ARM NEON provides 128-bit SIMD on mobile edge devices like Raspberry Pi",
        "Quantization reduces neural network model size with minimal quality degradation",
        "Cosine similarity measures the angle between high-dimensional embedding vectors",
    ];

    let queries = [
        "SIMD vector acceleration for CPUs",
        "Rust memory safety ownership",
        "stream cipher encryption TLS",
        "KV cache memory bandwidth LLM",
        "random projection distance preservation",
    ];

    // Benchmark insert: fill 1024-entry store
    let mut store = VectorStore::with_capacity(1024);
    let insert_iters = 10;

    // Warmup
    for text in &corpus {
        store.insert(text);
    }
    store.clear();

    let start = Instant::now();
    for _ in 0..insert_iters {
        store.clear();
        for i in 0..1024 {
            store.insert(corpus[i % corpus.len()]);
        }
    }
    let insert_elapsed = start.elapsed();
    let per_insert_us = insert_elapsed.as_micros() as f64 / (insert_iters * 1024) as f64;

    // Benchmark recall: search filled store
    let recall_iters = 1000;

    // Warmup
    for q in &queries {
        let _ = store.recall(q, 5);
    }

    let start = Instant::now();
    for _ in 0..recall_iters {
        for q in &queries {
            let _ = store.recall(q, 5);
        }
    }
    let recall_elapsed = start.elapsed();
    let total_recalls = recall_iters * queries.len();
    let per_recall_us = recall_elapsed.as_micros() as f64 / total_recalls as f64;

    let mem_per_vec = std::mem::size_of::<f32>() * crate::kernels::search::JL_DIM;
    let total_mem_kb = (mem_per_vec * 1024) as f64 / 1024.0;

    Ok(format!(
        "─── recall (JL 256→64, NEON batch_cosine, 1024 vecs = 256 KB) ───\n\
         Insert:\n  Per insert: {:.1} us\n  1024 inserts: {:.1} ms\n\
         Recall (top-5 from 1024):\n  Per recall: {:.1} us\n  {} recalls: {:.1} ms\n\
         Memory: {:.0} KB for 1024 vectors ({} B/vec)",
        per_insert_us,
        per_insert_us * 1024.0 / 1000.0,
        per_recall_us,
        total_recalls,
        recall_elapsed.as_secs_f64() * 1000.0,
        total_mem_kb,
        mem_per_vec,
    ))
}

pub fn bench_router() -> crate::error::Result<String> {
    use crate::kernels::command_router;

    let commands: &[&[u8]] = &[
        b"/help", b"/quit", b"/time", b"/calc 2+2", b"/shell ls",
        b"/read file.txt", b"/ls", b"/cpu", b"/json keys {}",
        b"hello world", b"/unknown", b"/tokens test",
    ];
    let iterations = 100_000;

    // Warmup
    for _ in 0..1000 {
        for cmd in commands {
            let _ = command_router::match_command_verified(cmd);
        }
    }

    let start = Instant::now();
    for _ in 0..iterations {
        for cmd in commands {
            let _ = command_router::match_command_verified(cmd);
        }
    }
    let elapsed = start.elapsed();

    let total_calls = iterations * commands.len();
    let per_call_ns = elapsed.as_nanos() as f64 / total_calls as f64;

    Ok(format!(
        "─── router (command_router — SIMD hash lookup, 2-stage verified) ───\n  {} commands × {} iterations\n  Per call: {:.0} ns\n  Total calls: {}\n  Total: {:.1} ms",
        commands.len(),
        iterations,
        per_call_ns,
        total_calls,
        elapsed.as_secs_f64() * 1000.0,
    ))
}

pub fn bench_vault() -> crate::error::Result<String> {
    use crate::vault::{EachachaCrypto, VaultCrypto};

    let lib_path = crate::vault::find_chacha_lib()
        .ok_or_else(|| crate::error::Error::Tool("libchacha20.so not found".into()))?;
    let crypto = EachachaCrypto::new(lib_path);

    let data = vec![0x42u8; 4096];
    let key: [u8; 32] = [0xAB; 32];
    let nonce: [u8; 12] = [0xCD; 12];
    let iterations = 10_000;

    // Warmup
    for _ in 0..100 {
        let enc = crypto.encrypt(&data, &key, &nonce).map_err(|e| crate::error::Error::Tool(e))?;
        let _ = crypto.decrypt(&enc, &key, &nonce);
    }

    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = crypto.encrypt(&data, &key, &nonce);
    }
    let enc_elapsed = start.elapsed();
    let enc_ns = enc_elapsed.as_nanos() as f64 / iterations as f64;

    let encrypted = crypto.encrypt(&data, &key, &nonce).map_err(|e| crate::error::Error::Tool(e))?;
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = crypto.decrypt(&encrypted, &key, &nonce);
    }
    let dec_elapsed = start.elapsed();
    let dec_ns = dec_elapsed.as_nanos() as f64 / iterations as f64;
    let throughput = data.len() as f64 * iterations as f64 / enc_elapsed.as_secs_f64() / 1e9;

    Ok(format!(
        "─── vault (eachacha — SIMD ChaCha20, 4-round interleaved) ───\n  Encrypt: {:.0} ns/4KB\n  Decrypt: {:.0} ns/4KB\n  Throughput: {:.1} GB/s\n  {} iterations",
        enc_ns, dec_ns, throughput, iterations
    ))
}

pub fn bench_search() -> crate::error::Result<String> {
    use crate::kernels::search::{batch_cosine, top_k, JL_DIM};

    let n_vecs = 1024;
    let iterations = 1000;

    let mut vecs = vec![0.0f32; n_vecs * JL_DIM];
    let mut val = 1.0f32;
    for v in vecs.iter_mut() {
        val = (val * 1.1 + 0.3) % 2.0 - 1.0;
        *v = val;
    }

    let mut query = vec![0.0f32; JL_DIM];
    for (i, q) in query.iter_mut().enumerate() {
        *q = (i as f32 * 0.1).sin();
    }
    let query_norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();

    for _ in 0..10 {
        let scores = batch_cosine(&query, query_norm, &vecs, JL_DIM, n_vecs);
        let _ = top_k(&scores, 5);
    }

    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let scores = batch_cosine(&query, query_norm, &vecs, JL_DIM, n_vecs);
        let _ = top_k(&scores, 5);
    }
    let elapsed = start.elapsed();
    let per_search_us = elapsed.as_micros() as f64 / iterations as f64;
    let searches_per_sec = iterations as f64 / elapsed.as_secs_f64();

    Ok(format!(
        "─── search (search kernel — NEON/AVX batch dot, branchless top-k) ───\n  Per search: {:.1} µs  ({} × {}-dim vectors)\n  Throughput: {:.0}K searches/s\n  {} iterations",
        per_search_us, n_vecs, JL_DIM, searches_per_sec / 1000.0, iterations
    ))
}

pub fn bench_jl() -> crate::error::Result<String> {
    use crate::kernels::search::{jl_project, JL_DIM};

    let in_dim = 256;
    let iterations = 100_000;

    let vec: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.1).sin()).collect();
    let signs: Vec<f32> = (0..in_dim).map(|i| if i % 3 == 0 { -1.0 } else { 1.0 }).collect();

    for _ in 0..100 {
        let _ = jl_project(&vec, &signs, JL_DIM);
    }

    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let _ = jl_project(&vec, &signs, JL_DIM);
    }
    let elapsed = start.elapsed();
    let per_ns = elapsed.as_nanos() as f64 / iterations as f64;
    let per_sec = iterations as f64 / elapsed.as_secs_f64();

    Ok(format!(
        "─── jl (turbo_rotate FWHT + sign-flip → jl_project 256→{}) ───\n  Per projection: {:.0} ns\n  Throughput: {:.1}M projections/s\n  {} iterations",
        JL_DIM, per_ns, per_sec / 1e6, iterations
    ))
}

pub fn bench_fused() -> crate::error::Result<String> {
    use crate::vault::{Vault, EachachaCrypto, find_chacha_lib};
    use crate::vault::fused_search::FusedSearcher;

    crate::kernels::ffi::init()
        .map_err(|e| crate::error::Error::Tool(e))?;

    let lib = find_chacha_lib()
        .ok_or_else(|| crate::error::Error::Tool("libchacha20.so not found".into()))?;
    let crypto = Box::new(EachachaCrypto::new(lib));

    let dir = std::env::temp_dir().join("olorin_bench_fused");
    std::fs::create_dir_all(&dir).ok();
    let vault_path = dir.join("bench.vault");
    let _ = std::fs::remove_file(&vault_path);

    let key = [0xABu8; 32];
    let mut vault = Vault::create(&vault_path, &key, crypto)
        .map_err(|e| crate::error::Error::Tool(e.to_string()))?;

    // Write a 4 KB block with searchable content
    let block = "User: How do I optimize x86 SIMD code for AVX-512?\n\
                 Olorin: Use 512-bit zmm registers. Key intrinsics include _mm512_fmadd_ps.\n"
        .repeat(30);
    vault.append_message(&block)
        .map_err(|e| crate::error::Error::Tool(e.to_string()))?;
    vault.flush()
        .map_err(|e| crate::error::Error::Tool(e.to_string()))?;

    let (ciphertext, nonce) = vault.read_encrypted_block(0)
        .map_err(|e| crate::error::Error::Tool(e.to_string()))?;

    let mut searcher = FusedSearcher::new();
    let needles: &[&[u8]] = &[b"AVX-512", b"zmm"];
    let iterations = 10_000;

    // Warmup — stabilize buffers in cache
    for _ in 0..100 {
        let _ = searcher.search(&ciphertext, needles, &key, &nonce);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = searcher.search(&ciphertext, needles, &key, &nonce);
    }
    let elapsed = start.elapsed();

    let per_call_us = elapsed.as_micros() as f64 / iterations as f64;
    let per_call_ns = elapsed.as_nanos() as f64 / iterations as f64;
    let throughput = ciphertext.len() as f64 * iterations as f64 / elapsed.as_secs_f64() / 1e9;

    let _ = std::fs::remove_file(&vault_path);

    Ok(format!(
        "─── fused (FusedSearcher — chacha20_search_v2, decrypt+search in SIMD registers) ───\n\
         \x20 {} B ciphertext, {} needles, {} iterations\n\
         \x20 Per call: {:.1} µs ({:.0} ns)\n\
         \x20 Throughput: {:.2} GB/s\n\
         \x20 Scratch: ~23 KB pre-allocated (L1d resident after warmup)",
        ciphertext.len(), needles.len(), iterations,
        per_call_us, per_call_ns, throughput,
    ))
}

pub fn bench_all() -> crate::error::Result<String> {
    let start = std::time::Instant::now();
    let mut results = Vec::new();
    results.push(bench_safety()?);
    results.push(bench_router()?);
    results.push(bench_recall()?);
    results.push(bench_vault()?);
    results.push(bench_search()?);
    results.push(bench_jl()?);
    results.push(bench_fused()?);
    let total = start.elapsed();
    results.push(format!("─── summary ───\n  7 benchmarks completed in {:.1} s", total.as_secs_f64()));
    Ok(results.join("\n\n"))
}

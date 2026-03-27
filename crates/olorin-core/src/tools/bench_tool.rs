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
        "Run a quick microbenchmark. Targets: safety, router, recall, vault, search, jl, all."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": ["safety", "router", "recall", "vault", "search", "jl", "all"],
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
        "Safety scan ({} B input, {} iterations):\n  Per call: {:.0} ns\n  Throughput: {:.2} GB/s\n  Total: {:.1} ms",
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
        "Recall benchmark (JL-projected {}-dim, 1024 entries):\n\
         Insert:\n  Per insert: {:.1} us\n  1024 inserts: {:.1} ms\n\
         Recall (top-5 from 1024):\n  Per recall: {:.1} us\n  {} recalls: {:.1} ms\n\
         Memory: {:.0} KB for 1024 vectors ({} B/vec)",
        crate::kernels::search::JL_DIM,
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
        "Command router ({} commands × {} iterations):\n  Per call: {:.0} ns\n  Total calls: {}\n  Total: {:.1} ms",
        commands.len(),
        iterations,
        per_call_ns,
        total_calls,
        elapsed.as_secs_f64() * 1000.0,
    ))
}

pub fn bench_vault() -> crate::error::Result<String> {
    use crate::vault::{EachachaCrypto, VaultCrypto};

    let lib_dir = crate::kernels::ffi::kernel_dir()
        .map_err(|e| crate::error::Error::Tool(e))?;
    let lib_path = lib_dir.join("libchacha20.so");
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

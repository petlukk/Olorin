use super::ToolResult;
use std::time::Instant;

pub fn run(args: &str) -> ToolResult {
    let target = args.trim();
    let result = match target {
        "safety" => bench_safety(),
        "router" => bench_router(),
        "recall" => bench_recall(),
        "vault" => bench_vault(),
        "search" => bench_search(),
        "jl" => bench_jl(),
        "fused" => bench_fused(),
        "all" => bench_all(),
        "" => return ToolResult {
            output: "usage: bench <safety|router|recall|vault|search|jl|fused|all>".to_string(),
            success: false,
        },
        _ => return ToolResult {
            output: format!("unknown target '{target}'. Use: safety, router, recall, vault, search, jl, fused, all"),
            success: false,
        },
    };
    match result {
        Ok(s) => ToolResult { output: s, success: true },
        Err(e) => ToolResult { output: format!("bench error: {e}"), success: false },
    }
}

fn bench_safety() -> Result<String, String> {
    use crate::core::safety;

    let input = "Hello, this is a normal user message for benchmarking purposes. ".repeat(16);
    let iterations = 10_000;

    // Warmup
    for _ in 0..100 {
        let _ = safety::scan(input.as_bytes());
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = safety::scan(input.as_bytes());
    }
    let elapsed = start.elapsed();

    let per_call_ns = elapsed.as_nanos() as f64 / iterations as f64;
    let bytes_per_sec = input.len() as f64 * iterations as f64 / elapsed.as_secs_f64();
    let gb_per_sec = bytes_per_sec / 1e9;

    Ok(format!(
        "─── safety (fused_safety — SIMD byte classifier + leak + injection, single pass) ───\n  {} B input, {} iterations\n  Per call: {:.0} ns\n  Throughput: {:.2} GB/s\n  Total: {:.1} ms",
        input.len(), iterations, per_call_ns, gb_per_sec, elapsed.as_secs_f64() * 1000.0,
    ))
}

fn bench_recall() -> Result<String, String> {
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

    let mut store = VectorStore::new(1024);
    let insert_iters = 10;

    // Warmup
    for text in &corpus { store.add(text); }
    store.clear();

    let start = Instant::now();
    for _ in 0..insert_iters {
        store.clear();
        for i in 0..1024 {
            store.add(corpus[i % corpus.len()]);
        }
    }
    let insert_elapsed = start.elapsed();
    let per_insert_us = insert_elapsed.as_micros() as f64 / (insert_iters * 1024) as f64;

    let recall_iters = 1000;

    // Warmup
    for q in &queries { let _ = store.search(q, 5); }

    let start = Instant::now();
    for _ in 0..recall_iters {
        for q in &queries { let _ = store.search(q, 5); }
    }
    let recall_elapsed = start.elapsed();
    let total_recalls = recall_iters * queries.len();
    let per_recall_us = recall_elapsed.as_micros() as f64 / total_recalls as f64;

    Ok(format!(
        "─── recall (JL 256→64, NEON batch_cosine, 1024 vecs) ───\n\
         Insert:\n  Per insert: {:.1} us\n  1024 inserts: {:.1} ms\n\
         Recall (top-5 from 1024):\n  Per recall: {:.1} us\n  {} recalls: {:.1} ms",
        per_insert_us,
        per_insert_us * 1024.0 / 1000.0,
        per_recall_us,
        total_recalls,
        recall_elapsed.as_secs_f64() * 1000.0,
    ))
}

fn bench_router() -> Result<String, String> {
    use crate::kernels::ffi;

    let commands: &[&[u8]] = &[
        b"/help", b"/quit", b"/time", b"/calc 2+2", b"/shell ls",
        b"/read file.txt", b"/ls", b"/cpu", b"/json keys {}",
        b"hello world", b"/unknown", b"/tokens test",
    ];
    let iterations = 100_000;

    // Warmup
    for _ in 0..1000 {
        for cmd in commands {
            let mut out_match: i32 = 0;
            unsafe { ffi::match_command(cmd.as_ptr(), cmd.len() as i32, &mut out_match); }
        }
    }

    let start = Instant::now();
    for _ in 0..iterations {
        for cmd in commands {
            let mut out_match: i32 = 0;
            unsafe { ffi::match_command(cmd.as_ptr(), cmd.len() as i32, &mut out_match); }
        }
    }
    let elapsed = start.elapsed();

    let total_calls = iterations * commands.len();
    let per_call_ns = elapsed.as_nanos() as f64 / total_calls as f64;

    Ok(format!(
        "─── router (command_router — SIMD hash lookup) ───\n  {} commands × {} iterations\n  Per call: {:.0} ns\n  Total calls: {}\n  Total: {:.1} ms",
        commands.len(), iterations, per_call_ns, total_calls, elapsed.as_secs_f64() * 1000.0,
    ))
}

fn bench_vault() -> Result<String, String> {
    use crate::storage::crypto;

    let mut data = vec![0x42u8; 4096];
    let key: [u8; 32] = [0xABu8; 32];
    let nonce: [u8; 12] = [0xCDu8; 12];
    let iterations = 10_000;

    // Warmup
    for _ in 0..100 {
        let mut buf = data.clone();
        crypto::encrypt(&key, &nonce, 0, &mut buf);
        crypto::decrypt(&key, &nonce, 0, &mut buf);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let mut buf = data.clone();
        crypto::encrypt(&key, &nonce, 0, &mut buf);
    }
    let enc_elapsed = start.elapsed();
    let enc_ns = enc_elapsed.as_nanos() as f64 / iterations as f64;

    crypto::encrypt(&key, &nonce, 0, &mut data);
    let start = Instant::now();
    for _ in 0..iterations {
        let mut buf = data.clone();
        crypto::decrypt(&key, &nonce, 0, &mut buf);
    }
    let dec_elapsed = start.elapsed();
    let dec_ns = dec_elapsed.as_nanos() as f64 / iterations as f64;
    let throughput = 4096.0 * iterations as f64 / enc_elapsed.as_secs_f64() / 1e9;

    Ok(format!(
        "─── vault (SIMD ChaCha20, 4-round interleaved) ───\n  Encrypt: {:.0} ns/4KB\n  Decrypt: {:.0} ns/4KB\n  Throughput: {:.1} GB/s\n  {} iterations",
        enc_ns, dec_ns, throughput, iterations
    ))
}

fn bench_search() -> Result<String, String> {
    use crate::kernels::ffi;

    const JL_DIM: usize = 64;
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
    let mut scores = vec![0.0f32; n_vecs];
    let mut indices = vec![0i32; 5];
    let mut top_scores = vec![0.0f32; 5];

    // Warmup
    for _ in 0..10 {
        unsafe {
            ffi::batch_cosine(query.as_ptr(), query_norm, vecs.as_ptr(), JL_DIM as i32, n_vecs as i32, scores.as_mut_ptr());
            ffi::top_k(scores.as_ptr(), n_vecs as i32, 5, indices.as_mut_ptr(), top_scores.as_mut_ptr());
        }
    }

    let start = Instant::now();
    for _ in 0..iterations {
        unsafe {
            ffi::batch_cosine(query.as_ptr(), query_norm, vecs.as_ptr(), JL_DIM as i32, n_vecs as i32, scores.as_mut_ptr());
            ffi::top_k(scores.as_ptr(), n_vecs as i32, 5, indices.as_mut_ptr(), top_scores.as_mut_ptr());
        }
    }
    let elapsed = start.elapsed();
    let per_search_us = elapsed.as_micros() as f64 / iterations as f64;
    let searches_per_sec = iterations as f64 / elapsed.as_secs_f64();

    Ok(format!(
        "─── search (search kernel — NEON/AVX batch dot, branchless top-k) ───\n  Per search: {:.1} µs  ({} × {}-dim vectors)\n  Throughput: {:.0}K searches/s\n  {} iterations",
        per_search_us, n_vecs, JL_DIM, searches_per_sec / 1000.0, iterations
    ))
}

fn bench_jl() -> Result<String, String> {
    use crate::kernels::ffi;

    const JL_DIM: usize = 64;
    let in_dim = 256;
    let iterations = 100_000;

    let vec: Vec<f32> = (0..in_dim).map(|i| (i as f32 * 0.1).sin()).collect();
    let signs: Vec<f32> = (0..in_dim).map(|i| if i % 3 == 0 { -1.0 } else { 1.0 }).collect();
    let mut out = vec![0.0f32; JL_DIM];
    let mut scratch = vec![0.0f32; in_dim];

    // Warmup
    for _ in 0..100 {
        unsafe {
            ffi::jl_project(vec.as_ptr(), signs.as_ptr(), in_dim as i32, JL_DIM as i32, out.as_mut_ptr(), scratch.as_mut_ptr());
        }
    }

    let start = Instant::now();
    for _ in 0..iterations {
        unsafe {
            ffi::jl_project(vec.as_ptr(), signs.as_ptr(), in_dim as i32, JL_DIM as i32, out.as_mut_ptr(), scratch.as_mut_ptr());
        }
    }
    let elapsed = start.elapsed();
    let per_ns = elapsed.as_nanos() as f64 / iterations as f64;
    let per_sec = iterations as f64 / elapsed.as_secs_f64();

    Ok(format!(
        "─── jl (turbo_rotate FWHT + sign-flip → jl_project 256→{}) ───\n  Per projection: {:.0} ns\n  Throughput: {:.1}M projections/s\n  {} iterations",
        JL_DIM, per_ns, per_sec / 1e6, iterations
    ))
}

fn bench_fused() -> Result<String, String> {
    use crate::storage::{vault::Vault, search::FusedSearcher};

    let dir = std::env::temp_dir().join("olorin_bench_fused");
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let bench_dir = dir.join("benchvault");
    std::fs::create_dir_all(&bench_dir).map_err(|e| e.to_string())?;

    let vault_path = bench_dir.join("vault.bin");
    let _ = std::fs::remove_file(&vault_path);

    let mut vault = Vault::open(&bench_dir).map_err(|e| e.to_string())?;

    // Write a 4 KB block with searchable content
    let block = "How do I optimize x86 SIMD code for AVX-512? Use 512-bit zmm registers.".repeat(40);
    vault.append(b"user", block.as_bytes()).map_err(|e| e.to_string())?;

    // Read back the encrypted block for benchmarking
    let plaintext = vault.decrypt_block(0).map_err(|e| e.to_string())?;
    // Re-encrypt with a known key for benchmarking
    let key = [0xABu8; 32];
    let nonce = [0xCDu8; 12];
    let mut ciphertext = plaintext.clone();
    crate::storage::crypto::encrypt(&key, &nonce, 0, &mut ciphertext);

    let mut searcher = FusedSearcher::new();
    let needles: &[&[u8]] = &[b"AVX-512", b"zmm"];
    let iterations = 10_000;

    // Warmup
    for _ in 0..100 {
        let _ = searcher.search(&key, &nonce, 0, &ciphertext, needles);
    }

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = searcher.search(&key, &nonce, 0, &ciphertext, needles);
    }
    let elapsed = start.elapsed();

    let per_call_us = elapsed.as_micros() as f64 / iterations as f64;
    let per_call_ns = elapsed.as_nanos() as f64 / iterations as f64;
    let throughput = ciphertext.len() as f64 * iterations as f64 / elapsed.as_secs_f64() / 1e9;

    let _ = std::fs::remove_dir_all(&bench_dir);

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

fn bench_all() -> Result<String, String> {
    let start = Instant::now();
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

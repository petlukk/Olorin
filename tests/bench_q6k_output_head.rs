//! Isolated single-thread q6k matvec GFLOPS at the output-head shape.
//!
//! Gemma 4's output head is m=262144 (vocab), k=1536 (hidden) Q6K —
//! ~330 MB of weight, dominant per-decode-step cost (~23 ms on 8T in
//! production). Measure the raw single-thread kernel rate so we know
//! the parallel ceiling.
//!
//! llama.cpp reference (`test-backend-ops perf -o MUL_MAT -p "type_a=q6_K,...m=4096"`,
//! default ggml threading): q6_K n=1 m=4096 k=14336 → 47.84 GFLOPS aggregate.
//! Different shape (wider, shorter) but q6_K inner loop is the same.
//!
//! Also runs q6k_matvec (the plain non-ws single-thread kernel) for
//! reference — production uses q6k_matvec_ws with ith=0 nth=1 which
//! adds work-stealing overhead that should be negligible for single
//! thread but worth confirming.
//!
//! Run: cargo test --release --test bench_q6k_output_head -- --nocapture

use std::path::Path;
use std::sync::atomic::AtomicI32;
use std::time::Instant;

fn model_path() -> String {
    std::env::var("OLORIN_MODEL").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")
    })
}

#[test]
fn q6k_output_head_throughput() {
    let h = std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(inner)
        .unwrap();
    h.join().unwrap();
}

fn inner() {
    if !Path::new(&model_path()).exists() {
        eprintln!("SKIP: model not present");
        return;
    }

    let gguf = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    // Output head = embed weight used as classifier. Q6K on Gemma 4.
    assert_eq!(
        model.embed_dtype,
        olorin::inference::matmul::GGML_TYPE_Q6_K,
        "expected Q6K output head"
    );

    let m = model.vocab_size;
    let k = model.hidden_dim;
    eprintln!("output-head shape: m={m} (vocab) k={k} (hidden)  Q6K");

    // Build a Q8K activation (the "last token's final-norm output" shape).
    let nb = k / 256;
    let mut input = vec![0.0f32; k];
    for i in 0..k { input[i] = 0.01 * (i % 97) as f32 - 0.5; }
    let mut qs = vec![0i8; k + 12];
    let mut d  = vec![0.0f32; nb];
    let mut bs = vec![0i16; nb * 16];
    unsafe {
        olorin::kernels::ffi_inference::quant_f32_q8k(
            input.as_ptr(), qs.as_mut_ptr(), d.as_mut_ptr(), bs.as_mut_ptr(), k as i32,
        );
    }

    let mut out = vec![0.0f32; m];
    // d_scratch for q6k_matvec_ws: n_blocks * 4 floats per thread
    let mut d_scratch = vec![0.0f32; nb * 4];

    // Production path: q6k_matvec_ws with ith=0, nth=1 (single-threaded sim).
    let current_chunk = AtomicI32::new(1);
    let weight = model.embed_weight;
    let bytes_per_weight_pass = (m as f64) * (nb as f64) * 210.0;

    let run_ws = |out: &mut [f32], d_scratch: &mut [f32]| {
        current_chunk.store(1, std::sync::atomic::Ordering::Relaxed);
        olorin::inference::matmul_graph::q6k_matvec_ws(
            weight,
            qs.as_ptr(), d.as_ptr(), bs.as_ptr(),
            out.as_mut_ptr(), d_scratch.as_mut_ptr(),
            m, k,
            &current_chunk, 0, 1,
        );
    };

    // Warm
    for _ in 0..2 { run_ws(&mut out, &mut d_scratch); }

    let mflop = 2.0 * m as f64 * k as f64 / 1e6;
    let iters = 5usize;
    let t0 = Instant::now();
    for _ in 0..iters { run_ws(&mut out, &mut d_scratch); }
    let sec = t0.elapsed().as_secs_f64() / iters as f64;
    let gflops = mflop / sec / 1000.0;
    let gbps = bytes_per_weight_pass / sec / 1e9;

    eprintln!();
    eprintln!("q6k_matvec_ws (single-threaded, ith=0 nth=1):");
    eprintln!("  ms/call:        {:>8.3}", sec * 1000.0);
    eprintln!("  GFLOPS:         {:>8.2}", gflops);
    eprintln!("  GB_w/s:         {:>8.2}  (weight {:.1} MB)", gbps, bytes_per_weight_pass / 1e6);
    eprintln!();
    eprintln!("Production per-decode-step (from GEMMA4_TIMING on 8T):  ~23 ms");
    let aggregate_gflops_8t = mflop / 0.023 / 1000.0;
    eprintln!("Implied 8T aggregate GFLOPS:  {:.2}  → per-thread {:.2}",
        aggregate_gflops_8t, aggregate_gflops_8t / 8.0);
    eprintln!();
    eprintln!("Reference (llama.cpp test-backend-ops): q6_K m=4096 k=14336 → 47.84 GFLOPS aggregate");
}

//! Isolated single-thread q4k_matvec_8x8 GFLOPS bench on the exact shapes
//! the decode hot path uses.
//!
//! Decode is dominated by ~6 q4k matvecs per layer (q/k/v/wo/gate+up/down)
//! times 35 layers. Each runs on 8 threads in production. This bench
//! measures the single-thread kernel ceiling at those shapes so we can
//! divide production throughput by 8 and compare per-thread to see where
//! parallel efficiency is lost.
//!
//! llama.cpp reference (`test-backend-ops perf -o MUL_MAT -p "type_a=q4_K,n=1"`,
//! default ggml threading ≈ 8 threads):
//!     m=4096 k=14336 n=1 → 67 GFLOPS aggregate → ~8.4 GFLOPS/thread
//!
//! Run: cargo test --release --test bench_q4k_matvec_shapes -- --nocapture

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
fn q4k_matvec_decode_shapes() {
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

    // Pick a layer with the largest ffn_dim (12288) so gate/up/down reflect the
    // hot layer shapes.
    let il = (0..model.n_layers)
        .max_by_key(|&i| model.ffn_dim[i])
        .unwrap();
    let lw = &model.layers[il];
    let hd = model.hidden_dim;
    let ffn = model.ffn_dim[il];
    let head_dim = model.head_dim_k[il];
    let n_heads = model.n_heads;

    eprintln!("layer {il}: hd={hd} ffn={ffn} n_heads={n_heads} head_dim={head_dim}");
    eprintln!(
        "{:<18}  {:>7}  {:>7}  {:>10}  {:>9}  {:>11}",
        "op", "m_rows", "k_cols", "ms/call", "GFLOPS", "GB_w/s"
    );
    eprintln!("{:-<70}", "");

    let shapes = [
        ("gemv_q",         n_heads * head_dim, hd,     lw.wq),
        ("gemv_wo",        hd,                 n_heads * head_dim, lw.wo),
        ("gemv_gate",      ffn,                hd,     lw.w_gate),
        ("gemv_up",        ffn,                hd,     lw.w_up),
        ("gemv_down",      hd,                 ffn,    lw.w_down),
    ];

    for (name, m, k, w_ptr) in shapes {
        if m % 8 != 0 || k % 256 != 0 {
            eprintln!("{:<18}  SKIP m={} k={} (not %8/%256)", name, m, k);
            continue;
        }
        bench_matvec(name, w_ptr, m, k);
    }
}

fn bench_matvec(name: &str, weight: *const u8, m: usize, k: usize) {
    // Repack into block_q4_Kx8 tiles (same as production).
    let packed = olorin::inference::repack::q4k_repack_8x8(weight, m, k);

    // Build Q8K input (single token's activation).
    let mut input = vec![0.0f32; k];
    for i in 0..k { input[i] = 0.01 * (i % 97) as f32 - 0.5; }
    let nb = k / 256;
    let mut qs = vec![0i8; k + 12];
    let mut d  = vec![0.0f32; nb];
    let mut bs = vec![0i16; nb * 16];
    unsafe {
        olorin::kernels::ffi_inference::quant_f32_q8k(
            input.as_ptr(), qs.as_mut_ptr(), d.as_mut_ptr(), bs.as_mut_ptr(), k as i32,
        );
    }

    let mut out = vec![0.0f32; m];

    // Single-threaded work-stealing: ith=0 nth=1.
    let current_chunk = AtomicI32::new(1);
    let do_call = |out: &mut [f32]| {
        current_chunk.store(1, std::sync::atomic::Ordering::Relaxed);
        olorin::inference::matmul_graph::q4k_matvec_8x8_ws(
            packed.as_ptr(),
            qs.as_ptr(), d.as_ptr(), bs.as_ptr(),
            out.as_mut_ptr(),
            m, k,
            &current_chunk, 0, 1,
        );
    };

    // Warm
    for _ in 0..3 { do_call(&mut out); }

    let mflop_per_call = 2.0 * m as f64 * k as f64 / 1e6;
    // Target ~300ms per data point
    let iters = ((300.0 / (mflop_per_call / 1000.0).max(1.0)).round() as usize).clamp(50, 10_000);

    let t0 = Instant::now();
    for _ in 0..iters { do_call(&mut out); }
    let sec = t0.elapsed().as_secs_f64() / iters as f64;

    let gflops = mflop_per_call / sec / 1000.0;
    let weight_bytes = packed.len() as f64;
    let weight_gbps = weight_bytes / sec / 1e9;

    eprintln!(
        "{:<18}  {:>7}  {:>7}  {:>10.3}  {:>9.2}  {:>11.2}",
        name, m, k, sec * 1000.0, gflops, weight_gbps
    );
}

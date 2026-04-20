//! Isolated single-thread GFLOPS bench for q5k_gemm + q6k_gemm kernels.
//!
//! Mirrors bench_q4k_gemm_scaling.rs structure. ARM-only because the GEMM
//! kernels are aarch64-gated.
//!
//! Comparison reference (today's llama.cpp test-backend-ops at M=4096 K=14336):
//!   q5_K MUL_MAT N=4: 59.92 GFLOPS aggregate (4T) = ~15 GFLOPS/thread
//!                N=8: 62.05                       = ~15.5/thread
//!              N=512: 63.52                       = ~15.9/thread
//!   q6_K MUL_MAT N=4: 48.33 GFLOPS aggregate (4T) = ~12 GFLOPS/thread
//!                N=8: 50.58                       = ~12.6/thread
//!              N=512: 51.64                       = ~12.9/thread
//!
//! llama.cpp shapes are M=4096 K=14336. Olorin shapes here are model-natural:
//!   Q5K wo:     K=qkv_dim, M=hidden_dim (per layer)
//!   Q6K w_down: K=ffn_dim,  M=hidden_dim (Q6K w_down layers only)

#![cfg(target_arch = "aarch64")]

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
fn q5k_q6k_gemm_scaling() {
    let h = std::thread::Builder::new()
        .stack_size(256 * 1024 * 1024)
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

    // Find a Q5K wo layer with the largest K (qkv_dim).
    let q5k_layer = (0..model.layers.len())
        .filter(|&il| model.layers[il].wo_dtype == olorin::inference::matmul::GGML_TYPE_Q5_K)
        .max_by_key(|&il| model.n_heads * model.head_dim_k[il])
        .expect("no Q5K wo");
    // Find a Q6K w_down layer with the largest ffn_dim.
    let q6k_layer = (0..model.layers.len())
        .filter(|&il| model.layers[il].w_down_dtype == olorin::inference::matmul::GGML_TYPE_Q6_K)
        .max_by_key(|&il| model.ffn_dim[il])
        .expect("no Q6K w_down");

    bench_q5k(&model, q5k_layer);
    eprintln!();
    bench_q6k(&model, q6k_layer);
}

fn bench_q5k(model: &olorin::inference::engine::Gemma4Model, il: usize) {
    let lw = &model.layers[il];
    let nc = model.hidden_dim;                                  // wo output dim
    let n_inner = model.n_heads * model.head_dim_k[il];         // wo inner dim (qkv)
    assert!(nc % 8 == 0 && n_inner % 256 == 0);
    eprintln!("=== Q5K wo (layer {il}): M={nc}, K={n_inner} ===");
    eprintln!("{:>6}  {:>10}  {:>10}", "N", "ms/call", "GFLOPS");
    eprintln!("{:-<32}", "");
    bench_kernel_q5k(lw.wo, nc, n_inner);
}

fn bench_q6k(model: &olorin::inference::engine::Gemma4Model, il: usize) {
    let lw = &model.layers[il];
    let nc = model.hidden_dim;          // w_down output dim
    let n_inner = model.ffn_dim[il];    // w_down inner dim
    assert!(nc % 8 == 0 && n_inner % 256 == 0);
    eprintln!("=== Q6K w_down (layer {il}): M={nc}, K={n_inner} ===");
    eprintln!("{:>6}  {:>10}  {:>10}", "N", "ms/call", "GFLOPS");
    eprintln!("{:-<32}", "");
    bench_kernel_q6k(lw.w_down, nc, n_inner);
}

fn bench_kernel_q5k(weight: *const u8, nc: usize, n_inner: usize) {
    for &batch_n in &[4usize, 8, 16, 32, 64, 128, 256] {
        let q8_a = build_q8_a(n_inner, batch_n);
        let mut out = vec![0.0f32; batch_n * nc];
        let current_chunk = AtomicI32::new(1);
        let mut call = || {
            current_chunk.store(1, std::sync::atomic::Ordering::Relaxed);
            olorin::inference::matmul_graph::q5k_gemm_batch_ws(
                weight, q8_a.as_ptr(), out.as_mut_ptr(),
                n_inner, nc, batch_n, nc,
                &current_chunk, 0, 1,
            );
        };
        call();
        let mflop = 2.0 * nc as f64 * n_inner as f64 * batch_n as f64 / 1e6;
        let iters = ((300.0 / (mflop / 1000.0).max(1.0)).round() as usize).clamp(10, 200);
        let t0 = Instant::now();
        for _ in 0..iters { call(); }
        let secs = t0.elapsed().as_secs_f64() / iters as f64;
        eprintln!("{:>6}  {:>10.3}  {:>10.2}", batch_n, secs * 1000.0, mflop / secs / 1000.0);
    }
}

fn bench_kernel_q6k(weight: *const u8, nc: usize, n_inner: usize) {
    for &batch_n in &[4usize, 8, 16, 32, 64, 128, 256] {
        let q8_a = build_q8_a(n_inner, batch_n);
        let mut out = vec![0.0f32; batch_n * nc];
        let current_chunk = AtomicI32::new(1);
        let mut call = || {
            current_chunk.store(1, std::sync::atomic::Ordering::Relaxed);
            olorin::inference::matmul_graph::q6k_gemm_batch_ws(
                weight, q8_a.as_ptr(), out.as_mut_ptr(),
                n_inner, nc, batch_n, nc,
                &current_chunk, 0, 1,
            );
        };
        call();
        let mflop = 2.0 * nc as f64 * n_inner as f64 * batch_n as f64 / 1e6;
        let iters = ((300.0 / (mflop / 1000.0).max(1.0)).round() as usize).clamp(10, 200);
        let t0 = Instant::now();
        for _ in 0..iters { call(); }
        let secs = t0.elapsed().as_secs_f64() / iters as f64;
        eprintln!("{:>6}  {:>10.3}  {:>10.2}", batch_n, secs * 1000.0, mflop / secs / 1000.0);
    }
}

/// Quantize batch_n synthetic activation rows into block_q8_Kx4 layout.
fn build_q8_a(n_inner: usize, batch_n: usize) -> Vec<u8> {
    assert!(batch_n % 4 == 0);
    let nb = n_inner / 256;
    let block_size = nb * 1168;
    let mut all_qs: Vec<Vec<i8>> = Vec::with_capacity(batch_n);
    let mut all_d: Vec<Vec<f32>> = Vec::with_capacity(batch_n);
    let mut all_bsums: Vec<Vec<i16>> = Vec::with_capacity(batch_n);
    for col in 0..batch_n {
        let mut input = vec![0.0f32; n_inner];
        for i in 0..n_inner {
            input[i] = 0.01 * ((col * 7 + i) % 97) as f32 - 0.5;
        }
        let mut qs = vec![0i8; n_inner + 12];
        let mut d = vec![0.0f32; nb];
        let mut bsums = vec![0i16; nb * 16];
        unsafe {
            olorin::kernels::ffi_inference::quant_f32_q8k(
                input.as_ptr(), qs.as_mut_ptr(), d.as_mut_ptr(),
                bsums.as_mut_ptr(), n_inner as i32,
            );
        }
        all_qs.push(qs);
        all_d.push(d);
        all_bsums.push(bsums);
    }
    let n_groups = batch_n / 4;
    let mut q8_a = vec![0u8; n_groups * block_size];
    for g in 0..n_groups {
        let r0 = g * 4;
        let mut row_d = vec![0.0f32; nb * 4];
        for b in 0..nb {
            for r in 0..4 { row_d[b * 4 + r] = all_d[r0 + r][b]; }
        }
        unsafe {
            olorin::kernels::ffi_inference::q8k_repack_4(
                all_qs[r0].as_ptr(), all_qs[r0 + 1].as_ptr(),
                all_qs[r0 + 2].as_ptr(), all_qs[r0 + 3].as_ptr(),
                row_d.as_ptr(),
                all_bsums[r0].as_ptr(), all_bsums[r0 + 1].as_ptr(),
                all_bsums[r0 + 2].as_ptr(), all_bsums[r0 + 3].as_ptr(),
                q8_a[g * block_size..].as_mut_ptr(), nb as i32,
            );
        }
    }
    q8_a
}

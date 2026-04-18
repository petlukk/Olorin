//! Isolated q4k_8x8_q8k_gemm GFLOPS scaling bench — single-threaded.
//!
//! Measures raw kernel throughput (single-thread, no barrier/dispatch overhead)
//! on the Gemma 4 ffn_down shape — the single biggest prefill op (~43% of
//! forward_batch layer time at N=257). Target comparison: llama.cpp's ggml
//! `test-backend-ops perf -o MUL_MAT` on q4_K × f32 (default ggml threading):
//!
//!   llama.cpp q4_K MUL_MAT at m=4096 k=14336 (similar tall-thin shape):
//!     n=1: 67 GFLOPS   n=2: 134   n=4: 176   n=8: 196
//!
//! Run: cargo test --release --test bench_q4k_gemm_scaling -- --nocapture

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
fn q4k_gemm_scaling() {
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

    // Prefer a layer with the largest ffn_dim (12288 on Gemma 4) and Q4K w_down.
    let (il, nc, n_inner) = {
        let mut best: Option<(usize, usize, usize)> = None;
        for (il, lw) in model.layers.iter().enumerate() {
            if lw.w_down_dtype != olorin::inference::matmul::GGML_TYPE_Q4_K { continue; }
            let nc = model.hidden_dim;
            let n_inner = model.ffn_dim[il];
            match best {
                None => best = Some((il, nc, n_inner)),
                Some((_, _, cur_k)) if n_inner > cur_k => best = Some((il, nc, n_inner)),
                _ => {}
            }
        }
        best.expect("no Q4K w_down")
    };

    assert!(nc % 8 == 0, "nc={nc} must be %8");
    assert!(n_inner % 256 == 0, "n_inner={n_inner} must be %256");

    eprintln!("shape: M (out rows) = {nc}, K (inner) = {n_inner}  (layer {il} w_down)");

    let lw = &model.layers[il];
    let nb = n_inner / 256;
    let block_size = nb * 1168; // block_q8_Kx4 stride (bytes per 4-row group per super-block)

    // Repack weight into block_q4_Kx8 tiles. Done once, outside timing.
    let packed = olorin::inference::repack::q4k_repack_8x8(lw.w_down, nc, n_inner);

    eprintln!(
        "{:>6}  {:>10}  {:>10}  {:>10}",
        "N", "ms/call", "GFLOPS", "GB_w/s"
    );
    eprintln!("{:-<42}", "");

    let bytes_per_weight_pass = packed.len() as f64;

    for &batch_n in &[4usize, 8, 16, 32, 64, 128, 256] {
        // Build Q8K inputs for batch_n synthetic activation rows.
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

        // block_q8_Kx4 layout — group rows into 4-row chunks.
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

        let mut out = vec![0.0f32; batch_n * nc];

        // Single-threaded work-stealing call: ith=0 is the only worker;
        // the wrapper iterates all tiles sequentially.
        let current_chunk = AtomicI32::new(1); // "next" chunk starts at 1, worker claims 0 first.
        let mut warm = || {
            current_chunk.store(1, std::sync::atomic::Ordering::Relaxed);
            olorin::inference::matmul_graph::q4k_gemm_8x8_batch_ws(
                packed.as_ptr(), q8_a.as_ptr(), out.as_mut_ptr(),
                n_inner, nc, batch_n, nc,
                &current_chunk, 0, 1,
            );
        };
        warm();

        let mflop_per_call = 2.0 * nc as f64 * n_inner as f64 * batch_n as f64 / 1e6;
        // Aim ~300ms per data point; cap iters so small N don't blow time.
        let iters = ((300.0 / (mflop_per_call / 1000.0).max(1.0)).round() as usize).clamp(10, 200);

        let t0 = Instant::now();
        for _ in 0..iters { warm(); }
        let secs_per_call = t0.elapsed().as_secs_f64() / iters as f64;

        let gflops = mflop_per_call / secs_per_call / 1000.0;
        let weight_gbps = bytes_per_weight_pass / secs_per_call / 1e9;

        eprintln!(
            "{:>6}  {:>10.3}  {:>10.1}  {:>10.2}",
            batch_n,
            secs_per_call * 1000.0,
            gflops,
            weight_gbps,
        );
    }
}

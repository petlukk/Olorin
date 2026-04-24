//! Isolated GFLOPS bench: q5k_8x8_q8k_matvec vs per-row q5k_dot_q8k.
//!
//! Measures sustained throughput of the 8x8 tile kernel against the
//! single-row loop that it's meant to replace. Uses layer-0 wk
//! (Q5K, 256 rows × 1536 cols) for deterministic shapes.
//!
//! ARM-only (both kernels gated on aarch64 for the tile variant).

#![cfg(target_arch = "aarch64")]

use std::path::Path;
use std::time::Instant;

fn model_path() -> String {
    let home = std::env::var("HOME").unwrap();
    format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")
}

#[test]
fn q5k_dot_8x8_throughput_vs_single_row() {
    let h = std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(inner)
        .unwrap();
    h.join().unwrap();
}

fn inner() {
    if !Path::new(&model_path()).exists() {
        eprintln!("SKIP: no model at {}", model_path());
        return;
    }
    let gguf = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    let lw = &model.layers[0];
    assert_eq!(lw.wk_dtype, olorin::inference::matmul::GGML_TYPE_Q5_K);

    let head_dim = model.head_dim_k[0];
    let nc = model.n_kv_heads * head_dim; // 256 rows
    let n = model.hidden_dim;             // 1536 inner dim
    let nb = n / 256;                     // 6 blocks
    let pow2 = olorin::inference::matmul::pow2_table();

    // Quantize synthetic input once
    let mut input = vec![0.0f32; n];
    for i in 0..n {
        input[i] = 0.01 * ((i * 13 + 7) % 97) as f32 - 0.5;
    }
    let mut qs = vec![0i8; n + 12];
    let mut q8_d = vec![0.0f32; nb];
    let mut bsums = vec![0i16; nb * 16];
    unsafe {
        olorin::kernels::ffi_inference::quant_f32_q8k(
            input.as_ptr(),
            qs.as_mut_ptr(),
            q8_d.as_mut_ptr(),
            bsums.as_mut_ptr(),
            n as i32,
        );
    }

    // Repack weights
    let tile_bytes = 1408usize;
    let dst_total = (nc / 8) * nb * tile_bytes;
    let mut repacked = vec![0u8; dst_total];
    unsafe {
        olorin::kernels::ffi_inference::q5k_repack_8x8(
            lw.wk,
            repacked.as_mut_ptr(),
            nc as i32,
            n as i32,
        );
    }

    // Per matvec call: 2 * n_rows * n_cols = flops (mul + add per element)
    let flops_per_call = 2.0 * (nc as f64) * (n as f64);
    let iters = 2000usize;

    // ── Warmup + bench: single-row loop ──
    let src_row_bytes = nb * 176;
    let mut ref_scores = vec![0.0f32; nc];
    // warmup
    for _ in 0..20 {
        for r in 0..nc {
            let row_ptr = unsafe { lw.wk.add(r * src_row_bytes) };
            ref_scores[r] = unsafe {
                olorin::kernels::ffi_inference::q5k_dot_q8k(
                    row_ptr,
                    qs.as_ptr(),
                    bsums.as_ptr(),
                    nb as i32,
                    q8_d.as_ptr(),
                    pow2.as_ptr(),
                )
            };
        }
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        for r in 0..nc {
            let row_ptr = unsafe { lw.wk.add(r * src_row_bytes) };
            ref_scores[r] = unsafe {
                olorin::kernels::ffi_inference::q5k_dot_q8k(
                    row_ptr,
                    qs.as_ptr(),
                    bsums.as_ptr(),
                    nb as i32,
                    q8_d.as_ptr(),
                    pow2.as_ptr(),
                )
            };
        }
    }
    let t_single = t0.elapsed().as_secs_f64();
    let gflops_single = (flops_per_call * iters as f64) / (t_single * 1.0e9);

    // ── Warmup + bench: 8x8 tile ──
    let mut tile_scores = vec![0.0f32; nc];
    let mut scratch = vec![0u8; 128];
    // warmup
    for _ in 0..20 {
        unsafe {
            olorin::kernels::ffi_inference::q5k_8x8_q8k_matvec(
                repacked.as_ptr(),
                qs.as_ptr(),
                q8_d.as_ptr(),
                bsums.as_ptr(),
                pow2.as_ptr(),
                scratch.as_mut_ptr(),
                tile_scores.as_mut_ptr(),
                nc as i32,
                n as i32,
            );
        }
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe {
            olorin::kernels::ffi_inference::q5k_8x8_q8k_matvec(
                repacked.as_ptr(),
                qs.as_ptr(),
                q8_d.as_ptr(),
                bsums.as_ptr(),
                pow2.as_ptr(),
                scratch.as_mut_ptr(),
                tile_scores.as_mut_ptr(),
                nc as i32,
                n as i32,
            );
        }
    }
    let t_tile = t0.elapsed().as_secs_f64();
    let gflops_tile = (flops_per_call * iters as f64) / (t_tile * 1.0e9);

    let speedup = t_single / t_tile;
    eprintln!();
    eprintln!("=== Q5K matvec throughput (256×1536, {iters} iters) ===");
    eprintln!("single-row loop: {:>7.3} s  {:>6.2} GFLOPS", t_single, gflops_single);
    eprintln!("8x8 tile:        {:>7.3} s  {:>6.2} GFLOPS", t_tile, gflops_tile);
    eprintln!("speedup:         {:>7.2}x", speedup);
    eprintln!();
}

//! Bench — q4k_8x8_q8k_gemm (one call for N input cols) vs N repeated
//! q4k_8x8_q8k_matvec calls on the same problem.
//!
//! Hard gate: N=8 gemm must be >= 1.15x faster than the matvec loop.
//! Gate is tuned for Pi 5 / native Linux; WSL CPU scheduling jitter at
//! sub-ms measurements makes it unreliable in the default suite.
//!
//! Run: cargo test --release --test bench_q4k_gemm -- --ignored --nocapture

use std::path::Path;
use std::time::Instant;

fn model_path() -> String {
    let home = std::env::var("HOME").unwrap();
    format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")
}

#[test]
#[ignore = "perf gate — run explicitly on target hardware with --ignored"]
fn bench_gemm_vs_matvec_loop() {
    let h = std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(bench_inner)
        .unwrap();
    h.join().unwrap();
}

fn bench_inner() {
    if !Path::new(&model_path()).exists() {
        eprintln!("SKIP: no model");
        return;
    }
    let gguf = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    let lw = &model.layers[0];
    assert_eq!(lw.w_gate_dtype, olorin::inference::matmul::GGML_TYPE_Q4_K);

    let nc = model.ffn_dim[0];   // output cols = 6144
    let n = model.hidden_dim;    // inner dim = 1536
    let nb = n / 256;
    let pow2 = olorin::inference::matmul::pow2_table();

    let packed = olorin::inference::repack::q4k_repack_8x8(lw.w_gate, nc, n);

    eprintln!("bench: nc={nc} n={n} nb={nb}");
    eprintln!("{:>6}  {:>14}  {:>14}  {:>9}", "N", "matvec_ms", "gemm_ms", "speedup");
    eprintln!("{:-<50}", "");

    for &batch_n in &[4usize, 8, 16, 32] {
        // Quantize batch_n synthetic input vectors
        let mut all_qs: Vec<Vec<i8>> = Vec::new();
        let mut all_d: Vec<Vec<f32>> = Vec::new();
        let mut all_bsums: Vec<Vec<i16>> = Vec::new();
        for col in 0..batch_n {
            let mut input = vec![0.0f32; n];
            for i in 0..n {
                input[i] = 0.01 * ((col * 7 + i) % 97) as f32 - 0.5;
            }
            let mut qs = vec![0i8; n + 12];
            let mut d = vec![0.0f32; nb];
            let mut bsums = vec![0i16; nb * 16];
            unsafe {
                olorin::kernels::ffi_inference::quant_f32_q8k(
                    input.as_ptr(), qs.as_mut_ptr(), d.as_mut_ptr(),
                    bsums.as_mut_ptr(), n as i32,
                );
            }
            all_qs.push(qs);
            all_d.push(d);
            all_bsums.push(bsums);
        }

        // Build block_q8_Kx4 A-side for gemm
        let block_size = nb * 1168;
        let mut q8_a = vec![0u8; (batch_n / 4) * block_size];
        for g in 0..(batch_n / 4) {
            let r0 = g * 4;
            let mut row_d = vec![0.0f32; nb * 4];
            for b in 0..nb {
                for r in 0..4 { row_d[b * 4 + r] = all_d[r0 + r][b]; }
            }
            unsafe {
                olorin::kernels::ffi_inference::q8k_repack_4(
                    all_qs[r0].as_ptr(), all_qs[r0+1].as_ptr(),
                    all_qs[r0+2].as_ptr(), all_qs[r0+3].as_ptr(),
                    row_d.as_ptr(),
                    all_bsums[r0].as_ptr(), all_bsums[r0+1].as_ptr(),
                    all_bsums[r0+2].as_ptr(), all_bsums[r0+3].as_ptr(),
                    q8_a[g * block_size..].as_mut_ptr(), nb as i32,
                );
            }
        }

        let mut out = vec![0.0f32; batch_n * nc];
        let mut mv_scratch = vec![0u8; 128];
        let mut gm_scratch = vec![0u8; 128];
        let iters = std::cmp::max(5, 200 / std::cmp::max(1, batch_n));

        // Warm up both paths
        for k in 0..batch_n {
            unsafe {
                olorin::kernels::ffi_inference::q4k_8x8_q8k_matvec(
                    packed.as_ptr(), all_qs[k].as_ptr(), all_d[k].as_ptr(),
                    all_bsums[k].as_ptr(), pow2.as_ptr(), mv_scratch.as_mut_ptr(),
                    out[k * nc..].as_mut_ptr(), nc as i32, n as i32,
                );
            }
        }
        unsafe {
            olorin::kernels::ffi_inference::q4k_8x8_q8k_gemm(
                packed.as_ptr(), q8_a.as_ptr(), gm_scratch.as_mut_ptr(),
                out.as_mut_ptr(), nc as i32, n as i32,
                batch_n as i32, nc as i32,
            );
        }

        // Path A: N matvec calls
        let t0 = Instant::now();
        for _ in 0..iters {
            for k in 0..batch_n {
                unsafe {
                    olorin::kernels::ffi_inference::q4k_8x8_q8k_matvec(
                        packed.as_ptr(), all_qs[k].as_ptr(), all_d[k].as_ptr(),
                        all_bsums[k].as_ptr(), pow2.as_ptr(), mv_scratch.as_mut_ptr(),
                        out[k * nc..].as_mut_ptr(), nc as i32, n as i32,
                    );
                }
            }
        }
        let t_mv = t0.elapsed().as_secs_f64() / iters as f64;

        // Path B: 1 gemm call
        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe {
                olorin::kernels::ffi_inference::q4k_8x8_q8k_gemm(
                    packed.as_ptr(), q8_a.as_ptr(), gm_scratch.as_mut_ptr(),
                    out.as_mut_ptr(), nc as i32, n as i32,
                    batch_n as i32, nc as i32,
                );
            }
        }
        let t_gm = t0.elapsed().as_secs_f64() / iters as f64;

        let speedup = t_mv / t_gm;
        eprintln!(
            "{:>6}  {:>14.3}  {:>14.3}  {:>8.2}x",
            batch_n, t_mv * 1000.0, t_gm * 1000.0, speedup,
        );

        if batch_n == 8 {
            assert!(
                speedup >= 1.15,
                "N=8 gemm speedup {:.2}x is below the 1.15x gate", speedup,
            );
        }
    }
}

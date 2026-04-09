//! Bench — q4k_8x8_q8k_gemm (one call for N input cols) vs N repeated
//! q4k_8x8_q8k_matvec calls on the same problem.
//!
//! Run: PATH="/root/dev/eacompute/target/release:$PATH" \
//!      cargo test --release --test bench_q4k_gemm -- --nocapture --test-threads=1
//!
//! The point of this bench is to measure how much the gemm's per-(tile,
//! row, block) unpack + scale extraction reuse actually buys us across N
//! input columns. Per the plan's acceptance gate, gemm at N=8 should be
//! at least 1.5× faster than the matvec loop. If it isn't, the kernel's
//! load reuse isn't paying off and we need to restructure.

use std::path::Path;
use std::time::Instant;

fn model_path() -> String {
    let home = std::env::var("HOME").unwrap();
    format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")
}

#[test]
fn bench_gemm_vs_matvec_loop() {
    if !Path::new(&model_path()).exists() {
        eprintln!("SKIP: no model");
        return;
    }
    let gguf = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    // Layer 0 ffn_gate is Q4K in this quant (same pick as batch2-4).
    // Shape: [hidden_dim=1536, ffn_dim=6144] → n_rows=6144, n_cols=1536.
    let lw = &model.layers[0];
    assert_eq!(
        lw.w_gate_dtype,
        olorin::inference::matmul::GGML_TYPE_Q4_K,
        "bench requires Q4K weight"
    );
    let n_rows = model.ffn_dim[0];
    let n_cols = model.hidden_dim;
    let n_blocks = n_cols / 256;
    let row_bytes = n_blocks * 144;

    // Repack the weight once (outside the timing loop).
    let mut packed = vec![0u8; n_rows * row_bytes];
    unsafe {
        olorin::kernels::ffi_inference::q4k_repack_8x8(
            lw.w_gate,
            packed.as_mut_ptr(),
            n_rows as i32,
            n_cols as i32,
        );
    }
    let pow2 = olorin::inference::matmul::pow2_table();

    eprintln!(
        "bench: n_rows={n_rows}, n_cols={n_cols}, n_blocks={n_blocks}, weight_bytes={}",
        n_rows * row_bytes
    );
    eprintln!(
        "{:>6}  {:>14}  {:>14}  {:>10}",
        "N", "matvec-loop ms", "gemm ms", "speedup"
    );

    for &n in &[1usize, 2, 8, 32, 128] {
        // Synthetic input — we're timing, not verifying correctness here.
        // batch4 already proved bit-exactness for the same kernel.
        let qs_stride = n_cols + 12;
        let mut q8_qs = vec![5i8; qs_stride * n];
        let mut q8_d = vec![0.01f32; n_blocks * n];
        let mut q8_bsums = vec![17i16; n_blocks * 16 * n];

        // Make the data non-trivially non-uniform so the compiler can't
        // constant-fold anything (it shouldn't anyway, but belt-and-braces).
        for k in 0..n {
            q8_qs[k * qs_stride] = ((k as i32) % 127 - 63) as i8;
            q8_d[k * n_blocks] = 0.01 + (k as f32) * 0.0013;
            q8_bsums[k * n_blocks * 16] = (k as i16) % 31 - 15;
        }

        let mut out = vec![0.0f32; n_rows * n];
        let mut scratch = vec![0u8; 144];

        // Iteration count — total work should be ~100 ms worth so timer
        // noise is small. For N=1 that's many iters; for N=128, fewer.
        let iters: usize = std::cmp::max(5, 200 / std::cmp::max(1, n));

        // Warm-up: run each path once so the kernel .so is paged in and
        // the L2 state is predictable.
        for k in 0..n {
            unsafe {
                olorin::kernels::ffi_inference::q4k_8x8_q8k_matvec(
                    packed.as_ptr(),
                    q8_qs[k * qs_stride..].as_ptr(),
                    q8_d[k * n_blocks..].as_ptr(),
                    q8_bsums[k * n_blocks * 16..].as_ptr(),
                    pow2.as_ptr(),
                    scratch.as_mut_ptr(),
                    out[k * n_rows..].as_mut_ptr(),
                    n_rows as i32,
                    n_cols as i32,
                );
            }
        }
        let mut acc_scratch = vec![0.0f32; 2 * n];
        unsafe {
            olorin::kernels::ffi_inference::q4k_8x8_q8k_gemm(
                packed.as_ptr(),
                q8_qs.as_ptr(),
                q8_d.as_ptr(),
                q8_bsums.as_ptr(),
                pow2.as_ptr(),
                scratch.as_mut_ptr(),
                acc_scratch.as_mut_ptr(),
                out.as_mut_ptr(),
                n_rows as i32,
                n_cols as i32,
                n as i32,
            );
        }

        // Path A: N matvec calls
        let t0 = Instant::now();
        for _ in 0..iters {
            for k in 0..n {
                unsafe {
                    olorin::kernels::ffi_inference::q4k_8x8_q8k_matvec(
                        packed.as_ptr(),
                        q8_qs[k * qs_stride..].as_ptr(),
                        q8_d[k * n_blocks..].as_ptr(),
                        q8_bsums[k * n_blocks * 16..].as_ptr(),
                        pow2.as_ptr(),
                        scratch.as_mut_ptr(),
                        out[k * n_rows..].as_mut_ptr(),
                        n_rows as i32,
                        n_cols as i32,
                    );
                }
            }
        }
        let t_matvec_loop = t0.elapsed().as_secs_f64() / iters as f64;

        // Path B: 1 gemm call
        let t0 = Instant::now();
        for _ in 0..iters {
            unsafe {
                olorin::kernels::ffi_inference::q4k_8x8_q8k_gemm(
                    packed.as_ptr(),
                    q8_qs.as_ptr(),
                    q8_d.as_ptr(),
                    q8_bsums.as_ptr(),
                    pow2.as_ptr(),
                    scratch.as_mut_ptr(),
                    acc_scratch.as_mut_ptr(),
                    out.as_mut_ptr(),
                    n_rows as i32,
                    n_cols as i32,
                    n as i32,
                );
            }
        }
        let t_gemm = t0.elapsed().as_secs_f64() / iters as f64;

        let speedup = t_matvec_loop / t_gemm;
        eprintln!(
            "{:>6}  {:>14.3}  {:>14.3}  {:>9.2}x",
            n,
            t_matvec_loop * 1000.0,
            t_gemm * 1000.0,
            speedup,
        );

        // Plan acceptance gate: at N=8, gemm must be ≥ 1.5× faster than
        // the matvec loop. If it isn't, the per-tile unpack/scale reuse
        // isn't paying off and the kernel needs restructuring.
        if n == 8 {
            assert!(
                speedup >= 1.5,
                "N=8 gemm speedup {:.2}x is below the 1.5x acceptance gate",
                speedup
            );
        }
    }
}

//! Bit-exact correctness gate for q4k_8x8_q8k_gemm.
//!
//! For each N in {4, 8, 16, 32}, runs the fused gemm on layer-0 ffn_gate
//! against N synthetic input columns and asserts per-output to_bits()
//! equality vs. running q4k_8x8_q8k_matvec N times.
//!
//! Naming map:
//!   nr = N (batch size, must be % 4 == 0)
//!   nc = weight n_rows (ffn_dim, the output dimension)
//!   n  = weight n_cols (hidden_dim, the inner dimension)

use std::path::Path;

fn model_path() -> String {
    let home = std::env::var("HOME").unwrap();
    format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")
}

#[test]
fn gemm_matches_matvec_loop_bitexact() {
    // Spawn with 128 MB stack — the gemm kernel + large Q8K buffers at N=32
    // exceed the default test thread stack.
    let h = std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(gemm_matches_matvec_loop_bitexact_inner)
        .unwrap();
    h.join().unwrap();
}

fn gemm_matches_matvec_loop_bitexact_inner() {
    if !Path::new(&model_path()).exists() {
        eprintln!("SKIP: no model at {}", model_path());
        return;
    }
    let gguf = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    let lw = &model.layers[0];
    assert_eq!(lw.w_gate_dtype, olorin::inference::matmul::GGML_TYPE_Q4_K);

    let nc = model.ffn_dim[0]; // output cols (weight rows) = 6144
    let n = model.hidden_dim;  // inner dim (weight cols) = 1536
    let nb = n / 256;
    let pow2 = olorin::inference::matmul::pow2_table();

    // Repack weights: standard Q4K → block_q4_Kx8
    let packed = olorin::inference::repack::q4k_repack_8x8(lw.w_gate, nc, n);

    for &batch_n in &[4, 8, 16, 32] {
        eprintln!("Testing N={batch_n}...");
        assert!(batch_n % 4 == 0);

        // Build N synthetic input vectors with distinct patterns
        let mut inputs: Vec<Vec<f32>> = Vec::new();
        for col in 0..batch_n {
            let mut v = vec![0.0f32; n];
            for i in 0..n {
                v[i] = 0.01 * ((col * 7 + i) % 97) as f32 - 0.5;
            }
            inputs.push(v);
        }

        // Quantize each input to Q8K
        let mut all_qs: Vec<Vec<i8>> = Vec::new();
        let mut all_d: Vec<Vec<f32>> = Vec::new();
        let mut all_bsums: Vec<Vec<i16>> = Vec::new();
        for col in 0..batch_n {
            let mut qs = vec![0i8; n + 12]; // extra for alignment
            let mut d = vec![0.0f32; nb];
            let mut bsums = vec![0i16; nb * 16];
            unsafe {
                olorin::kernels::ffi_inference::quant_f32_q8k(
                    inputs[col].as_ptr(),
                    qs.as_mut_ptr(),
                    d.as_mut_ptr(),
                    bsums.as_mut_ptr(),
                    n as i32,
                );
            }
            all_qs.push(qs);
            all_d.push(d);
            all_bsums.push(bsums);
        }

        // ── Reference: run matvec N times ──
        let mut ref_out = vec![0.0f32; batch_n * nc];
        let mut mv_scratch = vec![0u8; 128];
        for col in 0..batch_n {
            unsafe {
                olorin::kernels::ffi_inference::q4k_8x8_q8k_matvec(
                    packed.as_ptr(),
                    all_qs[col].as_ptr(),
                    all_d[col].as_ptr(),
                    all_bsums[col].as_ptr(),
                    pow2.as_ptr(),
                    mv_scratch.as_mut_ptr(),
                    ref_out[col * nc..].as_mut_ptr(),
                    nc as i32,
                    n as i32,
                );
            }
        }

        // ── Repack Q8K inputs into block_q8_Kx4 ──
        // Process in groups of 4 rows
        let block_q8_kx4_size = nb * 1168;
        let mut q8_a = vec![0u8; (batch_n / 4) * block_q8_kx4_size];
        for group in 0..(batch_n / 4) {
            let r0 = group * 4;
            // row_d: interleaved as [d_r0_b0, d_r1_b0, d_r2_b0, d_r3_b0, d_r0_b1, ...]
            let mut row_d = vec![0.0f32; nb * 4];
            for b in 0..nb {
                for r in 0..4 {
                    row_d[b * 4 + r] = all_d[r0 + r][b];
                }
            }
            let dst_off = group * block_q8_kx4_size;
            unsafe {
                olorin::kernels::ffi_inference::q8k_repack_4(
                    all_qs[r0].as_ptr(),
                    all_qs[r0 + 1].as_ptr(),
                    all_qs[r0 + 2].as_ptr(),
                    all_qs[r0 + 3].as_ptr(),
                    row_d.as_ptr(),
                    all_bsums[r0].as_ptr(),
                    all_bsums[r0 + 1].as_ptr(),
                    all_bsums[r0 + 2].as_ptr(),
                    all_bsums[r0 + 3].as_ptr(),
                    q8_a[dst_off..].as_mut_ptr(),
                    nb as i32,
                );
            }
        }

        // ── Run gemm ──
        let mut gemm_out = vec![0.0f32; batch_n * nc];
        let mut gemm_scratch = vec![0u8; 128];
        unsafe {
            olorin::kernels::ffi_inference::q4k_8x8_q8k_gemm(
                packed.as_ptr(),
                q8_a.as_ptr(),
                gemm_scratch.as_mut_ptr(),
                gemm_out.as_mut_ptr(),
                nc as i32,     // bs = row stride = nc
                n as i32,      // inner dimension
                batch_n as i32, // nr = batch size
                nc as i32,     // nc = output cols
            );
        }

        // ── Compare bit-exact ──
        let mut mismatches = 0;
        for row in 0..batch_n {
            for col_idx in 0..nc {
                let ref_val = ref_out[row * nc + col_idx];
                let gemm_val = gemm_out[row * nc + col_idx];
                if ref_val.to_bits() != gemm_val.to_bits() {
                    mismatches += 1;
                    if mismatches <= 5 {
                        eprintln!(
                            "  MISMATCH N={batch_n} row={row} col={col_idx}: \
                             ref={ref_val:.6} gemm={gemm_val:.6} \
                             diff={:.3e}",
                            (ref_val - gemm_val).abs()
                        );
                    }
                }
            }
        }
        assert_eq!(
            mismatches, 0,
            "N={batch_n}: {mismatches} bit-exact mismatches out of {}",
            batch_n * nc
        );
        eprintln!(
            "  PASS: N={batch_n}, {} outputs bit-exact",
            batch_n * nc
        );
    }
}

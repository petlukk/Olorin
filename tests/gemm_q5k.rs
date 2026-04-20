//! Bit-exact correctness gate for q5k_gemm (ARM only).
//!
//! For each N in {4, 8, 16, 32}, runs the gemm on layer-0 wk against N
//! synthetic input columns and asserts per-output to_bits() equality vs.
//! running q5k_dot_q8k N times.

#![cfg(target_arch = "aarch64")]

use std::path::Path;

fn model_path() -> String {
    let home = std::env::var("HOME").unwrap();
    format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")
}

#[test]
fn q5k_gemm_matches_matvec_loop_bitexact() {
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
    assert_eq!(lw.wk_dtype, olorin::inference::matmul::GGML_TYPE_Q5_K,
        "test assumes layer 0 wk is Q5K");

    let head_dim = model.head_dim_k[0];
    let nc = model.n_kv_heads * head_dim;  // kv_dim, output rows
    let n = model.hidden_dim;              // inner dim
    let nb = n / 256;
    let pow2 = olorin::inference::matmul::pow2_table();

    eprintln!("Q5K GEMM parity test: nc={nc}, n={n}, nb={nb}");

    for &batch_n in &[4, 8, 16, 32] {
        eprintln!("Testing N={batch_n}...");
        assert!(batch_n % 4 == 0);

        let mut inputs: Vec<Vec<f32>> = Vec::new();
        for col in 0..batch_n {
            let mut v = vec![0.0f32; n];
            for i in 0..n {
                v[i] = 0.01 * ((col * 11 + i * 3) % 89) as f32 - 0.4;
            }
            inputs.push(v);
        }

        let mut all_qs: Vec<Vec<i8>> = Vec::new();
        let mut all_d: Vec<Vec<f32>> = Vec::new();
        let mut all_bsums: Vec<Vec<i16>> = Vec::new();
        for col in 0..batch_n {
            let mut qs = vec![0i8; n + 12];
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

        // Reference: per-token, per-row q5k_dot_q8k.
        let row_bytes = nb * olorin::inference::matmul::Q5K_BLOCK_BYTES;
        let mut ref_out = vec![0.0f32; batch_n * nc];
        for col in 0..batch_n {
            for row in 0..nc {
                let v = unsafe {
                    olorin::kernels::ffi_inference::q5k_dot_q8k(
                        lw.wk.add(row * row_bytes),
                        all_qs[col].as_ptr(),
                        all_bsums[col].as_ptr(),
                        nb as i32,
                        all_d[col].as_ptr(),
                        pow2.as_ptr(),
                    )
                };
                ref_out[col * nc + row] = v;
            }
        }

        // Repack inputs into block_q8_Kx4.
        let block_q8_kx4_size = nb * 1168;
        let mut q8_a = vec![0u8; (batch_n / 4) * block_q8_kx4_size];
        for group in 0..(batch_n / 4) {
            let r0 = group * 4;
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

        // Run q5k_gemm.
        let mut gemm_out = vec![0.0f32; batch_n * nc];
        let mut gemm_scratch = vec![0u8; 256];
        unsafe {
            olorin::kernels::ffi_inference::q5k_gemm(
                lw.wk,
                q8_a.as_ptr(),
                gemm_scratch.as_mut_ptr(),
                gemm_out.as_mut_ptr(),
                nc as i32,         // output_stride = row stride = nc
                n as i32,          // inner dimension
                batch_n as i32,    // nr
                nc as i32,         // nc = output rows
            );
        }

        let mut mismatches = 0;
        let mut max_diff = 0.0f32;
        for row in 0..batch_n {
            for col_idx in 0..nc {
                let ref_val = ref_out[row * nc + col_idx];
                let gemm_val = gemm_out[row * nc + col_idx];
                if ref_val.to_bits() != gemm_val.to_bits() {
                    mismatches += 1;
                    let d = (ref_val - gemm_val).abs();
                    if d > max_diff { max_diff = d; }
                    if mismatches <= 5 {
                        eprintln!(
                            "  MISMATCH N={batch_n} tok={row} row={col_idx}: \
                             ref={ref_val:.6} gemm={gemm_val:.6} \
                             diff={d:.3e}"
                        );
                    }
                }
            }
        }
        assert_eq!(
            mismatches, 0,
            "N={batch_n}: {mismatches} bit-exact mismatches (max diff {max_diff:.3e}) out of {}",
            batch_n * nc
        );
        eprintln!("  PASS: N={batch_n}, {} outputs bit-exact", batch_n * nc);
    }
}

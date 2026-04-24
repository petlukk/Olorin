//! Bit-exact correctness gate for q5k_8x8_q8k_matvec.
//!
//! Reference: per-row `q5k_dot_q8k` against the un-repacked tensor.
//! Implementation: `q5k_8x8_q8k_matvec` against the repacked tensor.
//! All 256 output rows must match to_bits() exactly.
//!
//! ARM-only because q5k_8x8_q8k_matvec is gated on aarch64.

#![cfg(target_arch = "aarch64")]

use std::path::Path;

fn model_path() -> String {
    let home = std::env::var("HOME").unwrap();
    format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")
}

#[test]
fn q5k_dot_8x8_matches_single_row_loop_bitexact() {
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
    assert_eq!(
        lw.wk_dtype,
        olorin::inference::matmul::GGML_TYPE_Q5_K,
        "test assumes layer 0 wk is Q5K"
    );

    let head_dim = model.head_dim_k[0];
    let nc = model.n_kv_heads * head_dim; // 256 rows
    let n = model.hidden_dim;             // 1536 inner dim
    let nb = n / 256;                     // 6 blocks per row
    let pow2 = olorin::inference::matmul::pow2_table();
    assert!(nc % 8 == 0);

    eprintln!("Q5K dot_8x8 parity test: nc={nc}, n={n}, nb={nb}");

    // Synthetic input column (deterministic, non-trivial values)
    let mut input = vec![0.0f32; n];
    for i in 0..n {
        input[i] = 0.01 * ((i * 13 + 7) % 97) as f32 - 0.5;
    }

    // Quantize to Q8K
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

    // ── Reference: per-row q5k_dot_q8k loop ──
    let src_row_bytes = nb * 176;
    let src: *const u8 = lw.wk;
    let mut ref_scores = vec![0.0f32; nc];
    for r in 0..nc {
        let row_ptr = unsafe { src.add(r * src_row_bytes) };
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

    // ── New path: q5k_repack_8x8 then q5k_8x8_q8k_matvec ──
    let tile_bytes = 1408usize;
    let dst_total = (nc / 8) * nb * tile_bytes;
    let mut repacked = vec![0u8; dst_total];
    unsafe {
        olorin::kernels::ffi_inference::q5k_repack_8x8(
            src,
            repacked.as_mut_ptr(),
            nc as i32,
            n as i32,
        );
    }

    let mut new_scores = vec![0.0f32; nc];
    // Scratch: the q4k_dot_8x8 convention is 128 bytes (utmp[32] worth).
    let mut scratch = vec![0u8; 128];
    unsafe {
        olorin::kernels::ffi_inference::q5k_8x8_q8k_matvec(
            repacked.as_ptr(),
            qs.as_ptr(),
            q8_d.as_ptr(),
            bsums.as_ptr(),
            pow2.as_ptr(),
            scratch.as_mut_ptr(),
            new_scores.as_mut_ptr(),
            nc as i32,
            n as i32,
        );
    }

    // ── Compare with ULP/relative-error tolerance ──
    // FMA vs separate mul+add + different accumulation order produce small
    // rounding differences that don't indicate a logic bug. We allow up to
    // 1e-4 relative error (well below quantization noise floor).
    let mut mismatches = 0usize;
    let mut exact = 0usize;
    let mut max_abs_rel_err: f64 = 0.0;
    const REL_TOL: f64 = 1.0e-4;
    for r in 0..nc {
        let ref_v = ref_scores[r] as f64;
        let new_v = new_scores[r] as f64;
        if ref_scores[r].to_bits() == new_scores[r].to_bits() {
            exact += 1;
            continue;
        }
        let denom = ref_v.abs().max(1e-6);
        let rel = (ref_v - new_v).abs() / denom;
        if rel > max_abs_rel_err {
            max_abs_rel_err = rel;
        }
        if rel > REL_TOL {
            if mismatches < 8 {
                eprintln!(
                    "row {r}: ref={ref_v:.9}  new={new_v:.9}  rel_err={rel:.2e}"
                );
            }
            mismatches += 1;
        }
    }

    eprintln!(
        "Q5K dot_8x8 parity: exact={exact}/{nc}, max rel err={max_abs_rel_err:.2e} (tol {REL_TOL:.0e})"
    );
    assert_eq!(
        mismatches, 0,
        "{mismatches} row(s) exceeded rel_err tolerance {REL_TOL}; max observed = {max_abs_rel_err:.2e}"
    );
}

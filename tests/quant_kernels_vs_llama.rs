//! Step 7–11: per-quant-format dot kernel correctness vs llama.cpp scalar refs.
//! Each test exercises one Eä SIMD kernel against a scalar Rust port of the
//! corresponding llama.cpp generic implementation, using real model weights
//! and BOS-embedding inputs.
//!
//! Run: cargo test --release --test quant_kernels_vs_llama -- --nocapture
//!
//! Requires: ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf

mod common;
use common::*;
use common::llama_refs::{
    llama_q8k_ref_block, llama_q4k_dot_ref, llama_q5k_dot_ref, llama_q6k_dot_ref,
};

#[test]
fn step7_q8k_quant_kernel_vs_llama_ref() {
    olorin::kernels::ffi::init().unwrap();

    // Construct 256 values with controlled amax = 2.0, including values that
    // scale to exact halves and near-halves on both sides of zero.
    let amax = 2.0f32;
    let mut x = vec![0.0f32; 256];
    x[0] = amax;
    x[1] = -amax;
    for k in 0..32 {
        let target = k as f32 + 0.5;
        x[2 + k]    = target / 63.5;
        x[34 + k]   = -target / 63.5;
    }
    for i in 66..256 {
        x[i] = ((i as f32) * 0.0137 - 0.5) * 0.01;
    }

    let mut o_qs = vec![0i8; 256 + 12];
    let mut o_d = vec![0.0f32; 1];
    let mut o_bsums = vec![0i16; 16];
    olorin::inference::matmul::quant_input(&x, &mut o_qs, &mut o_d, &mut o_bsums);

    let (r_qs, r_d, _r_bsums) = llama_q8k_ref_block(&x);

    eprintln!("=== Step 7: olorin quant_input vs llama q8_K_ref ===");
    eprintln!("olorin d[0]={:.8}  llama_ref d[0]={:.8}", o_d[0], r_d);
    eprintln!("|olorin d| = {:.8}   |llama d| = {:.8}", o_d[0].abs(), r_d.abs());

    let mut mismatches = 0usize;
    for j in 0..256 {
        let o_mag = o_qs[j].unsigned_abs() as i32;
        let r_mag = r_qs[j].unsigned_abs() as i32;
        if o_mag != r_mag {
            if mismatches < 12 {
                eprintln!("  qs[{:>3}] x={:>10.6}  scaled_olorin={:>9.4}  olorin={:>4} llama={:>4}  (|Δ|={})",
                    j, x[j], x[j] * 63.5, o_qs[j], r_qs[j], (o_mag - r_mag).abs());
            }
            mismatches += 1;
        }
    }
    eprintln!("Total qs magnitude mismatches: {} / 256", mismatches);

    if mismatches == 0 {
        eprintln!("HYPOTHESIS REJECTED: kernels agree on this synthetic input");
    } else {
        eprintln!("HYPOTHESIS CONFIRMED: olorin's quant_input differs from llama's q8_K_ref");
    }
}

#[test]
fn step8_q8k_quant_real_embedding() {
    if !has_model() { eprintln!("SKIP: no model"); return; }
    olorin::kernels::ffi::init().unwrap();

    let gguf = olorin::inference::gguf::GgufFile::open(std::path::Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    let hd = model.hidden_dim;

    let mut embed = vec![0.0f32; hd];
    olorin::inference::dequant::embed_lookup(model.embed_weight, model.embed_dtype, 2, &mut embed, hd);
    let scale = (hd as f32).sqrt();
    for v in embed.iter_mut() { *v *= scale; }

    let n_blocks = hd / 256;

    let mut o_qs = vec![0i8; hd + 12];
    let mut o_d = vec![0.0f32; n_blocks];
    let mut o_bsums = vec![0i16; n_blocks * 16];
    olorin::inference::matmul::quant_input(&embed, &mut o_qs, &mut o_d, &mut o_bsums);

    let mut r_qs_all = vec![0i8; hd];
    let mut r_d_all = vec![0.0f32; n_blocks];
    let mut r_bsums_all = vec![0i16; n_blocks * 16];
    for b in 0..n_blocks {
        let (qs, d, bsums) = llama_q8k_ref_block(&embed[b*256..(b+1)*256]);
        r_qs_all[b*256..(b+1)*256].copy_from_slice(&qs);
        r_d_all[b] = d;
        r_bsums_all[b*16..(b+1)*16].copy_from_slice(&bsums);
    }

    eprintln!("=== Step 8: Q8K quant — real BOS embedding ({hd} floats, {n_blocks} blocks) ===");

    let mut d_mismatch = false;
    for b in 0..n_blocks {
        let od = o_d[b].abs();
        let rd = r_d_all[b].abs();
        if (od - rd).abs() > 1e-10 {
            eprintln!("  d[{b}] MISMATCH: olorin={od:.8} llama={rd:.8}");
            d_mismatch = true;
        }
    }
    if !d_mismatch { eprintln!("  d: all {n_blocks} blocks match (magnitude)"); }

    let mut qs_mismatches = 0usize;
    for j in 0..hd {
        let o_mag = o_qs[j].unsigned_abs() as i32;
        let r_mag = r_qs_all[j].unsigned_abs() as i32;
        if o_mag != r_mag {
            if qs_mismatches < 5 {
                let b = j / 256;
                eprintln!("  qs[{j}] (block {b}) MISMATCH: olorin={} llama={} x={:.6}",
                    o_qs[j], r_qs_all[j], embed[j]);
            }
            qs_mismatches += 1;
        }
    }
    eprintln!("  qs: {qs_mismatches} / {hd} magnitude mismatches");

    let mut bsums_mismatches = 0usize;
    for g in 0..n_blocks * 16 {
        let o_bs = o_bsums[g].abs();
        let r_bs = r_bsums_all[g].abs();
        if o_bs != r_bs {
            if bsums_mismatches < 5 {
                eprintln!("  bsums[{g}] MISMATCH: olorin={} llama={}", o_bsums[g], r_bsums_all[g]);
            }
            bsums_mismatches += 1;
        }
    }
    eprintln!("  bsums: {bsums_mismatches} / {} magnitude mismatches", n_blocks * 16);

    let mut max_recon_err = 0.0f32;
    for b in 0..n_blocks {
        for j in 0..256 {
            let idx = b * 256 + j;
            let o_val = o_d[b] * (o_qs[idx] as f32);
            let r_val = r_d_all[b] * (r_qs_all[idx] as f32);
            let err = (o_val - r_val).abs();
            if err > max_recon_err { max_recon_err = err; }
        }
    }
    eprintln!("  reconstituted max error: {max_recon_err:.10}");

    assert_eq!(qs_mismatches, 0, "qs magnitude mismatches");
    assert_eq!(bsums_mismatches, 0, "bsums magnitude mismatches");
    assert!(max_recon_err < 1e-5, "reconstituted values diverge: {max_recon_err}");
    eprintln!("PASS: Q8K quant bit-exact (magnitude) on real embedding data");
}

#[test]
fn step9_q4k_dot_vs_llama_ref() {
    if !has_model() { eprintln!("SKIP: no model"); return; }
    olorin::kernels::ffi::init().unwrap();

    let gguf = olorin::inference::gguf::GgufFile::open(std::path::Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    let hd = model.hidden_dim;
    let n_blocks = hd / 256;

    let mut embed = vec![0.0f32; hd];
    olorin::inference::dequant::embed_lookup(model.embed_weight, model.embed_dtype, 2, &mut embed, hd);
    let scale = (hd as f32).sqrt();
    for v in embed.iter_mut() { *v *= scale; }

    let mut normed = vec![0.0f32; hd];
    olorin::kernels::ffi_inference::gemma4_rmsnorm(
        embed.as_ptr(), model.layers[0].attn_norm, normed.as_mut_ptr(),
        hd as i32, model.rms_eps,
    );

    let mut q8_qs = vec![0i8; hd + 12];
    let mut q8_d = vec![0.0f32; n_blocks];
    let mut q8_bsums = vec![0i16; n_blocks * 16];
    olorin::inference::matmul::quant_input(&normed, &mut q8_qs, &mut q8_d, &mut q8_bsums);

    let lw = &model.layers[0];
    let wq_ptr = lw.wq as *const u8;
    let wq_dtype = lw.wq_dtype;

    eprintln!("=== Step 9: Q4K dot — Olorin kernel vs llama scalar ref ===");
    eprintln!("  wq_dtype={wq_dtype} (expect 12=Q4K or 14=Q6K)");

    if wq_dtype != olorin::inference::matmul::GGML_TYPE_Q4_K {
        eprintln!("  SKIP: Wq is not Q4K (dtype={wq_dtype}), testing with gate weight instead");
        let gate_ptr = lw.w_gate as *const u8;
        let gate_dtype = lw.w_gate_dtype;
        eprintln!("  gate_dtype={gate_dtype}");
        if gate_dtype != olorin::inference::matmul::GGML_TYPE_Q4_K {
            eprintln!("  SKIP: no Q4K weights found");
            return;
        }
        test_q4k_dot_rows(gate_ptr, &q8_qs, &q8_d, &q8_bsums, n_blocks, 8,
            olorin::inference::matmul::pow2_table());
        return;
    }

    test_q4k_dot_rows(wq_ptr, &q8_qs, &q8_d, &q8_bsums, n_blocks, 8,
        olorin::inference::matmul::pow2_table());
}

fn test_q4k_dot_rows(
    weight_ptr: *const u8, q8_qs: &[i8], q8_d: &[f32], q8_bsums: &[i16],
    n_blocks: usize, n_rows: usize, pow2: &[f32; 32],
) {
    let row_bytes = n_blocks * 144;

    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;

    for row in 0..n_rows {
        let row_ptr = unsafe { weight_ptr.add(row * row_bytes) };

        let olorin_result = unsafe {
            olorin::kernels::ffi_inference::q4k_dot_q8k(
                row_ptr, q8_qs.as_ptr(), q8_bsums.as_ptr(),
                n_blocks as i32, q8_d.as_ptr(), pow2.as_ptr(),
            )
        };

        let llama_result = llama_q4k_dot_ref(
            row_ptr, q8_qs, q8_d, q8_bsums, n_blocks,
        );

        let abs_err = (olorin_result - llama_result).abs();
        let rel_err = if llama_result.abs() > 1e-6 {
            abs_err / llama_result.abs()
        } else {
            abs_err
        };

        if abs_err > max_abs_err { max_abs_err = abs_err; }
        if rel_err > max_rel_err { max_rel_err = rel_err; }

        if row < 4 || abs_err > 0.01 {
            eprintln!("  row {row}: olorin={olorin_result:.6} llama={llama_result:.6} abs_err={abs_err:.8} rel_err={rel_err:.6}");
        }
    }

    eprintln!("  max_abs_err={max_abs_err:.8}  max_rel_err={max_rel_err:.8}");
    assert!(max_abs_err < 0.01, "Q4K dot abs error too large: {max_abs_err}");
    assert!(max_rel_err < 1e-4, "Q4K dot rel error too large: {max_rel_err}");
    eprintln!("PASS: Q4K dot matches llama scalar reference");
}

#[test]
fn step10_q5k_dot_vs_llama_ref() {
    if !has_model() { eprintln!("SKIP: no model"); return; }
    olorin::kernels::ffi::init().unwrap();

    let gguf = olorin::inference::gguf::GgufFile::open(std::path::Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    let hd = model.hidden_dim;
    let n_blocks = hd / 256;

    let mut embed = vec![0.0f32; hd];
    olorin::inference::dequant::embed_lookup(model.embed_weight, model.embed_dtype, 2, &mut embed, hd);
    let scale = (hd as f32).sqrt();
    for v in embed.iter_mut() { *v *= scale; }

    let mut normed = vec![0.0f32; hd];
    olorin::kernels::ffi_inference::gemma4_rmsnorm(
        embed.as_ptr(), model.layers[0].attn_norm, normed.as_mut_ptr(),
        hd as i32, model.rms_eps,
    );

    let mut q8_qs = vec![0i8; hd + 12];
    let mut q8_d = vec![0.0f32; n_blocks];
    let mut q8_bsums = vec![0i16; n_blocks * 16];
    olorin::inference::matmul::quant_input(&normed, &mut q8_qs, &mut q8_d, &mut q8_bsums);

    let lw = &model.layers[0];
    assert_eq!(lw.wk_dtype, olorin::inference::matmul::GGML_TYPE_Q5_K, "Wk should be Q5K");
    let wk_ptr = lw.wk as *const u8;
    let head_dim = model.head_dim_k[0];
    let kv_dim = model.n_kv_heads * head_dim;
    let row_bytes_q5k = n_blocks * 176;

    eprintln!("=== Step 10: Q5K dot — Olorin kernel vs llama scalar ref ===");
    eprintln!("  kv_dim={kv_dim} head_dim={head_dim} n_blocks={n_blocks}");

    let n_rows = kv_dim.min(8);
    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;
    let pow2 = olorin::inference::matmul::pow2_table();

    for row in 0..n_rows {
        let row_ptr = unsafe { wk_ptr.add(row * row_bytes_q5k) };

        let olorin_result = unsafe {
            olorin::kernels::ffi_inference::q5k_dot_q8k(
                row_ptr, q8_qs.as_ptr(), q8_bsums.as_ptr(),
                n_blocks as i32, q8_d.as_ptr(), pow2.as_ptr(),
            )
        };

        let llama_result = llama_q5k_dot_ref(row_ptr, &q8_qs, &q8_d, &q8_bsums, n_blocks);

        let abs_err = (olorin_result - llama_result).abs();
        let rel_err = if llama_result.abs() > 1e-6 { abs_err / llama_result.abs() } else { abs_err };
        if abs_err > max_abs_err { max_abs_err = abs_err; }
        if rel_err > max_rel_err { max_rel_err = rel_err; }

        if row < 4 || abs_err > 0.01 {
            eprintln!("  row {row}: olorin={olorin_result:.6} llama={llama_result:.6} abs={abs_err:.8} rel={rel_err:.6}");
        }
    }

    eprintln!("  max_abs_err={max_abs_err:.8}  max_rel_err={max_rel_err:.8}");
    assert!(max_abs_err < 0.01, "Q5K dot abs error too large: {max_abs_err}");
    assert!(max_rel_err < 1e-4, "Q5K dot rel error too large: {max_rel_err}");
    eprintln!("PASS: Q5K dot matches llama scalar reference");
}

#[test]
fn step11_q6k_dot_vs_llama_ref() {
    if !has_model() { eprintln!("SKIP: no model"); return; }
    olorin::kernels::ffi::init().unwrap();

    let gguf = olorin::inference::gguf::GgufFile::open(std::path::Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    let hd = model.hidden_dim;
    let n_blocks = hd / 256;

    let mut embed = vec![0.0f32; hd];
    olorin::inference::dequant::embed_lookup(model.embed_weight, model.embed_dtype, 2, &mut embed, hd);
    let scale = (hd as f32).sqrt();
    for v in embed.iter_mut() { *v *= scale; }

    let mut normed = vec![0.0f32; hd];
    olorin::kernels::ffi_inference::gemma4_rmsnorm(
        embed.as_ptr(), model.layers[0].attn_norm, normed.as_mut_ptr(),
        hd as i32, model.rms_eps,
    );

    let mut q8_qs = vec![0i8; hd + 12];
    let mut q8_d = vec![0.0f32; n_blocks];
    let mut q8_bsums = vec![0i16; n_blocks * 16];
    olorin::inference::matmul::quant_input(&normed, &mut q8_qs, &mut q8_d, &mut q8_bsums);

    let lw = &model.layers[0];
    assert_eq!(lw.wq_dtype, olorin::inference::matmul::GGML_TYPE_Q6_K, "Wq should be Q6K");
    let wq_ptr = lw.wq as *const u8;
    let row_bytes = n_blocks * 210;
    let n_heads = model.n_heads;
    let head_dim = model.head_dim_k[0];
    let n_rows = (n_heads * head_dim).min(8);

    eprintln!("=== Step 11: Q6K dot — Olorin kernel vs llama scalar ref ===");
    eprintln!("  n_rows={n_rows} n_blocks={n_blocks}");

    let mut d_arr = vec![0.0f32; n_blocks];

    let mut max_abs_err = 0.0f32;
    let mut max_rel_err = 0.0f32;

    for row in 0..n_rows {
        let row_ptr = unsafe { wq_ptr.add(row * row_bytes) };

        for blk in 0..n_blocks {
            let d_off = unsafe { row_ptr.add(blk * 210 + 208) };
            let raw = unsafe { u16::from_le_bytes([*d_off, *d_off.add(1)]) };
            d_arr[blk] = olorin::inference::matmul::f16_to_f32_scalar(raw) * q8_d[blk];
        }

        let olorin_result = unsafe {
            olorin::kernels::ffi_inference::q6k_dot_q8k(
                row_ptr, q8_qs.as_ptr(), q8_bsums.as_ptr(),
                n_blocks as i32, d_arr.as_ptr(),
            )
        };

        let llama_result = llama_q6k_dot_ref(row_ptr, &q8_qs, &q8_d, &q8_bsums, n_blocks);

        let abs_err = (olorin_result - llama_result).abs();
        let rel_err = if llama_result.abs() > 1e-6 { abs_err / llama_result.abs() } else { abs_err };
        if abs_err > max_abs_err { max_abs_err = abs_err; }
        if rel_err > max_rel_err { max_rel_err = rel_err; }

        if row < 4 || abs_err > 0.01 {
            eprintln!("  row {row}: olorin={olorin_result:.6} llama={llama_result:.6} abs={abs_err:.8} rel={rel_err:.6}");
        }
    }

    eprintln!("  max_abs_err={max_abs_err:.8}  max_rel_err={max_rel_err:.8}");
    assert!(max_abs_err < 0.01, "Q6K dot abs error too large: {max_abs_err}");
    assert!(max_rel_err < 1e-4, "Q6K dot rel error too large: {max_rel_err}");
    eprintln!("PASS: Q6K dot matches llama scalar reference");
}

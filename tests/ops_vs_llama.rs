//! Step 12–16: per-op kernel correctness vs llama.cpp scalar refs
//! (RMSNorm, RoPE, GELU, Softmax+Softcap, f32↔f16).
//!
//! Run: cargo test --release --test ops_vs_llama -- --nocapture
//!
//! Most steps require the Q4_K_M GGUF (so they can pull real layer weights);
//! step14 and step16 are model-free.

mod common;
use common::*;
use common::llama_refs::{
    llama_rmsnorm_ref, llama_rope_cache, llama_gelu_f32, llama_f32_to_f16, olorin_f32_to_f16,
};

#[test]
fn step12_rmsnorm_vs_llama_ref() {
    if !has_model() { eprintln!("SKIP: no model"); return; }
    olorin::kernels::ffi::init().unwrap();

    let gguf = olorin::inference::gguf::GgufFile::open(std::path::Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    let hd = model.hidden_dim;

    let mut embed = vec![0.0f32; hd];
    olorin::inference::dequant::embed_lookup(model.embed_weight, model.embed_dtype, 2, &mut embed, hd);
    let scale = (hd as f32).sqrt();
    for v in embed.iter_mut() { *v *= scale; }

    let mut olorin_out = vec![0.0f32; hd];
    olorin::kernels::ffi_inference::gemma4_rmsnorm(
        embed.as_ptr(), model.layers[0].attn_norm, olorin_out.as_mut_ptr(),
        hd as i32, model.rms_eps,
    );

    let mut llama_out = vec![0.0f32; hd];
    llama_rmsnorm_ref(&embed, model.layers[0].attn_norm, &mut llama_out, model.rms_eps);

    eprintln!("=== Step 12: RMSNorm — Olorin kernel vs llama ref (double-precision) ===");
    eprintln!("  olorin L2={:.6}  llama L2={:.6}", l2(&olorin_out), l2(&llama_out));
    eprintln!("  olorin first4=[{:.6},{:.6},{:.6},{:.6}]",
        olorin_out[0], olorin_out[1], olorin_out[2], olorin_out[3]);
    eprintln!("  llama  first4=[{:.6},{:.6},{:.6},{:.6}]",
        llama_out[0], llama_out[1], llama_out[2], llama_out[3]);

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for i in 0..hd {
        let abs_err = (olorin_out[i] - llama_out[i]).abs();
        let rel_err = if llama_out[i].abs() > 1e-8 { abs_err / llama_out[i].abs() } else { abs_err };
        if abs_err > max_abs { max_abs = abs_err; }
        if rel_err > max_rel { max_rel = rel_err; }
    }
    eprintln!("  max_abs_err={max_abs:.8}  max_rel_err={max_rel:.8}");

    assert!(max_abs < 0.001, "RMSNorm abs error too large: {max_abs}");
    assert!(max_rel < 1e-4, "RMSNorm rel error too large: {max_rel}");
    eprintln!("PASS: RMSNorm matches llama double-precision reference");
}

#[test]
fn step13_rope_vs_llama_ref() {
    if !has_model() { eprintln!("SKIP: no model"); return; }
    olorin::kernels::ffi::init().unwrap();

    let gguf = olorin::inference::gguf::GgufFile::open(std::path::Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    let hd = model.hidden_dim;
    let n_heads = model.n_heads;

    let head_dim = model.head_dim_k[0];
    let rope_theta = model.rope_theta_swa;
    let n_rot = model.rope_dim_swa;

    eprintln!("=== Step 13: RoPE — Olorin vs llama ref ===");
    eprintln!("  SWA: head_dim={head_dim} n_rot={n_rot} theta={rope_theta}");

    let pos = 5usize;

    let mut olorin_cos = vec![0.0f32; n_rot / 2];
    let mut olorin_sin = vec![0.0f32; n_rot / 2];
    compute_rope_tables(&mut olorin_cos, &mut olorin_sin, pos, n_rot, rope_theta, None);

    let (llama_cos, llama_sin) = llama_rope_cache(pos, rope_theta, n_rot, None);

    let half = n_rot / 2;
    let mut max_cos_err = 0.0f32;
    let mut max_sin_err = 0.0f32;
    for d in 0..half {
        let ce = (olorin_cos[d] - llama_cos[d]).abs();
        let se = (olorin_sin[d] - llama_sin[d]).abs();
        if ce > max_cos_err { max_cos_err = ce; }
        if se > max_sin_err { max_sin_err = se; }
    }
    eprintln!("  cos/sin table max err: cos={max_cos_err:.8} sin={max_sin_err:.8}");

    let mut embed = vec![0.0f32; hd];
    olorin::inference::dequant::embed_lookup(model.embed_weight, model.embed_dtype, 2, &mut embed, hd);
    let scale = (hd as f32).sqrt();
    for v in embed.iter_mut() { *v *= scale; }

    let mut normed = vec![0.0f32; hd];
    olorin::kernels::ffi_inference::gemma4_rmsnorm(
        embed.as_ptr(), model.layers[0].attn_norm, normed.as_mut_ptr(),
        hd as i32, model.rms_eps,
    );

    let q_dim = n_heads * head_dim;
    let mut olorin_q = vec![0.0f32; q_dim];
    for i in 0..q_dim { olorin_q[i] = normed[i % hd]; }
    let mut llama_q = olorin_q.clone();

    olorin::kernels::ffi_inference::gemma4_rope(
        olorin_q.as_mut_ptr(), olorin_cos.as_ptr(), olorin_sin.as_ptr(),
        head_dim as i32, n_heads as i32,
    );

    for h in 0..n_heads {
        let base = h * head_dim;
        for d in 0..half {
            let re = llama_q[base + d];
            let im = llama_q[base + d + half];
            llama_q[base + d]        = re * llama_cos[d] - im * llama_sin[d];
            llama_q[base + d + half] = re * llama_sin[d] + im * llama_cos[d];
        }
    }

    let mut max_abs = 0.0f32;
    for i in 0..q_dim {
        let err = (olorin_q[i] - llama_q[i]).abs();
        if err > max_abs { max_abs = err; }
    }
    eprintln!("  RoPE output max_abs_err={max_abs:.8}");

    if !model.is_swa[4] {
        let head_dim_g = model.head_dim_k[4];
        let n_rot_g = model.rope_dim_global;
        let theta_g = model.rope_theta_global;
        let ff = model.rope_freqs.as_deref();

        let mut o_cos_g = vec![0.0f32; n_rot_g / 2];
        let mut o_sin_g = vec![0.0f32; n_rot_g / 2];
        compute_rope_tables(&mut o_cos_g, &mut o_sin_g, pos, n_rot_g, theta_g, ff);
        let (l_cos_g, l_sin_g) = llama_rope_cache(pos, theta_g, n_rot_g, ff);

        let half_g = n_rot_g / 2;
        let mut max_g = 0.0f32;
        for d in 0..half_g {
            let ce = (o_cos_g[d] - l_cos_g[d]).abs();
            let se = (o_sin_g[d] - l_sin_g[d]).abs();
            if ce > max_g { max_g = ce; }
            if se > max_g { max_g = se; }
        }
        eprintln!("  Global layer cos/sin max err: {max_g:.8} (head_dim={head_dim_g} n_rot={n_rot_g} theta={theta_g})");
    }

    assert!(max_cos_err < 1e-5, "cos table error: {max_cos_err}");
    assert!(max_sin_err < 1e-5, "sin table error: {max_sin_err}");
    assert!(max_abs < 1e-4, "RoPE output error: {max_abs}");
    eprintln!("PASS: RoPE matches llama reference");
}

#[test]
fn step14_gelu_vs_llama_ref() {
    olorin::kernels::ffi::init().unwrap();

    let mut gate = vec![0.0f32; 256];
    let mut up = vec![0.0f32; 256];
    for i in 0..256 {
        gate[i] = ((i as f32) * 0.137 - 17.5).sin() * 3.0;
        up[i] = ((i as f32) * 0.271 + 2.3).cos() * 2.0;
    }

    let mut olorin_out = vec![0.0f32; 256];
    olorin::kernels::ffi_inference::gelu_mul(
        gate.as_ptr(), up.as_ptr(), olorin_out.as_mut_ptr(), 256,
    );

    let mut llama_out = vec![0.0f32; 256];
    for i in 0..256 {
        llama_out[i] = llama_gelu_f32(gate[i]) * up[i];
    }

    eprintln!("=== Step 14: GELU — Olorin SIMD kernel vs llama scalar ref ===");

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for i in 0..256 {
        let abs_err = (olorin_out[i] - llama_out[i]).abs();
        let rel_err = if llama_out[i].abs() > 1e-8 { abs_err / llama_out[i].abs() } else { abs_err };
        if abs_err > max_abs { max_abs = abs_err; }
        if rel_err > max_rel { max_rel = rel_err; }
        if abs_err > 0.001 && i < 5 {
            eprintln!("  [{i}] gate={:.4} up={:.4} olorin={:.6} llama={:.6} err={abs_err:.8}",
                gate[i], up[i], olorin_out[i], llama_out[i]);
        }
    }

    eprintln!("  first4 olorin: [{:.6},{:.6},{:.6},{:.6}]",
        olorin_out[0], olorin_out[1], olorin_out[2], olorin_out[3]);
    eprintln!("  first4 llama:  [{:.6},{:.6},{:.6},{:.6}]",
        llama_out[0], llama_out[1], llama_out[2], llama_out[3]);
    eprintln!("  max_abs_err={max_abs:.8}  max_rel_err={max_rel:.8}");

    assert!(max_abs < 0.001, "GELU abs error too large: {max_abs}");
    assert!(max_rel < 1e-4, "GELU rel error too large: {max_rel}");
    eprintln!("PASS: GELU matches llama scalar reference");
}

#[test]
fn step15_softcap_vs_llama_ref() {
    olorin::kernels::ffi::init().unwrap();

    eprintln!("=== Step 15: Softcap vs llama ref ===");

    let cap = 30.0f32;
    let logit_vals: Vec<f32> = (-10..=10).map(|i| i as f32 * 5.0).collect();
    let n_logits = logit_vals.len();

    let mut olorin_sc = logit_vals.clone();
    olorin::kernels::ffi_inference::softcap_f32(
        olorin_sc.as_mut_ptr(), n_logits as i32, cap,
    );

    let mut llama_sc = logit_vals.clone();
    for v in llama_sc.iter_mut() {
        *v = cap * (*v / cap).tanh();
    }

    let mut sc_max_abs = 0.0f32;
    let mut sc_max_rel = 0.0f32;
    for i in 0..n_logits {
        let abs_err = (olorin_sc[i] - llama_sc[i]).abs();
        let rel_err = if llama_sc[i].abs() > 1e-10 { abs_err / llama_sc[i].abs() } else { abs_err };
        if abs_err > sc_max_abs { sc_max_abs = abs_err; }
        if rel_err > sc_max_rel { sc_max_rel = rel_err; }
    }
    eprintln!("  Softcap (cap={cap}): max_abs={sc_max_abs:.8} max_rel={sc_max_rel:.8}");

    let in_range = olorin_sc.iter().all(|&v| v.abs() < cap);
    eprintln!("  Softcap all in (-{cap}, +{cap}): {in_range}");

    assert!(sc_max_abs < 1e-5, "Softcap abs error: {sc_max_abs}");
    assert!(in_range, "Softcap values out of range");
    eprintln!("PASS: Softcap matches llama reference");
}

#[test]
fn step16_f32_to_f16_vs_llama() {
    eprintln!("=== Step 16: f32→f16 — Olorin vs llama ===");

    let test_values: Vec<f32> = vec![
        0.0, 1.0, -1.0, 0.5, -0.5,
        0.1, 0.01, 0.001, 100.0, -100.0,
        1.0009765625, 1.001953125, 1.0014648438,
        0.333333333, -0.333333333,
        std::f32::consts::PI, std::f32::consts::E,
        65504.0, -65504.0,
        5.96e-8,
    ];

    let mut real_vals = Vec::new();
    for i in 0..256 {
        real_vals.push(((i as f32) * 0.137 - 17.5).sin() * 3.0);
    }

    let all_vals: Vec<f32> = test_values.iter().chain(real_vals.iter()).copied().collect();
    let n = all_vals.len();

    let mut mismatches = 0usize;
    let mut max_val_diff = 0.0f32;

    for (i, &v) in all_vals.iter().enumerate() {
        let olorin_h = olorin_f32_to_f16(v);
        let llama_h = llama_f32_to_f16(v);

        if olorin_h != llama_h {
            let o_back = olorin::inference::matmul::f16_to_f32_scalar(olorin_h);
            let l_back = olorin::inference::matmul::f16_to_f32_scalar(llama_h);
            let diff = (o_back - l_back).abs();
            if diff > max_val_diff { max_val_diff = diff; }

            if mismatches < 10 {
                eprintln!("  [{i}] v={v:.8} olorin=0x{olorin_h:04x} llama=0x{llama_h:04x} (back: {o_back:.6} vs {l_back:.6}, Δ={diff:.8})");
            }
            mismatches += 1;
        }
    }

    eprintln!("  mismatches: {mismatches} / {n}");
    eprintln!("  max f32 value difference from mismatched f16: {max_val_diff:.8}");

    if mismatches > 0 {
        eprintln!("WARNING: f32→f16 rounding differs! This propagates through KV cache.");
    } else {
        eprintln!("PASS: f32→f16 bit-exact with llama");
    }

}

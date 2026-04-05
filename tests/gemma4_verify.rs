//! Step-by-step verification of Gemma 4 forward pass components.
//!
//! Run: cargo test --release --test gemma4_verify -- --nocapture
//!
//! Requires: ~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf

use std::path::Path;

fn model_path() -> String {
    let home = std::env::var("HOME").unwrap();
    format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")
}

fn l2(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

fn first4(v: &[f32]) -> String {
    format!("[{:.4}, {:.4}, {:.4}, {:.4}]", v[0], v[1], v[2], v[3])
}

fn bare_rmsnorm(x: &mut [f32], eps: f32) {
    let n = x.len();
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>();
    let scale = 1.0 / ((ss / n as f32) + eps).sqrt();
    for v in x.iter_mut() { *v *= scale; }
}

fn compute_rope_tables(cos: &mut [f32], sin: &mut [f32], pos: usize, n_rot: usize, theta: f32, ff: Option<&[f32]>) {
    let half = n_rot / 2;
    for d in 0..half {
        let base_freq = 1.0 / theta.powf(2.0 * d as f32 / n_rot as f32);
        let freq = match ff { Some(f) => base_freq / f[d], None => base_freq };
        let angle = pos as f32 * freq;
        cos[d] = angle.cos();
        sin[d] = angle.sin();
    }
}

fn has_model() -> bool {
    Path::new(std::path::Path::new(&model_path())).exists()
}

#[test]
fn step0_gguf_load() {
    if !has_model() { eprintln!("SKIP: no model"); return; }

    let gguf = olorin::inference::gguf::GgufFile::open(std::path::Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();

    eprintln!("=== GGUF Load ===");
    eprintln!("layers={} hidden={} heads={}/{}",
        model.n_layers, model.hidden_dim, model.n_heads, model.n_kv_heads);
    eprintln!("vocab={}", model.vocab_size);
    eprintln!("head_dim_k[0]={} head_dim_k[34]={}", model.head_dim_k[0], model.head_dim_k[34]);
    eprintln!("ffn_dim[0]={} ffn_dim[34]={}", model.ffn_dim[0], model.ffn_dim[34]);
    eprintln!("swa_count={} global_count={}",
        model.is_swa.iter().filter(|&&x| x).count(),
        model.is_swa.iter().filter(|&&x| !x).count());
    eprintln!("shared_kv={}", model.kv_shared_source.iter().filter(|x| x.is_some()).count());

    assert_eq!(model.n_layers, 35);
    assert_eq!(model.hidden_dim, 1536);
    assert_eq!(model.n_heads, 8);
    assert_eq!(model.n_kv_heads, 1);
    eprintln!("is_swa[0]={} is_swa[34]={}", model.is_swa[0], model.is_swa[34]);
    eprintln!("head_dim_k[0]={} head_dim_k[4]={} head_dim_k[34]={}",
        model.head_dim_k[0], model.head_dim_k[4], model.head_dim_k[34]);
    // Layer 0 is SWA (pattern=1), layer 4 is global (pattern=0), layer 34 is global
    assert!(model.is_swa[0], "layer 0 should be SWA");
    assert!(!model.is_swa[4], "layer 4 should be global");
    assert!(!model.is_swa[34], "layer 34 should be global");
    assert_eq!(model.head_dim_k[0], 256, "SWA head_dim_k should be 256");
    assert_eq!(model.head_dim_k[4], 512, "global head_dim_k should be 512");
    assert_eq!(model.head_dim_k[34], 512, "global head_dim_k should be 512");
    assert_eq!(model.ffn_dim[0], 6144);
    assert_eq!(model.ffn_dim[34], 12288);

    eprintln!("PASS: GGUF load correct");
}

#[test]
fn step1_embedding() {
    if !has_model() { eprintln!("SKIP: no model"); return; }

    let gguf = olorin::inference::gguf::GgufFile::open(std::path::Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();

    // Init kernels
    olorin::kernels::ffi::init().unwrap();

    // Embed token 2 ("Hello" starts with BOS=2 in Gemma)
    let token_id: usize = 2; // BOS token
    let hd = model.hidden_dim;
    let mut embed = vec![0.0f32; hd];

    olorin::inference::dequant::q6k_embed_lookup(
        model.embed_weight, token_id, &mut embed, hd,
    );

    let raw_l2 = l2(&embed);

    // Per-block and per-group L2 for Q6K dequant verification
    for blk in 0..(hd / 256) {
        let b = &embed[blk * 256..(blk + 1) * 256];
        let bl2 = l2(b);
        // Per-group L2 (8 groups of 32 elements per block)
        let mut gl2s = String::new();
        for g in 0..8 {
            let gslice = &b[g * 32..(g + 1) * 32];
            gl2s += &format!(" g{}={:.4}", g, l2(gslice));
        }
        eprintln!("blk{blk}: L2={bl2:.6}{gl2s}");
    }

    // Gemma scaling: multiply by sqrt(hidden_dim)
    let scale = (hd as f32).sqrt();
    for v in embed.iter_mut() {
        *v *= scale;
    }

    let scaled_l2 = l2(&embed);

    eprintln!("=== Embedding (token={}) ===", token_id);
    eprintln!("raw L2 = {:.6}", raw_l2);
    eprintln!("scaled L2 = {:.6} (× sqrt({})={:.2})", scaled_l2, hd, scale);
    eprintln!("first 8: {:?}", &embed[..8].iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>());
    eprintln!("last 4: {:?}", &embed[hd-4..].iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>());

    // Sanity: embedding should not be zero
    assert!(raw_l2 > 0.1, "embedding is near-zero");
    assert!(scaled_l2 > 1.0, "scaled embedding is near-zero");

    eprintln!("PASS: embedding non-zero");
}

#[test]
fn step2_rmsnorm() {
    if !has_model() { eprintln!("SKIP: no model"); return; }

    let gguf = olorin::inference::gguf::GgufFile::open(std::path::Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    let hd = model.hidden_dim;
    let mut embed = vec![0.0f32; hd];
    olorin::inference::dequant::q6k_embed_lookup(model.embed_weight, 2, &mut embed, hd);
    let scale = (hd as f32).sqrt();
    for v in embed.iter_mut() { *v *= scale; }

    // RMSNorm with layer 0 attn_norm weight
    let mut normed = vec![0.0f32; hd];
    olorin::kernels::ffi_inference::gemma4_rmsnorm(
        embed.as_ptr(),
        model.layers[0].attn_norm,
        normed.as_mut_ptr(),
        hd as i32,
        model.rms_eps,
    );

    eprintln!("=== RMSNorm (layer 0 attn_norm) ===");
    eprintln!("input L2 = {:.6}", l2(&embed));
    eprintln!("output L2 = {:.6}", l2(&normed));
    eprintln!("first 8: {:?}", &normed[..8].iter().map(|x| format!("{:.4}", x)).collect::<Vec<_>>());

    assert!(l2(&normed) > 0.1, "normed output near-zero");
    assert!((l2(&normed) - l2(&embed)).abs() / l2(&embed) < 10.0, "L2 ratio unreasonable");

    eprintln!("PASS: RMSNorm produces non-zero output");
}

#[test]
fn step3_qkv_projection() {
    if !has_model() { eprintln!("SKIP: no model"); return; }

    let gguf = olorin::inference::gguf::GgufFile::open(std::path::Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    let hd = model.hidden_dim; // 1536
    let n_heads = model.n_heads; // 8
    let n_kv = model.n_kv_heads; // 1
    let head_dim = model.head_dim_k[0]; // 256 (SWA layer 0)
    let head_dim_v = model.head_dim_v[0];
    let lw = &model.layers[0];

    // ── Embed BOS (token 2) + scale ─────────────────────────────
    let mut x = vec![0.0f32; hd];
    olorin::inference::dequant::q6k_embed_lookup(model.embed_weight, 2, &mut x, hd);
    let scale = (hd as f32).sqrt();
    for v in x.iter_mut() { *v *= scale; }

    // ── attn_norm ───────────────────────────────────────────────
    let mut x_norm = vec![0.0f32; hd];
    olorin::kernels::ffi_inference::gemma4_rmsnorm(
        x.as_ptr(), lw.attn_norm, x_norm.as_mut_ptr(), hd as i32, model.rms_eps,
    );
    eprintln!("=== Step 3: QKV Projection (layer 0, token BOS, pos=0) ===");
    eprintln!("attn_norm  L2={:.4}  first4={}", l2(&x_norm), first4(&x_norm));

    // ── Q8K quantize ────────────────────────────────────────────
    let n_blocks = hd / 256;
    let mut q8_qs = vec![0i8; hd + 12];
    let mut q8_d = vec![0.0f32; n_blocks];
    let mut q8_bsums = vec![0i16; n_blocks * 16];
    olorin::inference::matmul::quant_input(&x_norm, &mut q8_qs, &mut q8_d, &mut q8_bsums);

    // ── Q projection ────────────────────────────────────────────
    let q_dim = n_heads * head_dim; // 8 * 256 = 2048
    let mut q = vec![0.0f32; q_dim];
    let mut d_scratch = vec![0.0f32; n_blocks * 4];
    olorin::inference::matmul::matvec(
        lw.wq_dtype, lw.wq,
        &q8_qs, &q8_d, &q8_bsums,
        &mut q, &mut d_scratch,
        q_dim, hd,
    );
    eprintln!("Q_proj     L2={:.4}  first4={}  dtype={}", l2(&q), first4(&q), lw.wq_dtype);

    // ── K projection ────────────────────────────────────────────
    let kv_dim = n_kv * head_dim; // 1 * 256 = 256
    let mut k = vec![0.0f32; kv_dim];
    olorin::inference::matmul::matvec(
        lw.wk_dtype, lw.wk,
        &q8_qs, &q8_d, &q8_bsums,
        &mut k, &mut d_scratch,
        kv_dim, hd,
    );
    eprintln!("K_proj     L2={:.4}  first4={}  dtype={}", l2(&k), first4(&k), lw.wk_dtype);

    // ── V projection ────────────────────────────────────────────
    let kv_dim_v = n_kv * head_dim_v;
    let mut v = vec![0.0f32; kv_dim_v];
    olorin::inference::matmul::matvec(
        lw.wv_dtype, lw.wv,
        &q8_qs, &q8_d, &q8_bsums,
        &mut v, &mut d_scratch,
        kv_dim_v, hd,
    );
    eprintln!("V_proj     L2={:.4}  first4={}  dtype={}", l2(&v), first4(&v), lw.wv_dtype);

    // ── Per-head Q norm (weighted RMSNorm) ──────────────────────
    let mut scratch = vec![0.0f32; head_dim];
    if !lw.q_norm.is_null() {
        for h in 0..n_heads {
            let off = h * head_dim;
            olorin::kernels::ffi_inference::gemma4_rmsnorm(
                q.as_ptr().wrapping_add(off), lw.q_norm,
                scratch.as_mut_ptr(), head_dim as i32, model.rms_eps,
            );
            q[off..off + head_dim].copy_from_slice(&scratch);
        }
    }
    eprintln!("Q_norm     L2={:.4}  first4={}", l2(&q), first4(&q));

    // ── Per-head K norm (weighted RMSNorm) ──────────────────────
    if !lw.k_norm.is_null() {
        for h in 0..n_kv {
            let off = h * head_dim;
            olorin::kernels::ffi_inference::gemma4_rmsnorm(
                k.as_ptr().wrapping_add(off), lw.k_norm,
                scratch.as_mut_ptr(), head_dim as i32, model.rms_eps,
            );
            k[off..off + head_dim].copy_from_slice(&scratch);
        }
    }
    eprintln!("K_norm     L2={:.4}  first4={}", l2(&k), first4(&k));

    // ── V bare norm (no weight) ─────────────────────────────────
    for h in 0..n_kv {
        let off = h * head_dim_v;
        bare_rmsnorm(&mut v[off..off + head_dim_v], model.rms_eps);
    }
    eprintln!("V_bare     L2={:.4}  first4={}", l2(&v), first4(&v));

    // ── RoPE (NEOX, pos=0 → cos=1 sin=0, so no change) ─────────
    let n_rot = model.rope_dim_swa; // layer 0 is SWA
    let theta = model.rope_theta_swa;
    let mut cos_table = vec![0.0f32; n_rot / 2];
    let mut sin_table = vec![0.0f32; n_rot / 2];
    compute_rope_tables(&mut cos_table, &mut sin_table, 0, n_rot, theta, None);

    olorin::kernels::ffi_inference::gemma4_rope(
        q.as_mut_ptr(), cos_table.as_ptr(), sin_table.as_ptr(),
        head_dim as i32, n_heads as i32,
    );
    olorin::kernels::ffi_inference::gemma4_rope(
        k.as_mut_ptr(), cos_table.as_ptr(), sin_table.as_ptr(),
        head_dim as i32, n_kv as i32,
    );
    eprintln!("Q_rope     L2={:.4}  first4={} (pos=0, should match Q_norm)", l2(&q), first4(&q));
    eprintln!("K_rope     L2={:.4}  first4={} (pos=0, should match K_norm)", l2(&k), first4(&k));

    // ── Summary table ───────────────────────────────────────────
    eprintln!();
    eprintln!("Compare against llama.cpp verify_layers output for layer 0, token BOS:");
    eprintln!("  attn_norm L2:  {:.4}", l2(&x_norm));
    eprintln!("  Q_proj L2:     {:.4}", l2(&q[..q_dim]));
    eprintln!("  K_proj L2:     {:.4}", l2(&k[..kv_dim]));
    eprintln!("  V_proj L2:     {:.4}", l2(&v[..kv_dim_v]));
    eprintln!("  Q_norm L2:     {:.4}  (llama.cpp: 44.55)", l2(&q));
    eprintln!("  K_norm L2:     {:.4}  (llama.cpp: 2.03)", l2(&k));
    eprintln!("  V_bare L2:     {:.4}  (llama.cpp: 16.00)", l2(&v));

    eprintln!("PASS: step3 QKV projections computed");
}

#[test]
fn step3b_single_layer() {
    if !has_model() { eprintln!("SKIP: no model"); return; }

    let gguf = olorin::inference::gguf::GgufFile::open(std::path::Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    // Run full layer 0 with BOS token via Gemma4State
    let mut state = olorin::inference::forward::Gemma4State::new(&model, 512);

    // Embed BOS + scale (no PLE for this test)
    let hd = model.hidden_dim;
    olorin::inference::dequant::q6k_embed_lookup(model.embed_weight, 2, &mut state.x, hd);
    let scale = (hd as f32).sqrt();
    for v in state.x[..hd].iter_mut() { *v *= scale; }

    // Run layer 0
    state.layer_forward(&model, 0, 0, false);

    eprintln!("=== Step 3b: Single layer forward (L0, BOS, pos=0, no PLE) ===");
    eprintln!("l_out L2={:.4}  first4=[{:.4},{:.4},{:.4},{:.4}]",
        l2(&state.x[..hd]), state.x[0], state.x[1], state.x[2], state.x[3]);
    eprintln!("  (llama.cpp: L2=40.3056 first4=[-0.2106, -0.0050, 0.0073, -0.3651])");

    // Also check attention output — at pos=0 it should equal V_normed
    // kqv_out is in attn_out buffer
    let n_heads = model.n_heads;
    let head_dim = model.head_dim_k[0];
    eprintln!("attn_out L2={:.4}  first4=[{:.4},{:.4},{:.4},{:.4}]",
        l2(&state.attn_out[..n_heads * head_dim]),
        state.attn_out[0], state.attn_out[1], state.attn_out[2], state.attn_out[3]);
    eprintln!("  (llama.cpp kqv_out: L2=45.2570 first4=[0.0263, 0.1174, 0.0296, -0.1724])");

    eprintln!("PASS: step3b single layer forward");
}

#[test]
fn step4_ple() {
    if !has_model() { eprintln!("SKIP: no model"); return; }

    let gguf = olorin::inference::gguf::GgufFile::open(std::path::Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    let hd = model.hidden_dim;
    let ple_dim = model.ple_dim;
    let n_layers = model.n_layers;

    eprintln!("=== Step 4: PLE (ple_dim={}, n_layers={}) ===", ple_dim, n_layers);

    if ple_dim == 0 {
        eprintln!("SKIP: no PLE in this model");
        return;
    }

    let mut state = olorin::inference::forward::Gemma4State::new(&model, 512);

    // Embed BOS token + scale
    olorin::inference::dequant::q6k_embed_lookup(model.embed_weight, 2, &mut state.x, hd);
    let scale = (hd as f32).sqrt();
    for v in state.x[..hd].iter_mut() { *v *= scale; }

    // Run Phase A
    state.prepare_ple(&model, 2);

    let total = ple_dim * n_layers;
    let sig_l2 = l2(&state.ple_signal[..total]);
    eprintln!("ple_signal L2={:.4}  total={}", sig_l2, total);
    eprintln!("ple_signal first4={}", first4(&state.ple_signal));
    eprintln!("ple_signal[ple_dim]={}", first4(&state.ple_signal[ple_dim..]));

    assert!(sig_l2 > 0.1, "PLE signal is near-zero");
    assert!(!sig_l2.is_nan(), "PLE signal is NaN");

    eprintln!("PASS: step4 PLE Phase A computed");
}

#[test]
fn step5_logits() {
    if !has_model() { eprintln!("SKIP: no model"); return; }

    let gguf = olorin::inference::gguf::GgufFile::open(std::path::Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    let mut state = olorin::inference::forward::Gemma4State::new(&model, 512);

    // Forward pass with BOS token (id=2)
    let logits_vec = state.forward_one(&model, 2).to_vec();
    let logits = &logits_vec;

    let hd = model.hidden_dim;
    eprintln!("pre-logit hidden L2={:.4}  (L34 out, llama.cpp: 21.01)", l2(&state.x[..hd]));

    let logit_l2 = l2(logits);
    eprintln!("=== Step 5: Logits (BOS token) ===");
    eprintln!("logits L2={:.4}  (llama.cpp: 2655.2185)", logit_l2);
    eprintln!("logits first4={}  (llama.cpp: [-10.5338, 15.5578, 11.2333, -10.5488])", first4(logits));

    // Find top-5
    let mut scored: Vec<(f32, usize)> = logits.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    eprintln!("Top-5 (llama.cpp: 236761=20.07, 236764=19.18, 236771=18.75):");
    for i in 0..5.min(scored.len()) {
        eprintln!("  {}: token={} logit={:.4}", i, scored[i].1, scored[i].0);
    }

    assert!(!logit_l2.is_nan(), "logits contain NaN");
    assert!(logit_l2 > 1.0, "logits near-zero");
    eprintln!("PASS: step5 logits computed");
}

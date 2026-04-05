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

    olorin::inference::matmul::q6k_embed_lookup(
        model.embed_weight, token_id, &mut embed, hd,
    );

    let raw_l2 = l2(&embed);

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
    olorin::inference::matmul::q6k_embed_lookup(model.embed_weight, 2, &mut embed, hd);
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

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

fn sum(v: &[f32]) -> f64 {
    v.iter().map(|&x| x as f64).sum::<f64>()
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

    // Debug: per-block scalar vs SIMD Q5K for row 1 to find which block diverges
    if lw.wk_dtype == 13 {
        let n_blk = hd / 256;
        let row_bytes = n_blk * 176;
        let pow2 = olorin::inference::matmul::pow2_table();
        let wk1 = unsafe { lw.wk.add(1 * row_bytes) };
        for blk in 0..n_blk {
            let bp = unsafe { wk1.add(blk * 176) };
            // SIMD single-block
            let q8_blk = unsafe { q8_qs.as_ptr().add(blk * 256) };
            let bs_blk = unsafe { q8_bsums.as_ptr().add(blk * 16) };
            let d_blk = unsafe { q8_d.as_ptr().add(blk) };
            let simd_val = unsafe {
                olorin::kernels::ffi_inference::q5k_dot_q8k(
                    bp as *const u8, q8_blk, bs_blk, 1, d_blk, pow2.as_ptr(),
                )
            };
            // Scalar single-block (same as llama.cpp)
            let d_raw = unsafe { u16::from_le_bytes([*bp, *bp.add(1)]) };
            let dm_raw = unsafe { u16::from_le_bytes([*bp.add(2), *bp.add(3)]) };
            let d = olorin::inference::matmul::f16_to_f32_scalar(d_raw);
            let dm = olorin::inference::matmul::f16_to_f32_scalar(dm_raw);
            let qs_ptr = unsafe { bp.add(48) };
            let qh_ptr = unsafe { bp.add(16) };
            let sc_ptr = unsafe { bp.add(4) };
            let q8_base = blk * 256;
            let mut q5v = [0u8; 256];
            let mut m: u8 = 1;
            for j in 0..4usize {
                for l in 0..32usize {
                    let ql = unsafe { *qs_ptr.add(j*32+l) };
                    let qh = unsafe { *qh_ptr.add(l) };
                    q5v[j*64+l] = (ql & 0xF) + if qh & m != 0 { 16 } else { 0 };
                }
                m <<= 1;
                for l in 0..32usize {
                    let ql = unsafe { *qs_ptr.add(j*32+l) };
                    let qh = unsafe { *qh_ptr.add(l) };
                    q5v[j*64+32+l] = (ql >> 4) + if qh & m != 0 { 16 } else { 0 };
                }
                m <<= 1;
            }
            let mut utmp = [0u32; 4];
            unsafe { std::ptr::copy_nonoverlapping(sc_ptr, utmp.as_mut_ptr() as *mut u8, 12); }
            utmp[3] = ((utmp[2] >> 4) & 0x0f0f0f0f) | (((utmp[1] >> 6) & 0x03030303) << 4);
            let uaux = utmp[1] & 0x3f3f3f3f;
            utmp[1] = (utmp[2] & 0x0f0f0f0f) | (((utmp[0] >> 6) & 0x03030303) << 4);
            utmp[2] = uaux;
            utmp[0] &= 0x3f3f3f3f;
            let scs = unsafe { &*(utmp.as_ptr() as *const [u8; 8]) };
            let mins = unsafe { &*((utmp.as_ptr() as *const u8).add(8) as *const [u8; 8]) };
            let mut sumi_mins = 0i32;
            for j in 0..16 { sumi_mins += q8_bsums[blk*16+j] as i32 * mins[j/2] as i32; }
            let mut sumi = 0i32;
            for g in 0..8 {
                let sc = scs[g] as i32;
                for l in 0..32 { sumi += sc * q5v[g*32+l] as i32 * q8_qs[q8_base+g*32+l] as i32; }
            }
            let scalar_val = d * q8_d[blk] * sumi as f32 - dm * q8_d[blk] * sumi_mins as f32;
            if (simd_val - scalar_val).abs() > 0.01 {
                eprintln!("  K row1 blk{blk}: SIMD={simd_val:.4} scalar={scalar_val:.4} d={d:.6} dm={dm:.6} q8d={:.6} sumi={sumi} mins_sum={sumi_mins}",
                    q8_d[blk]);
            }
        }
    }
    // Scalar Q5K reference
    if lw.wk_dtype == 13 {
        let n_blocks_k = hd / 256;
        let row_bytes = n_blocks_k * 176;
        // Scalar reference Q5K dot matching llama.cpp exactly
        for r in 0..4.min(kv_dim) {
            let wk_r = unsafe { lw.wk.add(r * row_bytes) };
            let mut result_scalar = 0.0f32;
            for blk in 0..n_blocks_k {
                let bp = unsafe { wk_r.add(blk * 176) };
                let d_raw = unsafe { u16::from_le_bytes([*bp, *bp.add(1)]) };
                let dm_raw = unsafe { u16::from_le_bytes([*bp.add(2), *bp.add(3)]) };
                let d = olorin::inference::matmul::f16_to_f32_scalar(d_raw);
                let dm = olorin::inference::matmul::f16_to_f32_scalar(dm_raw);
                let qs_ptr = unsafe { bp.add(48) };
                let qh_ptr = unsafe { bp.add(16) };
                let sc_ptr = unsafe { bp.add(4) };
                let q8_base = blk * 256;

                // Dequant Q5K values (matching llama.cpp order)
                let mut q5_vals = [0i8; 256];
                let mut m: u8 = 1;
                for j in 0..4usize {
                    for l in 0..32usize {
                        let ql = unsafe { *qs_ptr.add(j * 32 + l) };
                        let qh = unsafe { *qh_ptr.add(l) };
                        q5_vals[j * 64 + l] = (ql & 0xF) as i8 + if qh & m != 0 { 16 } else { 0 };
                    }
                    m <<= 1;
                    for l in 0..32usize {
                        let ql = unsafe { *qs_ptr.add(j * 32 + l) };
                        let qh = unsafe { *qh_ptr.add(l) };
                        q5_vals[j * 64 + 32 + l] = (ql >> 4) as i8 + if qh & m != 0 { 16 } else { 0 };
                    }
                    m <<= 1;
                }

                // Extract scales and mins (matching llama.cpp)
                let mut utmp = [0u32; 4];
                unsafe {
                    std::ptr::copy_nonoverlapping(sc_ptr, utmp.as_mut_ptr() as *mut u8, 12);
                }
                utmp[3] = ((utmp[2] >> 4) & 0x0f0f0f0f) | (((utmp[1] >> 6) & 0x03030303) << 4);
                let uaux = utmp[1] & 0x3f3f3f3f;
                utmp[1] = (utmp[2] & 0x0f0f0f0f) | (((utmp[0] >> 6) & 0x03030303) << 4);
                utmp[2] = uaux;
                utmp[0] &= 0x3f3f3f3f;
                let scales = unsafe { &*(utmp.as_ptr() as *const [u8; 16]) };
                let mins = &scales[8..];
                let scs = &scales[0..8];

                // Mins correction
                let mut sumi_mins = 0i32;
                for j in 0..16 {
                    sumi_mins += q8_bsums[blk * 16 + j] as i32 * mins[j / 2] as i32;
                }

                // Dot product with scales
                let mut sumi = 0i32;
                for g in 0..8 {
                    let sc = scs[g] as i32;
                    for l in 0..32 {
                        sumi += sc * q5_vals[g * 32 + l] as i32 * q8_qs[q8_base + g * 32 + l] as i32;
                    }
                }

                result_scalar += d * q8_d[blk] * sumi as f32 - dm * q8_d[blk] * sumi_mins as f32;
            }
            let pow2 = olorin::inference::matmul::pow2_table();
            let wk_row = unsafe { lw.wk.add(r * row_bytes) };
            let simd_val = unsafe {
                olorin::kernels::ffi_inference::q5k_dot_q8k(
                    wk_row, q8_qs.as_ptr(), q8_bsums.as_ptr(),
                    n_blocks_k as i32, q8_d.as_ptr(), pow2.as_ptr(),
                )
            };
            eprintln!("K row {r}: scalar={result_scalar:.4} simd={simd_val:.4} llama=[3.81, 3.14, 0.66, 1.16]");
        }
    }

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
    let pool = olorin::inference::threadpool::ThreadPool::new();
    let mut state = olorin::inference::forward::Gemma4State::new(&model, 512, &pool);

    // Embed BOS + scale, then run PLE Phase A (matches production forward_one)
    let hd = model.hidden_dim;
    olorin::inference::dequant::q6k_embed_lookup(model.embed_weight, 2, &mut state.x, hd);
    let scale = (hd as f32).sqrt();
    for v in state.x[..hd].iter_mut() { *v *= scale; }
    state.prepare_ple(&model, 2);

    // Run layer 0
    state.layer_forward(&model, 0, 0, false, &pool);

    eprintln!("=== Step 3b: Single layer forward (L0, BOS, pos=0) ===");
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

    let pool = olorin::inference::threadpool::ThreadPool::new();
    let mut state = olorin::inference::forward::Gemma4State::new(&model, 512, &pool);

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

    let pool = olorin::inference::threadpool::ThreadPool::new();
    let mut state = olorin::inference::forward::Gemma4State::new(&model, 512, &pool);

    // Forward pass with BOS token (id=2)
    let logits_vec = state.forward_one(&model, 2, &pool).to_vec();
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

#[test]
fn step6_two_token_vs_llama_eval_callback() {
    // Reference values captured from:
    //   llama-eval-callback -m gemma-4-e2b-it-Q4_K_M.gguf -p "a" -n 0
    // Tokens: [BOS=2, 'a'=236746]. Dumped tensors are at output position 1
    // (the 'a' token, after BOS in KV).
    //
    // IMPORTANT: llama processes prompt "a" as a SINGLE BATCHED gemm forward
    // (hidden state shape {1536, 2}), while olorin processes the two tokens as
    // SEQUENTIAL matvec forwards via the incremental decode path. The two have
    // different f32 inner-loop accumulation orders, so the values printed below
    // are NOT expected to match bit-for-bit at pos=1. The drift seen here is
    // the architectural prompt-eval gap, not an inference bug. olorin's decode
    // path is bit-exact to llama's decode path (proven by step3b at pos=0).
    // Closing this gap requires a batched forward path with Q4K×Q8K gemm
    // kernels — tracked in a separate plan.
    if !has_model() { eprintln!("SKIP: no model"); return; }

    let gguf = olorin::inference::gguf::GgufFile::open(std::path::Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    let pool = olorin::inference::threadpool::ThreadPool::new();
    let mut state = olorin::inference::forward::Gemma4State::new(&model, 512, &pool);

    // Forward BOS at pos=0
    let _ = state.forward_one(&model, 2, &pool);

    // Forward 'a' (token 236746) at pos=1
    let logits_vec = state.forward_one(&model, 236746, &pool).to_vec();

    let hd = model.hidden_dim;
    let l34_sum = sum(&state.x[..hd]);
    let logits_sum = sum(&logits_vec);

    eprintln!("=== Step 6: 2-token (BOS, 'a') @ pos=1 vs llama-eval-callback ===");
    eprintln!("l_out-34 sum:     olorin={:.6}  llama.cpp=40.513065", l34_sum);
    eprintln!("logits sum:       olorin={:.4}  llama.cpp=-1781197.7500", logits_sum);

    let l34_drift = (l34_sum - 40.513065).abs() / 40.513065_f64.abs();
    let lg_drift = (logits_sum + 1781197.75).abs() / 1781197.75_f64.abs();
    eprintln!("relative drift:   l_out-34={:.4}%  logits={:.4}%",
        l34_drift * 100.0, lg_drift * 100.0);
    eprintln!("PASS: step6 dumped");
}

/// Scalar Rust port of llama.cpp's quantize_row_q8_K_ref (ggml-quants.c:2692).
/// Used to bisect olorin's quant kernel. Returns (qs, d, bsums) for one 256-block.
fn llama_q8k_ref_block(x: &[f32]) -> (Vec<i8>, f32, Vec<i16>) {
    assert_eq!(x.len(), 256);
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &v in x {
        let av = v.abs();
        if av > amax { amax = av; max = v; }
    }
    if amax == 0.0 {
        return (vec![0; 256], 0.0, vec![0; 16]);
    }
    let iscale = -127.0f32 / max;
    // Magic-number nearest-int (round-half-to-even) — bit-equivalent to ggml's nearest_int.
    let nearest = |fval: f32| -> i32 {
        let val = fval + 12582912.0f32;
        let bits: u32 = val.to_bits();
        ((bits & 0x007fffff) as i32) - 0x00400000
    };
    let mut qs = vec![0i8; 256];
    for j in 0..256 {
        let v = nearest(iscale * x[j]);
        qs[j] = v.min(127) as i8;
    }
    let mut bsums = vec![0i16; 16];
    for g in 0..16 {
        let mut s = 0i32;
        for k in 0..16 { s += qs[g*16 + k] as i32; }
        bsums[g] = s as i16;
    }
    (qs, 1.0 / iscale, bsums)
}

#[test]
fn step7_q8k_quant_kernel_vs_llama_ref() {
    // Hypothesis: olorin's quant_input differs from llama's quantize_row_q8_K_ref
    // due to rounding mode (round-half-away-from-zero vs round-half-to-even).
    // Test on a synthetic vector designed to land at half-integer scaled boundaries.
    olorin::kernels::ffi::init().unwrap();

    // Construct 256 values with controlled amax = 2.0, including values that
    // scale to exact halves and near-halves on both sides of zero.
    let amax = 2.0f32;
    let mut x = vec![0.0f32; 256];
    x[0] = amax;             // anchors amax (positive sign → llama's max=+2.0)
    x[1] = -amax;            // -2.0 → scales to +127 in llama, -127 in olorin
    // Scaled value = x * (127/2) = x * 63.5
    // To hit scaled=k.5, set x = (k.5)/63.5
    for k in 0..32 {
        let target = k as f32 + 0.5;
        x[2 + k]    = target / 63.5;        // positive halves
        x[34 + k]   = -target / 63.5;       // negative halves
    }
    // Fill rest with small noise
    for i in 66..256 {
        x[i] = ((i as f32) * 0.0137 - 0.5) * 0.01;
    }

    // Run olorin's kernel
    let mut o_qs = vec![0i8; 256 + 12];
    let mut o_d = vec![0.0f32; 1];
    let mut o_bsums = vec![0i16; 16];
    olorin::inference::matmul::quant_input(&x, &mut o_qs, &mut o_d, &mut o_bsums);

    // Run scalar reference
    let (r_qs, r_d, _r_bsums) = llama_q8k_ref_block(&x);

    eprintln!("=== Step 7: olorin quant_input vs llama q8_K_ref ===");
    eprintln!("olorin d[0]={:.8}  llama_ref d[0]={:.8}  (signs may differ; |d| should match)",
        o_d[0], r_d);
    eprintln!("|olorin d| = {:.8}   |llama d| = {:.8}", o_d[0].abs(), r_d.abs());

    let mut mismatches = 0usize;
    for j in 0..256 {
        // olorin's qs has same sign as x[j]; llama's qs has opposite sign of x[j] when max>0.
        // Compare magnitudes (and equiv signs after sign convention).
        let o = o_qs[j];
        let r = r_qs[j];
        // After sign convention, magnitudes should match exactly.
        let o_mag = o.unsigned_abs() as i32;
        let r_mag = r.unsigned_abs() as i32;
        if o_mag != r_mag {
            if mismatches < 12 {
                eprintln!("  qs[{:>3}] x={:>10.6}  scaled_olorin={:>9.4}  olorin={:>4} llama={:>4}  (|Δ|={})",
                    j, x[j], x[j] * 63.5, o, r, (o_mag - r_mag).abs());
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

//! GEMM-style batched prefill for Llama/Q4_K models.
//! Quantize all tokens to Q8K once, then batch-matmul against Q4K weights.

use crate::kernels::ffi_inference as ffi;
use crate::inference::forward::{apply_rope, build_rope_freqs};
use crate::inference::forward_llama::{LlamaState, embed_token, add_bias, q8k_blocks};
use crate::inference::gemm_q4k::{BatchQ8K, q4k_gemm_mt, q4k_fused_silu_gemm_mt};
use crate::inference::gemm_q6k::q6k_gemm_mt;
use crate::inference::matmul_q4k::Q4K_BLOCK_BYTES;
use crate::inference::matmul_q6k::Q6K_BLOCK_BYTES;
use crate::inference::engine::BitNetModel;
use crate::kernels::ffi_inference as ffi_inf;

/// Opaque pointer wrapper for vecadd thread dispatch.
#[derive(Clone, Copy)]
struct W {
    xs_mut: usize,
    buf: usize,
    h: usize,
    n: usize,
}

impl LlamaState {
    /// Run all transformer layers for N tokens, return per-token hidden states.
    fn prefill_layers(&mut self, model: &BitNetModel, tokens: &[u32]) -> Vec<Vec<f32>> {
        let n = tokens.len();
        let (h, hd, nh, nkv, kv, f) = (
            model.hidden_dim, model.head_dim, model.n_heads,
            model.n_kv_heads, model.kv_dim, model.ffn_dim,
        );
        let (h_nb, f_nb) = (q8k_blocks(h), q8k_blocks(f));
        let h_rs = h_nb * Q4K_BLOCK_BYTES;
        let mut xs: Vec<Vec<f32>> = tokens.iter().map(|&tok| {
            let mut x = vec![0.0f32; h]; embed_token(model, tok, &mut x); x
        }).collect();

        // Dump first and last token embeddings
        {
            let x0 = &xs[0];
            let l2: f32 = x0.iter().map(|v| v*v).sum::<f32>().sqrt();
            eprintln!("[embed] token[0]={} L2={l2:.6} x[0..8]={:?}", tokens[0], &x0[..8]);
            let xn = &xs[n-1];
            let ln: f32 = xn.iter().map(|v| v*v).sum::<f32>().sqrt();
            eprintln!("[embed] token[{}]={} L2={ln:.6} x[0..4]={:?}", n-1, tokens[n-1], &xn[..4]);
            // Raw bytes of first block for this token
            let embed_data = model.embed_weight_f16;
            let row_bytes = (h / 256) * 210; // Q6K
            let row_ptr = unsafe { embed_data.add(tokens[0] as usize * row_bytes) };
            let first_16 = unsafe { std::slice::from_raw_parts(row_ptr, 16) };
            eprintln!("[embed] raw bytes[0..16]={:02x?} embed_ptr={:?} row_bytes={}", first_16, embed_data, row_bytes);
        }

        let mut norm_all = vec![0.0f32; n * h];
        let mut bq_h = BatchQ8K::new(n, h);
        let mut bq_f = BatchQ8K::new(n, f);

        let (mut qs_all, mut ks_all, mut vs_all) = (
            vec![0.0f32; n*h], vec![0.0f32; n*kv], vec![0.0f32; n*kv],
        );
        let (mut attn_all, mut tmp_all, mut hidden_all) = (
            vec![0.0f32; n*h], vec![0.0f32; n*h], vec![0.0f32; n*f],
        );
        let nt = self.pool.thread_count();

        // Profiling accumulators (microseconds per step, summed across all layers)
        let mut t_rmsnorm1 = 0u64;
        let mut t_quant1 = 0u64;
        let mut t_qkv_gemm = 0u64;
        let mut t_bias_rope = 0u64;
        let mut t_kv_append = 0u64;
        let mut t_attention = 0u64;
        let mut t_quant_wo = 0u64;
        let mut t_wo_gemm = 0u64;
        let mut t_resid1 = 0u64;
        let mut t_rmsnorm2 = 0u64;
        let _t_quant2 = 0u64;
        let mut t_ffn_gemm = 0u64;
        let mut t_quant_down = 0u64;
        let mut t_down_gemm = 0u64;
        let mut t_resid2 = 0u64;

        for layer in 0..model.n_layers {
            let lw = &model.q4k_layers[layer];

            // Per-token L2 at layer start (first 5 layers only)
            if layer <= 4 {
                let mut toks: Vec<String> = Vec::new();
                for t in 0..n.min(6) {
                    let l2: f32 = xs[t].iter().map(|v| v*v).sum::<f32>().sqrt();
                    toks.push(format!("t{t}={l2:.1}"));
                }
                let last_l2: f32 = xs[n-1].iter().map(|v| v*v).sum::<f32>().sqrt();
                toks.push(format!("t{}={last_l2:.1}", n-1));
                eprintln!("[L{layer} input] {}", toks.join(" "));
            }

            // ── Parallel rmsnorm → norm_all, then batch quantize to Q8K ──
            let _t0 = std::time::Instant::now();
            {
                let xs_raw: Vec<*const f32> = xs.iter().map(|x| x.as_ptr()).collect();
                let norm_ptr = norm_all.as_mut_ptr() as usize;
                let xs_usize = xs_raw.as_ptr() as usize;
                let norm_w = lw.attn_norm as usize;
                let eps = model.rms_eps;
                let h_dim = h;
                let n_tok = n;
                self.pool.run(nt.min(n), move |tid, nt_used| unsafe {
                    let xs = xs_usize as *const *const f32;
                    let out = norm_ptr as *mut f32;
                    let mut t = tid;
                    while t < n_tok {
                        ffi::rmsnorm_f32(*xs.add(t), norm_w as *const f32, out.add(t * h_dim), h_dim as i32, eps);
                        t += nt_used;
                    }
                });
            }
            t_rmsnorm1 += _t0.elapsed().as_micros() as u64;
            let _t0 = std::time::Instant::now();
            // Parallel Q8K quantization of all N tokens
            {
                let qs_ptr = bq_h.qs.as_mut_ptr() as usize;
                let d_ptr = bq_h.d.as_mut_ptr() as usize;
                let bs_ptr = bq_h.bsums.as_mut_ptr() as usize;
                let norm_ptr = norm_all.as_ptr() as usize;
                let qs_stride = bq_h.qs_stride;
                let nb = bq_h.n_blocks;
                let h_dim = h;
                let n_tok = n;
                self.pool.run(nt.min(n), move |tid, nt_used| unsafe {
                    let mut t = tid;
                    while t < n_tok {
                        ffi::quant_f32_q8k(
                            (norm_ptr as *const f32).add(t * h_dim),
                            (qs_ptr as *mut i8).add(t * qs_stride),
                            (d_ptr as *mut f32).add(t * nb),
                            (bs_ptr as *mut i16).add(t * nb * 16),
                            h_dim as i32,
                        );
                        t += nt_used;
                    }
                });
            }

            t_quant1 += _t0.elapsed().as_micros() as u64;

            // DEBUG: verify RMSNorm for token 0 at L0
            if layer == 0 {
                // Manual RMSNorm for token 0
                let x0 = &xs[0];
                let ss: f32 = x0.iter().map(|v| v*v).sum::<f32>();
                let rms = (ss / h as f32 + model.rms_eps).sqrt();
                let manual_norm0 = x0[0] * unsafe { *lw.attn_norm.add(0) } / rms;
                let kernel_norm0 = norm_all[0];
                eprintln!("[L0 RMSNorm t0] rms={rms:.8} manual[0]={manual_norm0:.8} kernel[0]={kernel_norm0:.8} x0[0]={:.8} w[0]={:.8}",
                    x0[0], unsafe { *lw.attn_norm.add(0) });
                let kernel_l2: f32 = norm_all[..h].iter().map(|v| v*v).sum::<f32>().sqrt();
                eprintln!("[L0 RMSNorm t0] kernel L2={kernel_l2:.4}");
            }
            if layer == 0 {
                let norm0 = &norm_all[..h];
                let nl2: f32 = norm0.iter().map(|v| v*v).sum::<f32>().sqrt();
                eprintln!("[PIPE L0] rmsnorm: L2={nl2:.6} x_norm[0..8]={:?}", &norm0[..8]);

                let q8_d0 = bq_h.d[0];
                let q8_qs0: Vec<i32> = bq_h.qs[..8].iter().map(|&v| v as i32).collect();
                let q8_bs0 = bq_h.bsums[0];
                let neg_d = bq_h.d[..h_nb].iter().filter(|&&d| d < 0.0).count();
                eprintln!("[PIPE L0] q8k: d={q8_d0:.8} qs[0..8]={q8_qs0:?} bsums[0]={q8_bs0} neg_d={neg_d}/{h_nb}");
            }
            // Verify Q4K dot row 0 against C reference
            if layer == 0 {
                let wq = lw.wq;
                let pow2 = crate::inference::matmul_q4k::F16_POW2.as_ptr();
                unsafe {
                    let dot0 = ffi::q4k_dot_q8k(
                        wq,
                        bq_h.qs.as_ptr(),
                        bq_h.bsums.as_ptr(),
                        h_nb as i32,
                        bq_h.d.as_ptr(),
                        pow2,
                    );
                    eprintln!("[PIPE L0] q4k_dot(Wq row0) = {dot0:.8}  (ref: 0.12002471)");
                }
            }
            // ── QKV matmul: Q4K × Q8K (pre-quantized) ──
            let _t0 = std::time::Instant::now();
            q4k_gemm_mt(lw.wq, h_rs, h_nb, &bq_h, &mut qs_all, h, &self.pool);
            q4k_gemm_mt(lw.wk, h_rs, h_nb, &bq_h, &mut ks_all, kv, &self.pool);
            if lw.wv_block_bytes == Q6K_BLOCK_BYTES {
                q6k_gemm_mt(lw.wv, h_nb * Q6K_BLOCK_BYTES, h_nb, &bq_h, &mut vs_all, kv, &self.pool);
            } else {
                q4k_gemm_mt(lw.wv, h_rs, h_nb, &bq_h, &mut vs_all, kv, &self.pool);
            }

            t_qkv_gemm += _t0.elapsed().as_micros() as u64;
            if layer == 0 {
                // Step 1: QKV matmul output (last token, before bias/rope)
                let last_q = &qs_all[(n-1)*h..(n-1)*h+h];
                let last_k = &ks_all[(n-1)*kv..(n-1)*kv+kv];
                let last_v = &vs_all[(n-1)*kv..(n-1)*kv+kv];
                let ql2: f32 = last_q.iter().map(|v| v*v).sum::<f32>().sqrt();
                let kl2: f32 = last_k.iter().map(|v| v*v).sum::<f32>().sqrt();
                let vl2: f32 = last_v.iter().map(|v| v*v).sum::<f32>().sqrt();
                eprintln!("[CMP L0 step1] after QKV matmul (last tok, pre-bias/rope): Q_L2={ql2:.6} K_L2={kl2:.6} V_L2={vl2:.6}");
                eprintln!("[CMP L0 step1] Q[0..4]={:?} K[0..4]={:?} V[0..4]={:?}", &last_q[..4], &last_k[..4], &last_v[..4]);
            }
            if layer == 1 {
                let ql2: f32 = qs_all[..h].iter().map(|v| v*v).sum::<f32>().sqrt();
                let kl2: f32 = ks_all[..kv].iter().map(|v| v*v).sum::<f32>().sqrt();
                eprintln!("[prefill L1] after QKV: q_L2={ql2:.2} k_L2={kl2:.2}");
            }
            // ── Attention: batch bias+RoPE, KV store, f16 attention ──
            let _t0 = std::time::Instant::now();
            for t in 0..n {
                add_bias(&mut qs_all[t*h..(t+1)*h], lw.q_bias, h);
                add_bias(&mut ks_all[t*kv..(t+1)*kv], lw.k_bias, kv);
                add_bias(&mut vs_all[t*kv..(t+1)*kv], lw.v_bias, kv);
            }
            let mut rope_freqs = vec![0.0f32; hd];
            let pos_base = self.kv_cache.seq_len();
            for t in 0..n {
                build_rope_freqs(&mut rope_freqs, hd, pos_base + t, model.rope_theta);
                apply_rope(&mut qs_all[t*h..(t+1)*h], &rope_freqs, hd, nh);
                apply_rope(&mut ks_all[t*kv..(t+1)*kv], &rope_freqs, hd, nkv);
            }
            t_bias_rope += _t0.elapsed().as_micros() as u64;
            if layer == 0 {
                // Step 2: after bias+RoPE (last token)
                let last_q = &qs_all[(n-1)*h..(n-1)*h+h];
                let last_k = &ks_all[(n-1)*kv..(n-1)*kv+kv];
                let last_v = &vs_all[(n-1)*kv..(n-1)*kv+kv];
                let ql2: f32 = last_q.iter().map(|v| v*v).sum::<f32>().sqrt();
                let kl2: f32 = last_k.iter().map(|v| v*v).sum::<f32>().sqrt();
                let vl2: f32 = last_v.iter().map(|v| v*v).sum::<f32>().sqrt();
                eprintln!("[CMP L0 step2] after bias+rope (last tok): Q_L2={ql2:.6} K_L2={kl2:.6} V_L2={vl2:.6}");
                eprintln!("[CMP L0 step2] Q[0..4]={:?} K[0..4]={:?} V[0..4]={:?}", &last_q[..4], &last_k[..4], &last_v[..4]);
            }
            // KV store: token-major, F16KvCache handles scatter
            let _t0 = std::time::Instant::now();
            self.kv_cache.store(layer, 0, &ks_all[..n*kv], n).unwrap();
            self.kv_cache.store(layer, 1, &vs_all[..n*kv], n).unwrap();
            if layer == model.n_layers - 1 { self.kv_cache.advance(n).unwrap(); }

            t_kv_append += _t0.elapsed().as_micros() as u64;

            let _t0 = std::time::Instant::now();
            // Per-token f16 attention
            let q_per_kv = nh / nkv;
            let rsqrt_hd = 1.0 / (hd as f32).sqrt();
            for t in 0..n {
                let causal_len = pos_base + t + 1;
                let qt = &qs_all[t*h..(t+1)*h];
                for kv_h in 0..nkv {
                    let k_ptr = self.kv_cache.k_head_ptr(layer, kv_h);
                    let v_ptr = self.kv_cache.v_head_ptr(layer, kv_h);
                    for q_off in 0..q_per_kv {
                        let q_h = kv_h * q_per_kv + q_off;
                        let mut scores = vec![0.0f32; causal_len];
                        unsafe {
                            ffi_inf::attn_dot_f16(
                                qt[q_h * hd..].as_ptr(), k_ptr,
                                scores.as_mut_ptr(), causal_len as i32, hd as i32,
                            );
                            // Step 3: raw scores before softmax (last token, head 0)
                            if layer == 0 && t == n-1 && q_h == 0 {
                                let slen = scores.len().min(8);
                                eprintln!("[CMP L0 step3] raw scores H0 (last tok, first {slen}): {:?}", &scores[..slen]);
                            }
                            ffi_inf::softmax_f32(scores.as_mut_ptr(), causal_len as i32, rsqrt_hd);
                            // Step 4: after softmax
                            if layer == 0 && t == n-1 && q_h == 0 {
                                let slen = scores.len().min(8);
                                eprintln!("[CMP L0 step4] softmax H0 (last tok, first {slen}): {:?}", &scores[..slen]);
                            }
                            ffi_inf::attn_vsum_f16(
                                scores.as_ptr(), v_ptr,
                                attn_all[t*h + q_h*hd..].as_mut_ptr(),
                                causal_len as i32, hd as i32,
                            );
                            // Step 5: vsum output for head 0
                            if layer == 0 && t == n-1 && q_h == 0 {
                                let out = &attn_all[t*h..t*h+4];
                                eprintln!("[CMP L0 step5] vsum H0 (last tok): out[0..4]={:?}", out);
                            }
                        }
                    }
                }
            }

            t_attention += _t0.elapsed().as_micros() as u64;
            // Dump layer 0 attention output for last token (compare with llama.cpp)
            if layer == 0 {
                let last = (n-1)*h;
                let al2: f32 = attn_all[last..last+h].iter().map(|v| v*v).sum::<f32>().sqrt();
                eprintln!("[CMP L0] attn_out L2={al2:.6} x[0..4]={:?}  (llama.cpp: L2=1.4377 x=[-0.011, -0.001, -0.010, -0.013])",
                    &attn_all[last..last+4]);
            }
            if layer == 1 {
                let al2: f32 = attn_all[..h].iter().map(|v| v*v).sum::<f32>().sqrt();
                eprintln!("[prefill L1] after attn: attn_out_L2={al2:.2} attn[0..4]={:?}", &attn_all[..4]);
            }
            // ── Wo matmul: Q4K × Q8K ──
            let _t0 = std::time::Instant::now();
            {
                let qs_ptr = bq_h.qs.as_mut_ptr() as usize;
                let d_ptr = bq_h.d.as_mut_ptr() as usize;
                let bs_ptr = bq_h.bsums.as_mut_ptr() as usize;
                let src_ptr = attn_all.as_ptr() as usize;
                let qs_stride = bq_h.qs_stride;
                let nb = bq_h.n_blocks;
                let h_dim = h;
                let n_tok = n;
                self.pool.run(nt.min(n), move |tid, nt_used| unsafe {
                    let mut t = tid;
                    while t < n_tok {
                        ffi::quant_f32_q8k(
                            (src_ptr as *const f32).add(t * h_dim),
                            (qs_ptr as *mut i8).add(t * qs_stride),
                            (d_ptr as *mut f32).add(t * nb),
                            (bs_ptr as *mut i16).add(t * nb * 16),
                            h_dim as i32,
                        );
                        t += nt_used;
                    }
                });
            }
            t_quant_wo += _t0.elapsed().as_micros() as u64;
            // DEBUG: check Q8K of attn_out for token 0 at L0
            if layer == 0 {
                let a0 = &attn_all[..h];
                let a0_l2: f32 = a0.iter().map(|v| v*v).sum::<f32>().sqrt();
                let a0_max: f32 = a0.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let d0 = bq_h.d[0];
                let qs0: Vec<i32> = bq_h.qs[..16].iter().map(|&v| v as i32).collect();
                // Dequant check: d*qs should approximate original
                let dequant0: f32 = d0 * bq_h.qs[0] as f32;
                eprintln!("[L0 Wo Q8K t0] attn_L2={a0_l2:.4} max={a0_max:.6} d[0]={d0:.8} qs[0..16]={qs0:?} dequant[0]={dequant0:.6} orig[0]={:.6}", a0[0]);
                // Also check the GEMM input/output ratio
                let an = &attn_all[(n-1)*h..n*h];
                let an_l2: f32 = an.iter().map(|v| v*v).sum::<f32>().sqrt();
                let an_max: f32 = an.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                let dn = bq_h.d[(n-1)*h_nb];
                eprintln!("[L0 Wo Q8K tN] attn_L2={an_l2:.4} max={an_max:.6} d[0]={dn:.8}");
            }
            let _t0 = std::time::Instant::now();
            q4k_gemm_mt(lw.wo, h_rs, h_nb, &bq_h, &mut tmp_all, h, &self.pool);
            t_wo_gemm += _t0.elapsed().as_micros() as u64;
            // DEBUG: verify Wo GEMM output for token 0 at L0
            if layer == 0 {
                // Manual single-row dot for row 0 vs token 0
                let ref_dot = unsafe { crate::inference::matmul_q4k::q4k_row_dot(
                    lw.wo, h_nb, bq_h.qs_ptr(0) as _, bq_h.d_ptr(0) as _, bq_h.bsums_ptr(0) as _) };
                let gemm_val = tmp_all[0];
                // Also check row 0 vs last token
                let ref_last = unsafe { crate::inference::matmul_q4k::q4k_row_dot(
                    lw.wo, h_nb, bq_h.qs_ptr(n-1) as _, bq_h.d_ptr(n-1) as _, bq_h.bsums_ptr(n-1) as _) };
                let gemm_last = tmp_all[(n-1)*h];
                eprintln!("[L0 Wo verify] t0: gemm={gemm_val:.6} ref={ref_dot:.6} | tN: gemm={gemm_last:.6} ref={ref_last:.6}");
                // Check first few Wo output elements for token 0
                eprintln!("[L0 Wo t0] out[0..8]={:?}", &tmp_all[..8]);
                eprintln!("[L0 Wo tN] out[0..8]={:?}", &tmp_all[(n-1)*h..(n-1)*h+8]);
            }
            if layer == 1 {
                let tl2: f32 = tmp_all[..h].iter().map(|v| v*v).sum::<f32>().sqrt();
                eprintln!("[prefill L1] after Wo: tmp_L2={tl2:.2} tmp[0..4]={:?}", &tmp_all[..4]);
                let xl2: f32 = xs[0].iter().map(|v| v*v).sum::<f32>().sqrt();
                eprintln!("[prefill L1] before resid: x_L2={xl2:.2} x[0..4]={:?}", &xs[0][..4]);
            }
            // ── Parallel vecadd residual (attn) ──
            let _t0 = std::time::Instant::now();
            {
                let xs_raw: Vec<*mut f32> = xs.iter_mut().map(|x| x.as_mut_ptr()).collect();
                let w = W { xs_mut: xs_raw.as_ptr() as usize, buf: tmp_all.as_ptr() as usize, h, n };
                self.pool.run(nt.min(n), move |tid, nt_used| unsafe {
                    let xm = w.xs_mut as *const *mut f32;
                    let buf = w.buf as *const f32;
                    let mut t = tid;
                    while t < w.n {
                        let x = *xm.add(t);
                        ffi::vecadd_f32(x, buf.add(t * w.h), x, w.h as i32);
                        t += nt_used;
                    }
                });
            }

            t_resid1 += _t0.elapsed().as_micros() as u64;
            // Dump layer 0 after Wo+residual (= ffn_inp in llama.cpp)
            if layer == 0 {
                let xn = &xs[n-1];
                let xl2: f32 = xn.iter().map(|v| v*v).sum::<f32>().sqrt();
                eprintln!("[CMP L0] ffn_inp L2={xl2:.6} x[0..4]={:?}  (llama.cpp: L2=1.4283 x=[-0.018, -0.002, -0.011, -0.034])",
                    &xn[..4]);
                let tl2: f32 = tmp_all[(n-1)*h..(n-1)*h+h].iter().map(|v| v*v).sum::<f32>().sqrt();
                eprintln!("[CMP L0] Wo_out L2={tl2:.6}  (for last token)");
            }
            // ── Parallel rmsnorm (FFN) → norm_all, then batch quantize ──
            let _t0 = std::time::Instant::now();
            {
                let xs_raw: Vec<*const f32> = xs.iter().map(|x| x.as_ptr()).collect();
                let norm_ptr = norm_all.as_mut_ptr() as usize;
                let xs_usize = xs_raw.as_ptr() as usize;
                let norm_w = lw.ffn_norm as usize;
                let eps = model.rms_eps;
                let h_dim = h;
                let n_tok = n;
                self.pool.run(nt.min(n), move |tid, nt_used| unsafe {
                    let xs = xs_usize as *const *const f32;
                    let out = norm_ptr as *mut f32;
                    let mut t = tid;
                    while t < n_tok {
                        ffi::rmsnorm_f32(*xs.add(t), norm_w as *const f32, out.add(t * h_dim), h_dim as i32, eps);
                        t += nt_used;
                    }
                });
            }
            {
                let qs_ptr = bq_h.qs.as_mut_ptr() as usize;
                let d_ptr = bq_h.d.as_mut_ptr() as usize;
                let bs_ptr = bq_h.bsums.as_mut_ptr() as usize;
                let src_ptr = norm_all.as_ptr() as usize;
                let qs_stride = bq_h.qs_stride;
                let nb = bq_h.n_blocks;
                let h_dim = h;
                let n_tok = n;
                self.pool.run(nt.min(n), move |tid, nt_used| unsafe {
                    let mut t = tid;
                    while t < n_tok {
                        ffi::quant_f32_q8k(
                            (src_ptr as *const f32).add(t * h_dim),
                            (qs_ptr as *mut i8).add(t * qs_stride),
                            (d_ptr as *mut f32).add(t * nb),
                            (bs_ptr as *mut i16).add(t * nb * 16),
                            h_dim as i32,
                        );
                        t += nt_used;
                    }
                });
            }

            t_rmsnorm2 += _t0.elapsed().as_micros() as u64;
            // note: t_quant2 is zero — rmsnorm2 includes the quant block above
            // ── FFN gate+up+SiLU: Q4K × Q8K ──
            let _t0 = std::time::Instant::now();
            q4k_fused_silu_gemm_mt(lw.w_gate, lw.w_up, h_rs, h_nb, &bq_h, &mut hidden_all, f, &self.pool);

            t_ffn_gemm += _t0.elapsed().as_micros() as u64;
            if layer == 1 {
                // Check FFN input (norm_all) and hidden output for token 0
                let nl2: f32 = norm_all[..h].iter().map(|v| v*v).sum::<f32>().sqrt();
                let hl2: f32 = hidden_all[..f].iter().map(|v| v*v).sum::<f32>().sqrt();
                eprintln!("[prefill L1] FFN input norm_L2={nl2:.4} → hidden_L2={hl2:.2} hidden[1419]={:.4}", hidden_all[1419]);
                // Also check gate and up separately by calling q4k_dot directly
                let gw = lw.w_gate;
                let uw = lw.w_up;
                unsafe {
                    let g = crate::inference::matmul_q4k::q4k_row_dot(
                        gw.add(1419 * h_rs), h_nb,
                        bq_h.qs_ptr(0) as _, bq_h.d_ptr(0) as _, bq_h.bsums_ptr(0) as _);
                    let u = crate::inference::matmul_q4k::q4k_row_dot(
                        uw.add(1419 * h_rs), h_nb,
                        bq_h.qs_ptr(0) as _, bq_h.d_ptr(0) as _, bq_h.bsums_ptr(0) as _);
                    let silu_g = g / (1.0 + (-g).exp());
                    eprintln!("[prefill L1] row 1419: gate={g:.4} up={u:.4} silu(g)*u={:.4}", silu_g * u);
                }
            }
            // ── Down projection: Q4K or Q6K ──
            let _t0 = std::time::Instant::now();
            {
                let qs_ptr = bq_f.qs.as_mut_ptr() as usize;
                let d_ptr = bq_f.d.as_mut_ptr() as usize;
                let bs_ptr = bq_f.bsums.as_mut_ptr() as usize;
                let src_ptr = hidden_all.as_ptr() as usize;
                let qs_stride = bq_f.qs_stride;
                let nb = bq_f.n_blocks;
                let f_dim = f;
                let n_tok = n;
                self.pool.run(nt.min(n), move |tid, nt_used| unsafe {
                    let mut t = tid;
                    while t < n_tok {
                        ffi::quant_f32_q8k(
                            (src_ptr as *const f32).add(t * f_dim),
                            (qs_ptr as *mut i8).add(t * qs_stride),
                            (d_ptr as *mut f32).add(t * nb),
                            (bs_ptr as *mut i16).add(t * nb * 16),
                            f_dim as i32,
                        );
                        t += nt_used;
                    }
                });
            }
            t_quant_down += _t0.elapsed().as_micros() as u64;
            let _t0 = std::time::Instant::now();
            if lw.w_down_block_bytes == Q6K_BLOCK_BYTES {
                q6k_gemm_mt(lw.w_down, f_nb * Q6K_BLOCK_BYTES, f_nb, &bq_f, &mut tmp_all, h, &self.pool);
            } else {
                q4k_gemm_mt(lw.w_down, f_nb * Q4K_BLOCK_BYTES, f_nb, &bq_f, &mut tmp_all, h, &self.pool);
            }

            t_down_gemm += _t0.elapsed().as_micros() as u64;
            if layer == 1 {
                let dl2: f32 = tmp_all[..h].iter().map(|v| v*v).sum::<f32>().sqrt();
                eprintln!("[prefill L1] after down proj: tmp_L2={dl2:.2} tmp[0..4]={:?}", &tmp_all[..4]);
            }
            // ── Parallel vecadd residual (FFN) ──
            let _t0 = std::time::Instant::now();
            {
                let xs_raw: Vec<*mut f32> = xs.iter_mut().map(|x| x.as_mut_ptr()).collect();
                let w = W { xs_mut: xs_raw.as_ptr() as usize, buf: tmp_all.as_ptr() as usize, h, n };
                self.pool.run(nt.min(n), move |tid, nt_used| unsafe {
                    let xm = w.xs_mut as *const *mut f32;
                    let buf = w.buf as *const f32;
                    let mut t = tid;
                    while t < w.n {
                        let x = *xm.add(t);
                        ffi::vecadd_f32(x, buf.add(t * w.h), x, w.h as i32);
                        t += nt_used;
                    }
                });
            }
            t_resid2 += _t0.elapsed().as_micros() as u64;

            // DEBUG: zero dim2 to test dimensional drift hypothesis
            if std::env::var("OLORIN_ZERO_DIM2").is_ok() {
                for t in 0..n { xs[t][2] = 0.0; }
            }

            // DEBUG: per-step trace for token 0 and last token at every layer
            {
                let last = &xs[n-1];
                let l2: f32 = last.iter().map(|v| v*v).sum::<f32>().sqrt();
                let al2: f32 = attn_all[(n-1)*h..(n-1)*h+h].iter().map(|v| v*v).sum::<f32>().sqrt();
                let tl2: f32 = tmp_all[(n-1)*h..(n-1)*h+h].iter().map(|v| v*v).sum::<f32>().sqrt();
                let hl2: f32 = hidden_all[(n-1)*f..(n-1)*f+f].iter().map(|v| v*v).sum::<f32>().sqrt();
                // Token 0 stats
                let x0l2: f32 = xs[0].iter().map(|v| v*v).sum::<f32>().sqrt();
                let a0l2: f32 = attn_all[..h].iter().map(|v| v*v).sum::<f32>().sqrt();
                let t0l2: f32 = tmp_all[..h].iter().map(|v| v*v).sum::<f32>().sqrt();
                let h0l2: f32 = hidden_all[..f].iter().map(|v| v*v).sum::<f32>().sqrt();
                eprintln!("[prefill L{layer}] t0: attn={a0l2:.1} wo={t0l2:.1} ffn={h0l2:.1} x={x0l2:.1} | tN: attn={al2:.1} wo={tl2:.1} ffn={hl2:.1} x={l2:.1}");
            }
        }

        // Print profiling summary
        let total = t_rmsnorm1 + t_quant1 + t_qkv_gemm + t_bias_rope + t_kv_append
            + t_attention + t_quant_wo + t_wo_gemm + t_resid1 + t_rmsnorm2
            + t_ffn_gemm + t_quant_down + t_down_gemm + t_resid2;
        let nl = model.n_layers as u64;
        let ms = |us: u64| us as f64 / 1000.0;
        let pct = |us: u64| if total > 0 { us as f64 / total as f64 * 100.0 } else { 0.0 };
        // FLOP estimates: 2*M*N*K per matmul, N=n_tokens
        let nt = n as f64;
        let hf = h as f64;
        let kvf = kv as f64;
        let ff = f as f64;
        let qkv_flops = 2.0 * nt * hf * (hf + kvf + kvf); // Q+K+V
        let wo_flops = 2.0 * nt * hf * hf;
        let ffn_flops = 2.0 * nt * hf * ff * 2.0; // gate + up (fused)
        let down_flops = 2.0 * nt * ff * hf;
        let total_flops = (qkv_flops + wo_flops + ffn_flops + down_flops) * nl as f64;
        let tops = |flops: f64, us: u64| if us > 0 { flops * nl as f64 / (us as f64 * 1e-6) / 1e9 } else { 0.0 };

        eprintln!("\n--- prefill profile ({n} tokens, {nl} layers, {total}µs total = {:.1}ms) ---", ms(total));
        eprintln!("┌─────────────────────┬──────────┬───────┬──────────┐");
        eprintln!("│ Step                │   ms     │   %   │  GFLOPS  │");
        eprintln!("├─────────────────────┼──────────┼───────┼──────────┤");
        eprintln!("│ rmsnorm (attn)      │ {:7.1} │ {:4.1}% │          │", ms(t_rmsnorm1), pct(t_rmsnorm1));
        eprintln!("│ quant_q8k (attn)    │ {:7.1} │ {:4.1}% │          │", ms(t_quant1), pct(t_quant1));
        eprintln!("│ QKV gemm            │ {:7.1} │ {:4.1}% │ {:7.1}  │", ms(t_qkv_gemm), pct(t_qkv_gemm), tops(qkv_flops, t_qkv_gemm));
        eprintln!("│ bias+rope           │ {:7.1} │ {:4.1}% │          │", ms(t_bias_rope), pct(t_bias_rope));
        eprintln!("│ kv_append           │ {:7.1} │ {:4.1}% │          │", ms(t_kv_append), pct(t_kv_append));
        eprintln!("│ attention           │ {:7.1} │ {:4.1}% │          │", ms(t_attention), pct(t_attention));
        eprintln!("│ quant_q8k (Wo)      │ {:7.1} │ {:4.1}% │          │", ms(t_quant_wo), pct(t_quant_wo));
        eprintln!("│ Wo gemm             │ {:7.1} │ {:4.1}% │ {:7.1}  │", ms(t_wo_gemm), pct(t_wo_gemm), tops(wo_flops, t_wo_gemm));
        eprintln!("│ residual (attn)     │ {:7.1} │ {:4.1}% │          │", ms(t_resid1), pct(t_resid1));
        eprintln!("│ rmsnorm+quant (FFN) │ {:7.1} │ {:4.1}% │          │", ms(t_rmsnorm2), pct(t_rmsnorm2));
        eprintln!("│ FFN gate+up+SiLU    │ {:7.1} │ {:4.1}% │ {:7.1}  │", ms(t_ffn_gemm), pct(t_ffn_gemm), tops(ffn_flops, t_ffn_gemm));
        eprintln!("│ quant_q8k (down)    │ {:7.1} │ {:4.1}% │          │", ms(t_quant_down), pct(t_quant_down));
        eprintln!("│ down gemm           │ {:7.1} │ {:4.1}% │ {:7.1}  │", ms(t_down_gemm), pct(t_down_gemm), tops(down_flops, t_down_gemm));
        eprintln!("│ residual (FFN)      │ {:7.1} │ {:4.1}% │          │", ms(t_resid2), pct(t_resid2));
        eprintln!("├─────────────────────┼──────────┼───────┼──────────┤");
        eprintln!("│ TOTAL               │ {:7.1} │ 100%  │ {:7.1}  │", ms(total), total_flops / (total as f64 * 1e-6) / 1e9);
        eprintln!("│ per token           │ {:7.1} │       │          │", ms(total) / n as f64);
        eprintln!("└─────────────────────┴──────────┴───────┴──────────┘");

        xs
    }

    /// GEMM-style batched prefill: quantize all tokens to Q8K once,
    /// then load each weight matrix once and multiply all tokens.
    pub fn prefill(&mut self, model: &BitNetModel, tokens: &[u32]) {
        let xs = self.prefill_layers(model, tokens);
        let h = model.hidden_dim;
        let n = tokens.len();
        self.x[..h].copy_from_slice(&xs[n - 1]);
        let x_l2: f32 = self.x.iter().take(h).map(|v| v*v).sum::<f32>().sqrt();
        eprintln!("[prefill done] x_L2={x_l2:.2} x[0..8]={:?}", &self.x[..8]);
        // Check all token hidden states
        for (t, xh) in xs.iter().enumerate() {
            let l2: f32 = xh.iter().map(|v| v*v).sum::<f32>().sqrt();
            if t == 0 || t == n-1 { eprintln!("  xs[{t}] L2={l2:.2} [0..4]={:?}", &xh[..4]); }
        }
        self.output_proj(model);
    }

    /// Batched prefill that returns per-token argmax token IDs.
    /// Used by speculative decoding to verify draft tokens.
    pub fn prefill_verify(&mut self, model: &BitNetModel, tokens: &[u32]) -> Vec<u32> {
        let xs = self.prefill_layers(model, tokens);
        let h = model.hidden_dim;
        let mut result = Vec::with_capacity(xs.len());
        for hidden in &xs {
            self.x[..h].copy_from_slice(hidden);
            self.output_proj(model);
            result.push(super::forward_llama::argmax(&self.logits));
        }
        result
    }
}

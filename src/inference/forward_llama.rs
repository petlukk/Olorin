//! Transformer forward pass for Llama models with Q4_K_M quantization.

use crate::kernels::ffi_inference as ffi;
use crate::inference::forward::{apply_rope, build_rope_freqs, sample};
use crate::inference::matmul::embed_f16_lookup;
use crate::inference::matmul_q4k::{Q4K_BLOCK_BYTES, q4k_matmul_mt, q4k_matmul_work, q4k_fused_gate_up_silu_work};
use crate::inference::matmul_q6k::{Q6K_BLOCK_BYTES, q6k_matmul_mt, q6k_matmul_work};
use crate::inference::matmul_q4k::q4k_embed_lookup;
use crate::inference::matmul_q6k::q6k_embed_lookup;
use crate::inference::engine::BitNetModel;
use crate::inference::cache::{self, EakvCache};
use crate::inference::ptr::{SendPtr, SendMutPtr};

/// Add bias vector to output buffer. No-op if bias is null (Llama models).
#[inline]
pub(crate) fn add_bias(buf: &mut [f32], bias: *const f32, n: usize) {
    if bias.is_null() { return; }
    for i in 0..n { buf[i] += unsafe { *bias.add(i) }; }
}

pub struct LlamaState {
    pub(crate) x: Vec<f32>,
    x_norm: Vec<f32>,
    x_q8_qs: Vec<i8>,
    x_q8_d: Vec<f32>,
    x_q8_bsums: Vec<i32>,
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) attn_out: Vec<f32>,
    attn_q8_qs: Vec<i8>,
    attn_q8_d: Vec<f32>,
    attn_q8_bsums: Vec<i32>,
    pub(crate) hidden: Vec<f32>,
    hidden_q8_qs: Vec<i8>,
    hidden_q8_d: Vec<f32>,
    hidden_q8_bsums: Vec<i32>,
    pub(crate) logits: Vec<f32>,
    pub(crate) tmp: Vec<f32>,
    pub(crate) eakv: EakvCache,
    pub(crate) attn_scores: Vec<f32>,
    pub(crate) rope_freqs: Vec<f32>,
    pub(crate) x_norm_pub: Vec<f32>,
    #[allow(dead_code)]
    pub(crate) max_seq_len: usize,
}

pub(crate) fn q8k_blocks(dim: usize) -> usize { dim / 256 }

pub(crate) fn softmax_rows(data: &mut [f32], n_rows: usize, seq_len: usize) {
    for r in 0..n_rows {
        let row = &mut data[r * seq_len..(r + 1) * seq_len];
        let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() { *v = (*v - max_v).exp(); sum += *v; }
        let inv = 1.0 / sum;
        for v in row.iter_mut() { *v *= inv; }
    }
}

pub(crate) fn embed_token(model: &BitNetModel, token: u32, out: &mut [f32]) {
    match model.embed_dtype {
        12 => q4k_embed_lookup(model.embed_weight_f16, token, out, model.hidden_dim),
        14 => q6k_embed_lookup(model.embed_weight_f16, token, out, model.hidden_dim),
        _ => embed_f16_lookup(model.embed_weight_f16, token, out, model.hidden_dim),
    }
}

impl LlamaState {
    pub fn new(model: &BitNetModel, max_seq_len: usize) -> Self {
        let h = model.hidden_dim;
        let f = model.ffn_dim;
        let v = model.vocab_size;
        let hb = q8k_blocks(h);
        let fb = q8k_blocks(f);
        let nh = model.n_heads;
        let kt = cache::KernelTable::init()
            .expect("eakv kernels not found");
        let eakv = EakvCache::new(
            model.n_layers as i32, model.n_kv_heads as i32,
            model.head_dim as i32, max_seq_len as i32, kt,
        ).expect("failed to create EakvCache");
        LlamaState {
            x: vec![0.0; h],
            x_norm: vec![0.0; h],
            x_q8_qs: vec![0; h + 16],
            x_q8_d: vec![0.0; hb],
            x_q8_bsums: vec![0; hb * 16],
            q: vec![0.0; h],
            k: vec![0.0; model.kv_dim],
            v: vec![0.0; model.kv_dim],
            attn_out: vec![0.0; h],
            attn_q8_qs: vec![0; h + 16],
            attn_q8_d: vec![0.0; hb],
            attn_q8_bsums: vec![0; hb * 16],
            hidden: vec![0.0; f],
            hidden_q8_qs: vec![0; f + 16],
            hidden_q8_d: vec![0.0; fb],
            hidden_q8_bsums: vec![0; fb * 16],
            logits: vec![0.0; v],
            tmp: vec![0.0; h.max(f)],
            eakv,
            attn_scores: vec![0.0; nh * max_seq_len],
            rope_freqs: vec![0.0; model.head_dim],
            x_norm_pub: vec![0.0; h],
            max_seq_len,
        }
    }

    /// Process one token through one layer.
    pub(crate) fn process_layer(&mut self, model: &BitNetModel, layer: usize, x: &mut [f32], pos: usize) {
        let h = model.hidden_dim;
        let hd = model.head_dim;
        let nh = model.n_heads;
        let nkv = model.n_kv_heads;
        let kv = model.kv_dim;
        let f = model.ffn_dim;
        let h_nb = q8k_blocks(h);
        let f_nb = q8k_blocks(f);
        let h_row_stride = h_nb * Q4K_BLOCK_BYTES;
        let lw = &model.q4k_layers[layer];

        unsafe {
            ffi::rmsnorm_f32(x.as_ptr(), lw.attn_norm, self.x_norm.as_mut_ptr(), h as i32, model.rms_eps);
            ffi::quant_f32_q8k(self.x_norm.as_ptr(), self.x_q8_qs.as_mut_ptr(),
                self.x_q8_d.as_mut_ptr(), self.x_q8_bsums.as_mut_ptr(), h as i32);
        }

        // QKV — concurrent dispatch via std::thread::scope
        let total = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let wv_bb = lw.wv_block_bytes;
        if total >= 3 {
            let q_t = (total / 2).max(1);
            let rem = total - q_t;
            let k_t = rem / 2;
            let v_t = rem - k_t;
            let q8p = SendPtr(self.x_q8_qs.as_ptr());
            let q8d = SendPtr(self.x_q8_d.as_ptr());
            let q8b = SendPtr(self.x_q8_bsums.as_ptr());
            let q_out = SendMutPtr(self.q.as_mut_ptr());
            let k_out = SendMutPtr(self.k.as_mut_ptr());
            let v_out = SendMutPtr(self.v.as_mut_ptr());
            let (wq, wk, wv) = (SendPtr(lw.wq), SendPtr(lw.wk), SendPtr(lw.wv));
            let h_q6k_stride = h_nb * Q6K_BLOCK_BYTES;
            std::thread::scope(|s| {
                for tid in 0..q_t {
                    s.spawn(move || unsafe { q4k_matmul_work(wq.ptr(), h_row_stride, h_nb, q8p.ptr(), q8d.ptr(), q8b.ptr(), q_out.ptr(), h, tid, q_t); });
                }
                for tid in 0..k_t {
                    s.spawn(move || unsafe { q4k_matmul_work(wk.ptr(), h_row_stride, h_nb, q8p.ptr(), q8d.ptr(), q8b.ptr(), k_out.ptr(), kv, tid, k_t); });
                }
                for tid in 0..v_t {
                    s.spawn(move || unsafe {
                        if wv_bb == Q6K_BLOCK_BYTES { q6k_matmul_work(wv.ptr(), h_q6k_stride, h_nb, q8p.ptr(), q8d.ptr(), q8b.ptr(), v_out.ptr(), kv, tid, v_t); }
                        else { q4k_matmul_work(wv.ptr(), h_row_stride, h_nb, q8p.ptr(), q8d.ptr(), q8b.ptr(), v_out.ptr(), kv, tid, v_t); }
                    });
                }
            });
        } else {
            q4k_matmul_mt(lw.wq, h_row_stride, h_nb, self.x_q8_qs.as_ptr(), self.x_q8_d.as_ptr(), self.x_q8_bsums.as_ptr(), &mut self.q, h);
            q4k_matmul_mt(lw.wk, h_row_stride, h_nb, self.x_q8_qs.as_ptr(), self.x_q8_d.as_ptr(), self.x_q8_bsums.as_ptr(), &mut self.k, kv);
            if wv_bb == Q6K_BLOCK_BYTES {
                q6k_matmul_mt(lw.wv, h_nb * Q6K_BLOCK_BYTES, h_nb, self.x_q8_qs.as_ptr(), self.x_q8_d.as_ptr(), self.x_q8_bsums.as_ptr(), &mut self.v, kv);
            } else {
                q4k_matmul_mt(lw.wv, h_row_stride, h_nb, self.x_q8_qs.as_ptr(), self.x_q8_d.as_ptr(), self.x_q8_bsums.as_ptr(), &mut self.v, kv);
            }
        }

        add_bias(&mut self.q, lw.q_bias, h);
        add_bias(&mut self.k, lw.k_bias, kv);
        add_bias(&mut self.v, lw.v_bias, kv);

        build_rope_freqs(&mut self.rope_freqs, hd, pos, model.rope_theta);
        apply_rope(&mut self.q, &self.rope_freqs, hd, nh);
        apply_rope(&mut self.k, &self.rope_freqs, hd, nkv);
        self.eakv.append(&self.k[..kv], layer as i32, 0, 1).unwrap();
        self.eakv.append(&self.v[..kv], layer as i32, 1, 1).unwrap();
        if layer == model.n_layers - 1 { self.eakv.advance(1).unwrap(); }

        let seq_len = pos + 1;
        let scores = &mut self.attn_scores[..nh * seq_len];
        cache::attention::attention_scores(&self.eakv, &self.q, layer as i32, nh as i32, nkv as i32, scores);
        softmax_rows(scores, nh, seq_len);
        cache::attention::attention_output(&self.eakv, scores, layer as i32, nh as i32, nkv as i32, &mut self.attn_out);

        unsafe {
            ffi::quant_f32_q8k(self.attn_out.as_ptr(), self.attn_q8_qs.as_mut_ptr(),
                self.attn_q8_d.as_mut_ptr(), self.attn_q8_bsums.as_mut_ptr(), h as i32);
        }
        q4k_matmul_mt(lw.wo, h_row_stride, h_nb, self.attn_q8_qs.as_ptr(), self.attn_q8_d.as_ptr(),
            self.attn_q8_bsums.as_ptr(), &mut self.tmp, h);
        unsafe { ffi::vecadd_f32(x.as_ptr(), self.tmp.as_ptr(), self.attn_out.as_mut_ptr(), h as i32); }
        x[..h].copy_from_slice(&self.attn_out[..h]);

        unsafe {
            ffi::rmsnorm_f32(x.as_ptr(), lw.ffn_norm, self.x_norm.as_mut_ptr(), h as i32, model.rms_eps);
            ffi::quant_f32_q8k(self.x_norm.as_ptr(), self.x_q8_qs.as_mut_ptr(),
                self.x_q8_d.as_mut_ptr(), self.x_q8_bsums.as_mut_ptr(), h as i32);
        }

        // Fused gate+up+SiLU
        let q8p = SendPtr(self.x_q8_qs.as_ptr());
        let q8d = SendPtr(self.x_q8_d.as_ptr());
        let q8b = SendPtr(self.x_q8_bsums.as_ptr());
        let h_out = SendMutPtr(self.hidden.as_mut_ptr());
        let (wg, wu) = (SendPtr(lw.w_gate), SendPtr(lw.w_up));
        std::thread::scope(|s| {
            for tid in 0..total {
                s.spawn(move || unsafe {
                    q4k_fused_gate_up_silu_work(wg.ptr(), wu.ptr(), h_row_stride, h_nb,
                        q8p.ptr(), q8d.ptr(), q8b.ptr(), h_out.ptr(), f, tid, total);
                });
            }
        });

        unsafe {
            ffi::quant_f32_q8k(self.hidden.as_ptr(), self.hidden_q8_qs.as_mut_ptr(),
                self.hidden_q8_d.as_mut_ptr(), self.hidden_q8_bsums.as_mut_ptr(), f as i32);
        }
        if lw.w_down_block_bytes == Q6K_BLOCK_BYTES {
            q6k_matmul_mt(lw.w_down, f_nb * Q6K_BLOCK_BYTES, f_nb, self.hidden_q8_qs.as_ptr(),
                self.hidden_q8_d.as_ptr(), self.hidden_q8_bsums.as_ptr(), &mut self.tmp, h);
        } else {
            q4k_matmul_mt(lw.w_down, f_nb * Q4K_BLOCK_BYTES, f_nb, self.hidden_q8_qs.as_ptr(),
                self.hidden_q8_d.as_ptr(), self.hidden_q8_bsums.as_ptr(), &mut self.tmp, h);
        }
        unsafe { ffi::vecadd_f32(x.as_ptr(), self.tmp.as_ptr(), self.attn_out.as_mut_ptr(), h as i32); }
        x[..h].copy_from_slice(&self.attn_out[..h]);
    }

    /// Output projection: RMSNorm + quantize + matmul -> logits.
    pub(crate) fn output_proj(&mut self, model: &BitNetModel) {
        let h = model.hidden_dim;
        let h_nb = q8k_blocks(h);
        let h_row_stride = h_nb * Q4K_BLOCK_BYTES;
        unsafe {
            ffi::rmsnorm_f32(self.x.as_ptr(), model.norm_weight, self.x_norm.as_mut_ptr(), h as i32, model.rms_eps);
            ffi::quant_f32_q8k(self.x_norm.as_ptr(), self.x_q8_qs.as_mut_ptr(),
                self.x_q8_d.as_mut_ptr(), self.x_q8_bsums.as_mut_ptr(), h as i32);
        }
        if model.output_block_bytes == Q6K_BLOCK_BYTES {
            q6k_matmul_mt(model.output_weight, h_nb * Q6K_BLOCK_BYTES, h_nb, self.x_q8_qs.as_ptr(),
                self.x_q8_d.as_ptr(), self.x_q8_bsums.as_ptr(), &mut self.logits, model.vocab_size);
        } else {
            q4k_matmul_mt(model.output_weight, h_row_stride, h_nb, self.x_q8_qs.as_ptr(),
                self.x_q8_d.as_ptr(), self.x_q8_bsums.as_ptr(), &mut self.logits, model.vocab_size);
        }
    }

    /// Single-token forward pass.
    pub fn forward(&mut self, model: &BitNetModel, token: u32, pos: usize) {
        embed_token(model, token, &mut self.x);
        let mut x = std::mem::take(&mut self.x);
        for layer in 0..model.n_layers {
            self.process_layer(model, layer, &mut x, pos);
        }
        self.x = x;
        self.output_proj(model);
    }

    pub fn apply_repetition_penalty(&mut self, generated: &[u32], penalty: f32) {
        if penalty == 1.0 { return; }
        for &tok in generated {
            let idx = tok as usize;
            if idx < self.logits.len() {
                if self.logits[idx] > 0.0 { self.logits[idx] /= penalty; }
                else { self.logits[idx] *= penalty; }
            }
        }
    }

    pub fn sample_logits(&self, temperature: f32, top_k: usize, top_p: f32) -> u32 {
        sample(&self.logits, temperature, top_k, top_p)
    }
}

pub fn generate(
    model: &BitNetModel,
    prompt_tokens: &[u32],
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    eos_id: u32,
    max_seq_len: usize,
    mut on_token: impl FnMut(u32),
) -> (Vec<u32>, f64, f64) {
    use std::time::Instant;
    let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let mut state = LlamaState::new(model, max_seq_len);
    let mut output = Vec::with_capacity(prompt_tokens.len() + max_tokens);

    let prefill_start = Instant::now();
    state.prefill(model, prompt_tokens);
    output.extend_from_slice(prompt_tokens);
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    let first_tok_start = Instant::now();
    let mut pos = prompt_tokens.len();
    let mut n_gen = 0u32;
    let mut first_tok_ms = 0.0;

    let decode_start = Instant::now();
    for step in 0..max_tokens {
        if pos >= max_seq_len { break; }
        state.apply_repetition_penalty(&output, repetition_penalty);
        let next = state.sample_logits(temperature, top_k, top_p);
        if next == eos_id { break; }
        output.push(next);
        on_token(next);
        if step == 0 { first_tok_ms = first_tok_start.elapsed().as_secs_f64() * 1000.0; }
        state.forward(model, next, pos);
        pos += 1;
        n_gen += 1;
    }
    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

    let ptps = prompt_tokens.len() as f64 / (prefill_ms / 1000.0);
    let dtps = if n_gen > 0 { n_gen as f64 / (decode_ms / 1000.0) } else { 0.0 };
    let avg = if n_gen > 0 { decode_ms / n_gen as f64 } else { 0.0 };
    eprintln!("\n--- perf ({n_threads} threads) ---");
    eprintln!("prefill:    {} tokens in {prefill_ms:.0}ms ({ptps:.1} tok/s)", prompt_tokens.len());
    eprintln!("first tok:  {first_tok_ms:.0}ms");
    eprintln!("decode:     {n_gen} tokens in {decode_ms:.0}ms ({dtps:.1} tok/s, {avg:.1}ms/tok)");
    (output, prefill_ms, decode_ms)
}

fn wipe_f32(buf: &mut [f32]) {
    unsafe { std::ptr::write_bytes(buf.as_mut_ptr(), 0, buf.len()); }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

fn wipe_i8(buf: &mut [i8]) {
    unsafe { std::ptr::write_bytes(buf.as_mut_ptr(), 0, buf.len()); }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

fn wipe_i32(buf: &mut [i32]) {
    unsafe { std::ptr::write_bytes(buf.as_mut_ptr(), 0, buf.len()); }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

impl Drop for LlamaState {
    fn drop(&mut self) {
        wipe_f32(&mut self.x);
        wipe_f32(&mut self.x_norm);
        wipe_i8(&mut self.x_q8_qs);
        wipe_f32(&mut self.x_q8_d);
        wipe_i32(&mut self.x_q8_bsums);
        wipe_f32(&mut self.q);
        wipe_f32(&mut self.k);
        wipe_f32(&mut self.v);
        wipe_f32(&mut self.attn_out);
        wipe_i8(&mut self.attn_q8_qs);
        wipe_f32(&mut self.attn_q8_d);
        wipe_i32(&mut self.attn_q8_bsums);
        wipe_f32(&mut self.hidden);
        wipe_i8(&mut self.hidden_q8_qs);
        wipe_f32(&mut self.hidden_q8_d);
        wipe_i32(&mut self.hidden_q8_bsums);
        wipe_f32(&mut self.logits);
        wipe_f32(&mut self.tmp);
    }
}

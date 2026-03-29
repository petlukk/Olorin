//! Transformer forward pass for BitNet b1.58 2B-4T.

use crate::kernels::ffi_inference as ffi;
use crate::inference::matmul::{embed_f16_lookup, i8_output_matmul_mt, ternary_matmul_mt, ternary_matmul_fused_pair, ternary_matmul_qkv};
#[cfg(target_arch = "aarch64")]
use crate::inference::matmul::i8_output_matmul_speculative;
use crate::inference::engine::BitNetModel;
use crate::inference::cache::{self, EakvCache};
use crate::inference::threadpool::ThreadPool;

pub struct InferenceState {
    pub(crate) pool: ThreadPool,
    pub(crate) x: Vec<f32>,
    pub(crate) x_norm: Vec<f32>,
    pub(crate) x_quant: Vec<i8>,
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) attn_out: Vec<f32>,
    pub(crate) attn_out_quant: Vec<i8>,
    pub(crate) gate: Vec<f32>,
    pub(crate) up: Vec<f32>,
    pub(crate) hidden: Vec<f32>,
    pub(crate) hidden_quant: Vec<i8>,
    pub(crate) logits: Vec<f32>,
    pub(crate) tmp: Vec<f32>,
    pub(crate) kv_cache: EakvCache,
    pub(crate) attn_scores: Vec<f32>,
    pub(crate) rope_freqs: Vec<f32>,
    #[allow(dead_code)]
    pub(crate) max_seq_len: usize,
}

pub(crate) fn softmax_rows(data: &mut [f32], n_rows: usize, seq_len: usize) {
    for r in 0..n_rows {
        let row = &mut data[r * seq_len..(r + 1) * seq_len];
        let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max_v).exp();
            sum += *v;
        }
        let inv = 1.0 / sum;
        for v in row.iter_mut() { *v *= inv; }
    }
}

pub(crate) fn build_rope_freqs(freqs: &mut [f32], head_dim: usize, pos: usize, theta: f32) {
    for i in 0..head_dim / 2 {
        let angle = pos as f32 / theta.powf(2.0 * i as f32 / head_dim as f32);
        freqs[2 * i] = angle.cos();
        freqs[2 * i + 1] = angle.sin();
    }
}

pub(crate) fn apply_rope(data: &mut [f32], freqs: &[f32], head_dim: usize, n_heads: usize) {
    unsafe {
        ffi::apply_rope_f32(
            data.as_ptr(), freqs.as_ptr(), data.as_mut_ptr(),
            head_dim as i32, n_heads as i32,
        );
    }
}

pub(crate) fn argmax(s: &[f32]) -> u32 {
    s.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32).unwrap_or(0)
}

pub(crate) fn apply_top_k(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() { return; }
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.select_nth_unstable_by(k - 1, |&a, &b| {
        logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    for &i in &indices[k..] { logits[i] = f32::NEG_INFINITY; }
}

pub(crate) fn apply_top_p(probs: &mut [f32], p: f32) -> usize {
    if p >= 1.0 { return probs.len(); }
    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut cumsum = 0.0;
    let mut kept = 0;
    for &i in &indices {
        cumsum += probs[i];
        kept += 1;
        if cumsum >= p { break; }
    }
    for &i in &indices[kept..] { probs[i] = 0.0; }
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for v in probs.iter_mut() { *v *= inv; }
    }
    kept
}

pub(crate) fn sample(logits: &[f32], temperature: f32, top_k: usize, top_p: f32) -> u32 {
    if temperature <= 0.0 { return argmax(logits); }
    let mut logits_buf: Vec<f32> = logits.to_vec();
    apply_top_k(&mut logits_buf, top_k);
    let max_val = logits_buf.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits_buf.iter().map(|&x| ((x - max_val) / temperature).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() { *p /= sum; }
    apply_top_p(&mut probs, top_p);
    let r = xorshift_f32();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum { return i as u32; }
    }
    (probs.len() - 1) as u32
}

/// Xorshift64 RNG returning f32 in [0, 1).
pub(crate) fn xorshift_f32() -> f32 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = Cell::new(0);
    }
    STATE.with(|s| {
        let mut v = s.get();
        if v == 0 {
            v = std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap_or_default().as_nanos() as u64;
            if v == 0 { v = 0xdeadbeef; }
        }
        v ^= v << 13;
        v ^= v >> 7;
        v ^= v << 17;
        s.set(v);
        (v >> 40) as f32 / (1u64 << 24) as f32
    })
}

impl InferenceState {
    pub fn new(model: &BitNetModel, max_seq_len: usize) -> Self {
        let h = model.hidden_dim;
        let f = model.ffn_dim;
        let v = model.vocab_size;
        let nh = model.n_heads;
        let kt = cache::KernelTable::init()
            .expect("kv_cache kernels not found");
        let kv_cache = EakvCache::new(
            model.n_layers as i32, model.n_kv_heads as i32,
            model.head_dim as i32, max_seq_len as i32, kt,
        ).expect("failed to create EakvCache");
        InferenceState {
            pool: ThreadPool::new(),
            x: vec![0.0; h],
            x_norm: vec![0.0; h],
            x_quant: vec![0; h + 12],
            q: vec![0.0; h],
            k: vec![0.0; model.kv_dim],
            v: vec![0.0; model.kv_dim],
            attn_out: vec![0.0; h],
            attn_out_quant: vec![0; h + 12],
            gate: vec![0.0; f],
            up: vec![0.0; f],
            hidden: vec![0.0; f],
            hidden_quant: vec![0; f + 12],
            logits: vec![0.0; v],
            tmp: vec![0.0; h.max(f)],
            kv_cache,
            attn_scores: vec![0.0; nh * max_seq_len],
            rope_freqs: vec![0.0; model.head_dim],
            max_seq_len,
        }
    }

    pub fn forward(&mut self, model: &BitNetModel, token: u32, pos: usize) {
        let h = model.hidden_dim;
        let hd = model.head_dim;
        let nh = model.n_heads;
        let nkv = model.n_kv_heads;
        let kv = model.kv_dim;
        let f = model.ffn_dim;
        let seq_len = pos + 1;
        embed_f16_lookup(model.embed_weight_f16, token, &mut self.x, h);

        for layer in 0..model.n_layers {
            let lw = &model.layers[layer];
            unsafe {
                ffi::rmsnorm_f32(
                    self.x.as_ptr(), lw.attn_norm, self.x_norm.as_mut_ptr(),
                    h as i32, model.rms_eps,
                );
            }
            let mut act_scale: f32 = 0.0;
            let mut act_sum: i32 = 0;
            unsafe {
                ffi::quant_f32_i8(
                    self.x_norm.as_ptr(), self.x_quant.as_mut_ptr(),
                    &mut act_scale, &mut act_sum, h as i32,
                );
            }
            ternary_matmul_qkv(
                lw.wq, lw.wq_scale, &mut self.q, h,
                lw.wk, lw.wk_scale, &mut self.k, kv,
                lw.wv, lw.wv_scale, &mut self.v,
                self.x_quant.as_ptr(), act_scale, act_sum, h,
                &self.pool,
            );
            build_rope_freqs(&mut self.rope_freqs, hd, pos, model.rope_theta);
            apply_rope(&mut self.q, &self.rope_freqs, hd, nh);
            apply_rope(&mut self.k, &self.rope_freqs, hd, nkv);
            self.kv_cache.append(&self.k[..kv], layer as i32, 0, 1).unwrap();
            self.kv_cache.append(&self.v[..kv], layer as i32, 1, 1).unwrap();
            if layer == model.n_layers - 1 { self.kv_cache.advance(1).unwrap(); }

            let scores = &mut self.attn_scores[..nh * seq_len];
            cache::attention::attention_scores(
                &self.kv_cache, &self.q, layer as i32, nh as i32, nkv as i32, seq_len as i32, scores,
            );
            softmax_rows(scores, nh, seq_len);
            cache::attention::attention_output(
                &self.kv_cache, scores, layer as i32, nh as i32, nkv as i32, seq_len as i32, &mut self.attn_out,
            );

            unsafe {
                ffi::rmsnorm_f32(
                    self.attn_out.as_ptr(), lw.attn_sub_norm, self.attn_out.as_mut_ptr(),
                    h as i32, model.rms_eps,
                );
            }
            let mut attn_scale: f32 = 0.0;
            let mut attn_sum: i32 = 0;
            unsafe {
                ffi::quant_f32_i8(
                    self.attn_out.as_ptr(), self.attn_out_quant.as_mut_ptr(),
                    &mut attn_scale, &mut attn_sum, h as i32,
                );
            }
            ternary_matmul_mt(
                lw.wo, self.attn_out_quant.as_ptr(), attn_scale, attn_sum, lw.wo_scale,
                &mut self.tmp, h, h,
                &self.pool,
            );
            unsafe {
                ffi::vecadd_f32(
                    self.x.as_ptr(), self.tmp.as_ptr(),
                    self.attn_out.as_mut_ptr(), h as i32,
                );
            }
            self.x[..h].copy_from_slice(&self.attn_out[..h]);
            unsafe {
                ffi::rmsnorm_f32(
                    self.x.as_ptr(), lw.ffn_norm, self.x_norm.as_mut_ptr(),
                    h as i32, model.rms_eps,
                );
            }
            let mut ffn_scale: f32 = 0.0;
            let mut ffn_sum: i32 = 0;
            unsafe {
                ffi::quant_f32_i8(
                    self.x_norm.as_ptr(), self.x_quant.as_mut_ptr(),
                    &mut ffn_scale, &mut ffn_sum, h as i32,
                );
            }
            ternary_matmul_fused_pair(
                lw.w_gate, lw.w_gate_scale,
                lw.w_up, lw.w_up_scale,
                self.x_quant.as_ptr(), ffn_scale, ffn_sum,
                &mut self.gate, &mut self.up,
                f, h,
                &self.pool,
            );
            unsafe {
                ffi::squared_relu_mul_f32(
                    self.gate.as_ptr(), self.up.as_ptr(),
                    self.hidden.as_mut_ptr(), f as i32,
                );
                ffi::rmsnorm_f32(
                    self.hidden.as_ptr(), lw.ffn_sub_norm, self.hidden.as_mut_ptr(),
                    f as i32, model.rms_eps,
                );
            }
            let mut down_scale: f32 = 0.0;
            let mut down_sum: i32 = 0;
            unsafe {
                ffi::quant_f32_i8(
                    self.hidden.as_ptr(), self.hidden_quant.as_mut_ptr(),
                    &mut down_scale, &mut down_sum, f as i32,
                );
            }
            ternary_matmul_mt(
                lw.w_down, self.hidden_quant.as_ptr(), down_scale, down_sum, lw.w_down_scale,
                &mut self.tmp, h, f,
                &self.pool,
            );
            unsafe {
                ffi::vecadd_f32(
                    self.x.as_ptr(), self.tmp.as_ptr(),
                    self.attn_out.as_mut_ptr(), h as i32,
                );
            }
            self.x[..h].copy_from_slice(&self.attn_out[..h]);
        }

        unsafe {
            ffi::rmsnorm_f32(
                self.x.as_ptr(), model.norm_weight, self.x_norm.as_mut_ptr(),
                h as i32, model.rms_eps,
            );
        }

        #[cfg(target_arch = "aarch64")]
        {
            if !model.embed_sketch.is_empty() {
                i8_output_matmul_speculative(
                    &model.embed_weight_i8, &model.embed_row_scales,
                    &model.embed_sketch, model.embed_sketch_dim,
                    &self.x_norm, &mut self.logits,
                    model.vocab_size, h,
                );
            } else {
                i8_output_matmul_mt(
                    &model.embed_weight_i8, &model.embed_row_scales,
                    &self.x_norm, &mut self.logits,
                    model.vocab_size, h,
                    &self.pool,
                );
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        i8_output_matmul_mt(
            &model.embed_weight_i8, &model.embed_row_scales,
            &self.x_norm, &mut self.logits,
            model.vocab_size, h,
            &self.pool,
        );
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
        let mut state = InferenceState::new(model, max_seq_len);
        let mut output = Vec::with_capacity(prompt_tokens.len() + max_tokens);

        let prefill_start = Instant::now();
        if prompt_tokens.len() >= 8 {
            state.prefill(model, prompt_tokens);
        } else {
            for (i, &tok) in prompt_tokens.iter().enumerate() {
                state.forward(model, tok, i);
            }
        }
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

        let n_threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let ptps = prompt_tokens.len() as f64 / (prefill_ms / 1000.0);
        let dtps = if n_gen > 0 { n_gen as f64 / (decode_ms / 1000.0) } else { 0.0 };
        let avg = if n_gen > 0 { decode_ms / n_gen as f64 } else { 0.0 };
        eprintln!("\n--- perf ({n_threads} threads) ---");
        eprintln!("prefill:    {} tokens in {prefill_ms:.0}ms ({ptps:.1} tok/s)", prompt_tokens.len());
        eprintln!("first tok:  {first_tok_ms:.0}ms");
        eprintln!("decode:     {n_gen} tokens in {decode_ms:.0}ms ({dtps:.1} tok/s, {avg:.1}ms/tok)");
        (output, prefill_ms, decode_ms)
    }
}

fn wipe_f32(buf: &mut [f32]) {
    unsafe { std::ptr::write_bytes(buf.as_mut_ptr(), 0, buf.len()); }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

fn wipe_i8(buf: &mut [i8]) {
    unsafe { std::ptr::write_bytes(buf.as_mut_ptr(), 0, buf.len()); }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

impl Drop for InferenceState {
    fn drop(&mut self) {
        wipe_f32(&mut self.x);
        wipe_f32(&mut self.x_norm);
        wipe_i8(&mut self.x_quant);
        wipe_f32(&mut self.q);
        wipe_f32(&mut self.k);
        wipe_f32(&mut self.v);
        wipe_f32(&mut self.attn_out);
        wipe_i8(&mut self.attn_out_quant);
        wipe_f32(&mut self.gate);
        wipe_f32(&mut self.up);
        wipe_f32(&mut self.hidden);
        wipe_i8(&mut self.hidden_quant);
        wipe_f32(&mut self.logits);
        wipe_f32(&mut self.tmp);
    }
}

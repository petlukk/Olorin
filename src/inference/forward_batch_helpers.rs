//! Batch forward helpers — timing struct + quant/repack utilities.

use std::sync::atomic::AtomicI32;
use crate::inference::matmul;
use crate::inference::matmul_graph;
use crate::kernels::ffi_inference;

/// Accumulated per-op timing across all layers (microseconds).
/// Only thread 0 writes/reads these.
pub(crate) struct BatchLayerTiming {
    pub attn_norm: u64,
    pub quant_input: u64,
    pub repack_q8: u64,
    pub gemm_q: u64,
    pub q_norm_rope: u64,
    pub gemm_k: u64,
    pub gemm_v: u64,
    pub kv_norm_rope_cache: u64,
    pub attention: u64,
    pub quant_wo: u64,
    pub repack_wo: u64,
    pub gemm_wo: u64,
    pub post_attn_ffn_norm: u64,
    pub quant_ffn: u64,
    pub repack_ffn: u64,
    pub gemm_gate: u64,
    pub gemm_up: u64,
    pub gelu_mul: u64,
    pub quant_down: u64,
    pub repack_down: u64,
    pub gemm_down: u64,
    pub post_ffn_residual: u64,
    pub ple_total: u64,
    pub output_scale: u64,
}

impl BatchLayerTiming {
    pub fn new() -> Self {
        Self {
            attn_norm: 0, quant_input: 0, repack_q8: 0,
            gemm_q: 0, q_norm_rope: 0, gemm_k: 0, gemm_v: 0,
            kv_norm_rope_cache: 0, attention: 0,
            quant_wo: 0, repack_wo: 0, gemm_wo: 0,
            post_attn_ffn_norm: 0, quant_ffn: 0, repack_ffn: 0,
            gemm_gate: 0, gemm_up: 0,
            gelu_mul: 0, quant_down: 0, repack_down: 0, gemm_down: 0,
            post_ffn_residual: 0,
            ple_total: 0,
            output_scale: 0,
        }
    }

    pub fn print_summary(&self, n_layers: usize, n_tokens: usize) {
        let total = self.attn_norm + self.quant_input + self.repack_q8
            + self.gemm_q + self.q_norm_rope + self.gemm_k + self.gemm_v
            + self.kv_norm_rope_cache + self.attention
            + self.quant_wo + self.repack_wo + self.gemm_wo
            + self.post_attn_ffn_norm + self.quant_ffn + self.repack_ffn
            + self.gemm_gate + self.gemm_up
            + self.gelu_mul + self.quant_down + self.repack_down + self.gemm_down
            + self.post_ffn_residual + self.ple_total + self.output_scale;
        let ms = |us: u64| us as f64 / 1000.0;
        let pct = |us: u64| if total > 0 { us as f64 / total as f64 * 100.0 } else { 0.0 };
        eprintln!("[batch-timing] {n_layers} layers × {n_tokens} tokens, total {:.1}ms", ms(total));
        eprintln!("  attn_norm       {:7.1}ms  ({:4.1}%)", ms(self.attn_norm), pct(self.attn_norm));
        eprintln!("  quant_input     {:7.1}ms  ({:4.1}%)", ms(self.quant_input), pct(self.quant_input));
        eprintln!("  repack_q8       {:7.1}ms  ({:4.1}%)", ms(self.repack_q8), pct(self.repack_q8));
        eprintln!("  gemm_q          {:7.1}ms  ({:4.1}%)", ms(self.gemm_q), pct(self.gemm_q));
        eprintln!("  q_norm_rope     {:7.1}ms  ({:4.1}%)", ms(self.q_norm_rope), pct(self.q_norm_rope));
        eprintln!("  gemm_k          {:7.1}ms  ({:4.1}%)", ms(self.gemm_k), pct(self.gemm_k));
        eprintln!("  gemm_v          {:7.1}ms  ({:4.1}%)", ms(self.gemm_v), pct(self.gemm_v));
        eprintln!("  kv_norm_rope    {:7.1}ms  ({:4.1}%)", ms(self.kv_norm_rope_cache), pct(self.kv_norm_rope_cache));
        eprintln!("  attention       {:7.1}ms  ({:4.1}%)", ms(self.attention), pct(self.attention));
        eprintln!("  quant_wo        {:7.1}ms  ({:4.1}%)", ms(self.quant_wo), pct(self.quant_wo));
        eprintln!("  repack_wo       {:7.1}ms  ({:4.1}%)", ms(self.repack_wo), pct(self.repack_wo));
        eprintln!("  gemm_wo         {:7.1}ms  ({:4.1}%)", ms(self.gemm_wo), pct(self.gemm_wo));
        eprintln!("  post_attn+norm  {:7.1}ms  ({:4.1}%)", ms(self.post_attn_ffn_norm), pct(self.post_attn_ffn_norm));
        eprintln!("  quant_ffn       {:7.1}ms  ({:4.1}%)", ms(self.quant_ffn), pct(self.quant_ffn));
        eprintln!("  repack_ffn      {:7.1}ms  ({:4.1}%)", ms(self.repack_ffn), pct(self.repack_ffn));
        eprintln!("  gemm_gate       {:7.1}ms  ({:4.1}%)", ms(self.gemm_gate), pct(self.gemm_gate));
        eprintln!("  gemm_up         {:7.1}ms  ({:4.1}%)", ms(self.gemm_up), pct(self.gemm_up));
        eprintln!("  gelu_mul        {:7.1}ms  ({:4.1}%)", ms(self.gelu_mul), pct(self.gelu_mul));
        eprintln!("  quant_down      {:7.1}ms  ({:4.1}%)", ms(self.quant_down), pct(self.quant_down));
        eprintln!("  repack_down     {:7.1}ms  ({:4.1}%)", ms(self.repack_down), pct(self.repack_down));
        eprintln!("  gemm_down       {:7.1}ms  ({:4.1}%)", ms(self.gemm_down), pct(self.gemm_down));
        eprintln!("  post_ffn_res    {:7.1}ms  ({:4.1}%)", ms(self.post_ffn_residual), pct(self.post_ffn_residual));
        eprintln!("  ple             {:7.1}ms  ({:4.1}%)", ms(self.ple_total), pct(self.ple_total));
        eprintln!("  output_scale    {:7.1}ms  ({:4.1}%)", ms(self.output_scale), pct(self.output_scale));
    }
}

/// All threads quantize tokens in parallel. Tokens [n..n_pad) get zero-filled Q8K.
#[inline]
pub(crate) fn parallel_batch_quant(
    src: &[f32], dim: usize, n: usize, n_pad: usize,
    qs: &mut [i8], d: &mut [f32], bsums: &mut [i16],
    ith: usize, nth: usize,
) {
    let nb = dim / 256;
    let qs_stride = dim + 12;
    let mut t = ith;
    while t < n_pad {
        if t < n {
            matmul::quant_input(
                &src[t * dim..(t + 1) * dim],
                &mut qs[t * qs_stride..(t + 1) * qs_stride],
                &mut d[t * nb..(t + 1) * nb],
                &mut bsums[t * nb * 16..(t + 1) * nb * 16],
            );
        } else {
            qs[t * qs_stride..(t + 1) * qs_stride].fill(0);
            d[t * nb..(t + 1) * nb].fill(0.0);
            bsums[t * nb * 16..(t + 1) * nb * 16].fill(0);
        }
        t += nth;
    }
}

/// Repack Q8K -> block_q8_Kx4 tiles for GEMM. Thread 0 only.
#[inline]
pub(crate) fn repack_q8_for_gemm(
    qs: &[i8], d: &[f32], bsums: &[i16], q8_a: &mut [u8],
    dim: usize, n_pad: usize,
) {
    let nb = dim / 256;
    let qs_stride = dim + 12;
    let tile_size = nb * 1168;
    for group in 0..(n_pad / 4) {
        let r0 = group * 4;
        let mut row_d = [0.0f32; 192];
        for b in 0..nb {
            for r in 0..4 { row_d[b * 4 + r] = d[(r0 + r) * nb + b]; }
        }
        unsafe {
            ffi_inference::q8k_repack_4(
                qs.as_ptr().add(r0 * qs_stride),
                qs.as_ptr().add((r0 + 1) * qs_stride),
                qs.as_ptr().add((r0 + 2) * qs_stride),
                qs.as_ptr().add((r0 + 3) * qs_stride),
                row_d.as_ptr(),
                bsums.as_ptr().add(r0 * nb * 16),
                bsums.as_ptr().add((r0 + 1) * nb * 16),
                bsums.as_ptr().add((r0 + 2) * nb * 16),
                bsums.as_ptr().add((r0 + 3) * nb * 16),
                q8_a.as_mut_ptr().add(group * tile_size),
                nb as i32,
            );
        }
    }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn matvec_batch_step(
    repacked: Option<&[u8]>, dtype: u32, weight: *const u8,
    q8_a: *const u8, q8_qs: *const i8, q8_d: *const f32, q8_bsums: *const i16,
    output: *mut f32, d_scratch: *mut f32,
    n_rows: usize, n_cols: usize, n: usize, n_pad: usize, output_stride: usize,
    current_chunk: &AtomicI32, ith: usize, nth: usize,
) {
    if let Some(p) = repacked {
        matmul_graph::q4k_gemm_8x8_batch_ws(
            p.as_ptr(), q8_a, output,
            n_cols, n_rows, n_pad, output_stride,
            current_chunk, ith, nth,
        );
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if dtype == matmul::GGML_TYPE_Q5_K {
        matmul_graph::q5k_gemm_batch_ws(
            weight, q8_a, output,
            n_cols, n_rows, n_pad, output_stride,
            current_chunk, ith, nth,
        );
        return;
    }
    matmul_graph::matvec_batch_ws(
        dtype, weight, q8_qs, q8_d, q8_bsums, output, d_scratch,
        n_rows, n_cols, n, output_stride,
        current_chunk, ith, nth,
    );
}

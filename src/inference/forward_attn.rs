//! Gemma 4 attention + FFN helpers, split from forward.rs.

use crate::inference::engine::Gemma4Model;
use crate::inference::matmul;
use crate::kernels::ffi_inference;

use super::forward::Gemma4State;

impl Gemma4State {
    /// GQA attention decode: for each Q head, dot with cached K, softmax, weighted V sum.
    pub(crate) fn attention_decode(
        &mut self,
        n_heads: usize,
        _n_kv_heads: usize,
        gqa_ratio: usize,
        head_dim: usize,
        kv_dim: usize,
        attn_len: usize,
        scale: f32,
        k_ptr: *const u16,
        v_ptr: *const u16,
    ) {
        let stride = kv_dim; // n_kv_heads * head_dim per position

        for h in 0..n_heads {
            let kv_h = h / gqa_ratio;
            let q_off = h * head_dim;
            let q_slice = &self.q[q_off..q_off + head_dim];

            // Compute attention scores: Q dot K for each cached position
            for p in 0..attn_len {
                let k_offset = p * stride + kv_h * head_dim;
                let k_src = unsafe { k_ptr.add(k_offset) };
                unsafe {
                    ffi_inference::f16_to_f32(
                        k_src,
                        self.kv_f32_scratch.as_mut_ptr(),
                        head_dim as i32,
                    );
                }
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_slice[d] * self.kv_f32_scratch[d];
                }
                self.attn_scores[p] = dot;
            }

            // Softmax with scale = 1/sqrt(head_dim)
            unsafe {
                ffi_inference::softmax_f32(
                    self.attn_scores.as_mut_ptr(),
                    attn_len as i32,
                    scale,
                );
            }

            // Weighted V sum
            let out_off = q_off;
            for d in 0..head_dim {
                self.attn_out[out_off + d] = 0.0;
            }
            for p in 0..attn_len {
                let v_offset = p * stride + kv_h * head_dim;
                let v_src = unsafe { v_ptr.add(v_offset) };
                unsafe {
                    ffi_inference::f16_to_f32(
                        v_src,
                        self.kv_f32_scratch.as_mut_ptr(),
                        head_dim as i32,
                    );
                }
                let s = self.attn_scores[p];
                for d in 0..head_dim {
                    self.attn_out[out_off + d] += s * self.kv_f32_scratch[d];
                }
            }
        }
    }

    /// FFN: GeGLU — gate/up dual matmul, GELU(gate)*up, down projection.
    pub(crate) fn ffn(&mut self, model: &Gemma4Model, layer: usize) {
        let hd = model.hidden_dim;
        let ffn_dim = model.ffn_dim[layer];
        let lw = &model.layers[layer];

        // Quantize x_norm for gate/up matmul
        matmul::quant_input(
            &self.x_norm,
            &mut self.q8_qs,
            &mut self.q8_d,
            &mut self.q8_bsums,
        );

        // Fused gate + up projection
        matmul::q4k_matvec_dual(
            lw.w_gate,
            lw.w_up,
            &self.q8_qs, &self.q8_d, &self.q8_bsums,
            &mut self.gate, &mut self.up,
            ffn_dim, hd,
        );

        // GELU(gate) * up -> gate buffer
        ffi_inference::gelu_mul(
            self.gate.as_ptr(),
            self.up.as_ptr(),
            self.gate.as_mut_ptr(),
            ffn_dim as i32,
        );

        // Quantize gate (ffn_dim) for down projection
        matmul::quant_input(
            &self.gate[..ffn_dim],
            &mut self.ffn_q8_qs,
            &mut self.ffn_q8_d,
            &mut self.ffn_q8_bsums,
        );

        // Down projection: ffn_dim -> hidden_dim
        matmul::q4k_matvec(
            lw.w_down,
            &self.ffn_q8_qs, &self.ffn_q8_d, &self.ffn_q8_bsums,
            &mut self.down, hd, ffn_dim,
        );
    }
}

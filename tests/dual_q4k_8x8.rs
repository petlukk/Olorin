//! Bit-exact regression for q4k_8x8_q8k_matvec_dual.
//!
//! Runs the dual kernel on (packed_gate, packed_up, synthetic Q8K) and
//! compares each output f32 bit pattern against running the single 8x8
//! matvec kernel twice (once per weight). This is the correctness gate
//! for Phase B.2 — subsequent tasks trust this to_bits() equality.
//!
//! Run: PATH="$HOME/projects/eacompute/target/release:$PATH" \
//!      cargo test --release --test dual_q4k_8x8 -- --nocapture --test-threads=1

use std::path::Path;

fn model_path() -> String {
    let home = std::env::var("HOME").unwrap();
    format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")
}

#[test]
fn dual_matches_two_single_calls_bitexact() {
    if !Path::new(&model_path()).exists() {
        eprintln!("SKIP: no model at {}", model_path());
        return;
    }
    let gguf = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    // ffn_gate + ffn_up from layer 0. Both Q4K in Q4_K_M, identical shape.
    let lw = &model.layers[0];
    assert_eq!(
        lw.w_gate_dtype,
        olorin::inference::matmul::GGML_TYPE_Q4_K,
        "test requires Q4K ffn_gate"
    );
    assert_eq!(
        lw.w_up_dtype,
        olorin::inference::matmul::GGML_TYPE_Q4_K,
        "test requires Q4K ffn_up"
    );

    let n_rows = model.ffn_dim[0];
    let n_cols = model.hidden_dim;
    let n_blocks = n_cols / 256;
    let tile_bytes = n_blocks * 1152; // 1152 B per 8-row tile group
    let n_tiles = n_rows / 8;
    let total = n_tiles * tile_bytes;

    let mut packed_gate = vec![0u8; total];
    let mut packed_up = vec![0u8; total];
    unsafe {
        olorin::kernels::ffi_inference::q4k_repack_8x8(
            lw.w_gate, packed_gate.as_mut_ptr(), n_rows as i32, n_cols as i32,
        );
        olorin::kernels::ffi_inference::q4k_repack_8x8(
            lw.w_up, packed_up.as_mut_ptr(), n_rows as i32, n_cols as i32,
        );
    }

    // Non-trivial synthetic Q8K input. Pattern from tests/bench_q4k_gemm.rs:
    // non-zero entries in every field, non-constant across blocks so the
    // compiler cannot hoist or constant-fold anything.
    let mut q8_qs = vec![5i8; n_cols + 12];
    let mut q8_d = vec![0.01f32; n_blocks];
    let mut q8_bsums = vec![17i16; n_blocks * 16];
    for i in 0..n_cols {
        q8_qs[i] = ((i as i32) % 127 - 63) as i8;
    }
    for i in 0..n_blocks {
        q8_d[i] = 0.01 + (i as f32) * 0.0013;
    }
    for i in 0..(n_blocks * 16) {
        q8_bsums[i] = ((i as i16) % 31) - 15;
    }

    let pow2 = olorin::inference::matmul::pow2_table();

    // Reference: two separate calls to q4k_8x8_q8k_matvec.
    let mut ref_gate = vec![0f32; n_rows];
    let mut ref_up = vec![0f32; n_rows];
    let mut scratch_ref = [0u8; 128];
    unsafe {
        olorin::kernels::ffi_inference::q4k_8x8_q8k_matvec(
            packed_gate.as_ptr(),
            q8_qs.as_ptr(), q8_d.as_ptr(), q8_bsums.as_ptr(),
            pow2.as_ptr(), scratch_ref.as_mut_ptr(),
            ref_gate.as_mut_ptr(),
            n_rows as i32, n_cols as i32,
        );
        olorin::kernels::ffi_inference::q4k_8x8_q8k_matvec(
            packed_up.as_ptr(),
            q8_qs.as_ptr(), q8_d.as_ptr(), q8_bsums.as_ptr(),
            pow2.as_ptr(), scratch_ref.as_mut_ptr(),
            ref_up.as_mut_ptr(),
            n_rows as i32, n_cols as i32,
        );
    }

    // Candidate: one fused dual call.
    let mut fused_gate = vec![0f32; n_rows];
    let mut fused_up = vec![0f32; n_rows];
    let mut scratch_fused = [0u8; 128];
    unsafe {
        olorin::kernels::ffi_inference::q4k_8x8_q8k_matvec_dual(
            packed_gate.as_ptr(), packed_up.as_ptr(),
            q8_qs.as_ptr(), q8_d.as_ptr(), q8_bsums.as_ptr(),
            pow2.as_ptr(), scratch_fused.as_mut_ptr(),
            fused_gate.as_mut_ptr(), fused_up.as_mut_ptr(),
            n_rows as i32, n_cols as i32,
        );
    }

    // Per-output bit-exact equality on both channels.
    let mut gate_mismatches = 0usize;
    let mut up_mismatches = 0usize;
    for i in 0..n_rows {
        if ref_gate[i].to_bits() != fused_gate[i].to_bits() {
            if gate_mismatches < 5 {
                eprintln!(
                    "gate[{i}] MISMATCH: ref={} ({:#x}) fused={} ({:#x})",
                    ref_gate[i], ref_gate[i].to_bits(),
                    fused_gate[i], fused_gate[i].to_bits(),
                );
            }
            gate_mismatches += 1;
        }
        if ref_up[i].to_bits() != fused_up[i].to_bits() {
            if up_mismatches < 5 {
                eprintln!(
                    "up[{i}] MISMATCH: ref={} ({:#x}) fused={} ({:#x})",
                    ref_up[i], ref_up[i].to_bits(),
                    fused_up[i], fused_up[i].to_bits(),
                );
            }
            up_mismatches += 1;
        }
    }

    if gate_mismatches > 0 || up_mismatches > 0 {
        panic!(
            "bit-exact failure: gate {}/{} mismatches, up {}/{} mismatches",
            gate_mismatches, n_rows, up_mismatches, n_rows,
        );
    }
    eprintln!(
        "PASS: n_rows={n_rows}, n_cols={n_cols}, bit-exact on both channels \
         ({n_rows} gate elements + {n_rows} up elements)"
    );
}

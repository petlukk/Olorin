//! Bit-exact parity: q6k_repacked_batch_ws_pre_d (new) vs
//! q6k_repacked_batch_ws (existing). Both feed the same
//! q6k_dot_q8k_4row_repacked tile kernel; the only difference is where
//! the scale array (`d * q8_d`) is computed. Pre-computing at load time
//! must not change any f32 bit pattern.

use std::path::Path;
use std::sync::atomic::AtomicI32;

fn model_path() -> String {
    std::env::var("OLORIN_MODEL").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")
    })
}

#[test]
fn q6k_pre_d_matches_in_kernel_extract() {
    if !Path::new(&model_path()).exists() {
        eprintln!("SKIP: model not present");
        return;
    }

    let h = std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(inner)
        .unwrap();
    h.join().unwrap();
}

fn inner() {
    let gguf = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    // We need embed_q6k_repacked and embed_q6k_d_arr both populated by the
    // model loader.
    let packed = model
        .embed_q6k_repacked
        .as_ref()
        .expect("embed_q6k_repacked should be populated for Q6K output head");
    let d_arr = model
        .embed_q6k_d_arr
        .as_ref()
        .expect("embed_q6k_d_arr should be populated for Q6K output head");

    let n_rows = model.vocab_size;
    let n_cols = model.hidden_dim;
    let n_blocks = n_cols / 256;
    let n_tokens = 1usize; // output-head is always last-token-only.

    // Build a single Q8K activation row.
    let mut input = vec![0.0f32; n_cols];
    for i in 0..n_cols { input[i] = 0.01 * (i % 97) as f32 - 0.5; }
    let mut qs = vec![0i8; n_cols + 12];
    let mut d = vec![0.0f32; n_blocks];
    let mut bs = vec![0i16; n_blocks * 16];
    unsafe {
        olorin::kernels::ffi_inference::quant_f32_q8k(
            input.as_ptr(), qs.as_mut_ptr(), d.as_mut_ptr(), bs.as_mut_ptr(), n_cols as i32,
        );
    }

    // One scratch area large enough for single-threaded ith=0.
    let scratch_per_thread = n_blocks * 4;
    let mut d_scratch = vec![0.0f32; scratch_per_thread];

    // Run path A: pre-d variant (what ships now)
    let mut out_a = vec![0.0f32; n_rows];
    let cc_a = AtomicI32::new(1);
    olorin::inference::matmul_graph::q6k_repacked_batch_ws_pre_d(
        packed.as_ptr(), d_arr.as_ptr(),
        qs.as_ptr(), d.as_ptr(), bs.as_ptr(),
        out_a.as_mut_ptr(), d_scratch.as_mut_ptr(),
        n_rows, n_cols, n_tokens, n_rows,
        &cc_a, 0, 1,
    );

    // Run path B: in-kernel extract variant (legacy)
    let mut out_b = vec![0.0f32; n_rows];
    let cc_b = AtomicI32::new(1);
    olorin::inference::matmul_graph::q6k_repacked_batch_ws(
        packed.as_ptr(), model.embed_weight,
        qs.as_ptr(), d.as_ptr(), bs.as_ptr(),
        out_b.as_mut_ptr(), d_scratch.as_mut_ptr(),
        n_rows, n_cols, n_tokens, n_rows,
        &cc_b, 0, 1,
    );

    // Bit-exact: same tile kernel, identical scratch values.
    let differ: Vec<usize> = out_a.iter().zip(out_b.iter())
        .enumerate()
        .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
        .map(|(i, _)| i)
        .take(8)
        .collect();
    if !differ.is_empty() {
        for &i in &differ {
            eprintln!("  row {i:7}: pre_d={:>12.6}  kernel_extract={:>12.6}  Δ={:+.6e}",
                out_a[i], out_b[i], out_a[i] - out_b[i]);
        }
    }
    assert!(differ.is_empty(), "q6k pre-d path diverges from in-kernel-extract path ({} mismatches)", differ.len());
}

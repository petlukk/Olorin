//! Bit-exact parity: q5k_matvec_repacked_ws vs q5k_matvec_ws on a real
//! Gemma 4 E2B Q5K tensor. Both paths should produce identical f32 output
//! — the only difference is weight memory layout, not arithmetic.

use std::path::Path;
use std::sync::atomic::AtomicI32;

fn model_path() -> String {
    std::env::var("OLORIN_MODEL").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")
    })
}

#[test]
fn q5k_repacked_matches_per_row() {
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
    olorin::kernels::ffi::init().unwrap();

    // Pick any Q5K 2D tensor with n_rows % 4 == 0 and n_cols % 256 == 0.
    // Gemma 4 E2B Q4_K_M has attn_k and attn_output as Q5K on every layer.
    let q5k = 13u32;
    let mut pick: Option<(String, usize, usize)> = None;
    for (name, &idx) in gguf.tensor_map.iter() {
        let t = &gguf.tensors[idx];
        if t.dtype != q5k { continue; }
        if t.dims.len() != 2 { continue; }
        let n_cols = t.dims[0] as usize;
        let n_rows = t.dims[1] as usize;
        if n_rows % 4 != 0 || n_cols % 256 != 0 { continue; }
        pick = Some((name.clone(), n_rows, n_cols));
        break;
    }
    let (name, n_rows, n_cols) = pick.expect("no Q5K tensor with eligible shape");
    eprintln!("parity: {name} n_rows={n_rows} n_cols={n_cols}");

    let data = gguf.tensor_data(&name).expect("tensor_data");
    let n_blocks = n_cols / 256;
    let byte_len = n_rows * n_blocks * 176;
    assert!(data.len() >= byte_len, "truncated tensor data");
    let raw: *const u8 = data.as_ptr();

    // Repack once.
    let packed = olorin::inference::repack::q5k_repack_4row(raw, n_rows, n_cols);

    // Synthesize a Q8K activation row. Same quantization as production.
    let mut input = vec![0.0f32; n_cols];
    for i in 0..n_cols { input[i] = 0.01 * (i % 97) as f32 - 0.5; }
    let mut qs = vec![0i8; n_cols + 12];
    let mut d  = vec![0.0f32; n_blocks];
    let mut bs = vec![0i16; n_blocks * 16];
    unsafe {
        olorin::kernels::ffi_inference::quant_f32_q8k(
            input.as_ptr(), qs.as_mut_ptr(), d.as_mut_ptr(), bs.as_mut_ptr(), n_cols as i32,
        );
    }

    // Path A: non-repacked.
    let mut out_a = vec![0.0f32; n_rows];
    let cc_a = AtomicI32::new(1);
    olorin::inference::matmul_graph::q5k_matvec_ws(
        raw, qs.as_ptr(), d.as_ptr(), bs.as_ptr(),
        out_a.as_mut_ptr(), n_rows, n_cols,
        &cc_a, 0, 1,
    );

    // Path B: repacked.
    let mut out_b = vec![0.0f32; n_rows];
    let cc_b = AtomicI32::new(1);
    olorin::inference::matmul_graph::q5k_matvec_repacked_ws(
        packed.as_ptr(), qs.as_ptr(), d.as_ptr(), bs.as_ptr(),
        out_b.as_mut_ptr(), n_rows, n_cols,
        &cc_b, 0, 1,
    );

    // Bit-exact comparison.
    let differ: Vec<usize> = out_a.iter().zip(out_b.iter())
        .enumerate()
        .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
        .map(|(i, _)| i)
        .take(8)
        .collect();
    if !differ.is_empty() {
        for &i in &differ {
            eprintln!("  row {i:5}: non-repacked={:>12.6}  repacked={:>12.6}  Δ={:+.6e}",
                out_a[i], out_b[i], out_a[i] - out_b[i]);
        }
    }
    assert!(differ.is_empty(),
        "q5k repacked path diverges from non-repacked ({} of {} rows)", differ.len(), n_rows);
}

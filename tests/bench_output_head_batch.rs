//! Output-head matmul cost as a function of batch size K.
//!
//! Spec-dec research question: when verification runs the output head on
//! K positions in one batched call, does cost scale sublinearly (batching
//! amortizes weight loads) or linearly (bandwidth-bound, no amortization)?
//!
//! At K=1 production path reports ~23ms per decode-step on 8 threads.
//! If cost(K=2) ≈ 1.1×cost(K=1), self-speculative at L=30 K=1 is a clear win.
//! If cost(K=2) ≈ 2.0×cost(K=1), speculative pays as much for verify-head
//! as it saves in skipped layers, killing the speedup.
//!
//! Uses `q6k_repacked_batch_ws_pre_d` — the exact production kernel
//! (forward_graph.rs:130) — with 8 threads via std::thread::scope.
//!
//! Run: cargo test --release --test bench_output_head_batch -- --nocapture

use std::path::Path;
use std::sync::Barrier;
use std::sync::atomic::{AtomicI32, Ordering};
use std::time::Instant;

const N_THREADS: usize = 8;
const WARMUP: usize = 2;
const ITERS: usize = 5;
const K_VALUES: &[usize] = &[1, 2, 4, 8];

fn model_path() -> String {
    std::env::var("OLORIN_MODEL").unwrap_or_else(|_| {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")
    })
}

#[test]
fn output_head_batch_scaling() {
    let h = std::thread::Builder::new()
        .stack_size(128 * 1024 * 1024)
        .spawn(inner)
        .unwrap();
    h.join().unwrap();
}

fn inner() {
    if !Path::new(&model_path()).exists() {
        eprintln!("SKIP: model not present");
        return;
    }

    let gguf = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    assert_eq!(
        model.embed_dtype,
        olorin::inference::matmul::GGML_TYPE_Q6_K,
        "expected Q6K output head"
    );
    let q6k_buf = model.embed_q6k_repacked.as_ref()
        .expect("q6k repacked buffer must be populated (normal model load does this)");
    let d_arr = model.embed_q6k_d_arr.as_ref()
        .expect("q6k d_arr must be populated");

    let m = model.vocab_size;  // 262144
    let k = model.hidden_dim;  // 1536
    let nb = k / 256;          // 6

    eprintln!();
    eprintln!("Output-head shape: m={m} (vocab) k={k} (hidden)  Q6K  threads={N_THREADS}");
    eprintln!();

    // Build activations for the maximum K we'll test (8).
    let k_max = *K_VALUES.iter().max().unwrap();
    let qs_stride = k + 12;
    let mut batch_qs = vec![0i8; k_max * qs_stride];
    let mut batch_d  = vec![0.0f32; k_max * nb];
    let mut batch_bs = vec![0i16; k_max * nb * 16];
    for t in 0..k_max {
        let mut input = vec![0.0f32; k];
        for i in 0..k {
            // Vary per-token so the matmul doesn't hit any cache trick.
            input[i] = 0.01 * ((i + t * 17) % 97) as f32 - 0.5;
        }
        unsafe {
            olorin::kernels::ffi_inference::quant_f32_q8k(
                input.as_ptr(),
                batch_qs.as_mut_ptr().add(t * qs_stride),
                batch_d.as_mut_ptr().add(t * nb),
                batch_bs.as_mut_ptr().add(t * nb * 16),
                k as i32,
            );
        }
    }

    let current_chunk = AtomicI32::new(0);
    let d_scratch_per = nb * 4;
    let mut d_scratch = vec![0.0f32; N_THREADS * d_scratch_per];

    let mut k1_time: Option<f64> = None;
    eprintln!("{:>4}  {:>10}  {:>10}  {:>10}  {:>12}", "K", "ms/call", "ms/token", "ratio_vs_K1", "head_GFLOPS");
    eprintln!("{:-<56}", "");

    for &k_batch in K_VALUES {
        let mut output = vec![0.0f32; k_batch * m];

        // The bench closure: fresh current_chunk each iter, 8 threads all call the kernel.
        let run = |output: &mut [f32], d_scratch: &mut [f32]| {
            let barrier = Barrier::new(N_THREADS);
            current_chunk.store(N_THREADS as i32, Ordering::Relaxed);
            std::thread::scope(|s| {
                for ith in 0..N_THREADS {
                    // Raw pointers carried across the boundary as usize.
                    let qs_p = batch_qs.as_ptr() as usize;
                    let d_p  = batch_d.as_ptr()  as usize;
                    let bs_p = batch_bs.as_ptr() as usize;
                    let out_p = output.as_mut_ptr() as usize;
                    let sc_p  = d_scratch.as_mut_ptr() as usize;
                    let weight_p = q6k_buf.as_ptr() as usize;
                    let darr_p   = d_arr.as_ptr() as usize;
                    let barrier = &barrier;
                    let chunk = &current_chunk;
                    s.spawn(move || {
                        barrier.wait();
                        olorin::inference::matmul_graph::q6k_repacked_batch_ws_pre_d(
                            weight_p as *const u8,
                            darr_p as *const f32,
                            qs_p as *const i8,
                            d_p  as *const f32,
                            bs_p as *const i16,
                            out_p as *mut f32,
                            sc_p as *mut f32,
                            m, k, k_batch, m,
                            chunk, ith, N_THREADS,
                        );
                        barrier.wait();
                    });
                }
            });
        };

        for _ in 0..WARMUP { run(&mut output, &mut d_scratch); }

        let t0 = Instant::now();
        for _ in 0..ITERS { run(&mut output, &mut d_scratch); }
        let sec = t0.elapsed().as_secs_f64() / ITERS as f64;

        let ms = sec * 1000.0;
        let ms_per_tok = ms / k_batch as f64;
        let ratio = match k1_time {
            Some(t1) => ms / t1,
            None => { k1_time = Some(ms); 1.0 }
        };
        let flops = 2.0 * m as f64 * k as f64 * k_batch as f64;
        let gflops = flops / sec / 1e9;

        eprintln!("{k_batch:>4}  {ms:>10.3}  {ms_per_tok:>10.3}  {ratio:>10.3}x  {gflops:>12.2}");
    }

    eprintln!();
    eprintln!("Interpretation for speculative decoding at L=30, K=1 draft + K+1=2 verify:");
    eprintln!("  verify head cost = ratio(K=2) × single-token head cost");
    eprintln!("  spec-dec speedup ∝ 1 / (draft + verify), breakeven vs baseline ~77ms/tok");
    eprintln!("  if ratio(K=2) < 1.5× → spec-dec at L=30 K=1 is a clear win");
    eprintln!("  if ratio(K=2) > 1.9× → no meaningful head batching, re-evaluate design");
}

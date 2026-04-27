//! Microbenchmark for gemma4_gelu kernel — measures absolute throughput
//! of the current `gelu_mul` so two builds (libm exp vs exp_poly_f32) can
//! be compared by running this bench at each commit and diffing the numbers.
//!
//! No model load — just buffer fill, kernel call, time.
//!
//! Shapes reflect production: ple_dim=256 (PLE inp_gate output) and
//! ffn_dim=12288 (largest layer FFN gate, decode dominator).
//! Decode invokes gelu_mul once per layer per token, so per-call latency
//! at ffn_dim is on the inference critical path.
//!
//! Run: cargo test --release --test bench_gemma4_gelu -- --nocapture
//!
//! Compare runs:
//!   git checkout <baseline-commit>; cargo test --release --test bench_gemma4_gelu -- --nocapture
//!   git checkout <swap-commit>;     cargo test --release --test bench_gemma4_gelu -- --nocapture

use std::time::Instant;

#[test]
fn bench_gemma4_gelu() {
    let h = std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(inner)
        .unwrap();
    h.join().unwrap();
}

fn inner() {
    olorin::kernels::ffi::init().unwrap();

    eprintln!("{:<10}  {:>10}  {:>12}  {:>10}  {:>9}", "n", "iters", "ns/call", "ns/elem", "Gelem/s");
    eprintln!("{:-<60}", "");

    // Cover: tiny (overhead-dominated), PLE gate, layer FFN.
    for &n in &[64usize, 256, 2048, 12288] {
        bench_at(n);
    }
}

fn bench_at(n: usize) {
    // Realistic activation distribution: roughly N(0, 1) with some outliers.
    // Use deterministic LCG so runs are comparable.
    let mut state: u64 = 0xDEADBEEFCAFEBABE;
    let mut rng = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
    };

    let mut gate = vec![0.0f32; n];
    let mut up   = vec![0.0f32; n];
    for i in 0..n {
        // Box-Muller-ish: combine two uniforms to get something heavier-tailed than uniform.
        let a = rng();
        let b = rng();
        gate[i] = a * 2.0 + b;            // ~stddev 2.2, occasional outliers
        up[i]   = rng() * 1.5;
    }
    let mut out = vec![0.0f32; n];

    // Warmup
    for _ in 0..50 {
        olorin::kernels::ffi_inference::gelu_mul(gate.as_ptr(), up.as_ptr(), out.as_mut_ptr(), n as i32);
    }

    // Target ~200ms per data point. Tiny n needs many iters to escape clock noise.
    let target_ns: u64 = 200_000_000;
    let probe_iters = 1000;
    let t0 = Instant::now();
    for _ in 0..probe_iters {
        olorin::kernels::ffi_inference::gelu_mul(gate.as_ptr(), up.as_ptr(), out.as_mut_ptr(), n as i32);
    }
    let probe_ns = t0.elapsed().as_nanos() as u64;
    let ns_per_call_probe = probe_ns / probe_iters as u64;
    let iters = (target_ns / ns_per_call_probe.max(1)).clamp(1000, 5_000_000) as usize;

    let t0 = Instant::now();
    for _ in 0..iters {
        olorin::kernels::ffi_inference::gelu_mul(gate.as_ptr(), up.as_ptr(), out.as_mut_ptr(), n as i32);
    }
    let total_ns = t0.elapsed().as_nanos() as u64;

    let ns_per_call = total_ns as f64 / iters as f64;
    let ns_per_elem = ns_per_call / n as f64;
    let gelem_per_s = (n as f64 * iters as f64) / total_ns as f64;

    eprintln!(
        "{:<10}  {:>10}  {:>12.1}  {:>10.3}  {:>9.3}",
        n, iters, ns_per_call, ns_per_elem, gelem_per_s
    );

    // Sanity: check first few outputs aren't NaN/Inf.
    for i in 0..n.min(8) {
        assert!(out[i].is_finite(), "gelu_mul produced non-finite at i={}: {}", i, out[i]);
    }
}

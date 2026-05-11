//! Manual perf demo for the ealog rune. Run with:
//!   cargo test --release --test ealog_perf_demo perf_demo -- --nocapture --ignored
//!
//! Expects /tmp/ealog_bench.log to exist (synthesized by an external
//! script). This test is ignored by default so the normal test sweep
//! doesn't depend on the fixture.

use olorin::kernels::ffi;
use std::time::Instant;

#[test]
#[ignore]
fn perf_demo_kernel_only() {
    ffi::init().unwrap();
    let path = "/tmp/ealog_bench.log";
    let bytes = std::fs::read(path).expect("run /tmp/ealog_bench.py first");
    println!("file: {} bytes", bytes.len());

    let mut counts = [0i32; 6];
    let mut positions = [0i32; 16];
    let mut n_pos = 0i32;
    let mut scratch = [0u8; 16];
    let mut best_ns = u128::MAX;
    for trial in 0..5 {
        let t = Instant::now();
        unsafe {
            ffi::log_level_scan(
                bytes.as_ptr(), bytes.len() as i32,
                counts.as_mut_ptr(),
                positions.as_mut_ptr(), 16, &mut n_pos,
                scratch.as_mut_ptr(),
            );
        }
        let ns = t.elapsed().as_nanos();
        println!(
            "trial {trial}: kernel {:.1} ms, {:.2} GB/s, counts={counts:?}",
            ns as f64 / 1e6,
            (bytes.len() as f64) / (ns as f64),
        );
        if ns < best_ns {
            best_ns = ns;
        }
    }
    let best_ms = best_ns as f64 / 1e6;
    let best_gbs = (bytes.len() as f64) / (best_ns as f64);
    println!("best: {best_ms:.1} ms  ({best_gbs:.2} GB/s)");
    println!("counts: DEBUG={} INFO={} WARN={} ERROR={} FATAL={} NL={}",
        counts[0], counts[1], counts[2], counts[3], counts[4], counts[5]);
}

//! Statistical smoke test for constant-time `poly1305_verify`.
//!
//! Compares median wall-clock verify times for two tag-difference profiles:
//! a flip at byte 0 vs a flip at byte 15.  An early-exit memcmp-style
//! implementation would terminate immediately on byte 0 and run to the end
//! on byte 15, producing a large median divergence; a constant-time
//! OR-reduce + branchless `is_zero` (what we ship) should see the medians
//! land within ~20% of each other on a quiet machine.
//!
//! `#[ignore]`d by default — wall-clock measurements are noisy under
//! parallel test execution and on contended CPUs.  Run explicitly:
//!
//!   cargo test --test poly1305_constant_time -- --ignored --test-threads=1
//!
//! On a noisy workstation expect the occasional false negative; the bar
//! here is "no large systematic divergence", not "rigorous side-channel
//! analysis".

use olorin::kernels::ffi;
use std::time::Instant;

fn measure_verify_ns(key: &[u8; 32], msg: &[u8], tag: &[u8; 16]) -> u128 {
    let start = Instant::now();
    let _ = unsafe {
        ffi::poly1305_verify(key.as_ptr(), msg.as_ptr(), msg.len() as i32, tag.as_ptr())
    };
    start.elapsed().as_nanos()
}

#[test]
#[ignore]
fn verify_time_does_not_depend_on_differ_position() {
    ffi::init().expect("kernel init");

    let key = [0x42u8; 32];
    let msg = b"timing leak test message";

    // Compute the correct tag once.
    let mut correct = [0u8; 16];
    unsafe {
        ffi::poly1305_mac(key.as_ptr(), msg.as_ptr(), msg.len() as i32, correct.as_mut_ptr());
    }

    const N: usize = 10_000;
    let mut t_byte0 = Vec::with_capacity(N);
    let mut t_byte15 = Vec::with_capacity(N);

    // Warmup so the first samples don't include cold-cache effects.
    for _ in 0..1000 {
        let mut t = correct;
        t[0] ^= 0x01;
        let _ = measure_verify_ns(&key, msg, &t);
    }

    for _ in 0..N {
        let mut t = correct;
        t[0] ^= 0x01;
        t_byte0.push(measure_verify_ns(&key, msg, &t));
    }
    for _ in 0..N {
        let mut t = correct;
        t[15] ^= 0x01;
        t_byte15.push(measure_verify_ns(&key, msg, &t));
    }

    t_byte0.sort();
    t_byte15.sort();
    let median_byte0 = t_byte0[N / 2];
    let median_byte15 = t_byte15[N / 2];

    let lo = median_byte0.min(median_byte15) as f64;
    let hi = median_byte0.max(median_byte15) as f64;
    let ratio = hi / lo;
    assert!(
        ratio < 1.2,
        "median verify time depends on differ position: byte0={median_byte0}ns byte15={median_byte15}ns (ratio {ratio:.3})",
    );
}

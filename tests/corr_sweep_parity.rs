//! corr_sweep kernel parity — the SIMD lag sweep must agree with a
//! scalar reference across shapes and random data, recover a planted
//! lag exactly, and honor its always-fully-written output contract.
//!
//! Each score divides reduce_add's lane-wise f32x4 sum, whose summation
//! order differs from a sequential scalar sum, so scores are compared
//! with a relative tolerance rather than bit-exact (per the
//! SIMD-parity-use-ULP rule). Argmax-over-lags is a selection on well-
//! separated values and must match exactly.

use olorin::kernels::ffi;

fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Scalar reference: same lag convention as the kernel —
/// out[max_lag + lag] pairs a[i + lag] with b[i] over the overlap,
/// divided by the overlap length.
fn ref_corr_sweep(a: &[f32], b: &[f32], max_lag: i32) -> Vec<f32> {
    let n = a.len() as i32;
    let mut out = vec![0.0f32; (2 * max_lag + 1) as usize];
    for lag in -max_lag..=max_lag {
        let (off_a, off_b, m) = if lag >= 0 {
            (lag, 0, n - lag)
        } else {
            (0, -lag, n + lag)
        };
        if m < 1 {
            continue;
        }
        let mut s = 0.0f32;
        for i in 0..m {
            s += a[(off_a + i) as usize] * b[(off_b + i) as usize];
        }
        out[(max_lag + lag) as usize] = s / (m as f32);
    }
    out
}

fn call_kernel(a: &[f32], b: &[f32], max_lag: i32) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    let mut out = vec![f32::NAN; (2 * max_lag + 1) as usize];
    unsafe {
        ffi::corr_sweep(
            a.as_ptr(), b.as_ptr(),
            a.len() as i32, max_lag,
            out.as_mut_ptr(),
        );
    }
    out
}

fn assert_scores_close(kernel: &[f32], reference: &[f32], ctx: &str) {
    assert_eq!(kernel.len(), reference.len(), "{ctx}: length mismatch");
    for (i, (&kv, &rv)) in kernel.iter().zip(reference).enumerate() {
        let denom = rv.abs().max(1.0);
        let rel = (kv - rv).abs() / denom;
        assert!(
            rel < 1e-4,
            "{ctx}: lag slot {i}: kernel={kv} ref={rv} rel={rel:.2e}"
        );
    }
}

#[test]
fn parity_random_sweep() {
    ffi::init().expect("kernel init");
    let mut state: u64 = 0x1357_9bdf_2468_ace0;

    // (n, max_lag) shapes: the eacorrelate production shape, the eatime
    // grid, odd sizes exercising scalar tails, max_lag >= n (zero-overlap
    // lags), and a minimal series.
    let shapes = [
        (512usize, 128i32),
        (120, 60),
        (333, 17),
        (8, 16),   // |lag| >= n slots must stay exactly 0
        (1, 0),
    ];

    for &(n, max_lag) in &shapes {
        let gen = |state: &mut u64| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    let r = (xorshift64(state) >> 40) as f32 / 16_777_216.0;
                    r * 4.0 - 2.0 // z-score-ish range, crosses zero
                })
                .collect()
        };
        let a = gen(&mut state);
        let b = gen(&mut state);

        let kernel = call_kernel(&a, &b, max_lag);
        let reference = ref_corr_sweep(&a, &b, max_lag);
        assert_scores_close(&kernel, &reference, &format!("n={n} max_lag={max_lag}"));
    }
}

#[test]
fn recovers_planted_lag() {
    ffi::init().expect("kernel init");
    // b carries an impulse train; a is b shifted 7 buckets later
    // (events in a happen AFTER events in b). The sweep must peak at
    // lag = +7 under the kernel's sign convention.
    const N: usize = 512;
    const SHIFT: usize = 7;
    let mut b = vec![0.0f32; N];
    let mut a = vec![0.0f32; N];
    let mut p = 13usize;
    while p + SHIFT < N {
        b[p] = 1.0;
        a[p + SHIFT] = 1.0;
        p += 31;
    }

    let max_lag = 32;
    let out = call_kernel(&a, &b, max_lag);
    let argmax = out
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
        .unwrap()
        .0 as i32;
    assert_eq!(
        argmax - max_lag,
        SHIFT as i32,
        "peak lag mismatch: scores={out:?}"
    );
}

#[test]
fn symmetry_swapping_inputs_mirrors_lags() {
    ffi::init().expect("kernel init");
    let mut state: u64 = 0xdead_beef_cafe_f00d;
    let n = 200usize;
    let series = |state: &mut u64| -> Vec<f32> {
        (0..n)
            .map(|_| (xorshift64(state) >> 40) as f32 / 16_777_216.0 - 0.5)
            .collect()
    };
    let a = series(&mut state);
    let b = series(&mut state);

    let max_lag = 40i32;
    let ab = call_kernel(&a, &b, max_lag);
    let ba = call_kernel(&b, &a, max_lag);
    // corr(a,b) at lag L equals corr(b,a) at lag -L: identical pair sets,
    // identical overlap, only the roles swap. Same summation order in the
    // kernel, but lanes split differently when offsets differ — compare
    // with the usual relative tolerance.
    let mirrored: Vec<f32> = ba.iter().rev().copied().collect();
    assert_scores_close(&ab, &mirrored, "swap symmetry");
}

#[test]
fn empty_input_writes_zeros() {
    ffi::init().expect("kernel init");
    // n == 0: the kernel must still fully write the output (all zeros),
    // never leaving the caller's sentinels behind.
    let a: [f32; 0] = [];
    let b: [f32; 0] = [];
    let mut out = [f32::NAN; 9];
    unsafe {
        ffi::corr_sweep(a.as_ptr(), b.as_ptr(), 0, 4, out.as_mut_ptr());
    }
    assert_eq!(out, [0.0; 9]);
}

#[test]
fn negative_max_lag_writes_nothing() {
    ffi::init().expect("kernel init");
    let a = [1.0f32, 2.0, 3.0];
    let b = [4.0f32, 5.0, 6.0];
    let mut out = [f32::NAN; 3];
    unsafe {
        ffi::corr_sweep(a.as_ptr(), b.as_ptr(), 3, -1, out.as_mut_ptr());
    }
    assert!(out.iter().all(|v| v.is_nan()), "sentinels must survive");
}

#[test]
fn zero_lag_is_normalized_dot() {
    ffi::init().expect("kernel init");
    // max_lag == 0: single slot, the plain dot product over n divided by n.
    let a = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let b = [2.0f32, 2.0, 2.0, 2.0, 2.0];
    let out = call_kernel(&a, &b, 0);
    let expected = (1.0 + 2.0 + 3.0 + 4.0 + 5.0) * 2.0 / 5.0;
    assert!((out[0] - expected).abs() < 1e-6, "got {} want {expected}", out[0]);
}

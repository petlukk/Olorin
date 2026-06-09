//! col_reduce kernel parity — the SIMD per-column (min, max, mean)
//! envelope must agree with a scalar reference across a sweep of
//! (len, n_cols) shapes and random data.
//!
//! min/max are selections (order-independent, no rounding) so they must
//! match bit-exactly. The mean divides reduce_add's lane-wise f32x4 sum,
//! whose summation order differs from a sequential scalar sum, so it is
//! compared with a relative tolerance rather than bit-exact (per the
//! SIMD-parity-use-ULP rule).

use olorin::kernels::ffi;

fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// Scalar reference: integer column bounds identical to the kernel, then
/// a sequential min/max/sum over each column's slice.
fn ref_col_reduce(data: &[f32], n_cols: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = data.len();
    let mut mins = vec![0.0f32; n_cols];
    let mut maxs = vec![0.0f32; n_cols];
    let mut means = vec![0.0f32; n_cols];
    for c in 0..n_cols {
        let lo = (c * len) / n_cols;
        let hi = ((c + 1) * len) / n_cols;
        let mut mn = data[lo];
        let mut mx = data[lo];
        let mut s = 0.0f32;
        for &v in &data[lo..hi] {
            if v < mn { mn = v; }
            if v > mx { mx = v; }
            s += v;
        }
        mins[c] = mn;
        maxs[c] = mx;
        means[c] = s / ((hi - lo) as f32);
    }
    (mins, maxs, means)
}

fn call_kernel(data: &[f32], n_cols: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut mins = vec![f32::NAN; n_cols];
    let mut maxs = vec![f32::NAN; n_cols];
    let mut means = vec![f32::NAN; n_cols];
    unsafe {
        ffi::col_reduce(
            data.as_ptr(),
            data.len() as i32,
            n_cols as i32,
            mins.as_mut_ptr(),
            maxs.as_mut_ptr(),
            means.as_mut_ptr(),
        );
    }
    (mins, maxs, means)
}

#[test]
fn parity_random_sweep() {
    ffi::init().expect("kernel init");
    let mut state: u64 = 0x0102_0304_0506_0708;

    // (len, n_cols) shapes: 1:1 mapping, single column, heavy downsample,
    // non-divisible ratios (uneven column widths), and a vector-tail mix.
    let shapes = [
        (64usize, 64usize),
        (200, 1),
        (10_000, 80),
        (10_000, 137),  // non-divisible → uneven column widths
        (4096, 200),
        (333, 17),      // small + odd, exercises scalar tails per column
    ];

    for &(len, n_cols) in &shapes {
        // Random data in a range that spans negatives and crosses zero.
        let data: Vec<f32> = (0..len)
            .map(|_| {
                let r = (xorshift64(&mut state) >> 40) as f32 / 16_777_216.0;
                r * 2000.0 - 500.0
            })
            .collect();

        let (km, kx, kmean) = call_kernel(&data, n_cols);
        let (rm, rx, rmean) = ref_col_reduce(&data, n_cols);

        for c in 0..n_cols {
            // min/max: selections, must be bit-exact.
            assert_eq!(
                km[c].to_bits(), rm[c].to_bits(),
                "min mismatch len={len} n_cols={n_cols} col={c}: kernel={} ref={}",
                km[c], rm[c]
            );
            assert_eq!(
                kx[c].to_bits(), rx[c].to_bits(),
                "max mismatch len={len} n_cols={n_cols} col={c}: kernel={} ref={}",
                kx[c], rx[c]
            );
            // mean: divide of reorder-summed sum, relative tolerance.
            let denom = rmean[c].abs().max(1.0);
            let rel = (kmean[c] - rmean[c]).abs() / denom;
            assert!(
                rel < 1e-4,
                "mean drift len={len} n_cols={n_cols} col={c}: kernel={} ref={} rel={rel:.2e}",
                kmean[c], rmean[c]
            );
        }
    }
}

#[test]
fn one_sample_per_column_is_identity() {
    ffi::init().expect("kernel init");
    // n_cols == len: every column owns exactly one sample, so min == max
    // == mean == that sample, exactly (no summation reorder possible).
    let data: Vec<f32> = (0..50).map(|i| (i as f32) * 3.5 - 17.0).collect();
    let (mins, maxs, means) = call_kernel(&data, data.len());
    for (i, &v) in data.iter().enumerate() {
        assert_eq!(mins[i].to_bits(), v.to_bits());
        assert_eq!(maxs[i].to_bits(), v.to_bits());
        assert_eq!(means[i].to_bits(), v.to_bits());
    }
}

#[test]
fn empty_input_leaves_outputs_untouched() {
    ffi::init().expect("kernel init");
    // len == 0: kernel must return without writing, so the caller's
    // pre-filled sentinels survive.
    let data: [f32; 0] = [];
    let mut mins = [-1.0f32; 4];
    let mut maxs = [-1.0f32; 4];
    let mut means = [-1.0f32; 4];
    unsafe {
        ffi::col_reduce(data.as_ptr(), 0, 4, mins.as_mut_ptr(), maxs.as_mut_ptr(), means.as_mut_ptr());
    }
    assert_eq!(mins, [-1.0; 4]);
    assert_eq!(maxs, [-1.0; 4]);
    assert_eq!(means, [-1.0; 4]);
}

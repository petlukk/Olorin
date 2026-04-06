use olorin::kernels::ffi;
use olorin::kernels::ffi_inference;

fn init() {
    ffi::init().expect("kernel init failed");
}

// ---- vec_add_f32 ----

#[test]
fn test_vec_add_f32_1536() {
    init();
    let n = 1536;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.2 + 1.0).collect();
    let mut out = vec![0.0f32; n];
    let mut ref_out = vec![0.0f32; n];
    for i in 0..n { ref_out[i] = a[i] + b[i]; }
    ffi_inference::vec_add_f32(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), n as i32);
    for i in 0..n {
        assert!((out[i] - ref_out[i]).abs() < 1e-5, "vec_add mismatch at {i}: {} vs {}", out[i], ref_out[i]);
    }
}

#[test]
fn test_vec_add_f32_8960() {
    init();
    let n = 8960;
    let a: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32).cos()).collect();
    let mut out = vec![0.0f32; n];
    let mut ref_out = vec![0.0f32; n];
    for i in 0..n { ref_out[i] = a[i] + b[i]; }
    ffi_inference::vec_add_f32(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), n as i32);
    for i in 0..n {
        assert!((out[i] - ref_out[i]).abs() < 1e-5, "vec_add mismatch at {i}");
    }
}

// ---- vec_scale_f32 ----

#[test]
fn test_vec_scale_f32_1536() {
    init();
    let n = 1536;
    let s = 2.5f32;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.3 - 50.0).collect();
    let mut out = vec![0.0f32; n];
    let mut ref_out = vec![0.0f32; n];
    for i in 0..n { ref_out[i] = a[i] * s; }
    ffi_inference::vec_scale_f32(a.as_ptr(), out.as_mut_ptr(), s, n as i32);
    for i in 0..n {
        assert!((out[i] - ref_out[i]).abs() < 1e-4, "vec_scale mismatch at {i}: {} vs {}", out[i], ref_out[i]);
    }
}

#[test]
fn test_vec_scale_f32_8960() {
    init();
    let n = 8960;
    let s = -0.7f32;
    let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
    let mut out = vec![0.0f32; n];
    let mut ref_out = vec![0.0f32; n];
    for i in 0..n { ref_out[i] = a[i] * s; }
    ffi_inference::vec_scale_f32(a.as_ptr(), out.as_mut_ptr(), s, n as i32);
    for i in 0..n {
        assert!((out[i] - ref_out[i]).abs() < 1e-5, "vec_scale mismatch at {i}");
    }
}

// ---- vec_fma_f32 ----

#[test]
fn test_vec_fma_f32_1536() {
    init();
    let n = 1536;
    let s = 1.5f32;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..n).map(|i| i as f32 * -0.05 + 3.0).collect();
    let mut out = vec![0.0f32; n];
    let mut ref_out = vec![0.0f32; n];
    for i in 0..n { ref_out[i] = (a[i] + b[i]) * s; }
    ffi_inference::vec_fma_f32(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), s, n as i32);
    for i in 0..n {
        assert!((out[i] - ref_out[i]).abs() < 1e-4, "vec_fma mismatch at {i}: {} vs {}", out[i], ref_out[i]);
    }
}

#[test]
fn test_vec_fma_f32_8960() {
    init();
    let n = 8960;
    let s = 0.333f32;
    let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).sin()).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01).cos()).collect();
    let mut out = vec![0.0f32; n];
    let mut ref_out = vec![0.0f32; n];
    for i in 0..n { ref_out[i] = (a[i] + b[i]) * s; }
    ffi_inference::vec_fma_f32(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), s, n as i32);
    for i in 0..n {
        assert!((out[i] - ref_out[i]).abs() < 1e-5, "vec_fma mismatch at {i}");
    }
}

// ---- vec_acc_f32 ----

#[test]
fn test_vec_acc_f32_1536() {
    init();
    let n = 1536;
    let s = 0.5f32;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.2).collect();
    let mut out: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let mut ref_out = out.clone();
    for i in 0..n { ref_out[i] = ref_out[i] + a[i] * s; }
    ffi_inference::vec_acc_f32(out.as_mut_ptr(), a.as_ptr(), s, n as i32);
    for i in 0..n {
        assert!((out[i] - ref_out[i]).abs() < 1e-4, "vec_acc mismatch at {i}: {} vs {}", out[i], ref_out[i]);
    }
}

#[test]
fn test_vec_acc_f32_8960() {
    init();
    let n = 8960;
    let s = -1.2f32;
    let a: Vec<f32> = (0..n).map(|i| (i as f32 * 0.005).sin()).collect();
    let mut out: Vec<f32> = (0..n).map(|i| (i as f32 * 0.003).cos()).collect();
    let mut ref_out = out.clone();
    for i in 0..n { ref_out[i] = ref_out[i] + a[i] * s; }
    ffi_inference::vec_acc_f32(out.as_mut_ptr(), a.as_ptr(), s, n as i32);
    for i in 0..n {
        assert!((out[i] - ref_out[i]).abs() < 1e-5, "vec_acc mismatch at {i}");
    }
}

// ---- Odd sizes ----

#[test]
fn test_vec_ops_odd_sizes() {
    init();
    let sizes = [1, 3, 7, 13, 17, 31, 100, 255];
    let s = 2.0f32;

    for &n in &sizes {
        let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 + 1.0).collect();
        let b: Vec<f32> = (0..n).map(|i| i as f32 * -0.2 + 0.5).collect();

        // vec_add
        let mut out = vec![0.0f32; n];
        ffi_inference::vec_add_f32(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), n as i32);
        for i in 0..n {
            let expected = a[i] + b[i];
            assert!((out[i] - expected).abs() < 1e-5, "vec_add n={n} i={i}");
        }

        // vec_scale
        let mut out = vec![0.0f32; n];
        ffi_inference::vec_scale_f32(a.as_ptr(), out.as_mut_ptr(), s, n as i32);
        for i in 0..n {
            let expected = a[i] * s;
            assert!((out[i] - expected).abs() < 1e-5, "vec_scale n={n} i={i}");
        }

        // vec_fma
        let mut out = vec![0.0f32; n];
        ffi_inference::vec_fma_f32(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), s, n as i32);
        for i in 0..n {
            let expected = (a[i] + b[i]) * s;
            assert!((out[i] - expected).abs() < 1e-4, "vec_fma n={n} i={i}");
        }

        // vec_acc
        let mut out: Vec<f32> = (0..n).map(|i| i as f32 * 0.05).collect();
        let ref_out: Vec<f32> = out.iter().zip(a.iter()).map(|(&o, &ai)| o + ai * s).collect();
        ffi_inference::vec_acc_f32(out.as_mut_ptr(), a.as_ptr(), s, n as i32);
        for i in 0..n {
            assert!((out[i] - ref_out[i]).abs() < 1e-4, "vec_acc n={n} i={i}");
        }
    }
}

// ---- f32_dot ----

#[test]
fn test_f32_dot() {
    init();
    let a: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
    let b: Vec<f32> = (0..256).map(|i| 1.0 - (i as f32) * 0.005).collect();
    let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
    let got = ffi_inference::f32_dot(a.as_ptr(), b.as_ptr(), 256);
    assert!((expected - got).abs() < 0.01, "f32_dot: {expected} vs {got}");
}

#[test]
fn test_f32_dot_head_dims() {
    init();
    for n in [128, 256, 512] {
        let a: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 100) as f32 * 0.01).collect();
        let b: Vec<f32> = (0..n).map(|i| ((i * 13 + 5) % 100) as f32 * 0.01).collect();
        let expected: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let got = ffi_inference::f32_dot(a.as_ptr(), b.as_ptr(), n as i32);
        assert!((expected - got).abs() / expected.abs().max(1.0) < 1e-4,
            "f32_dot size {n}: {expected} vs {got}");
    }
}

// ---- f32_dot_acc ----

#[test]
fn test_f32_dot_acc() {
    init();
    let mut out = vec![1.0f32; 512];
    let a: Vec<f32> = (0..512).map(|i| i as f32 * 0.01).collect();
    let mut expected = vec![1.0f32; 512];
    for (o, x) in expected.iter_mut().zip(&a) { *o += x * 0.5; }
    ffi_inference::f32_dot_acc(out.as_mut_ptr(), a.as_ptr(), 0.5, 512);
    for (i, (e, g)) in expected.iter().zip(&out).enumerate() {
        assert!((e - g).abs() < 1e-5, "f32_dot_acc mismatch at {i}");
    }
}

// ---- bare_rmsnorm_f32 ----

#[test]
fn test_bare_rmsnorm() {
    init();
    let input: Vec<f32> = (0..512).map(|i| (i as f32 - 256.0) * 0.01).collect();
    let eps = 1e-6f32;
    let ss: f32 = input.iter().map(|v| v * v).sum();
    let scale = 1.0 / ((ss / 512.0) + eps).sqrt();
    let expected: Vec<f32> = input.iter().map(|v| v * scale).collect();
    let mut got = input.clone();
    ffi_inference::bare_rmsnorm_f32(got.as_mut_ptr(), 512, eps);
    for (i, (e, g)) in expected.iter().zip(&got).enumerate() {
        assert!((e - g).abs() < 1e-5, "bare_rmsnorm mismatch at {i}: {e} vs {g}");
    }
}

// ---- softcap_f32 ----

#[test]
fn test_softcap() {
    init();
    let cap = 30.0f32;
    let mut data: Vec<f32> = (-500..500).map(|i| i as f32 * 0.1).collect();
    let expected: Vec<f32> = data.iter().map(|&x| cap * (x / cap).tanh()).collect();
    ffi_inference::softcap_f32(data.as_mut_ptr(), data.len() as i32, cap);
    for (i, (e, g)) in expected.iter().zip(&data).enumerate() {
        assert!((e - g).abs() < 1e-4, "softcap mismatch at {i}: expected {e} got {g}");
    }
}

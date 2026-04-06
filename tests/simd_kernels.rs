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

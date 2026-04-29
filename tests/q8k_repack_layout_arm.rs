//! Verify q8k_repack_4_arm's output layout matches its docstring claim:
//!   "Group g: dst[16 + g*16 + r*4 .. 16 + g*16 + r*4 + 3] = row_r_qs[g*4..g*4+3]"
//!
//! Runs on aarch64 only.

#![cfg(target_arch = "aarch64")]

#[test]
fn q8k_repack_4_arm_layout_matches_docstring() {
    olorin::kernels::ffi::init().unwrap();

    let nb = 1usize;
    let n = 256usize;
    // Construct distinct qs per row so layout errors are detectable
    let mut qs0 = vec![0i8; n + 12];
    let mut qs1 = vec![0i8; n + 12];
    let mut qs2 = vec![0i8; n + 12];
    let mut qs3 = vec![0i8; n + 12];
    for i in 0..n {
        qs0[i] = ((i      ) % 127) as i8;
        qs1[i] = ((i + 100) % 127) as i8 + 1;
        qs2[i] = ((i + 200) % 127) as i8 + 2;
        qs3[i] = ((i + 300) % 127) as i8 + 3;
    }
    let row_d = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
    let bsums = vec![0i16; 16];

    let mut q8_a = vec![0u8; nb * 1168];
    unsafe {
        olorin::kernels::ffi_inference::q8k_repack_4(
            qs0.as_ptr(), qs1.as_ptr(), qs2.as_ptr(), qs3.as_ptr(),
            row_d.as_ptr(),
            bsums.as_ptr(), bsums.as_ptr(), bsums.as_ptr(), bsums.as_ptr(),
            q8_a.as_mut_ptr(), nb as i32,
        );
    }

    let qs = [&qs0[..], &qs1[..], &qs2[..], &qs3[..]];
    let mut errors = 0;
    for g in 0..64 {
        for r in 0..4 {
            for p in 0..4 {
                let off = 16 + g * 16 + r * 4 + p;
                let actual = q8_a[off] as i8;
                let expected = qs[r][g * 4 + p];
                if actual != expected {
                    errors += 1;
                    if errors <= 20 {
                        eprintln!("MISMATCH: g={} r={} p={} byte_off={}: actual={} expected={} (qs[{}][{}])",
                                  g, r, p, off, actual, expected, r, g*4+p);
                    }
                }
            }
        }
    }
    eprintln!("layout errors: {} / {}", errors, 64*4*4);
    assert_eq!(errors, 0, "q8k_repack_4_arm layout doesn't match docstring");
}

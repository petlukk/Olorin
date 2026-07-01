//! Standalone parity test for `attn_fused_batched` — isolates the NEON vs SSE2
//! precision divergence localized to this kernel (BOS-logit bisection, 2026-07-01).
//!
//! The kernel is bit-exact across arches at head_dim=256 (sliding-window layers)
//! but diverges at head_dim=512 (global layers), accumulating ~2.5% over the 35
//! layers — enough to flip tool-call emission on the Pi.
//!
//! Inputs are generated from a pure-integer LCG (bit-identical on every arch),
//! K/V are pre-encoded to f16 u16 (both the kernel and the f64 reference decode
//! the SAME bits), and the reference runs the kernel's exact algorithm in f64
//! ("truth"). Each arch's kernel error vs that truth is printed; NEON's error at
//! head_dim=512 is the deficit to fix.
//!
//!   cargo test --release --test attn_fused_parity -- --ignored --nocapture

use olorin::kernels::{ffi, ffi_inference};

const ARCH: &str = if cfg!(target_arch = "aarch64") {
    "aarch64-NEON"
} else {
    "x86_64-SSE2"
};

/// Deterministic value in [-1, 1] from an index — pure integer ops, so the
/// exact same f32 bits are produced on every architecture (no transcendentals).
fn det(i: u64) -> f32 {
    let mut s = i.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    s ^= s >> 33;
    s = s.wrapping_mul(0xff51afd7ed558ccd);
    s ^= s >> 33;
    ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

/// Exact f16 (u16) -> f32. f16 is a subset of f32, so this is lossless and
/// matches the kernel's `cvt_f16_f32`.
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // subnormal -> normalize
            let mut e = -1i32;
            let mut m = mant;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            let fe = (127 - 15 + 1 + e) as u32;
            (sign << 31) | (fe << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        let fe = exp + (127 - 15);
        (sign << 31) | (fe << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// f32 -> f16 (round toward zero is fine; both kernel and reference read the
/// SAME resulting bits, so the rounding mode never causes a kernel/ref mismatch).
fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x7fffff;
    if exp == 0xff {
        return sign | 0x7c00 | ((mant != 0) as u16 * 0x200);
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflow -> inf
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = (mant | 0x800000) >> (14 - e);
        return sign | (m as u16);
    }
    sign | ((e as u16) << 10) | ((mant >> 13) as u16)
}

/// f64 "truth" reference implementing the kernel's exact algorithm for one query
/// (qi=0) attending over `valid` keys: scaled QK dot, max-subtract softmax,
/// weighted V sum. All accumulation in f64.
fn reference(q: &[f32], k16: &[u16], v16: &[u16], head_dim: usize, valid: usize, scale: f32) -> Vec<f64> {
    let mut scores = vec![0f64; valid];
    for (j, s) in scores.iter_mut().enumerate() {
        let mut dot = 0f64;
        for d in 0..head_dim {
            dot += q[d] as f64 * f16_to_f32(k16[j * head_dim + d]) as f64;
        }
        *s = dot * scale as f64;
    }
    let m = scores.iter().cloned().fold(f64::MIN, f64::max);
    let mut sum = 0f64;
    for s in scores.iter_mut() {
        *s = (*s - m).exp();
        sum += *s;
    }
    for s in scores.iter_mut() {
        *s /= sum;
    }
    let mut out = vec![0f64; head_dim];
    for j in 0..valid {
        let w = scores[j];
        for d in 0..head_dim {
            out[d] += w * f16_to_f32(v16[j * head_dim + d]) as f64;
        }
    }
    out
}

/// Returns (max_abs_err, max_rel_err) of the kernel output vs the f64 reference.
fn run_case(head_dim: usize, n_kv: usize) -> (f64, f64) {
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..head_dim).map(|d| det(d as u64)).collect();
    let k16: Vec<u16> = (0..n_kv * head_dim)
        .map(|i| f32_to_f16(det(1_000_000 + i as u64)))
        .collect();
    let v16: Vec<u16> = (0..n_kv * head_dim)
        .map(|i| f32_to_f16(det(2_000_000 + i as u64)))
        .collect();

    let mut dst = vec![0f32; head_dim];
    let mut scores_buf = vec![0f32; n_kv + 8];
    let mut kv_scratch = vec![0f32; head_dim];

    unsafe {
        ffi_inference::attn_fused_batched(
            q.as_ptr(),
            k16.as_ptr(),
            v16.as_ptr(),
            dst.as_mut_ptr(),
            scores_buf.as_mut_ptr(),
            kv_scratch.as_mut_ptr(),
            head_dim as i32,
            head_dim as i32,        // q_stride
            head_dim as i32,        // out_stride
            head_dim as i32,        // stride_kv (one head, contiguous)
            0,                      // kv_head_offset
            n_kv as i32,
            1,                      // n_batch
            (n_kv - 1) as i32,      // cache_start -> valid = cache_start+0+1 = n_kv
            scale,
        );
    }

    let refv = reference(&q, &k16, &v16, head_dim, n_kv, scale);
    let mut max_abs = 0f64;
    let mut max_rel = 0f64;
    for d in 0..head_dim {
        let e = (dst[d] as f64 - refv[d]).abs();
        max_abs = max_abs.max(e);
        let denom = refv[d].abs().max(1e-9);
        max_rel = max_rel.max(e / denom);
    }
    (max_abs, max_rel)
}

/// Kernel error must stay under this vs the f64 truth. SSE2's worst case is
/// ~6.7e-5 (head_dim=512, n_kv=64); the NEON kernel currently hits ~2.0e-4 in
/// that same case (~3× worse) — the fix target. When the NEON `attn_fused_batched`
/// reduction/unroll is brought to SSE2 parity at head_dim=512, this passes on
/// both arches. Runs on x86 today (green baseline); RED on the Pi until fixed.
const TOL: f64 = 1.0e-4;

#[test]
#[ignore = "manual cross-arch kernel-parity probe"]
fn attn_fused_parity() {
    ffi::init().unwrap();
    println!("=== ATTN-FUSED-PARITY arch={ARCH} (kernel vs f64 truth, TOL={TOL:.0e}) ===");
    let mut violations = Vec::new();
    for &hd in &[256usize, 512] {
        for &nkv in &[1usize, 4, 64] {
            let (a, r) = run_case(hd, nkv);
            let flag = if r > TOL { " OVER-TOL" } else { "" };
            println!("head_dim={hd:>3} n_kv={nkv:>3}  max_abs={a:.3e}  max_rel={r:.3e}{flag}");
            if r > TOL {
                violations.push(format!("head_dim={hd} n_kv={nkv}: max_rel={r:.3e} > {TOL:.0e}"));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "attn_fused_batched exceeds parity tolerance on {ARCH}:\n  {}",
        violations.join("\n  ")
    );
}

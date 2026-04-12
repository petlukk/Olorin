use olorin::kernels::ffi;
use olorin::kernels::ffi_inference;

fn init() {
    ffi::init().expect("kernel init failed");
}

/// Convert f32 to f16 bits (u16) — same algorithm as cache.rs.
fn to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x007F_FFFF;

    if exp == 255 {
        return (sign | 0x7C00 | if mantissa != 0 { 0x0200 } else { 0 }) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return (sign | 0x7C00) as u16;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign as u16;
        }
        let m = mantissa | 0x0080_0000;
        let shift = 1 - new_exp;
        let half_bit = (m >> (shift + 13 - 1)) & 1;
        let result = m >> (shift + 13);
        return (sign | (result + half_bit)) as u16;
    }
    let half_bit = (mantissa >> 12) & 1;
    let result = ((new_exp as u32) << 10) | (mantissa >> 13);
    (sign | result + half_bit) as u16
}

/// Reference attention for one query token against positions 0..attn_len.
/// K and V caches are f16 (u16 bits), layout: [position * stride_kv + kv_head_offset].
unsafe fn reference_attention(
    q: *const f32,
    k_cache: *const u16,
    v_cache: *const u16,
    out: *mut f32,
    head_dim: i32,
    stride_kv: i32,
    kv_head_offset: i32,
    attn_len: i32,
    attn_scale: f32,
) {
    let hd = head_dim as usize;
    let mut scratch = vec![0.0f32; hd];
    let mut scores = vec![0.0f32; attn_len as usize];

    // Score phase: dot(q, k[p]) for each position
    for p in 0..attn_len {
        let k_ptr = k_cache.add((p * stride_kv + kv_head_offset) as usize);
        ffi_inference::f16_to_f32(k_ptr, scratch.as_mut_ptr(), head_dim);
        scores[p as usize] = ffi_inference::f32_dot(q, scratch.as_ptr(), head_dim);
    }

    // Softmax with scale
    ffi_inference::softmax_f32(scores.as_mut_ptr(), attn_len, attn_scale);

    // Zero output
    for i in 0..hd {
        *out.add(i) = 0.0;
    }

    // Weighted V sum
    for p in 0..attn_len {
        let v_ptr = v_cache.add((p * stride_kv + kv_head_offset) as usize);
        ffi_inference::f16_to_f32(v_ptr, scratch.as_mut_ptr(), head_dim);
        ffi_inference::f32_dot_acc(out, scratch.as_ptr(), scores[p as usize], head_dim);
    }
}

// ---- Test 1: N=1 bit-exact match ----

#[test]
fn test_attn_fused_batched_n1_bit_exact() {
    init();

    let head_dim: i32 = 256;
    let stride_kv: i32 = 256;
    let n_kv: i32 = 20;
    let n_batch: i32 = 1;
    let cache_start: i32 = 19;
    let attn_scale: f32 = 1.0;
    let kv_head_offset: i32 = 0;
    let hd = head_dim as usize;

    // Generate deterministic Q (f32)
    let q: Vec<f32> = (0..hd).map(|i| ((i as f32) * 0.1).sin() * 0.5).collect();

    // Generate deterministic K/V cache (f16 as u16), n_kv positions
    let cache_elems = (n_kv as usize) * (stride_kv as usize);
    let k_cache: Vec<u16> = (0..cache_elems)
        .map(|i| to_f16_bits(((i as f32) * 0.03).cos() * 0.4))
        .collect();
    let v_cache: Vec<u16> = (0..cache_elems)
        .map(|i| to_f16_bits(((i as f32) * 0.07 + 1.0).sin() * 0.3))
        .collect();

    // Reference: existing per-position loop
    // attn_len = cache_start + 1 = 20 (all n_kv positions are valid for the single token)
    let attn_len = cache_start + 1;
    let mut ref_out = vec![0.0f32; hd];
    unsafe {
        reference_attention(
            q.as_ptr(),
            k_cache.as_ptr(),
            v_cache.as_ptr(),
            ref_out.as_mut_ptr(),
            head_dim,
            stride_kv,
            kv_head_offset,
            attn_len,
            attn_scale,
        );
    }

    // Fused kernel — pad buffers for SIMD overread (8-wide vectors)
    let mut fused_out = vec![0.0f32; hd + 8];
    let mut scores_buf = vec![0.0f32; n_kv as usize + 8];
    let mut kv_scratch = vec![0.0f32; hd + 8];
    unsafe {
        ffi_inference::attn_fused_batched(
            q.as_ptr(),
            k_cache.as_ptr(),
            v_cache.as_ptr(),
            fused_out.as_mut_ptr(),
            scores_buf.as_mut_ptr(),
            kv_scratch.as_mut_ptr(),
            head_dim,
            head_dim,      // q_stride = head_dim (contiguous)
            head_dim,      // out_stride = head_dim
            stride_kv,
            kv_head_offset,
            n_kv,
            n_batch,
            cache_start,
            attn_scale,
        );
    }

    // Bit-exact comparison
    let mut mismatches = 0;
    for i in 0..hd {
        if ref_out[i].to_bits() != fused_out[i].to_bits() {
            if mismatches < 5 {
                eprintln!(
                    "N=1 mismatch at [{}]: ref={} (0x{:08x}) fused={} (0x{:08x})",
                    i, ref_out[i], ref_out[i].to_bits(), fused_out[i], fused_out[i].to_bits(),
                );
            }
            mismatches += 1;
        }
    }
    if mismatches > 0 {
        panic!("N=1 bit-exact: {mismatches}/{hd} mismatches");
    }
    eprintln!("PASS: N=1 bit-exact match ({hd} elements, {n_kv} KV positions)");
}

// ---- Test 2: N=4 batched with causal masking ----

#[test]
fn test_attn_fused_batched_n4_causal() {
    init();

    let head_dim: i32 = 256;
    let cache_start: i32 = 10;
    let n_batch: i32 = 4;
    let n_kv: i32 = 14; // 10 cached + 4 new
    let stride_kv: i32 = 256;
    let q_stride: i32 = 256;
    let out_stride: i32 = 256;
    let attn_scale: f32 = 1.0;
    let kv_head_offset: i32 = 0;
    let hd = head_dim as usize;
    let nb = n_batch as usize;

    // Generate Q: 4 contiguous query vectors
    let q: Vec<f32> = (0..hd * nb)
        .map(|i| ((i as f32) * 0.13 + 0.5).sin() * 0.6)
        .collect();

    // Generate K/V cache: 14 positions
    let cache_elems = (n_kv as usize) * (stride_kv as usize);
    let k_cache: Vec<u16> = (0..cache_elems)
        .map(|i| to_f16_bits(((i as f32) * 0.031).cos() * 0.35))
        .collect();
    let v_cache: Vec<u16> = (0..cache_elems)
        .map(|i| to_f16_bits(((i as f32) * 0.071 + 2.0).sin() * 0.25))
        .collect();

    // Reference: per-token with causal masking
    let mut ref_out = vec![0.0f32; hd * nb];
    for b in 0..nb {
        let causal_limit = cache_start + (b as i32) + 1; // 11, 12, 13, 14
        let q_ptr = unsafe { q.as_ptr().add(b * hd) };
        let out_ptr = unsafe { ref_out.as_mut_ptr().add(b * hd) };
        unsafe {
            reference_attention(
                q_ptr,
                k_cache.as_ptr(),
                v_cache.as_ptr(),
                out_ptr,
                head_dim,
                stride_kv,
                kv_head_offset,
                causal_limit,
                attn_scale,
            );
        }
    }

    // Fused kernel — pad buffers for SIMD overread
    let mut fused_out = vec![0.0f32; hd * nb + 8];
    let mut scores_buf = vec![0.0f32; n_kv as usize + 8];
    let mut kv_scratch = vec![0.0f32; hd + 8];
    unsafe {
        ffi_inference::attn_fused_batched(
            q.as_ptr(),
            k_cache.as_ptr(),
            v_cache.as_ptr(),
            fused_out.as_mut_ptr(),
            scores_buf.as_mut_ptr(),
            kv_scratch.as_mut_ptr(),
            head_dim,
            q_stride,
            out_stride,
            stride_kv,
            kv_head_offset,
            n_kv,
            n_batch,
            cache_start,
            attn_scale,
        );
    }

    // Bit-exact comparison per batch element
    let mut total_mismatches = 0;
    for b in 0..nb {
        let mut mismatches = 0;
        for i in 0..hd {
            let ri = b * hd + i;
            if ref_out[ri].to_bits() != fused_out[ri].to_bits() {
                if mismatches < 3 {
                    eprintln!(
                        "N=4 batch[{b}] mismatch at [{i}]: ref={} (0x{:08x}) fused={} (0x{:08x})",
                        ref_out[ri], ref_out[ri].to_bits(),
                        fused_out[ri], fused_out[ri].to_bits(),
                    );
                }
                mismatches += 1;
            }
        }
        if mismatches > 0 {
            eprintln!("  batch[{b}]: {mismatches}/{hd} mismatches (causal_limit={})", cache_start + (b as i32) + 1);
        }
        total_mismatches += mismatches;
    }
    if total_mismatches > 0 {
        panic!("N=4 causal: {total_mismatches}/{} total mismatches", hd * nb);
    }
    eprintln!("PASS: N=4 causal bit-exact ({nb} tokens, {n_kv} KV positions, causal limits 11..14)");
}

// ---- Test 3: Strided Q (N=2, q_stride != head_dim) ----

#[test]
fn test_attn_fused_batched_strided() {
    init();

    let head_dim: i32 = 256;
    let q_stride: i32 = 512; // simulate head 0 in wider buffer
    let out_stride: i32 = 512;
    let stride_kv: i32 = 256;
    let n_batch: i32 = 2;
    let cache_start: i32 = 5;
    let n_kv: i32 = 7; // 5 cached + 2 new
    let attn_scale: f32 = 1.0;
    let kv_head_offset: i32 = 0;
    let hd = head_dim as usize;
    let qs = q_stride as usize;
    let os = out_stride as usize;
    let nb = n_batch as usize;

    // Generate Q in strided layout: [q_stride * n_batch] f32s
    // Only first head_dim elements per stride are used
    let q_total = qs * nb;
    let q: Vec<f32> = (0..q_total)
        .map(|i| ((i as f32) * 0.17 + 3.0).sin() * 0.4)
        .collect();

    // Generate K/V cache: 7 positions
    let cache_elems = (n_kv as usize) * (stride_kv as usize);
    let k_cache: Vec<u16> = (0..cache_elems)
        .map(|i| to_f16_bits(((i as f32) * 0.041).cos() * 0.3))
        .collect();
    let v_cache: Vec<u16> = (0..cache_elems)
        .map(|i| to_f16_bits(((i as f32) * 0.083 + 5.0).sin() * 0.2))
        .collect();

    // Reference: per-token with causal masking, reading Q at stride offsets
    let out_total = os * nb;
    let mut ref_out = vec![0.0f32; out_total];
    for b in 0..nb {
        let causal_limit = cache_start + (b as i32) + 1; // 6, 7
        let q_ptr = unsafe { q.as_ptr().add(b * qs) };
        let out_ptr = unsafe { ref_out.as_mut_ptr().add(b * os) };
        unsafe {
            reference_attention(
                q_ptr,
                k_cache.as_ptr(),
                v_cache.as_ptr(),
                out_ptr,
                head_dim,
                stride_kv,
                kv_head_offset,
                causal_limit,
                attn_scale,
            );
        }
    }

    // Fused kernel — pad buffers for SIMD overread
    let mut fused_out = vec![0.0f32; out_total + 8];
    let mut scores_buf = vec![0.0f32; n_kv as usize + 8];
    let mut kv_scratch = vec![0.0f32; hd + 8];
    unsafe {
        ffi_inference::attn_fused_batched(
            q.as_ptr(),
            k_cache.as_ptr(),
            v_cache.as_ptr(),
            fused_out.as_mut_ptr(),
            scores_buf.as_mut_ptr(),
            kv_scratch.as_mut_ptr(),
            head_dim,
            q_stride,
            out_stride,
            stride_kv,
            kv_head_offset,
            n_kv,
            n_batch,
            cache_start,
            attn_scale,
        );
    }

    // Bit-exact comparison — only check first head_dim elements per stride
    let mut total_mismatches = 0;
    for b in 0..nb {
        let mut mismatches = 0;
        for i in 0..hd {
            let ri = b * os + i;
            if ref_out[ri].to_bits() != fused_out[ri].to_bits() {
                if mismatches < 3 {
                    eprintln!(
                        "Strided batch[{b}] mismatch at [{i}]: ref={} (0x{:08x}) fused={} (0x{:08x})",
                        ref_out[ri], ref_out[ri].to_bits(),
                        fused_out[ri], fused_out[ri].to_bits(),
                    );
                }
                mismatches += 1;
            }
        }
        if mismatches > 0 {
            eprintln!("  batch[{b}]: {mismatches}/{hd} mismatches (causal_limit={})", cache_start + (b as i32) + 1);
        }
        total_mismatches += mismatches;
    }
    if total_mismatches > 0 {
        panic!("Strided: {total_mismatches}/{} total mismatches", hd * nb);
    }
    eprintln!("PASS: Strided Q bit-exact (q_stride={qs}, out_stride={os}, {nb} tokens, {n_kv} KV positions)");
}

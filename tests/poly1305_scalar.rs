//! Scalar Poly1305 reference implementation (RFC 8439 §2.5).
//!
//! This module exists in `tests/` only — it is NOT shipped in the binary.
//! Its sole purpose is to serve as a byte-for-byte correct oracle that the
//! Eä SIMD kernel in subsequent tasks will be validated against.
//!
//! Algorithm: accumulator-based MAC over GF(2^130 - 5).
//! We use 5-limb radix-2^26 arithmetic; each limb holds at most 26 bits
//! after carry propagation, making overflow analysis straightforward.

/// Poly1305 prime p = 2^130 - 5.
/// In 5-limb radix-2^26: each limb is 26 bits, value = sum(limb[i] * 2^(26*i)).

/// Compute a Poly1305 MAC.
///
/// `key`     — 32-byte one-time key: bytes `[0..16]` give `r` (after clamping),
///             bytes `[16..32]` give `s`.
/// `msg`     — message of arbitrary length (including empty).
/// `tag_out` — 16-byte output buffer, filled with the little-endian tag.
pub fn poly1305_mac(key: &[u8; 32], msg: &[u8], tag_out: &mut [u8; 16]) {
    // ---- Key setup -------------------------------------------------------
    // r: clamp per RFC 8439 §2.5.1, then split into 5 × 26-bit limbs.
    let r = limbs_from_r(&key[..16]);
    // Precompute 5*r (used in reduction: 2^130 ≡ 5 mod p, so folding the
    // high limb back uses multiplication by 5).
    let r5: [u64; 5] = [r[0] * 5, r[1] * 5, r[2] * 5, r[3] * 5, r[4] * 5];

    // s: raw 16 bytes, little-endian u128 (added to acc at the end).
    let s = u128::from_le_bytes(key[16..32].try_into().unwrap());

    // ---- Accumulator (5 × 26-bit limbs, stored as u64 for overflow room) -
    let mut h: [u64; 5] = [0; 5];

    // ---- Process each 16-byte block --------------------------------------
    let mut offset = 0usize;
    while offset < msg.len() {
        let end = (offset + 16).min(msg.len());
        let chunk = &msg[offset..end];

        // Build block as 5 limbs (radix 2^26) from the chunk bytes,
        // with the implicit 1-bit placed just above the last message byte.
        let n = block_to_limbs(chunk);

        // h += n
        for i in 0..5 {
            h[i] += n[i];
        }

        // h *= r  (mod 2^130 - 5)
        h = mulmod(h, r, r5);

        offset += 16;
    }

    // ---- Final reduction: h mod p (canonical) ----------------------------
    h = propagate_carries(h);

    // Compare h to p = 2^130 - 5 to decide if we need to subtract p.
    // In 5-limb form: p[0] = 2^26-5 = 67108859, p[i] = 2^26-1 for i=1..4,
    // wait — actually p = 2^130-5 in base 2^26:
    //   p = (2^26-5), (2^26-1), (2^26-1), (2^26-1), (2^26-1)
    // No: 2^130-5 = 4*(2^128) - 5; in radix 2^26 the limbs are found by:
    //   limb0 = (2^130-5) mod 2^26 = 2^26 - 5 = 67108859  (since 2^26 mod 2^26 = 0, need (0-5) mod 2^26)
    //   actually 2^130 = (2^26)^5, so 2^130 mod 2^26 = 0, thus (2^130-5) mod 2^26 = 2^26-5.
    //   Each higher limb: ((2^130-5) >> (26*k)) mod 2^26 = 2^26-1 for k=1..4.
    // Actually let's verify:  p = 2^130-5.
    //   limb0 = p & ((1<<26)-1) = (2^130-5) mod 2^26.
    //   Since 2^130 ≡ 0 mod 2^26, limb0 = (-5) mod 2^26 = 2^26-5 = 67108859. ✓
    //   limb1..4 = each are (2^26-1) since filling bits above that. ✓
    // p_limbs: [67108859, 67108863, 67108863, 67108863, 67108863]
    // 67108863 = (1<<26)-1, 67108859 = (1<<26)-5.

    const MASK26: u64 = (1 << 26) - 1;
    // subtract p from h if h >= p:
    // We do this with a conditional borrow chain.
    let g0 = h[0].wrapping_add(5); // attempt h + 5 (which is h - p + 2^130)
    let c = g0 >> 26;
    let g0 = g0 & MASK26;
    let g1 = h[1] + c;
    let c = g1 >> 26;
    let g1 = g1 & MASK26;
    let g2 = h[2] + c;
    let c = g2 >> 26;
    let g2 = g2 & MASK26;
    let g3 = h[3] + c;
    let c = g3 >> 26;
    let g3 = g3 & MASK26;
    let g4 = h[4] + c;
    // If g4 >> 26 == 1 then the addition carried out, meaning h + 5 >= 2^130,
    // i.e., h >= p = 2^130 - 5.  Use g in that case, else h.
    let mask = !((g4 >> 26).wrapping_sub(1)); // all-ones if carry, all-zeros if not
    let g4 = g4 & MASK26;

    h[0] = (h[0] & !mask) | (g0 & mask);
    h[1] = (h[1] & !mask) | (g1 & mask);
    h[2] = (h[2] & !mask) | (g2 & mask);
    h[3] = (h[3] & !mask) | (g3 & mask);
    h[4] = (h[4] & !mask) | (g4 & mask);

    // ---- Serialize h as little-endian 128-bit value ----------------------
    // h is now in canonical [0, p) range; collapse into a u128.
    let hval: u128 = (h[0] as u128)
        | ((h[1] as u128) << 26)
        | ((h[2] as u128) << 52)
        | ((h[3] as u128) << 78)
        | ((h[4] as u128) << 104);

    // ---- Final tag: (h + s) mod 2^128 ------------------------------------
    let tag_val = hval.wrapping_add(s);
    *tag_out = tag_val.to_le_bytes();
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Clamp the `r` bytes and decompose into 5 × 26-bit limbs (stored as u64).
///
/// Clamping mask (RFC 8439 §2.5.1, little-endian byte offsets):
///   r[3,7,11,15] &= 0x0f
///   r[4,8,12]    &= 0xfc
fn limbs_from_r(r_bytes: &[u8]) -> [u64; 5] {
    let mut b = [0u8; 16];
    b.copy_from_slice(r_bytes);
    b[3] &= 0x0f;
    b[7] &= 0x0f;
    b[11] &= 0x0f;
    b[15] &= 0x0f;
    b[4] &= 0xfc;
    b[8] &= 0xfc;
    b[12] &= 0xfc;
    // Convert to u128, then extract 26-bit limbs.
    let v = u128::from_le_bytes(b);
    limbs_from_u130(v, 0)
}

/// Split a 130-bit integer (value, hi_bit) into 5 × 26-bit limbs.
/// `hi` contributes 2 bits at position 128.
fn limbs_from_u130(lo128: u128, hi: u64) -> [u64; 5] {
    const MASK26: u128 = (1 << 26) - 1;
    [
        (lo128 & MASK26) as u64,
        ((lo128 >> 26) & MASK26) as u64,
        ((lo128 >> 52) & MASK26) as u64,
        ((lo128 >> 78) & MASK26) as u64,
        (((lo128 >> 104) as u64) | (hi << 24)) & ((1 << 26) - 1),
    ]
}

/// Convert a message chunk (1–16 bytes) to 5 × 26-bit limbs with the
/// implicit 1-bit placed at position `chunk.len() * 8`.
fn block_to_limbs(chunk: &[u8]) -> [u64; 5] {
    let mut buf = [0u8; 16];
    buf[..chunk.len()].copy_from_slice(chunk);
    let lo128 = u128::from_le_bytes(buf);
    // The 1-bit at position chunk.len()*8.
    let bit_pos = chunk.len() * 8;
    if bit_pos < 128 {
        let lo128_with_bit = lo128 | (1u128 << bit_pos);
        limbs_from_u130(lo128_with_bit, 0)
    } else {
        // bit_pos == 128: hi = 1, lo = lo128.
        limbs_from_u130(lo128, 1)
    }
}

/// Propagate carries so every limb is in [0, 2^26).
fn propagate_carries(mut h: [u64; 5]) -> [u64; 5] {
    const MASK26: u64 = (1 << 26) - 1;
    for _ in 0..2 {
        // Two passes to be safe after mulmod where limbs can be large.
        let c = h[0] >> 26;
        h[0] &= MASK26;
        h[1] += c;
        let c = h[1] >> 26;
        h[1] &= MASK26;
        h[2] += c;
        let c = h[2] >> 26;
        h[2] &= MASK26;
        h[3] += c;
        let c = h[3] >> 26;
        h[3] &= MASK26;
        h[4] += c;
        let c = h[4] >> 26;
        h[4] &= MASK26;
        // Wrap: 2^130 ≡ 5 mod p, so the carry from limb4 is multiplied by 5.
        h[0] += c * 5;
    }
    h
}

/// Multiply h (5-limb) by r (5-limb) and reduce mod (2^130 - 5).
/// Uses the standard schoolbook multiply with the 2^130≡5 shortcut for
/// the top limbs.  r5[i] = r[i]*5 must be pre-computed by the caller.
fn mulmod(h: [u64; 5], r: [u64; 5], r5: [u64; 5]) -> [u64; 5] {
    // Standard schoolbook: d[k] = sum_{i+j≡k mod 5} h[i]*r[j]
    // but for j >= 5-i (i.e., i+j >= 5), the contribution wraps with a
    // factor of 5 (since 2^130 ≡ 5 mod p, each extra limb gets *5).
    //
    // This is the standard Poly1305 SIMD trick, written here without SIMD.
    //
    // Accumulate each output limb d[k] as a u64, then propagate carries.

    let mut d: [u64; 5] = [0; 5];

    // For each pair (i, j), the product h[i] * r[j] contributes to limb (i+j) mod 5,
    // with an extra factor of 5 if i+j >= 5.
    for i in 0..5 {
        for j in 0..5 {
            let sum = i + j;
            if sum < 5 {
                d[sum] += h[i] * r[j];
            } else {
                d[sum - 5] += h[i] * r5[j];
            }
        }
    }

    // Carry-propagate two rounds to normalize.
    propagate_carries(d)
}

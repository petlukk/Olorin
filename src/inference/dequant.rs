//! Dequantization routines for embedding lookups.
//!
//! Separate from matmul.rs to keep files under 500 lines.

use super::matmul::{
    GGML_TYPE_Q4_K, GGML_TYPE_Q6_K,
    Q4K_BLOCK_SIZE, Q4K_BLOCK_BYTES,
    Q6K_BLOCK_SIZE, Q6K_BLOCK_BYTES,
    f16_to_f32_scalar,
};

/// Dispatch a single-row embedding dequantization by dtype.
/// Q6K and Q4K are the dtypes used for `token_embd.weight` across the
/// supported model variants (Q4_K_M baseline = Q6K; Q4K-embed variant = Q4K).
pub fn embed_lookup(weight: *const u8, dtype: u32, token_id: usize, output: &mut [f32], hidden_dim: usize) {
    match dtype {
        GGML_TYPE_Q6_K => q6k_embed_lookup(weight, token_id, output, hidden_dim),
        GGML_TYPE_Q4_K => q4k_embed_lookup(weight, token_id, output, hidden_dim),
        other => panic!("embed_lookup: unsupported dtype {other} for token_embd"),
    }
}

// ---------------------------------------------------------------------------
// Q6K embedding dequantization
// ---------------------------------------------------------------------------

/// Dequantize a single row from a Q6K embedding table to f32.
///
/// Q6K block layout (210 bytes per 256 elements):
///   ql[128]@0  qh[64]@128  scales[16]@192  d(f16)@208
///
/// Each element is 6-bit: low 4 bits from ql, high 2 bits from qh.
/// Value = d * scale[group] * (q6 - 32).
pub fn q6k_embed_lookup(
    weight: *const u8,
    token_id: usize,
    output: &mut [f32],
    hidden_dim: usize,
) {
    debug_assert!(hidden_dim % Q6K_BLOCK_SIZE == 0);
    let n_blocks = hidden_dim / Q6K_BLOCK_SIZE;
    let row_bytes = n_blocks * Q6K_BLOCK_BYTES;
    let row_base = unsafe { weight.add(token_id * row_bytes) };

    for blk in 0..n_blocks {
        let block = unsafe { row_base.add(blk * Q6K_BLOCK_BYTES) };
        let out_base = blk * Q6K_BLOCK_SIZE;

        // Extract d (f16 at offset 208)
        let d_raw = unsafe { u16::from_le_bytes([*block.add(208), *block.add(209)]) };
        let d = f16_to_f32_scalar(d_raw);

        // Q6K layout per block (256 elements):
        //   ql[128]@0  qh[64]@128  scales[16]@192  d(f16)@208
        //
        // Processed in 2 halves of 128 elements each.
        // Each half has 4 groups of 32 elements:
        //   Group 0: ql[0..32] low nibble  + qh bits [0..1]
        //   Group 1: ql[0..32] high nibble + qh bits [2..3]
        //   Group 2: ql[32..64] low nibble + qh bits [4..5]
        //   Group 3: ql[32..64] high nibble + qh bits [6..7]
        //
        // Each group's 32 elements split into 2 subgroups of 16
        // with separate scale entries.

        for half in 0..2usize {
            let ql_ptr = unsafe { block.add(half * 64) };
            let qh_ptr = unsafe { block.add(128 + half * 32) };
            let sc_ptr = unsafe { block.add(192 + half * 8) };
            let elem_base = out_base + half * 128;

            // Group 0: ql[0..32] low nibble, qh bits 0-1
            let sc0a = d * unsafe { *sc_ptr.add(0) as i8 as f32 };
            let sc0b = d * unsafe { *sc_ptr.add(1) as i8 as f32 };
            for j in 0..16usize {
                let ql = unsafe { *ql_ptr.add(j) };
                let qh = unsafe { *qh_ptr.add(j) };
                let q6 = (ql & 0x0f) as i32 | (((qh & 0x03) as i32) << 4);
                output[elem_base + j] = sc0a * ((q6 - 32) as f32);
            }
            for j in 0..16usize {
                let ql = unsafe { *ql_ptr.add(16 + j) };
                let qh = unsafe { *qh_ptr.add(16 + j) };
                let q6 = (ql & 0x0f) as i32 | (((qh & 0x03) as i32) << 4);
                output[elem_base + 16 + j] = sc0b * ((q6 - 32) as f32);
            }

            // Group 1: ql[32..64] low nibble, qh bits 2-3
            // (llama.cpp: y[l+32] = ql[l+32] & 0xF | qh >> 2)
            let sc1a = d * unsafe { *sc_ptr.add(2) as i8 as f32 };
            let sc1b = d * unsafe { *sc_ptr.add(3) as i8 as f32 };
            for j in 0..16usize {
                let ql = unsafe { *ql_ptr.add(32 + j) };
                let qh = unsafe { *qh_ptr.add(j) };
                let q6 = (ql & 0x0f) as i32 | ((((qh >> 2) & 0x03) as i32) << 4);
                output[elem_base + 32 + j] = sc1a * ((q6 - 32) as f32);
            }
            for j in 0..16usize {
                let ql = unsafe { *ql_ptr.add(32 + 16 + j) };
                let qh = unsafe { *qh_ptr.add(16 + j) };
                let q6 = (ql & 0x0f) as i32 | ((((qh >> 2) & 0x03) as i32) << 4);
                output[elem_base + 32 + 16 + j] = sc1b * ((q6 - 32) as f32);
            }

            // Group 2: ql[0..32] high nibble, qh bits 4-5
            // (llama.cpp: y[l+64] = ql[l] >> 4 | qh >> 4)
            let sc2a = d * unsafe { *sc_ptr.add(4) as i8 as f32 };
            let sc2b = d * unsafe { *sc_ptr.add(5) as i8 as f32 };
            for j in 0..16usize {
                let ql = unsafe { *ql_ptr.add(j) };
                let qh = unsafe { *qh_ptr.add(j) };
                let q6 = ((ql >> 4) & 0x0f) as i32 | ((((qh >> 4) & 0x03) as i32) << 4);
                output[elem_base + 64 + j] = sc2a * ((q6 - 32) as f32);
            }
            for j in 0..16usize {
                let ql = unsafe { *ql_ptr.add(16 + j) };
                let qh = unsafe { *qh_ptr.add(16 + j) };
                let q6 = ((ql >> 4) & 0x0f) as i32 | ((((qh >> 4) & 0x03) as i32) << 4);
                output[elem_base + 64 + 16 + j] = sc2b * ((q6 - 32) as f32);
            }

            // Group 3: ql[32..64] high nibble, qh bits 6-7
            let sc3a = d * unsafe { *sc_ptr.add(6) as i8 as f32 };
            let sc3b = d * unsafe { *sc_ptr.add(7) as i8 as f32 };
            for j in 0..16usize {
                let ql = unsafe { *ql_ptr.add(32 + j) };
                let qh = unsafe { *qh_ptr.add(j) };
                let q6 = ((ql >> 4) & 0x0f) as i32 | ((((qh >> 6) & 0x03) as i32) << 4);
                output[elem_base + 96 + j] = sc3a * ((q6 - 32) as f32);
            }
            for j in 0..16usize {
                let ql = unsafe { *ql_ptr.add(32 + 16 + j) };
                let qh = unsafe { *qh_ptr.add(16 + j) };
                let q6 = ((ql >> 4) & 0x0f) as i32 | ((((qh >> 6) & 0x03) as i32) << 4);
                output[elem_base + 96 + 16 + j] = sc3b * ((q6 - 32) as f32);
            }
        }
    }
}

/// Dequantize a single row from a Q6K table with configurable row width.
///
/// Same algorithm as q6k_embed_lookup but named for clarity when used with
/// PLE token embeddings where row_dim = ple_dim * n_layers (not hidden_dim).
#[inline]
pub fn q6k_dequant_row(
    weight: *const u8,
    row_id: usize,
    output: &mut [f32],
    row_dim: usize,
) {
    q6k_embed_lookup(weight, row_id, output, row_dim)
}

// ---------------------------------------------------------------------------
// Q4K embedding dequantization
// ---------------------------------------------------------------------------

/// Dequantize a single row from a Q4K embedding table to f32.
///
/// Q4K block layout (144 bytes per 256 elements):
///   d(f16)@0  dmin(f16)@2  scales[12]@4  qs[128]@16
///
/// Each element is 4-bit. The 256-element block is split into 8 sub-blocks
/// of 32 elements; each sub-block has its own 6-bit scale and 6-bit min,
/// packed across `scales[12]`. Element value = d*sc[j] * q4 - dmin*m[j].
pub fn q4k_embed_lookup(
    weight: *const u8,
    token_id: usize,
    output: &mut [f32],
    hidden_dim: usize,
) {
    debug_assert!(hidden_dim % Q4K_BLOCK_SIZE == 0);
    let n_blocks = hidden_dim / Q4K_BLOCK_SIZE;
    let row_bytes = n_blocks * Q4K_BLOCK_BYTES;
    let row_base = unsafe { weight.add(token_id * row_bytes) };

    for blk in 0..n_blocks {
        let block = unsafe { row_base.add(blk * Q4K_BLOCK_BYTES) };
        let out_base = blk * Q4K_BLOCK_SIZE;

        let d_raw = unsafe { u16::from_le_bytes([*block.add(0), *block.add(1)]) };
        let dmin_raw = unsafe { u16::from_le_bytes([*block.add(2), *block.add(3)]) };
        let d = f16_to_f32_scalar(d_raw);
        let dmin = f16_to_f32_scalar(dmin_raw);

        // Read sub-block scale/min pairs once and dequantize 32 elements at a time.
        // Pairs (0,1), (2,3), (4,5), (6,7) share 32 bytes of qs each.
        for pair in 0..4usize {
            let j_lo = pair * 2;
            let j_hi = pair * 2 + 1;
            let (sc_lo, m_lo) = q4k_get_scale_min(j_lo, block);
            let (sc_hi, m_hi) = q4k_get_scale_min(j_hi, block);
            let d1 = d * sc_lo as f32;
            let d2 = d * sc_hi as f32;
            let mn1 = dmin * m_lo as f32;
            let mn2 = dmin * m_hi as f32;

            let qs = unsafe { block.add(16 + pair * 32) };
            let elem = out_base + pair * 64;
            for l in 0..32usize {
                let q = unsafe { *qs.add(l) };
                output[elem + l]      = d1 * (q & 0x0F) as f32 - mn1;
                output[elem + 32 + l] = d2 * (q >> 4)   as f32 - mn2;
            }
        }
    }
}

/// Read sub-block 6-bit scale and min from `scales[12]` packed at block+4.
/// Mirrors llama.cpp `get_scale_min_k4`.
#[inline]
fn q4k_get_scale_min(j: usize, block: *const u8) -> (u8, u8) {
    let s = |i: usize| unsafe { *block.add(4 + i) };
    if j < 4 {
        (s(j) & 63, s(j + 4) & 63)
    } else {
        let d = (s(j + 4) & 0x0F) | ((s(j - 4) >> 6) << 4);
        let m = (s(j + 4) >> 4)   | ((s(j) >> 6) << 4);
        (d, m)
    }
}

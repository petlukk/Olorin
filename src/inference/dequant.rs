//! Dequantization routines for embedding lookups.
//!
//! Separate from matmul.rs to keep files under 500 lines.

use super::matmul::{Q6K_BLOCK_SIZE, Q6K_BLOCK_BYTES, f16_to_f32_scalar};

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

            // Group 1: ql[0..32] high nibble, qh bits 2-3
            let sc1a = d * unsafe { *sc_ptr.add(2) as i8 as f32 };
            let sc1b = d * unsafe { *sc_ptr.add(3) as i8 as f32 };
            for j in 0..16usize {
                let ql = unsafe { *ql_ptr.add(j) };
                let qh = unsafe { *qh_ptr.add(j) };
                let q6 = ((ql >> 4) & 0x0f) as i32 | ((((qh >> 2) & 0x03) as i32) << 4);
                output[elem_base + 32 + j] = sc1a * ((q6 - 32) as f32);
            }
            for j in 0..16usize {
                let ql = unsafe { *ql_ptr.add(16 + j) };
                let qh = unsafe { *qh_ptr.add(16 + j) };
                let q6 = ((ql >> 4) & 0x0f) as i32 | ((((qh >> 2) & 0x03) as i32) << 4);
                output[elem_base + 32 + 16 + j] = sc1b * ((q6 - 32) as f32);
            }

            // Group 2: ql[32..64] low nibble, qh bits 4-5
            let sc2a = d * unsafe { *sc_ptr.add(4) as i8 as f32 };
            let sc2b = d * unsafe { *sc_ptr.add(5) as i8 as f32 };
            for j in 0..16usize {
                let ql = unsafe { *ql_ptr.add(32 + j) };
                let qh = unsafe { *qh_ptr.add(j) };
                let q6 = (ql & 0x0f) as i32 | ((((qh >> 4) & 0x03) as i32) << 4);
                output[elem_base + 64 + j] = sc2a * ((q6 - 32) as f32);
            }
            for j in 0..16usize {
                let ql = unsafe { *ql_ptr.add(32 + 16 + j) };
                let qh = unsafe { *qh_ptr.add(16 + j) };
                let q6 = (ql & 0x0f) as i32 | ((((qh >> 4) & 0x03) as i32) << 4);
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

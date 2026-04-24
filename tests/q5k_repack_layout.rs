//! Round-trip layout correctness test for q5k_repack_8x8.
//!
//! Takes layer-0 wk (Q5K, 256 rows × 1536 cols in Gemma 4 E2B), calls
//! q5k_repack_8x8, then reads the repacked output at the documented
//! offsets and asserts bit-for-bit equality with the source data.
//!
//! Covers: d[8] @ 0, dmin[8] @ 16, qh[256] @ 128, qs[1024] @ 384.
//! Scales @ 32 are NOT covered here — they involve bit-repacking that's
//! easier to verify via the eventual q5k_dot_8x8 parity test.

use std::path::Path;

fn model_path() -> String {
    let home = std::env::var("HOME").unwrap();
    format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf")
}

#[test]
fn q5k_repack_8x8_layout_matches_spec() {
    if !Path::new(&model_path()).exists() {
        eprintln!("SKIP: no model at {}", model_path());
        return;
    }
    let gguf = olorin::inference::gguf::GgufFile::open(Path::new(&model_path())).unwrap();
    let model = olorin::inference::engine::Gemma4Model::from_gguf(&gguf).unwrap();
    olorin::kernels::ffi::init().unwrap();

    let lw = &model.layers[0];
    assert_eq!(
        lw.wk_dtype,
        olorin::inference::matmul::GGML_TYPE_Q5_K,
        "test assumes layer 0 wk is Q5K"
    );

    let head_dim = model.head_dim_k[0];
    let nc = model.n_kv_heads * head_dim; // 1 × 256 = 256 rows
    let n = model.hidden_dim;             // 1536 inner dim
    let nb = n / 256;                     // 6 blocks per row
    assert!(nc % 8 == 0, "n_rows must be multiple of 8 for this test");

    let src_row_bytes = nb * 176;
    let src_total = nc * src_row_bytes;
    let tile_bytes = 1408usize;
    let dst_total = (nc / 8) * nb * tile_bytes;
    assert_eq!(
        src_total, dst_total,
        "repack is a permutation — sizes must match"
    );

    let src: *const u8 = lw.wk;
    let mut dst = vec![0u8; dst_total];

    unsafe {
        olorin::kernels::ffi_inference::q5k_repack_8x8(
            src,
            dst.as_mut_ptr(),
            nc as i32,
            n as i32,
        );
    }

    let src_slice: &[u8] = unsafe { std::slice::from_raw_parts(src, src_total) };

    // Verify every 8-row tile, every block within.
    for tile in 0..(nc / 8) {
        for blk in 0..nb {
            let d_offset = (tile * nb + blk) * tile_bytes;
            let row_base = tile * 8 * src_row_bytes;

            // d[8] at dst+d+0 — 8 × 2 bytes, one per row.
            // Source: d at offset 0 of each block (row_base + r*row + blk*176).
            for r in 0..8 {
                let src_d_off = row_base + r * src_row_bytes + blk * 176 + 0;
                let dst_d_off = d_offset + r * 2;
                assert_eq!(
                    dst[dst_d_off], src_slice[src_d_off],
                    "d[0] byte tile={tile} blk={blk} row={r}"
                );
                assert_eq!(
                    dst[dst_d_off + 1], src_slice[src_d_off + 1],
                    "d[1] byte tile={tile} blk={blk} row={r}"
                );
            }

            // dmin[8] at dst+d+16 — src has dmin at offset 2.
            for r in 0..8 {
                let src_dmin_off = row_base + r * src_row_bytes + blk * 176 + 2;
                let dst_dmin_off = d_offset + 16 + r * 2;
                assert_eq!(
                    dst[dst_dmin_off], src_slice[src_dmin_off],
                    "dmin[0] byte tile={tile} blk={blk} row={r}"
                );
                assert_eq!(
                    dst[dst_dmin_off + 1], src_slice[src_dmin_off + 1],
                    "dmin[1] byte tile={tile} blk={blk} row={r}"
                );
            }

            // qh[256] at dst+d+128, row-sequential (32 bytes per row).
            // Source: qh at offset 16 of each block.
            for r in 0..8 {
                for b in 0..32 {
                    let src_qh_off = row_base + r * src_row_bytes + blk * 176 + 16 + b;
                    let dst_qh_off = d_offset + 128 + r * 32 + b;
                    assert_eq!(
                        dst[dst_qh_off], src_slice[src_qh_off],
                        "qh tile={tile} blk={blk} row={r} byte={b}"
                    );
                }
            }

            // qs[1024] at dst+d+384, interleaved in CHUNK-sized groups per row.
            // On ARM: 4-byte chunks. On x86: 8-byte chunks. Either way, every
            // source byte should appear SOMEWHERE in dst — verify via per-row
            // chunk reconstruction:
            // - For each row r, the row's 128 source qs bytes get split into
            //   128 / CHUNK groups of CHUNK bytes; group g of row r lands at
            //   dst offset d + 384 + g * (CHUNK * 8) + r * CHUNK.
            let chunk = if cfg!(target_arch = "aarch64") { 4usize } else { 8usize };
            let groups = 128 / chunk;
            for r in 0..8 {
                for g in 0..groups {
                    for b in 0..chunk {
                        let src_qs_off =
                            row_base + r * src_row_bytes + blk * 176 + 48 + g * chunk + b;
                        let dst_qs_off = d_offset + 384 + g * (chunk * 8) + r * chunk + b;
                        assert_eq!(
                            dst[dst_qs_off], src_slice[src_qs_off],
                            "qs tile={tile} blk={blk} row={r} group={g} byte={b}"
                        );
                    }
                }
            }
        }
    }

    eprintln!(
        "q5k_repack layout verified: {} tiles × {} blocks, chunk={}",
        nc / 8,
        nb,
        if cfg!(target_arch = "aarch64") { 4 } else { 8 }
    );
}

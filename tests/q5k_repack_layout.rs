//! Byte-level layout parity for `q5k_repack_4row`.
//!
//! The repack interleaves 4 consecutive rows of Q5K weights into tile layout
//! so each superblock's 4 row-slices sit contiguous in memory (cache win for
//! the repacked 4-row dot kernel). This test proves the repacked buffer
//! contains exactly the original source bytes, just permuted — i.e. the
//! repack is lossless and the layout math matches what a repacked kernel
//! would expect.
//!
//! Checks on both a synthetic Q5K buffer (deterministic values so offset
//! errors show up as off-by-one) and, when present, on a real Gemma 4 weight
//! tensor loaded from the shipped gguf.

use std::path::Path;

/// Q5K superblock: 176 bytes / 256 elements.
const Q5K_BLOCK_BYTES: usize = 176;

fn synth_buf(n_rows: usize, n_blocks: usize) -> Vec<u8> {
    // Deterministic byte per (row, block, offset): any off-by-one in the
    // repack indexing produces a bit-exact mismatch at a specific byte.
    let mut buf = vec![0u8; n_rows * n_blocks * Q5K_BLOCK_BYTES];
    for row in 0..n_rows {
        for blk in 0..n_blocks {
            for off in 0..Q5K_BLOCK_BYTES {
                let idx = (row * n_blocks + blk) * Q5K_BLOCK_BYTES + off;
                buf[idx] = ((row.wrapping_mul(37)
                    ^ blk.wrapping_mul(131)
                    ^ off.wrapping_mul(7)) & 0xff) as u8;
            }
        }
    }
    buf
}

fn assert_repack_layout(src: &[u8], n_rows: usize, n_cols: usize) {
    let n_blocks = n_cols / 256;
    let row_bytes = n_blocks * Q5K_BLOCK_BYTES;
    let tile_bytes = 4 * Q5K_BLOCK_BYTES;
    let n_quads = n_rows / 4;

    let packed = olorin::inference::repack::q5k_repack_4row(src.as_ptr(), n_rows, n_cols);
    assert_eq!(packed.len(), n_quads * n_blocks * tile_bytes, "packed size");

    // For every (quad, block, row-in-quad), the tile slot must contain the
    // original row's block bytes.
    for quad in 0..n_quads {
        for blk in 0..n_blocks {
            for r in 0..4usize {
                let src_off = (quad * 4 + r) * row_bytes + blk * Q5K_BLOCK_BYTES;
                let dst_off = (quad * n_blocks + blk) * tile_bytes + r * Q5K_BLOCK_BYTES;
                let src_slice = &src[src_off..src_off + Q5K_BLOCK_BYTES];
                let dst_slice = &packed[dst_off..dst_off + Q5K_BLOCK_BYTES];
                assert_eq!(
                    src_slice, dst_slice,
                    "mismatch at quad={quad} blk={blk} r={r}"
                );
            }
        }
    }
}

#[test]
fn synthetic_small_shape() {
    // Smallest valid shape exercising multiple quads and multiple blocks.
    let n_rows = 8;
    let n_cols = 512; // 2 blocks
    let n_blocks = n_cols / 256;
    let buf = synth_buf(n_rows, n_blocks);
    assert_repack_layout(&buf, n_rows, n_cols);
}

#[test]
fn synthetic_gemma4_ffn_down_shape() {
    // Actual ffn_down shape in Gemma 4 E2B: hidden_dim=1536, ffn_dim=12288.
    // For ffn_down: rows=hidden_dim=1536, cols=ffn_dim=12288. n_cols must be
    // multiple of 256 (12288 is, and 12288/256=48 blocks).
    let n_rows = 1536;
    let n_cols = 12288;
    let n_blocks = n_cols / 256;
    let buf = synth_buf(n_rows, n_blocks);
    assert_repack_layout(&buf, n_rows, n_cols);
}

#[test]
fn real_weight_when_model_present() {
    // Use actual weight bytes from the shipped gguf if available. Only asserts
    // for a Q5K tensor — if Gemma 4 E2B in this checkout has no Q5K layers,
    // the test skips with a clear note.
    let home = match std::env::var("HOME") { Ok(h) => h, Err(_) => return };
    let path = format!("{home}/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf");
    if !Path::new(&path).exists() {
        eprintln!("SKIP: model not present");
        return;
    }

    let gguf = match olorin::inference::gguf::GgufFile::open(Path::new(&path)) {
        Ok(g) => g,
        Err(_) => { eprintln!("SKIP: gguf open failed"); return; }
    };

    // Enumerate ALL Q5K tensors so we understand which weights actually use
    // Q5K in this checkout — the perf signal from `q5k_dot_q8k_4row` (8% of
    // decode cycles) is cumulative across however many tensors fall into this
    // bucket. Q5_K == 13.
    let q5k = 13u32;
    let mut q5k_names: Vec<(String, usize, usize)> = Vec::new();
    for (name, &idx) in gguf.tensor_map.iter() {
        let t = &gguf.tensors[idx];
        if t.dtype != q5k { continue; }
        if t.dims.len() != 2 { continue; }
        let n_cols = t.dims[0] as usize;
        let n_rows = t.dims[1] as usize;
        q5k_names.push((name.clone(), n_cols, n_rows));
    }
    q5k_names.sort();
    eprintln!("Q5K tensors in {path}: {}", q5k_names.len());
    for (n, c, r) in &q5k_names {
        eprintln!("  {n}: dims=[{c}, {r}]");
    }
    if q5k_names.is_empty() {
        eprintln!("SKIP: no Q5K tensor in this gguf");
        return;
    }
    // Validate repack layout on the first eligible tensor (all share shape).
    let mut validated = false;
    for (name, n_cols, n_rows) in &q5k_names {
        if *n_rows % 4 != 0 || *n_cols % 256 != 0 { continue; }
        let data = match gguf.tensor_data(name) {
            Some(d) => d,
            None => continue,
        };
        let n_blocks = *n_cols / 256;
        let byte_len = *n_rows * n_blocks * Q5K_BLOCK_BYTES;
        if data.len() < byte_len { continue; }
        eprintln!("validating repack layout on {name}");
        assert_repack_layout(&data[..byte_len], *n_rows, *n_cols);
        validated = true;
        break;
    }
    assert!(validated, "no Q5K tensor had valid repack shape");
}

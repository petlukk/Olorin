//! Binary IO for `.eakv` files — byte-compatible with csrc/io.c.
//!
//! Format:
//! - 512-byte header (packed, EAKV magic, version=1, dimensions)
//! - Index table: n_layers * 2 * u64 offsets (K offset, V offset)
//! - Data at align64(512 + index_table_size)
//! - Per KV slot: weights (n_groups * 32) + scales (n_groups * f32) + biases (n_groups * f32),
//!   padded to 64-byte alignment

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::cache::EakvCache;
use crate::KernelTable;

const MAGIC: [u8; 4] = *b"EAKV";
const HEADER_SIZE: usize = 512;

const fn align64(x: usize) -> usize {
    (x + 63) & !63
}

/// Packed header — must match C `eakv_header_t` exactly.
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct EakvHeader {
    magic: [u8; 4],
    version: u16,
    quant_scheme: u16,
    group_size: u32,
    orig_dtype: u16,
    n_layers: u32,
    n_heads: u32,
    head_dim: u32,
    seq_len: u32,
    max_seq_len: u32,
    compression: i16,
    model_hash: [u8; 32],
    tokenizer_hash: [u8; 32],
    checksum: u64,
}

/// Save an `EakvCache` to a `.eakv` file.
pub fn save(cache: &EakvCache, path: &Path) -> Result<(), String> {
    let n_layers = cache.n_layers() as usize;
    let n_heads = cache.n_heads() as usize;
    let head_dim = cache.head_dim() as usize;
    let seq_len = cache.seq_len() as usize;

    let n_groups = (n_heads * head_dim * seq_len) / 64;
    let weights_size = n_groups * 32;
    let scales_size = n_groups * 4; // f32
    let biases_size = n_groups * 4;
    let block_raw = weights_size + scales_size + biases_size;
    let block_aligned = align64(block_raw);

    // Build header
    let mut header_buf = [0u8; HEADER_SIZE];
    let h = EakvHeader {
        magic: MAGIC,
        version: 1,
        quant_scheme: 0,
        group_size: 64,
        orig_dtype: 0,
        n_layers: n_layers as u32,
        n_heads: n_heads as u32,
        head_dim: head_dim as u32,
        seq_len: seq_len as u32,
        max_seq_len: seq_len as u32,
        compression: 0,
        model_hash: [0u8; 32],
        tokenizer_hash: [0u8; 32],
        checksum: 0,
    };
    // SAFETY: EakvHeader is repr(C, packed) and fits in HEADER_SIZE.
    unsafe {
        let src = &h as *const EakvHeader as *const u8;
        std::ptr::copy_nonoverlapping(src, header_buf.as_mut_ptr(), std::mem::size_of::<EakvHeader>());
    }

    let mut f = File::create(path).map_err(|e| format!("save: create {}: {e}", path.display()))?;

    f.write_all(&header_buf).map_err(|e| format!("save: write header: {e}"))?;

    // Index table
    let idx_table_size = n_layers * 2 * 8;
    let data_start = align64(HEADER_SIZE + idx_table_size);

    let mut cur = data_start;
    for _l in 0..n_layers {
        let k_off = cur as u64;
        cur += block_aligned;
        let v_off = cur as u64;
        cur += block_aligned;
        f.write_all(&k_off.to_le_bytes()).map_err(|e| format!("save: write idx: {e}"))?;
        f.write_all(&v_off.to_le_bytes()).map_err(|e| format!("save: write idx: {e}"))?;
    }

    // Padding between index table and data
    let pos = HEADER_SIZE + idx_table_size;
    if pos < data_start {
        let zeros = vec![0u8; data_start - pos];
        f.write_all(&zeros).map_err(|e| format!("save: write pad: {e}"))?;
    }

    // Per-slot data
    let gpd = head_dim / 64; // groups per dim (per token per head)
    let max_seq = cache.max_seq_len() as usize;
    let gph = max_seq * gpd; // groups per head in buffer (stride)
    let groups_per_head_file = seq_len * gpd; // groups per head in file
    let pad_size = block_aligned - block_raw;
    let zeros = vec![0u8; if pad_size > 0 { pad_size } else { 1 }];

    for l in 0..n_layers as i32 {
        for kv in 0..2i32 {
            let w = cache.weights(l, kv);
            let s = cache.scales(l, kv);
            let b = cache.biases(l, kv);

            // Write weights: gather groups_per_head_file * 32 bytes from each head
            for h_idx in 0..n_heads {
                let base = h_idx * gph * 32;
                let len = groups_per_head_file * 32;
                f.write_all(&w[base..base + len])
                    .map_err(|e| format!("save: write weights: {e}"))?;
            }

            // Write scales: gather groups_per_head_file f32s from each head
            for h_idx in 0..n_heads {
                let base = h_idx * gph;
                let len = groups_per_head_file;
                // Convert f32 slice to bytes
                let byte_slice = unsafe {
                    std::slice::from_raw_parts(
                        s[base..base + len].as_ptr() as *const u8,
                        len * 4,
                    )
                };
                f.write_all(byte_slice)
                    .map_err(|e| format!("save: write scales: {e}"))?;
            }

            // Write biases
            for h_idx in 0..n_heads {
                let base = h_idx * gph;
                let len = groups_per_head_file;
                let byte_slice = unsafe {
                    std::slice::from_raw_parts(
                        b[base..base + len].as_ptr() as *const u8,
                        len * 4,
                    )
                };
                f.write_all(byte_slice)
                    .map_err(|e| format!("save: write biases: {e}"))?;
            }

            // Padding
            if pad_size > 0 {
                f.write_all(&zeros[..pad_size])
                    .map_err(|e| format!("save: write pad: {e}"))?;
            }
        }
    }

    Ok(())
}

/// Load an `EakvCache` from a `.eakv` file.
pub fn load(path: &Path, kernels: KernelTable) -> Result<EakvCache, String> {
    let mut f = File::open(path).map_err(|e| format!("load: open {}: {e}", path.display()))?;

    // Read header
    let mut header_buf = [0u8; HEADER_SIZE];
    f.read_exact(&mut header_buf).map_err(|e| format!("load: read header: {e}"))?;

    let h: EakvHeader = unsafe { std::ptr::read_unaligned(header_buf.as_ptr() as *const EakvHeader) };

    if h.magic != MAGIC {
        return Err("load: bad magic".into());
    }
    let version = h.version;
    if version != 1 {
        return Err(format!("load: unsupported version {version}"));
    }

    let n_layers = h.n_layers as i32;
    let n_heads = h.n_heads as i32;
    let head_dim = h.head_dim as i32;
    let seq_len = h.seq_len as i32;

    // Create cache with max_seq_len = seq_len (matches C)
    let mut cache = EakvCache::new(n_layers, n_heads, head_dim, seq_len, kernels)
        .ok_or_else(|| format!("load: failed to create cache ({n_layers}L, {n_heads}H, {head_dim}D, seq={seq_len})"))?;

    let n_groups = (n_heads as usize * head_dim as usize * seq_len as usize) / 64;
    let weights_size = n_groups * 32;

    // Read index table
    let idx_table_size = n_layers as usize * 2 * 8;
    let mut idx_buf = vec![0u8; idx_table_size];
    f.read_exact(&mut idx_buf).map_err(|e| format!("load: read idx: {e}"))?;

    let mut offsets = Vec::with_capacity(n_layers as usize * 2);
    for i in 0..(n_layers as usize * 2) {
        let off = u64::from_le_bytes(idx_buf[i * 8..(i + 1) * 8].try_into().unwrap());
        offsets.push(off);
    }

    // Since max_seq_len == seq_len, max_groups == n_groups, so we can write directly.
    for l in 0..n_layers {
        for kv in 0..2i32 {
            let off = offsets[l as usize * 2 + kv as usize];
            f.seek(SeekFrom::Start(off)).map_err(|e| format!("load: seek: {e}"))?;

            // Read weights directly into cache buffer
            let w = cache.weights_mut(l, kv);
            f.read_exact(&mut w[..weights_size])
                .map_err(|e| format!("load: read weights L{l} kv{kv}: {e}"))?;

            // Read scales
            let s = cache.scales_mut(l, kv);
            let scales_bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    s.as_mut_ptr() as *mut u8,
                    n_groups * 4,
                )
            };
            f.read_exact(scales_bytes)
                .map_err(|e| format!("load: read scales L{l} kv{kv}: {e}"))?;

            // Read biases
            let b = cache.biases_mut(l, kv);
            let biases_bytes = unsafe {
                std::slice::from_raw_parts_mut(
                    b.as_mut_ptr() as *mut u8,
                    n_groups * 4,
                )
            };
            f.read_exact(biases_bytes)
                .map_err(|e| format!("load: read biases L{l} kv{kv}: {e}"))?;
        }
    }

    cache.set_seq_len(seq_len);
    Ok(cache)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels;

    fn load_kernels() -> Option<KernelTable> {
        let dir = kernels::find_kernel_dir().ok()?;
        KernelTable::load(&dir).ok()
    }

    #[test]
    fn test_save_load_roundtrip() {
        let kt = match load_kernels() {
            Some(k) => k,
            None => {
                eprintln!("skipping — kernels not available");
                return;
            }
        };

        let n_layers = 2i32;
        let n_heads = 2i32;
        let head_dim = 64i32;
        let seq = 4i32;

        let mut cache = EakvCache::new(n_layers, n_heads, head_dim, seq, kt).unwrap();

        // Fill with patterned data via load_raw
        let elems = n_layers as usize * 2 * n_heads as usize * seq as usize * head_dim as usize;
        let mut data = vec![0.0f32; elems];
        for (i, v) in data.iter_mut().enumerate() {
            *v = (i as f32) * 0.01;
        }
        cache.load_raw(&data, seq).unwrap();
        assert_eq!(cache.seq_len(), seq);

        // Save
        let tmp = std::env::temp_dir().join("eakv_io_test.eakv");
        save(&cache, &tmp).expect("save failed");

        // Load — need fresh kernels since KernelTable doesn't implement Clone
        let kt2 = load_kernels().unwrap();
        let loaded = load(&tmp, kt2).expect("load failed");

        // Verify dimensions and seq_len
        assert_eq!(loaded.n_layers(), n_layers);
        assert_eq!(loaded.n_heads(), n_heads);
        assert_eq!(loaded.head_dim(), head_dim);
        assert_eq!(loaded.seq_len(), seq);
        assert_eq!(loaded.max_seq_len(), seq);

        // Verify data matches byte-for-byte
        let n_groups = (n_heads as usize * head_dim as usize * seq as usize) / 64;
        for l in 0..n_layers {
            for kv in 0..2i32 {
                let orig_w = cache.weights(l, kv);
                let load_w = loaded.weights(l, kv);
                assert_eq!(
                    &orig_w[..n_groups * 32],
                    &load_w[..n_groups * 32],
                    "weights mismatch L{l} kv{kv}"
                );

                let orig_s = cache.scales(l, kv);
                let load_s = loaded.scales(l, kv);
                assert_eq!(
                    &orig_s[..n_groups],
                    &load_s[..n_groups],
                    "scales mismatch L{l} kv{kv}"
                );

                let orig_b = cache.biases(l, kv);
                let load_b = loaded.biases(l, kv);
                assert_eq!(
                    &orig_b[..n_groups],
                    &load_b[..n_groups],
                    "biases mismatch L{l} kv{kv}"
                );
            }
        }

        // Cleanup
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_load_bad_magic() {
        let tmp = std::env::temp_dir().join("eakv_bad_magic.eakv");
        let mut buf = vec![0u8; HEADER_SIZE];
        buf[0..4].copy_from_slice(b"NOPE");
        std::fs::write(&tmp, &buf).unwrap();

        let kt = match load_kernels() {
            Some(k) => k,
            None => {
                eprintln!("skipping — kernels not available");
                return;
            }
        };
        let res = load(&tmp, kt);
        match res {
            Err(e) => assert!(e.contains("bad magic"), "unexpected error: {e}"),
            Ok(_) => panic!("expected error for bad magic"),
        }

        let _ = std::fs::remove_file(&tmp);
    }
}

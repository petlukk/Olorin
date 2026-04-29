//! Offline tool: requantize all Q4K transformer-body weights to Q3K.
//!
//! Run: `cargo test --release --test requant_q4k_to_q3k -- --ignored --nocapture`
//!
//! Reads  `~/.olorin/models/gemma-4-e2b-it-Q4_K_M-q4kembed.gguf` (or set
//!        OLORIN_SOURCE_GGUF to override).
//! Writes `~/.olorin/models/gemma-4-e2b-it-Q4_K_M-q3kbucket.gguf`.
//!
//! Requants every Q4K tensor EXCEPT `token_embd.weight` (kept Q4K so the
//! existing embed_lookup keeps working — Q3K embed_lookup is not wired and
//! token_embd has negligible decode bandwidth anyway). Other dtypes (Q5K,
//! Q6K, F32, BF16) copy through byte-for-byte.
//!
//! Quantization is a faithful port of llama.cpp `quantize_row_q3_K_ref` +
//! `make_q3_quants` from `ggml/src/ggml-quants.c` (do_rmse=true; no imatrix).

use olorin::inference::dequant::q4k_embed_lookup;
use olorin::inference::gguf::GgufFile;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const ALIGNMENT: u64 = 32;
const QK_K: usize = 256;
const Q3K_BLOCK_BYTES: usize = 110;
const Q4K_BLOCK_BYTES: usize = 144;
const GGML_TYPE_Q3_K: u32 = 11;
const GGML_TYPE_Q4_K: u32 = 12;

/// Tensors whose name == this string are kept at their source dtype.
/// `token_embd.weight` is excluded because embed_lookup only handles Q6K/Q4K.
const KEEP_AS_IS: &[&str] = &["token_embd.weight"];

/// Selective requant: only Q4K tensors whose name ends with one of these
/// suffixes get converted to Q3K. Empty list = requant ALL Q4K tensors.
///
/// First-cut "all Q4K → Q3K" was too aggressive (gemma4_smoke produced empty
/// output, 27.9% logits drift). Per the port plan's pitfall section, Q3K
/// accuracy is meaningfully worse than Q4K and likely needs selective
/// application. Default: the two biggest per-layer tensors (ffn_up,
/// ffn_gate at ~10.6 MB each × 35 layers = ~742 MB of the Q4K bucket).
///
/// Override with `OLORIN_REQUANT_SUFFIXES=ffn_up.weight` (comma-separated)
/// for selective-revert experiments — e.g. drop one arm of the GeGLU back
/// to Q4K to recover prefill while keeping the decode bandwidth saving on
/// the other arm.
const REQUANT_SUFFIXES_DEFAULT: &[&str] = &["ffn_up.weight", "ffn_gate.weight"];

#[test]
#[ignore]
fn requant_q4k_to_q3k() {
    let home = std::env::var("HOME").expect("HOME not set");
    let src = std::env::var("OLORIN_SOURCE_GGUF")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| Path::new(&home)
            .join(".olorin/models/gemma-4-e2b-it-Q4_K_M-q4kembed.gguf"));
    let dst = std::env::var("OLORIN_DEST_GGUF")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| Path::new(&home)
            .join(".olorin/models/gemma-4-e2b-it-Q4_K_M-q3kbucket.gguf"));
    assert!(src.exists(), "source not found: {}", src.display());

    let suffix_env = std::env::var("OLORIN_REQUANT_SUFFIXES").ok();
    let suffixes: Vec<&str> = match suffix_env.as_deref() {
        Some(s) => s.split(',').map(str::trim).filter(|s| !s.is_empty()).collect(),
        None => REQUANT_SUFFIXES_DEFAULT.to_vec(),
    };
    eprintln!("source:   {}", src.display());
    eprintln!("dest:     {}", dst.display());
    eprintln!("suffixes: {:?}", suffixes);

    let gguf = GgufFile::open(&src).expect("open source gguf");

    // Identify every Q4K tensor that's NOT in the keep-as-is list.
    let mut to_requant: Vec<usize> = Vec::new();
    let mut bytes_before: u64 = 0;
    let mut bytes_after: u64 = 0;
    for i in 0..gguf.tensors.len() {
        let t = &gguf.tensors[i];
        if t.dtype != GGML_TYPE_Q4_K { continue; }
        if KEEP_AS_IS.contains(&gguf.tensor_names[i].as_str()) { continue; }
        if !suffixes.is_empty() {
            let name = gguf.tensor_names[i].as_str();
            if !suffixes.iter().any(|s| name.ends_with(s)) { continue; }
        }
        let n_elements: u64 = t.dims.iter().product();
        let n_blocks = (n_elements as usize + QK_K - 1) / QK_K;
        bytes_before += (n_blocks * Q4K_BLOCK_BYTES) as u64;
        bytes_after  += (n_blocks * Q3K_BLOCK_BYTES) as u64;
        to_requant.push(i);
    }
    eprintln!("requant target: {} Q4K tensors", to_requant.len());
    eprintln!("Q4K bytes:  {} ({:.1} MB)", bytes_before, bytes_before as f64 / (1024.0 * 1024.0));
    eprintln!("Q3K bytes:  {} ({:.1} MB)", bytes_after,  bytes_after  as f64 / (1024.0 * 1024.0));
    eprintln!("savings:    {} ({:.1} MB)", bytes_before - bytes_after,
              (bytes_before - bytes_after) as f64 / (1024.0 * 1024.0));

    // For each tensor: dequant row-by-row to f32, requant row-by-row to Q3K bytes.
    // We store all the requant outputs in a HashMap<tensor_idx, Vec<u8>>.
    let mut new_data: std::collections::HashMap<usize, Vec<u8>> = std::collections::HashMap::new();
    let raw_ptr = gguf.raw().as_ptr();
    let data_offset = gguf.data_offset as usize;

    let t0 = std::time::Instant::now();
    for (count, &idx) in to_requant.iter().enumerate() {
        let t = &gguf.tensors[idx];
        let name = &gguf.tensor_names[idx];
        // GGUF tensor dims convention: dims[0] = inner (contracted), dims[1..] = rows.
        // For 2D weight matrix [n_inner, n_rows], we treat each "n_inner-long" slice as one row.
        let n_inner = t.dims[0] as usize;
        let n_rows: usize = t.dims[1..].iter().map(|&d| d as usize).product();
        assert!(n_inner % QK_K == 0, "{name}: n_inner {n_inner} not multiple of {QK_K}");
        let blocks_per_row = n_inner / QK_K;
        let q4k_row_bytes = blocks_per_row * Q4K_BLOCK_BYTES;
        let q3k_row_bytes = blocks_per_row * Q3K_BLOCK_BYTES;

        let weight_ptr = unsafe { raw_ptr.add(data_offset + t.offset as usize) };
        let mut out = vec![0u8; n_rows * q3k_row_bytes];
        let mut row_f32 = vec![0f32; n_inner];

        for row in 0..n_rows {
            // q4k_embed_lookup interprets `token_id * row_bytes` as the row offset,
            // which is exactly what we want for any row-major Q4K weight matrix.
            q4k_embed_lookup(weight_ptr, row, &mut row_f32, n_inner);
            let lo = row * q3k_row_bytes;
            let hi = lo + q3k_row_bytes;
            quantize_row_q3_k(&row_f32, &mut out[lo..hi]);
        }

        // Sanity assertion: did we fill what we expected?
        assert_eq!(out.len(), n_rows * q3k_row_bytes);
        let _ = q4k_row_bytes;  // kept for clarity even though unused after assert above.
        new_data.insert(idx, out);

        let elapsed = t0.elapsed().as_secs_f32();
        let total = to_requant.len();
        let eta = if count > 0 { elapsed * (total - count - 1) as f32 / (count + 1) as f32 } else { 0.0 };
        eprintln!("[{:>3}/{}] {} {:?} done ({:.1}s elapsed, {:.1}s eta)",
                  count + 1, total, name, t.dims, elapsed, eta);
    }
    eprintln!("requant complete in {:.1}s", t0.elapsed().as_secs_f32());

    write_output(&gguf, &new_data, &dst);

    let dst_size = std::fs::metadata(&dst).expect("stat dest").len();
    let src_size = std::fs::metadata(&src).expect("stat src").len();
    eprintln!("source:  {:.1} MB", src_size as f64 / (1024.0 * 1024.0));
    eprintln!("output:  {:.1} MB", dst_size as f64 / (1024.0 * 1024.0));
    eprintln!("savings: {:.1} MB", (src_size as i64 - dst_size as i64) as f64 / (1024.0 * 1024.0));
}

fn write_output(
    gguf: &GgufFile,
    new_data: &std::collections::HashMap<usize, Vec<u8>>,
    dst: &Path,
) {
    let n = gguf.tensors.len();
    let raw = gguf.raw();

    let new_size: Vec<u64> = (0..n).map(|i| match new_data.get(&i) {
        Some(buf) => buf.len() as u64,
        None => tensor_byte_size(&gguf.tensors[i].dims, gguf.tensors[i].dtype) as u64,
    }).collect();

    let mut new_offsets = vec![0u64; n];
    let mut cursor: u64 = 0;
    for i in 0..n {
        new_offsets[i] = cursor;
        cursor = align_up(cursor + new_size[i], ALIGNMENT);
    }

    let new_tensor_info_size: u64 = (0..n).map(|i| {
        let name = &gguf.tensor_names[i];
        let dims = &gguf.tensors[i].dims;
        8 + name.len() as u64 + 4 + 8 * dims.len() as u64 + 4 + 8
    }).sum();
    let unpadded_end = gguf.meta_end + new_tensor_info_size;
    let new_data_offset = align_up(unpadded_end, ALIGNMENT);

    let f = File::create(dst).expect("create dest");
    let mut w = BufWriter::with_capacity(1 << 22, f);

    // Header (24 bytes) + KV metadata up to meta_end — copy verbatim from source.
    w.write_all(&raw[0..24]).unwrap();
    w.write_all(&raw[24..gguf.meta_end as usize]).unwrap();

    // Tensor info table — same names/dims, but rewrite dtype + offset for requanted entries.
    for i in 0..n {
        let name = &gguf.tensor_names[i];
        let dims = &gguf.tensors[i].dims;
        let dtype = if new_data.contains_key(&i) { GGML_TYPE_Q3_K } else { gguf.tensors[i].dtype };
        w.write_all(&(name.len() as u64).to_le_bytes()).unwrap();
        w.write_all(name.as_bytes()).unwrap();
        w.write_all(&(dims.len() as u32).to_le_bytes()).unwrap();
        for &d in dims { w.write_all(&d.to_le_bytes()).unwrap(); }
        w.write_all(&dtype.to_le_bytes()).unwrap();
        w.write_all(&new_offsets[i].to_le_bytes()).unwrap();
    }

    // Pad to data alignment.
    let pad = new_data_offset - unpadded_end;
    if pad > 0 { w.write_all(&vec![0u8; pad as usize]).unwrap(); }

    // Tensor data — requanted entries from new_data, others copied from source mmap.
    let zeros = [0u8; 32];
    let mut written: u64 = 0;
    for i in 0..n {
        assert_eq!(written, new_offsets[i],
            "data cursor mismatch at tensor {} ({}): expected {}, got {}",
            i, gguf.tensor_names[i], new_offsets[i], written);
        if let Some(buf) = new_data.get(&i) {
            w.write_all(buf).unwrap();
        } else {
            let t = &gguf.tensors[i];
            let start = gguf.data_offset as usize + t.offset as usize;
            let len = new_size[i] as usize;
            w.write_all(&raw[start..start + len]).unwrap();
        }
        written += new_size[i];
        let aligned = align_up(written, ALIGNMENT);
        let pad = (aligned - written) as usize;
        if pad > 0 { w.write_all(&zeros[..pad]).unwrap(); }
        written = aligned;
    }

    w.flush().unwrap();
}

fn align_up(offset: u64, alignment: u64) -> u64 {
    (offset + alignment - 1) & !(alignment - 1)
}

fn tensor_byte_size(dims: &[u64], dtype: u32) -> usize {
    let n: u64 = dims.iter().product();
    match dtype {
        0 => (n * 4) as usize,
        1 | 30 => (n * 2) as usize,
        11 => ((n as usize + 255) / 256) * 110,  // Q3_K
        12 => ((n as usize + 255) / 256) * 144,  // Q4_K
        13 => ((n as usize + 255) / 256) * 176,  // Q5_K
        14 => ((n as usize + 255) / 256) * 210,  // Q6_K
        24 => n as usize,
        25 => (n * 2) as usize,
        26 => (n * 4) as usize,
        27 | 28 => (n * 8) as usize,
        other => panic!("requant tool: unhandled dtype {other}"),
    }
}

// ---------------------------------------------------------------------------
// Q3K quantization — port of llama.cpp ggml-quants.c quantize_row_q3_K_impl
// + make_qx_quants (rmse_type=1, no imatrix). The _impl + make_qx_quants
// path is what llama-quantize uses with --imatrix; without imatrix llama.cpp
// falls back to _ref (line 1379-1381 in ggml-quants.c). Empirically the
// _impl/make_qx_quants chain still produces noticeably better quality than
// _ref even without an imatrix because it does a 19-point iscale grid
// search per sub-block AND a second-level optimization across scales.
// ---------------------------------------------------------------------------

const GROUP_MAX_EPS: f32 = 1e-15;

#[inline]
fn nearest_int(fval: f32) -> i32 {
    // Same fast trick as the Q4K tool (avoids round() ties-to-even fenv reliance).
    let v = fval + 12_582_912.0_f32;
    let i = v.to_bits() as i32;
    (i & 0x007f_ffff) - 0x0040_0000
}

/// Port of llama.cpp `make_qx_quants(n, nmax, x, L, rmse_type, qw)`.
/// Always called with rmse_type=1 from quantize_row_q3_K_impl. We hardcode that.
/// Returns the per-sub-block scale; writes L in `[0..2*nmax]` (after `+ nmax` shift).
fn make_qx_quants(nmax: i32, x: &[f32], l: &mut [i8], qw: &[f32]) -> f32 {
    let n = x.len();
    debug_assert_eq!(l.len(), n);
    debug_assert_eq!(qw.len(), n);

    let mut max = 0.0_f32;
    let mut amax = 0.0_f32;
    for i in 0..n {
        let ax = x[i].abs();
        if ax > amax { amax = ax; max = x[i]; }
    }
    if amax < GROUP_MAX_EPS {
        for li in l.iter_mut() { *li = 0; }
        return 0.0;
    }
    let mut iscale = -(nmax as f32) / max;
    let mut sumlx = 0.0_f32;
    let mut suml2 = 0.0_f32;
    for i in 0..n {
        let mut li = nearest_int(iscale * x[i]);
        if li < -nmax { li = -nmax; }
        if li > nmax - 1 { li = nmax - 1; }
        l[i] = (li + nmax) as i8;  // store shifted [0..2*nmax]
        let w = qw[i];             // rmse_type=1 with explicit qw → just qw[i]
        sumlx += w * x[i] * li as f32;
        suml2 += w * (li as f32) * (li as f32);
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    let mut best = scale * sumlx;

    // 19-point iscale grid search around the initial estimate (skip is=0 = original).
    for is in -9..=9 {
        if is == 0 { continue; }
        let trial_iscale = -((nmax as f32) + 0.1 * is as f32) / max;
        let mut t_sumlx = 0.0_f32;
        let mut t_suml2 = 0.0_f32;
        for i in 0..n {
            let mut li = nearest_int(trial_iscale * x[i]);
            if li < -nmax { li = -nmax; }
            if li > nmax - 1 { li = nmax - 1; }
            let w = qw[i];
            t_sumlx += w * x[i] * li as f32;
            t_suml2 += w * (li as f32) * (li as f32);
        }
        if t_suml2 > 0.0 && t_sumlx * t_sumlx > best * t_suml2 {
            for i in 0..n {
                let mut li = nearest_int(trial_iscale * x[i]);
                if li < -nmax { li = -nmax; }
                if li > nmax - 1 { li = nmax - 1; }
                l[i] = (li + nmax) as i8;
            }
            scale = t_sumlx / t_suml2;
            best = scale * t_sumlx;
            iscale = trial_iscale;
        }
    }
    let _ = iscale;  // kept assigned for clarity even though final value is unused.
    scale
}

/// Port of llama.cpp `quantize_row_q3_K_impl(..., quant_weights=NULL)`.
/// With NULL quant_weights, the per-sub-block weight is just `x[i]^2`.
/// Two-level optimization: per-sub-block scales via make_qx_quants(nmax=4),
/// then scales themselves are quantized by make_qx_quants(nmax=32) with sub-block
/// weights = sum(x^2) per sub-block.
fn quantize_row_q3_k(x: &[f32], dst: &mut [u8]) {
    assert!(x.len() % QK_K == 0);
    assert_eq!(dst.len(), (x.len() / QK_K) * Q3K_BLOCK_BYTES);
    let nb = x.len() / QK_K;
    let mut l = [0i8; QK_K];
    let mut scales = [0f32; QK_K / 16];
    let mut sw = [0f32; QK_K / 16];
    let mut ls = [0i8; QK_K / 16];
    let mut weight = [0f32; 16];

    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let block = &mut dst[i * Q3K_BLOCK_BYTES..(i + 1) * Q3K_BLOCK_BYTES];
        for b in block.iter_mut() { *b = 0; }

        for j in 0..(QK_K / 16) {
            let xs = &xb[j * 16..(j + 1) * 16];
            // No imatrix: weight[l] = x[l]^2
            let mut sumw = 0.0_f32;
            for k in 0..16 {
                weight[k] = xs[k] * xs[k];
                sumw += weight[k];
            }
            sw[j] = sumw;
            scales[j] = make_qx_quants(4, xs, &mut l[j * 16..(j + 1) * 16], &weight);
        }

        // Quantize the scales themselves: nmax=32 → range [-32..31], then +32 → [0..63].
        // Output L is the shifted [0..63] form ready to pack.
        let d_block = make_qx_quants(32, &scales, &mut ls, &sw);

        // Pack ls[0..15] into block[96..107] (12 bytes encoding 16 × 6-bit signed scales).
        for j in 0..(QK_K / 16) {
            let li = ls[j] as i32;
            if j < 8 {
                block[96 + j] = (li & 0xF) as u8;
            } else {
                block[96 + j - 8] |= ((li & 0xF) as u8) << 4;
            }
            let li_high = li >> 4;
            block[96 + (j % 4) + 8] |= (li_high as u8) << (2 * (j / 4));
        }
        block[108..110].copy_from_slice(&f32_to_f16(d_block).to_le_bytes());

        // Re-extract effective per-sub-block scales (from packed bytes) and re-quant L
        // — necessary because the scale quantization above may have changed sc[j] from
        // the value used in the per-sub-block quant.
        for j in 0..(QK_K / 16) {
            let sc_low = if j < 8 { block[96 + j] & 0xF } else { block[96 + j - 8] >> 4 };
            let sc_high = (block[96 + 8 + (j % 4)] >> (2 * (j / 4))) & 0x3;
            let sc = ((sc_low | (sc_high << 4)) as i32) - 32;
            let d = d_block * sc as f32;
            if d == 0.0 {
                // Match llama.cpp's `continue;` — preserves the L values that
                // make_qx_quants(nmax=4) wrote during the first per-sub-block
                // pass. Don't overwrite them with q3=0; that loses the
                // per-element quantization on degenerate scale = 0 sub-blocks.
                continue;
            }
            for ii in 0..16 {
                let mut li = nearest_int(xb[16 * j + ii] / d);
                if li < -4 { li = -4; }
                if li > 3 { li = 3; }
                l[16 * j + ii] = (li + 4) as i8;
            }
        }

        // Build hmask: bit b of hm[m] is set if L[m + b*32] > 3 (and we subtract 4 in that case).
        let mut m = 0;
        let mut hm: u8 = 1;
        for j in 0..QK_K {
            if l[j] > 3 {
                block[m] |= hm;
                l[j] -= 4;
            }
            m += 1;
            if m == QK_K / 8 {
                m = 0;
                hm <<= 1;
            }
        }

        // Pack qs: 4 values per byte at shifts 0,2,4,6 across element groups (j..j+96 step 32).
        for j in (0..QK_K).step_by(128) {
            for ll in 0..32 {
                let v = (l[j + ll] as u8)
                      | ((l[j + ll + 32] as u8) << 2)
                      | ((l[j + ll + 64] as u8) << 4)
                      | ((l[j + ll + 96] as u8) << 6);
                block[32 + j / 4 + ll] = v;
            }
        }
    }
}

fn f32_to_f16(x: f32) -> u16 {
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
        if new_exp < -10 { return sign as u16; }
        let m = mantissa | 0x0080_0000;
        let shift = (1 - new_exp) as u32;
        let half = (m >> (shift + 13 - 1)) & 1;
        let result = m >> (shift + 13);
        return (sign | (result + half)) as u16;
    }
    let half = (mantissa >> 12) & 1;
    let result = ((new_exp as u32) << 10) | (mantissa >> 13);
    (sign | (result + half)) as u16
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 { sign } else {
            let mut m = mant; let mut e = 0i32;
            while (m & 0x400) == 0 { m <<= 1; e -= 1; }
            m &= 0x3FF;
            sign | (((127 - 15 + e + 1) as u32) << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | 0x7F800000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

// ---------------------------------------------------------------------------
// Self-test: round-trip f32 → Q3K → f32, check error bounds. Runs in plain
// `cargo test --release --test requant_q4k_to_q3k` (no model file needed).
// ---------------------------------------------------------------------------

#[test]
fn q3k_roundtrip_unit_block_smoke() {
    // Deterministic input bounded to typical weight magnitudes.
    let mut x = [0f32; QK_K];
    for i in 0..QK_K {
        let t = i as f32 / QK_K as f32;
        x[i] = 0.18 * (2.0 * std::f32::consts::PI * 3.0 * t).sin()
             + 0.06 * (2.0 * std::f32::consts::PI * 11.0 * t).cos();
    }
    let mut block = [0u8; Q3K_BLOCK_BYTES];
    quantize_row_q3_k(&x, &mut block);

    // Reconstruct via the dequant rule used by q3k_dot.ea / dequantize_row_q3_K.
    // Mirror ggml-quants.c:1243 exactly with an explicit output index that
    // increments per element (the original uses *y++).
    let d = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));
    let mut recon = [0f32; QK_K];
    let hm = &block[0..32];
    let qs = &block[32..96];
    let mut scales_byte = [0u8; 16];
    for j in 0..16 {
        let lo = if j < 8 { block[96 + j] & 0xF } else { block[96 + j - 8] >> 4 };
        let hi = (block[96 + 8 + (j % 4)] >> (2 * (j / 4))) & 0x3;
        scales_byte[j] = lo | (hi << 4);
    }

    let mut m: u8 = 1;
    let mut is = 0;
    let mut out_i = 0usize;
    let mut q_base = 0usize;
    for _n in (0..QK_K).step_by(128) {
        let mut shift = 0;
        for _j in 0..4 {
            let dl1 = d * (scales_byte[is] as i32 - 32) as f32;
            for l in 0..16 {
                let q = ((qs[q_base + l] >> shift) & 3) as i32
                      - if (hm[l] & m) != 0 { 0 } else { 4 };
                recon[out_i] = dl1 * q as f32;
                out_i += 1;
            }
            is += 1;
            let dl2 = d * (scales_byte[is] as i32 - 32) as f32;
            for l in 0..16 {
                let q = ((qs[q_base + l + 16] >> shift) & 3) as i32
                      - if (hm[l + 16] & m) != 0 { 0 } else { 4 };
                recon[out_i] = dl2 * q as f32;
                out_i += 1;
            }
            is += 1;
            shift += 2;
            m <<= 1;
        }
        q_base += 32;
    }

    let mut max_err = 0.0_f32;
    let mut sum_sq = 0.0_f32;
    for i in 0..QK_K {
        let e = (recon[i] - x[i]).abs();
        if e > max_err { max_err = e; }
        sum_sq += e * e;
    }
    let rmse = (sum_sq / QK_K as f32).sqrt();
    eprintln!("q3k roundtrip sinusoid: max_err={max_err:.4} rmse={rmse:.4}");
    // Q3K is meaningfully lossier than Q4K. Set tolerance ~3-4× looser than the Q4K
    // smoke test (which uses 0.02 / 0.01); empirically Q3K hits ~0.04 / 0.012 here.
    assert!(max_err < 0.06, "Q3K max abs err {max_err} too high");
    assert!(rmse < 0.025, "Q3K rmse {rmse} too high");
}

//! Offline tool: requantize `token_embd.weight` from Q6K to Q4K.
//!
//! Run: `cargo test --release --test requant_token_embd_q6k_to_q4k -- --ignored --nocapture`
//!
//! Reads  `~/.olorin/models/gemma-4-e2b-it-Q4_K_M.gguf`
//! Writes `~/.olorin/models/gemma-4-e2b-it-Q4_K_M-q4kembed.gguf`
//!
//! Only `token_embd.weight` changes (Q6K [1536,262144]→Q4K [1536,262144],
//! 315→216 MB). Every other tensor is byte-for-byte identical with the source.
//!
//! Quantization is a faithful port of llama.cpp `quantize_row_q4_K_ref` +
//! `make_qkx2_quants` from `ggml/src/ggml-quants.c` (rmse mode, no imatrix).

use olorin::inference::dequant::q6k_embed_lookup;
use olorin::inference::gguf::GgufFile;

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const ALIGNMENT: u64 = 32;
const QK_K: usize = 256;
const Q4K_BLOCK_BYTES: usize = 144;
const Q6K_BLOCK_BYTES: usize = 210;
const GGML_TYPE_Q4_K: u32 = 12;
const GGML_TYPE_Q6_K: u32 = 14;

#[test]
#[ignore]
fn requant_token_embd_q6k_to_q4k() {
    let home = std::env::var("HOME").expect("HOME not set");
    let src = Path::new(&home).join(".olorin/models/gemma-4-e2b-it-Q4_K_M.gguf");
    let dst = Path::new(&home).join(".olorin/models/gemma-4-e2b-it-Q4_K_M-q4kembed.gguf");
    assert!(src.exists(), "source not found: {}", src.display());
    eprintln!("source: {}", src.display());
    eprintln!("dest:   {}", dst.display());

    let gguf = GgufFile::open(&src).expect("open source gguf");
    let token_embd_idx = *gguf.tensor_map.get("token_embd.weight")
        .expect("source has no token_embd.weight");
    let te = &gguf.tensors[token_embd_idx];
    assert_eq!(te.dtype, GGML_TYPE_Q6_K, "token_embd is not Q6K in source (dtype {})", te.dtype);
    assert_eq!(te.dims.len(), 2, "expected 2D token_embd, got {:?}", te.dims);
    let hidden = te.dims[0] as usize;
    let vocab = te.dims[1] as usize;
    assert!(hidden % QK_K == 0, "hidden_dim {} not multiple of {}", hidden, QK_K);
    let q4k_row_blocks = hidden / QK_K;
    let q4k_row_bytes = q4k_row_blocks * Q4K_BLOCK_BYTES;
    let q6k_row_bytes = (hidden / QK_K) * Q6K_BLOCK_BYTES;
    eprintln!("token_embd.weight Q6K [{},{}] = {} MB → Q4K = {} MB",
        hidden, vocab,
        (vocab * q6k_row_bytes) >> 20,
        (vocab * q4k_row_bytes) >> 20);

    let mut new_data = vec![0u8; vocab * q4k_row_bytes];
    let mut row_f32 = vec![0f32; hidden];
    let src_q6k_ptr = unsafe {
        gguf.raw().as_ptr().add(gguf.data_offset as usize + te.offset as usize)
    };
    let t0 = std::time::Instant::now();
    for token_id in 0..vocab {
        q6k_embed_lookup(src_q6k_ptr, token_id, &mut row_f32, hidden);
        let lo = token_id * q4k_row_bytes;
        let hi = lo + q4k_row_bytes;
        quantize_row_q4_k(&row_f32, &mut new_data[lo..hi]);
        if token_id % 32768 == 0 && token_id != 0 {
            let elapsed = t0.elapsed().as_secs_f32();
            let eta = elapsed * (vocab - token_id) as f32 / token_id as f32;
            eprintln!("[{:>6}/{}] {:.1}s elapsed, {:.1}s eta", token_id, vocab, elapsed, eta);
        }
    }
    eprintln!("requant complete in {:.1}s", t0.elapsed().as_secs_f32());

    write_output(&gguf, token_embd_idx, &new_data, &dst);

    let dst_size = std::fs::metadata(&dst).expect("stat dest").len();
    eprintln!("wrote {} ({:.1} MB)", dst.display(), dst_size as f64 / (1024.0 * 1024.0));
}

fn write_output(gguf: &GgufFile, token_embd_idx: usize, new_token_embd: &[u8], dst: &Path) {
    let n = gguf.tensors.len();
    let raw = gguf.raw();

    let new_size: Vec<u64> = (0..n).map(|i| {
        if i == token_embd_idx {
            new_token_embd.len() as u64
        } else {
            tensor_byte_size(&gguf.tensors[i].dims, gguf.tensors[i].dtype) as u64
        }
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

    w.write_all(&raw[0..24]).unwrap();
    w.write_all(&raw[24..gguf.meta_end as usize]).unwrap();

    for i in 0..n {
        let name = &gguf.tensor_names[i];
        let dims = &gguf.tensors[i].dims;
        let dtype = if i == token_embd_idx { GGML_TYPE_Q4_K } else { gguf.tensors[i].dtype };
        w.write_all(&(name.len() as u64).to_le_bytes()).unwrap();
        w.write_all(name.as_bytes()).unwrap();
        w.write_all(&(dims.len() as u32).to_le_bytes()).unwrap();
        for &d in dims { w.write_all(&d.to_le_bytes()).unwrap(); }
        w.write_all(&dtype.to_le_bytes()).unwrap();
        w.write_all(&new_offsets[i].to_le_bytes()).unwrap();
    }

    let pad = new_data_offset - unpadded_end;
    if pad > 0 { w.write_all(&vec![0u8; pad as usize]).unwrap(); }

    let zeros = [0u8; 32];
    let mut written: u64 = 0;
    for i in 0..n {
        assert_eq!(written, new_offsets[i],
            "data cursor mismatch at tensor {} ({}): expected {}, got {}",
            i, gguf.tensor_names[i], new_offsets[i], written);
        if i == token_embd_idx {
            w.write_all(new_token_embd).unwrap();
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
        12 => ((n as usize + 255) / 256) * 144,
        13 => ((n as usize + 255) / 256) * 176,
        14 => ((n as usize + 255) / 256) * 210,
        24 => n as usize,
        25 => (n * 2) as usize,
        26 => (n * 4) as usize,
        27 => (n * 8) as usize,
        28 => (n * 8) as usize,
        other => panic!("requant tool: unhandled dtype {other}"),
    }
}

// ---------------------------------------------------------------------------
// Q4K quantization — port of llama.cpp ggml-quants.c quantize_row_q4_K_ref.
// ---------------------------------------------------------------------------

#[inline]
fn nearest_int(fval: f32) -> i32 {
    let v = fval + 12_582_912.0_f32;
    let i = v.to_bits() as i32;
    (i & 0x007f_ffff) - 0x0040_0000
}

fn make_qkx2_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    l_out: &mut [u8],
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> (f32, f32) {
    let (mut min_v, mut max_v) = (x[0], x[0]);
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min_v { min_v = x[i]; }
        if x[i] > max_v { max_v = x[i]; }
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if min_v > 0.0 { min_v = 0.0; }
    if max_v == min_v {
        for i in 0..n { l_out[i] = 0; }
        return (0.0, -min_v);
    }
    let mut iscale = nmax as f32 / (max_v - min_v);
    let mut scale = 1.0 / iscale;
    let mut min = min_v;
    let mut best_err = 0.0_f32;
    for i in 0..n {
        let l = nearest_int(iscale * (x[i] - min)).max(0).min(nmax);
        l_out[i] = l as u8;
        let diff = scale * l as f32 + min - x[i];
        let diff = if use_mad { diff.abs() } else { diff * diff };
        best_err += weights[i] * diff;
    }
    if nstep < 1 {
        return (scale, -min);
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max_v - min_v);
        let mut sum_l = 0.0_f32;
        let mut sum_l2 = 0.0_f32;
        let mut sum_xl = 0.0_f32;
        for i in 0..n {
            let l = nearest_int(iscale * (x[i] - min)).max(0).min(nmax);
            laux[i] = l as u8;
            let w = weights[i];
            let lf = l as f32;
            sum_l  += w * lf;
            sum_l2 += w * lf * lf;
            sum_xl += w * lf * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min   = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut cur_err = 0.0_f32;
            for i in 0..n {
                let diff = this_scale * laux[i] as f32 + this_min - x[i];
                let diff = if use_mad { diff.abs() } else { diff * diff };
                cur_err += weights[i] * diff;
            }
            if cur_err < best_err {
                l_out[..n].copy_from_slice(&laux[..n]);
                best_err = cur_err;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    (scale, -min)
}

fn quantize_row_q4_k(x: &[f32], dst: &mut [u8]) {
    assert!(x.len() % QK_K == 0);
    assert_eq!(dst.len(), (x.len() / QK_K) * Q4K_BLOCK_BYTES);
    let nb = x.len() / QK_K;
    let mut l_arr = [0u8; QK_K];
    let mut laux  = [0u8; 32];
    let mut weights = [0f32; 32];
    let mut mins   = [0f32; QK_K / 32];
    let mut scales = [0f32; QK_K / 32];

    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let block = &mut dst[i * Q4K_BLOCK_BYTES..(i + 1) * Q4K_BLOCK_BYTES];
        for s in &mut block[4..16] { *s = 0; }

        let mut max_scale = 0.0_f32;
        let mut max_min = 0.0_f32;
        for j in 0..(QK_K / 32) {
            let xs = &xb[j * 32..(j + 1) * 32];
            let mut sum_x2 = 0.0_f32;
            for &v in xs { sum_x2 += v * v; }
            let av_x = (sum_x2 / 32.0).sqrt();
            for l in 0..32 { weights[l] = av_x + xs[l].abs(); }
            let (sc, mn) = make_qkx2_quants(
                32, 15, xs, &weights,
                &mut l_arr[j * 32..(j + 1) * 32], &mut laux,
                -1.0, 0.1, 20, false,
            );
            scales[j] = sc;
            mins[j] = mn;
            if sc > max_scale { max_scale = sc; }
            if mn > max_min { max_min = mn; }
        }

        let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
        let inv_min   = if max_min   > 0.0 { 63.0 / max_min   } else { 0.0 };
        for j in 0..(QK_K / 32) {
            let ls = (nearest_int(inv_scale * scales[j]) as u8).min(63);
            let lm = (nearest_int(inv_min   * mins[j]  ) as u8).min(63);
            if j < 4 {
                block[4 + j] = ls;
                block[4 + j + 4] = lm;
            } else {
                block[4 + j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                block[4 + j - 4] |= (ls >> 4) << 6;
                block[4 + j]     |= (lm >> 4) << 6;
            }
        }

        let d = if max_scale > 0.0 { max_scale / 63.0 } else { 0.0 };
        let dmin = if max_min > 0.0 { max_min / 63.0 } else { 0.0 };
        block[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        block[2..4].copy_from_slice(&f32_to_f16(dmin).to_le_bytes());

        for j in 0..(QK_K / 32) {
            let (sc, m) = get_scale_min_k4(j, &block[4..16]);
            let d_eff = d * sc as f32;
            if d_eff == 0.0 {
                for ii in 0..32 { l_arr[j * 32 + ii] = 0; }
                continue;
            }
            let dm_eff = dmin * m as f32;
            for ii in 0..32 {
                let l = nearest_int((xb[j * 32 + ii] + dm_eff) / d_eff)
                    .max(0).min(15);
                l_arr[j * 32 + ii] = l as u8;
            }
        }

        let qs = &mut block[16..16 + 128];
        let mut q_off = 0;
        let mut j = 0;
        while j < QK_K {
            for l in 0..32 {
                qs[q_off + l] = l_arr[j + l] | (l_arr[j + l + 32] << 4);
            }
            q_off += 32;
            j += 64;
        }
    }
}

fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
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

// ---------------------------------------------------------------------------
// Self-tests: round-trip a handful of f32 values through Q4K to catch obvious
// breakage without needing the model file.
// ---------------------------------------------------------------------------

#[test]
fn q4k_roundtrip_unit_block_smoke() {
    // Deterministic input — sinusoid bounded to embedding-weight magnitudes.
    let mut x = [0f32; QK_K];
    for i in 0..QK_K {
        let t = i as f32 / QK_K as f32;
        x[i] = 0.18 * (2.0 * std::f32::consts::PI * 3.0 * t).sin()
             + 0.06 * (2.0 * std::f32::consts::PI * 11.0 * t).cos();
    }
    let mut block = [0u8; Q4K_BLOCK_BYTES];
    quantize_row_q4_k(&x, &mut block);

    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let mut recon = [0f32; QK_K];
    let qs = &block[16..16 + 128];
    let mut is = 0;
    let mut q_off = 0;
    let mut elem = 0;
    while elem < QK_K {
        let (s1, m1) = get_scale_min_k4(is, &block[4..16]);
        let (s2, m2) = get_scale_min_k4(is + 1, &block[4..16]);
        let d1 = d * s1 as f32; let mn1 = dmin * m1 as f32;
        let d2 = d * s2 as f32; let mn2 = dmin * m2 as f32;
        for l in 0..32 {
            recon[elem + l]      = d1 * (qs[q_off + l] & 0x0F) as f32 - mn1;
            recon[elem + 32 + l] = d2 * (qs[q_off + l] >> 4)   as f32 - mn2;
        }
        q_off += 32;
        is += 2;
        elem += 64;
    }
    let mut max_err = 0.0_f32;
    let mut sum_sq = 0.0_f32;
    for i in 0..QK_K {
        let e = (recon[i] - x[i]).abs();
        if e > max_err { max_err = e; }
        sum_sq += e * e;
    }
    let rmse = (sum_sq / QK_K as f32).sqrt();
    eprintln!("q4k roundtrip uniform[-.2,.2]: max_err={max_err:.4} rmse={rmse:.4}");
    assert!(max_err < 0.02, "max abs err {max_err} too high");
    assert!(rmse < 0.01, "rmse {rmse} too high");
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

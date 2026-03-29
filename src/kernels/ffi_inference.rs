//! FFI layer for Eä SIMD kernels — cougar inference + eakv KV-cache.
//!
//! Initialized by `ffi::init()` via `init_from(&dir)`.
//! Call `ffi::init()` at startup; these wrappers are then available.

use libloading::{Library, Symbol};
use std::path::Path;
use std::sync::OnceLock;

// ── Type aliases — cougar ─────────────────────────────────────────────────────

type I2DotI8Fn = unsafe extern "C" fn(*const u8, *const i8, i32) -> i32;
type I2DotI8_4RowFn = unsafe extern "C" fn(
    *const u8, *const u8, *const u8, *const u8, *const i8, *mut i32, i32);
type I2DotI8_4RowDualFn = unsafe extern "C" fn(
    *const u8, *const u8, *const u8, *const u8,
    *const u8, *const u8, *const u8, *const u8,
    *const i8, *mut i32, *mut i32, i32);
type QuantF32I8Fn   = unsafe extern "C" fn(*const f32, *mut i8, *mut f32, *mut i32, i32);
type RmsnormFn      = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32, f32);
type FusedAttnF32Fn = unsafe extern "C" fn(*const f32, *const f32, *const f32, *mut f32, i32, i32, f32);
type I8Dot1RowFn    = unsafe extern "C" fn(*const i8, *const u8, i32) -> i32;
type I8Dot4RowFn    = unsafe extern "C" fn(*const i8, *const u8, *const u8, *const u8, *const u8, *mut i32, i32);
type SquaredReluFn  = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32);
type VecAddFn       = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32);
type QuantF32Q8kFn  = unsafe extern "C" fn(*const f32, *mut i8, *mut f32, *mut i32, i32);
type Q4kDotQ8kFn    = unsafe extern "C" fn(
    *const u8, *const i8, *const i32, *const u8, *const u8, i32, f32, f32) -> f32;
type Q4kDot4RowFn = unsafe extern "C" fn(
    *const u8, *const u8, *const u8, *const u8,
    *const i8, *const i32,
    *const u8, *const u8, *const u8, *const u8,
    *const u8, *const u8, *const u8, *const u8,
    *mut f32, i32, f32, f32, f32, f32, f32, f32, f32, f32);
type Q4kDot4RowDualFn = unsafe extern "C" fn(
    *const u8, *const u8, *const u8, *const u8,
    *const u8, *const u8, *const u8, *const u8,
    *const i8, *const i32,
    *const u8, *const u8, *const u8, *const u8,
    *const u8, *const u8, *const u8, *const u8,
    *const u8, *const u8, *const u8, *const u8,
    *const u8, *const u8, *const u8, *const u8,
    *mut f32, *mut f32, i32,
    f32, f32, f32, f32, f32, f32, f32, f32,
    f32, f32, f32, f32, f32, f32, f32, f32);
type Q6kDotQ8kFn = unsafe extern "C" fn(
    *const u8, *const u8, *const i8, *const i8, *const i32, i32, f32) -> f32;
type Q6kDot4RowFn = unsafe extern "C" fn(
    *const u8, *const u8, *const u8, *const u8,
    *const u8, *const u8, *const u8, *const u8,
    *const i8, *const i8, *const i8, *const i8,
    *const i8, *const i32, *mut f32, i32, f32, f32, f32, f32);
type ApplyRopeFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32, i32);

// ── Type aliases — eakv ───────────────────────────────────────────────────────

type QuantizeSIMDFn   = unsafe extern "C" fn(*const f32, *mut i32, *mut f32, *mut f32, i32);
type DequantizeSIMDFn = unsafe extern "C" fn(*const u8, *const f32, *const f32, *mut f32, i32);
type KScoreMhaFn = unsafe extern "C" fn(
    *const f32, *const u8, *const f32, *const f32, *mut f32, i32, i32, i32);
type KScoreGqaFn = unsafe extern "C" fn(
    *const f32, *const u8, *const f32, *const f32, *mut f32, i32, i32, i32, i32);
type VSumMhaFn = unsafe extern "C" fn(
    *const f32, *const u8, *const f32, *const f32, *mut f32, i32, i32, i32);
type VSumGqaFn = unsafe extern "C" fn(
    *const f32, *const u8, *const f32, *const f32, *mut f32, i32, i32, i32, i32);
type FusedAttentionFn = unsafe extern "C" fn(
    *const f32,
    *const u8, *const f32, *const f32,
    *const u8, *const f32, *const f32,
    *mut f32, i32, i32, i32);
type ValidateFn = unsafe extern "C" fn(*const f32, *const f32, *const i32, *const i32, i32) -> i32;

// ── KernelTableInference ──────────────────────────────────────────────────────

pub struct KernelTableInference {
    pub libs: Vec<Library>,
    pub i2_dot_i8:             I2DotI8Fn,
    pub i2_dot_i8_4row:        I2DotI8_4RowFn,
    pub i2_dot_i8_4row_dual:   I2DotI8_4RowDualFn,
    pub quant_f32_i8:          QuantF32I8Fn,
    pub rmsnorm_f32:           RmsnormFn,
    pub fused_attention_f32:   FusedAttnF32Fn,
    pub i8dot_1row:            I8Dot1RowFn,
    pub i8dot_4row:            I8Dot4RowFn,
    pub squared_relu_mul_f32:  SquaredReluFn,
    pub vecadd_f32:            VecAddFn,
    pub quant_f32_q8k:         QuantF32Q8kFn,
    pub q4k_dot_q8k:           Q4kDotQ8kFn,
    pub q4k_dot_q8k_4row:      Q4kDot4RowFn,
    pub q4k_dot_q8k_4row_dual: Q4kDot4RowDualFn,
    pub q6k_dot_q8k:           Q6kDotQ8kFn,
    pub q6k_dot_q8k_4row:      Q6kDot4RowFn,
    pub apply_rope_f32:        ApplyRopeFn,
    pub quantize_simd:         QuantizeSIMDFn,
    pub dequantize_simd:       DequantizeSIMDFn,
    pub fused_k_score:         KScoreMhaFn,
    pub fused_k_score_64:      KScoreMhaFn,
    pub fused_k_score_gqa:     KScoreGqaFn,
    pub fused_k_score_gqa_64:  KScoreGqaFn,
    pub fused_v_sum:           VSumMhaFn,
    pub fused_v_sum_64:        VSumMhaFn,
    pub fused_v_sum_gqa:       VSumGqaFn,
    pub fused_v_sum_gqa_64:    VSumGqaFn,
    pub fused_attention:       FusedAttentionFn,
    pub validate:              ValidateFn,
}

// SAFETY: KernelTableInference holds function pointers and library handles.
// Function pointers are valid for the lifetime of the libraries.
// Libraries are never unloaded (held in OnceLock for program lifetime).
unsafe impl Send for KernelTableInference {}
unsafe impl Sync for KernelTableInference {}

static KERNELS: OnceLock<KernelTableInference> = OnceLock::new();

fn k() -> &'static KernelTableInference {
    KERNELS.get().expect("inference kernels not initialized — call ffi::init() first")
}

/// Initialize inference kernels from a pre-extracted directory.
/// Called by `ffi::init()`. Safe to call multiple times.
pub fn init_from(lib_dir: &Path) -> Result<(), String> {
    if KERNELS.get().is_some() {
        return Ok(());
    }
    let table = load_inference_kernels(lib_dir)?;
    let _ = KERNELS.set(table);
    Ok(())
}

fn load_inference_kernels(lib_dir: &Path) -> Result<KernelTableInference, String> {
    let load = |name: &str| -> Result<Library, String> {
        let path = lib_dir.join(format!("lib{name}.so"));
        unsafe {
            Library::new(&path)
                .map_err(|e| format!("failed to load {}: {e}", path.display()))
        }
    };

    let i2s  = load("bitnet_i2s")?;
    let quant = load("bitnet_quant")?;
    let rms  = load("bitnet_rmsnorm")?;
    let attn = load("bitnet_fused_attn")?;
    let i8d  = load("bitnet_i8dot")?;
    let act  = load("bitnet_activate")?;
    let vadd = load("bitnet_vecadd")?;
    let q4kq = load("q4k_quant")?;
    let q4kd = load("q4k_dot")?;
    let q6kd = load("q6k_dot")?;
    let rope = load("rope")?;

    let quantize_lib      = load("quantize_simd")?;
    let k_score_lib       = load("fused_k_score")?;
    let k_score_64_lib    = load("fused_k_score_64")?;
    let k_score_gqa_lib   = load("fused_k_score_gqa")?;
    let k_score_gqa64_lib = load("fused_k_score_gqa_64")?;
    let v_sum_lib         = load("fused_v_sum")?;
    let v_sum_64_lib      = load("fused_v_sum_64")?;
    let fused_attn_lib    = load("fused_attention")?;
    let validate_lib      = load("validate")?;

    // CPU-dispatch for dequantize: prefer AVX-512 > AVX2 > SIMD
    #[cfg(target_arch = "x86_64")]
    let (deq_lib, deq_sym): (Library, &[u8]) =
        if is_x86_feature_detected!("avx512f") {
            if let Ok(lib) = load("dequantize_avx512") {
                (lib, b"q4_dequantize_avx512_f32\0")
            } else if let Ok(lib) = load("dequantize_avx2") {
                (lib, b"q4_dequantize_avx2_f32\0")
            } else {
                (load("dequantize_simd")?, b"q4_dequantize_simd_f32\0")
            }
        } else if is_x86_feature_detected!("avx2") {
            if let Ok(lib) = load("dequantize_avx2") {
                (lib, b"q4_dequantize_avx2_f32\0")
            } else {
                (load("dequantize_simd")?, b"q4_dequantize_simd_f32\0")
            }
        } else {
            (load("dequantize_simd")?, b"q4_dequantize_simd_f32\0")
        };
    #[cfg(not(target_arch = "x86_64"))]
    let (deq_lib, deq_sym): (Library, &[u8]) =
        (load("dequantize_simd")?, b"q4_dequantize_simd_f32\0");

    unsafe {
        let sym = |lib: &Library, name: &[u8]| -> Result<usize, String> {
            let s: Symbol<*const ()> = lib
                .get(name)
                .map_err(|e| format!("symbol {:?}: {e}",
                    std::str::from_utf8(&name[..name.len()-1]).unwrap_or("?")))?;
            Ok(*s as usize)
        };

        let t = KernelTableInference {
            i2_dot_i8:             std::mem::transmute(sym(&i2s,  b"i2_dot_i8\0")?),
            i2_dot_i8_4row:        std::mem::transmute(sym(&i2s,  b"i2_dot_i8_4row\0")?),
            i2_dot_i8_4row_dual:   std::mem::transmute(sym(&i2s,  b"i2_dot_i8_4row_dual\0")?),
            quant_f32_i8:          std::mem::transmute(sym(&quant, b"quant_f32_i8\0")?),
            rmsnorm_f32:           std::mem::transmute(sym(&rms,  b"rmsnorm_f32\0")?),
            fused_attention_f32:   std::mem::transmute(sym(&attn, b"fused_attention_f32\0")?),
            i8dot_1row:            std::mem::transmute(sym(&i8d,  b"i8dot_1row\0")?),
            i8dot_4row:            std::mem::transmute(sym(&i8d,  b"i8dot_4row\0")?),
            squared_relu_mul_f32:  std::mem::transmute(sym(&act,  b"squared_relu_mul_f32\0")?),
            vecadd_f32:            std::mem::transmute(sym(&vadd, b"vecadd_f32\0")?),
            quant_f32_q8k:         std::mem::transmute(sym(&q4kq, b"quant_f32_q8k\0")?),
            q4k_dot_q8k:           std::mem::transmute(sym(&q4kd, b"q4k_dot_q8k\0")?),
            q4k_dot_q8k_4row:      std::mem::transmute(sym(&q4kd, b"q4k_dot_q8k_4row\0")?),
            q4k_dot_q8k_4row_dual: std::mem::transmute(sym(&q4kd, b"q4k_dot_q8k_4row_dual\0")?),
            q6k_dot_q8k:           std::mem::transmute(sym(&q6kd, b"q6k_dot_q8k\0")?),
            q6k_dot_q8k_4row:      std::mem::transmute(sym(&q6kd, b"q6k_dot_q8k_4row\0")?),
            apply_rope_f32:        std::mem::transmute(sym(&rope, b"apply_rope_f32\0")?),
            quantize_simd:         std::mem::transmute(sym(&quantize_lib, b"q4_quantize_split_f32\0")?),
            dequantize_simd:       std::mem::transmute(sym(&deq_lib, deq_sym)?),
            fused_k_score:         std::mem::transmute(sym(&k_score_lib,     b"q4_fused_k_score_multi_f32\0")?),
            fused_k_score_64:      std::mem::transmute(sym(&k_score_64_lib,  b"q4_fused_k_score_multi_64_f32\0")?),
            fused_k_score_gqa:     std::mem::transmute(sym(&k_score_gqa_lib,   b"q4_k_score_gqa_f32\0")?),
            fused_k_score_gqa_64:  std::mem::transmute(sym(&k_score_gqa64_lib, b"q4_k_score_gqa_64_f32\0")?),
            fused_v_sum:           std::mem::transmute(sym(&v_sum_lib,    b"q4_fused_v_sum_multi_f32\0")?),
            fused_v_sum_64:        std::mem::transmute(sym(&v_sum_64_lib, b"q4_fused_v_sum_multi_64_f32\0")?),
            fused_v_sum_gqa:       std::mem::transmute(sym(&k_score_gqa_lib,   b"q4_v_sum_gqa_f32\0")?),
            fused_v_sum_gqa_64:    std::mem::transmute(sym(&k_score_gqa64_lib, b"q4_v_sum_gqa_64_f32\0")?),
            fused_attention:       std::mem::transmute(sym(&fused_attn_lib, b"q4_fused_attention_multi_f32\0")?),
            validate:              std::mem::transmute(sym(&validate_lib, b"q4_validate\0")?),
            libs: vec![
                i2s, quant, rms, attn, i8d, act, vadd,
                q4kq, q4kd, q6kd, rope,
                quantize_lib, deq_lib,
                k_score_lib, k_score_64_lib,
                k_score_gqa_lib, k_score_gqa64_lib,
                v_sum_lib, v_sum_64_lib,
                fused_attn_lib, validate_lib,
            ],
        };
        Ok(t)
    }
}

// ── Public wrappers — cougar ──────────────────────────────────────────────────

pub unsafe fn i2_dot_i8(weights: *const u8, activations: *const i8, n: i32) -> i32 {
    (k().i2_dot_i8)(weights, activations, n)
}

pub unsafe fn i2_dot_i8_4row(
    w0: *const u8, w1: *const u8, w2: *const u8, w3: *const u8,
    activations: *const i8, scores: *mut i32, n: i32,
) {
    (k().i2_dot_i8_4row)(w0, w1, w2, w3, activations, scores, n)
}

pub unsafe fn i2_dot_i8_4row_dual(
    gw0: *const u8, gw1: *const u8, gw2: *const u8, gw3: *const u8,
    uw0: *const u8, uw1: *const u8, uw2: *const u8, uw3: *const u8,
    activations: *const i8, gate_scores: *mut i32, up_scores: *mut i32, n: i32,
) {
    (k().i2_dot_i8_4row_dual)(
        gw0, gw1, gw2, gw3, uw0, uw1, uw2, uw3,
        activations, gate_scores, up_scores, n)
}

pub unsafe fn quant_f32_i8(
    src: *const f32, dst: *mut i8, out_scale: *mut f32, out_sum: *mut i32, n: i32,
) {
    (k().quant_f32_i8)(src, dst, out_scale, out_sum, n)
}

pub unsafe fn rmsnorm_f32(x: *const f32, weight: *const f32, out: *mut f32, n: i32, eps: f32) {
    (k().rmsnorm_f32)(x, weight, out, n, eps)
}

pub unsafe fn fused_attention_f32(
    q: *const f32, k_cache: *const f32, v_cache: *const f32,
    out: *mut f32, head_dim: i32, seq_len: i32, scale: f32,
) {
    (k().fused_attention_f32)(q, k_cache, v_cache, out, head_dim, seq_len, scale)
}

pub unsafe fn i8dot_1row(act: *const i8, w: *const u8, n: i32) -> i32 {
    (k().i8dot_1row)(act, w, n)
}

pub unsafe fn i8dot_4row(
    act: *const i8, w0: *const u8, w1: *const u8, w2: *const u8, w3: *const u8,
    scores: *mut i32, n: i32,
) {
    (k().i8dot_4row)(act, w0, w1, w2, w3, scores, n)
}

pub unsafe fn squared_relu_mul_f32(gate: *const f32, up: *const f32, out: *mut f32, n: i32) {
    (k().squared_relu_mul_f32)(gate, up, out, n)
}

pub unsafe fn vecadd_f32(a: *const f32, b: *const f32, out: *mut f32, n: i32) {
    (k().vecadd_f32)(a, b, out, n)
}

pub unsafe fn quant_f32_q8k(
    src: *const f32, dst_qs: *mut i8, dst_d: *mut f32, dst_bsums: *mut i32, n: i32,
) {
    (k().quant_f32_q8k)(src, dst_qs, dst_d, dst_bsums, n)
}

pub unsafe fn q4k_dot_q8k(
    q4: *const u8, q8: *const i8, bsums: *const i32,
    scales: *const u8, mins: *const u8, n_blocks: i32, d: f32, dmin: f32,
) -> f32 {
    (k().q4k_dot_q8k)(q4, q8, bsums, scales, mins, n_blocks, d, dmin)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn q4k_dot_q8k_4row(
    rw0: *const u8, rw1: *const u8, rw2: *const u8, rw3: *const u8,
    q8: *const i8, bsums: *const i32,
    sc0: *const u8, sc1: *const u8, sc2: *const u8, sc3: *const u8,
    mn0: *const u8, mn1: *const u8, mn2: *const u8, mn3: *const u8,
    scores: *mut f32, n_blocks: i32,
    d0: f32, d1: f32, d2: f32, d3: f32,
    dm0: f32, dm1: f32, dm2: f32, dm3: f32,
) {
    (k().q4k_dot_q8k_4row)(
        rw0, rw1, rw2, rw3, q8, bsums,
        sc0, sc1, sc2, sc3, mn0, mn1, mn2, mn3,
        scores, n_blocks, d0, d1, d2, d3, dm0, dm1, dm2, dm3)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn q4k_dot_q8k_4row_dual(
    gw0: *const u8, gw1: *const u8, gw2: *const u8, gw3: *const u8,
    uw0: *const u8, uw1: *const u8, uw2: *const u8, uw3: *const u8,
    q8: *const i8, bsums: *const i32,
    gsc0: *const u8, gsc1: *const u8, gsc2: *const u8, gsc3: *const u8,
    gmn0: *const u8, gmn1: *const u8, gmn2: *const u8, gmn3: *const u8,
    usc0: *const u8, usc1: *const u8, usc2: *const u8, usc3: *const u8,
    umn0: *const u8, umn1: *const u8, umn2: *const u8, umn3: *const u8,
    gate_scores: *mut f32, up_scores: *mut f32, n_blocks: i32,
    gd0: f32, gd1: f32, gd2: f32, gd3: f32,
    gdm0: f32, gdm1: f32, gdm2: f32, gdm3: f32,
    ud0: f32, ud1: f32, ud2: f32, ud3: f32,
    udm0: f32, udm1: f32, udm2: f32, udm3: f32,
) {
    (k().q4k_dot_q8k_4row_dual)(
        gw0, gw1, gw2, gw3, uw0, uw1, uw2, uw3,
        q8, bsums,
        gsc0, gsc1, gsc2, gsc3, gmn0, gmn1, gmn2, gmn3,
        usc0, usc1, usc2, usc3, umn0, umn1, umn2, umn3,
        gate_scores, up_scores, n_blocks,
        gd0, gd1, gd2, gd3, gdm0, gdm1, gdm2, gdm3,
        ud0, ud1, ud2, ud3, udm0, udm1, udm2, udm3)
}

pub unsafe fn q6k_dot_q8k(
    ql: *const u8, qh: *const u8, scales: *const i8,
    q8: *const i8, bsums: *const i32, n_blocks: i32, d: f32,
) -> f32 {
    (k().q6k_dot_q8k)(ql, qh, scales, q8, bsums, n_blocks, d)
}

pub unsafe fn q6k_dot_q8k_4row(
    ql0: *const u8, ql1: *const u8, ql2: *const u8, ql3: *const u8,
    qh0: *const u8, qh1: *const u8, qh2: *const u8, qh3: *const u8,
    sc0: *const i8, sc1: *const i8, sc2: *const i8, sc3: *const i8,
    q8: *const i8, bsums: *const i32, scores: *mut f32, n_blocks: i32,
    d0: f32, d1: f32, d2: f32, d3: f32,
) {
    (k().q6k_dot_q8k_4row)(
        ql0, ql1, ql2, ql3, qh0, qh1, qh2, qh3,
        sc0, sc1, sc2, sc3, q8, bsums,
        scores, n_blocks, d0, d1, d2, d3)
}

pub unsafe fn apply_rope_f32(
    data: *const f32, freqs: *const f32, out: *mut f32, head_dim: i32, n_heads: i32,
) {
    (k().apply_rope_f32)(data, freqs, out, head_dim, n_heads)
}

// ── Public wrappers — eakv ────────────────────────────────────────────────────

pub unsafe fn quantize_simd(
    src: *const f32, weights_out: *mut i32,
    scales_out: *mut f32, biases_out: *mut f32, n_groups: i32,
) {
    (k().quantize_simd)(src, weights_out, scales_out, biases_out, n_groups)
}

pub unsafe fn dequantize_simd(
    weights: *const u8, scales: *const f32, biases: *const f32, out: *mut f32, n_groups: i32,
) {
    (k().dequantize_simd)(weights, scales, biases, out, n_groups)
}

pub unsafe fn fused_k_score(
    q_vecs: *const f32, k_packed: *const u8, k_scales: *const f32, k_biases: *const f32,
    all_scores: *mut f32, seq_len: i32, n_heads: i32, groups_per_head: i32,
) {
    (k().fused_k_score)(q_vecs, k_packed, k_scales, k_biases,
        all_scores, seq_len, n_heads, groups_per_head)
}

pub unsafe fn fused_k_score_64(
    q_vecs: *const f32, k_packed: *const u8, k_scales: *const f32, k_biases: *const f32,
    all_scores: *mut f32, seq_len: i32, n_heads: i32, groups_per_head: i32,
) {
    (k().fused_k_score_64)(q_vecs, k_packed, k_scales, k_biases,
        all_scores, seq_len, n_heads, groups_per_head)
}

pub unsafe fn fused_k_score_gqa(
    q_vecs: *const f32, k_packed: *const u8, k_scales: *const f32, k_biases: *const f32,
    all_scores: *mut f32, seq_len: i32, n_q_heads: i32, n_kv_heads: i32, groups_per_head: i32,
) {
    (k().fused_k_score_gqa)(q_vecs, k_packed, k_scales, k_biases,
        all_scores, seq_len, n_q_heads, n_kv_heads, groups_per_head)
}

pub unsafe fn fused_k_score_gqa_64(
    q_vecs: *const f32, k_packed: *const u8, k_scales: *const f32, k_biases: *const f32,
    all_scores: *mut f32, seq_len: i32, n_q_heads: i32, n_kv_heads: i32, groups_per_head: i32,
) {
    (k().fused_k_score_gqa_64)(q_vecs, k_packed, k_scales, k_biases,
        all_scores, seq_len, n_q_heads, n_kv_heads, groups_per_head)
}

pub unsafe fn fused_v_sum(
    all_weights: *const f32, v_packed: *const u8, v_scales: *const f32, v_biases: *const f32,
    all_out: *mut f32, seq_len: i32, n_heads: i32, groups_per_head: i32,
) {
    (k().fused_v_sum)(all_weights, v_packed, v_scales, v_biases,
        all_out, seq_len, n_heads, groups_per_head)
}

pub unsafe fn fused_v_sum_64(
    all_weights: *const f32, v_packed: *const u8, v_scales: *const f32, v_biases: *const f32,
    all_out: *mut f32, seq_len: i32, n_heads: i32, groups_per_head: i32,
) {
    (k().fused_v_sum_64)(all_weights, v_packed, v_scales, v_biases,
        all_out, seq_len, n_heads, groups_per_head)
}

pub unsafe fn fused_v_sum_gqa(
    all_weights: *const f32, v_packed: *const u8, v_scales: *const f32, v_biases: *const f32,
    all_out: *mut f32, seq_len: i32, n_q_heads: i32, n_kv_heads: i32, groups_per_head: i32,
) {
    (k().fused_v_sum_gqa)(all_weights, v_packed, v_scales, v_biases,
        all_out, seq_len, n_q_heads, n_kv_heads, groups_per_head)
}

pub unsafe fn fused_v_sum_gqa_64(
    all_weights: *const f32, v_packed: *const u8, v_scales: *const f32, v_biases: *const f32,
    all_out: *mut f32, seq_len: i32, n_q_heads: i32, n_kv_heads: i32, groups_per_head: i32,
) {
    (k().fused_v_sum_gqa_64)(all_weights, v_packed, v_scales, v_biases,
        all_out, seq_len, n_q_heads, n_kv_heads, groups_per_head)
}

pub unsafe fn fused_attention(
    q_vecs: *const f32,
    k_packed: *const u8, k_scales: *const f32, k_biases: *const f32,
    v_packed: *const u8, v_scales: *const f32, v_biases: *const f32,
    all_out: *mut f32, seq_len: i32, n_heads: i32, groups_per_head: i32,
) {
    (k().fused_attention)(
        q_vecs,
        k_packed, k_scales, k_biases,
        v_packed, v_scales, v_biases,
        all_out, seq_len, n_heads, groups_per_head)
}

pub unsafe fn validate(
    scales: *const f32, biases: *const f32,
    scales_bits: *const i32, biases_bits: *const i32, n_groups: i32,
) -> i32 {
    (k().validate)(scales, biases, scales_bits, biases_bits, n_groups)
}

//! FFI layer for Eä SIMD kernels — inference + KV-cache.

use libloading::{Library, Symbol};
use std::path::Path;
use std::sync::OnceLock;
use crate::kernels::ffi_inference_types::*;

// I8MM (ARMv8.6+ integer matrix multiply) detection
#[cfg(target_arch = "aarch64")]
fn detect_i8mm() -> bool {
    // HWCAP2 = 26, HWCAP2_I8MM = (1 << 13)
    unsafe { libc::getauxval(26) & (1 << 13) != 0 }
}

#[cfg(not(target_arch = "aarch64"))]
fn detect_i8mm() -> bool { false }

static I8MM_AVAILABLE: OnceLock<bool> = OnceLock::new();

pub fn has_i8mm() -> bool {
    *I8MM_AVAILABLE.get_or_init(detect_i8mm)
}

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
    pub q4k_fused_dot:          Q4kFusedDotFn,
    pub q4k_fused_dot_4row:     Q4kFusedDot4RowFn,
    pub q6k_dot_q8k:           Q6kDotQ8kFn,
    pub q6k_dot_q8k_4row:      Q6kDot4RowFn,
    pub apply_rope_f32:        ApplyRopeFn,
    pub q4k_gemm_4x4:         Q4kGemm4x4Fn,
    pub attn_dot_f16:          AttnDotF16Fn,
    pub attn_vsum_f16:         AttnVsumF16Fn,
    pub f32_to_f16:            F32ToF16Fn,
    pub f16_to_f32:            F16ToF32Fn,
    pub softmax_f32:           SoftmaxF32Fn,
    pub silu_mul_f32:          SiluMulFn,
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
    let has_i8mm_hw = detect_i8mm();
    let _ = I8MM_AVAILABLE.set(has_i8mm_hw);

    let load = |name: &str| -> Result<Library, String> {
        let path = lib_dir.join(format!("lib{name}.so"));
        unsafe {
            Library::new(&path)
                .map_err(|e| format!("failed to load {}: {e}", path.display()))
        }
    };

    // Try I8MM variant first on ARM, fall back to base
    let load_best = |name: &str| -> Result<Library, String> {
        #[cfg(target_arch = "aarch64")]
        if has_i8mm_hw {
            let i8mm_name = format!("{name}_i8mm");
            if let Ok(lib) = load(&i8mm_name) {
                eprintln!("olorin: {name}=i8mm");
                return Ok(lib);
            }
        }
        load(name)
    };

    let i2s  = load("bitnet_i2s")?;
    let quant = load("bitnet_quant")?;
    let rms  = load("bitnet_rmsnorm")?;
    let attn = load("bitnet_fused_attn")?;
    let i8d  = load("bitnet_i8dot")?;
    let act  = load("bitnet_activate")?;
    let vadd = load("bitnet_vecadd")?;
    let q4kq = load("q4k_quant")?;
    let q4kd = load_best("q4k_dot")?;
    let q4kfg = load_best("q4k_fused_gemm")?;
    let q6kd = load("q6k_dot")?;
    let rope = load("rope")?;
    let gemm_tile = load_best("q4k_gemm_tile")?;

    let attn_f16_lib      = load("attn_f16")?;
    let f16_conv_lib      = load("f16_convert")?;
    let softmax_lib       = load("softmax")?;
    let silu_lib          = load("silu_mul")?;
    let validate_lib      = load("validate")?;

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
            q4k_fused_dot:          std::mem::transmute(sym(&q4kfg, b"q4k_fused_dot\0")?),
            q4k_fused_dot_4row:     std::mem::transmute(sym(&q4kfg, b"q4k_fused_dot_4row\0")?),
            q6k_dot_q8k:           std::mem::transmute(sym(&q6kd, b"q6k_dot_q8k\0")?),
            q6k_dot_q8k_4row:      std::mem::transmute(sym(&q6kd, b"q6k_dot_q8k_4row\0")?),
            apply_rope_f32:        std::mem::transmute(sym(&rope, b"apply_rope_f32\0")?),
            q4k_gemm_4x4:         std::mem::transmute(sym(&gemm_tile, b"q4k_gemm_4x4\0")?),
            attn_dot_f16:   std::mem::transmute(sym(&attn_f16_lib, b"attn_dot_f16\0")?),
            attn_vsum_f16:  std::mem::transmute(sym(&attn_f16_lib, b"attn_vsum_f16\0")?),
            f32_to_f16:     std::mem::transmute(sym(&f16_conv_lib, b"f32_to_f16\0")?),
            f16_to_f32:     std::mem::transmute(sym(&f16_conv_lib, b"f16_to_f32\0")?),
            softmax_f32:    std::mem::transmute(sym(&softmax_lib, b"softmax_f32\0")?),
            silu_mul_f32:   std::mem::transmute(sym(&silu_lib, b"silu_mul_f32\0")?),
            validate:              std::mem::transmute(sym(&validate_lib, b"q4_validate\0")?),
            libs: {
                let v = vec![
                    i2s, quant, rms, attn, i8d, act, vadd,
                    q4kq, q4kd, q4kfg, q6kd, rope, gemm_tile,
                    attn_f16_lib, f16_conv_lib, softmax_lib, silu_lib,
                    validate_lib,
                ];
                v
            },
        };
        Ok(t)
    }
}

// Public wrappers — inference

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
    src: *const f32, dst_qs: *mut i8, dst_d: *mut f32, dst_bsums: *mut i16, n: i32,
) {
    (k().quant_f32_q8k)(src, dst_qs, dst_d, dst_bsums, n)
}

pub unsafe fn q4k_dot_q8k(
    q4: *const u8, q8: *const i8, bsums: *const i16,
    n_blocks: i32, q8_d: *const f32, pow2: *const f32,
) -> f32 {
    (k().q4k_dot_q8k)(q4, q8, bsums, n_blocks, q8_d, pow2)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn q4k_dot_q8k_4row(
    rw0: *const u8, rw1: *const u8, rw2: *const u8, rw3: *const u8,
    q8: *const i8, bsums: *const i16,
    scores: *mut f32, n_blocks: i32, q8_d: *const f32, pow2: *const f32,
) {
    (k().q4k_dot_q8k_4row)(rw0, rw1, rw2, rw3, q8, bsums, scores, n_blocks, q8_d, pow2)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn q4k_dot_q8k_4row_dual(
    gw0: *const u8, gw1: *const u8, gw2: *const u8, gw3: *const u8,
    uw0: *const u8, uw1: *const u8, uw2: *const u8, uw3: *const u8,
    q8: *const i8, bsums: *const i16,
    gate_scores: *mut f32, up_scores: *mut f32, n_blocks: i32,
    q8_d: *const f32, pow2: *const f32,
) {
    (k().q4k_dot_q8k_4row_dual)(
        gw0, gw1, gw2, gw3, uw0, uw1, uw2, uw3,
        q8, bsums, gate_scores, up_scores, n_blocks, q8_d, pow2)
}

pub unsafe fn q4k_fused_dot(
    q4: *const u8, act: *const f32,
    n_blocks: i32, d_w: *const f32, dm_w: *const f32,
    scratch: *mut f32, bs: *mut i16,
) -> f32 {
    (k().q4k_fused_dot)(q4, act, n_blocks, d_w, dm_w, scratch, bs)
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub unsafe fn q4k_fused_dot_4row(
    rw0: *const u8, rw1: *const u8, rw2: *const u8, rw3: *const u8,
    act: *const f32,
    scores: *mut f32, n_blocks: i32,
    d0_w: *const f32, d1_w: *const f32, d2_w: *const f32, d3_w: *const f32,
    dm0_w: *const f32, dm1_w: *const f32, dm2_w: *const f32, dm3_w: *const f32,
    scratch: *mut f32, bs: *mut i16,
) {
    (k().q4k_fused_dot_4row)(
        rw0, rw1, rw2, rw3, act,
        scores, n_blocks, d0_w, d1_w, d2_w, d3_w, dm0_w, dm1_w, dm2_w, dm3_w,
        scratch, bs)
}

pub unsafe fn q6k_dot_q8k(
    weight: *const u8, q8: *const i8, bsums: *const i16,
    n_blocks: i32, d_arr: *const f32,
) -> f32 {
    (k().q6k_dot_q8k)(weight, q8, bsums, n_blocks, d_arr)
}

pub unsafe fn q6k_dot_q8k_4row(
    w0: *const u8, w1: *const u8, w2: *const u8, w3: *const u8,
    q8: *const i8, bsums: *const i16, scores: *mut f32, n_blocks: i32,
    d0: *const f32, d1: *const f32, d2: *const f32, d3: *const f32,
) {
    (k().q6k_dot_q8k_4row)(
        w0, w1, w2, w3, q8, bsums, scores, n_blocks, d0, d1, d2, d3)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn q4k_gemm_4x4(
    rw0: *const u8, rw1: *const u8, rw2: *const u8, rw3: *const u8,
    q8_0: *const i8, q8_1: *const i8, q8_2: *const i8, q8_3: *const i8,
    bs0: *const i16, bs1: *const i16, bs2: *const i16, bs3: *const i16,
    sc0: *const u8, sc1: *const u8, sc2: *const u8, sc3: *const u8,
    mn0: *const u8, mn1: *const u8, mn2: *const u8, mn3: *const u8,
    d0: *const f32, d1: *const f32, d2: *const f32, d3: *const f32,
    dm0: *const f32, dm1: *const f32, dm2: *const f32, dm3: *const f32,
    q8d0: *const f32, q8d1: *const f32, q8d2: *const f32, q8d3: *const f32,
    scores: *mut f32, n_blocks: i32,
) {
    (k().q4k_gemm_4x4)(
        rw0, rw1, rw2, rw3,
        q8_0, q8_1, q8_2, q8_3,
        bs0, bs1, bs2, bs3,
        sc0, sc1, sc2, sc3,
        mn0, mn1, mn2, mn3,
        d0, d1, d2, d3,
        dm0, dm1, dm2, dm3,
        q8d0, q8d1, q8d2, q8d3,
        scores, n_blocks,
    );
}

pub unsafe fn apply_rope_f32(
    data: *const f32, freqs: *const f32, out: *mut f32, head_dim: i32, n_heads: i32,
) {
    (k().apply_rope_f32)(data, freqs, out, head_dim, n_heads)
}

// Public wrappers — f16 attention + utilities

pub unsafe fn attn_dot_f16(
    query: *const f32, k_cache: *const u16, scores_out: *mut f32,
    seq_len: i32, head_dim: i32,
) { (k().attn_dot_f16)(query, k_cache, scores_out, seq_len, head_dim) }

pub unsafe fn attn_vsum_f16(
    weights: *const f32, v_cache: *const u16, out: *mut f32,
    seq_len: i32, head_dim: i32,
) { (k().attn_vsum_f16)(weights, v_cache, out, seq_len, head_dim) }

pub unsafe fn f32_to_f16(src: *const f32, dst: *mut u16, n: i32) {
    (k().f32_to_f16)(src, dst, n)
}

pub unsafe fn f16_to_f32(src: *const u16, dst: *mut f32, n: i32) {
    (k().f16_to_f32)(src, dst, n)
}

pub unsafe fn softmax_f32(data: *mut f32, n: i32, scale: f32) {
    (k().softmax_f32)(data, n, scale)
}

pub unsafe fn silu_mul_f32(gate: *const f32, up: *const f32, out: *mut f32, n: i32) {
    (k().silu_mul_f32)(gate, up, out, n)
}

pub unsafe fn validate(
    scales: *const f32, biases: *const f32,
    scales_bits: *const i32, biases_bits: *const i32, n_groups: i32,
) -> i32 { (k().validate)(scales, biases, scales_bits, biases_bits, n_groups) }

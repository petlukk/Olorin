//! FFI layer for Ea SIMD kernels — inference.

use libloading::{Library, Symbol};
use std::path::Path;
use std::sync::OnceLock;
use crate::kernels::ffi_inference_types::*;

pub struct KernelTableInference {
    pub libs: Vec<Library>,
    pub quant_f32_q8k:         QuantF32Q8kFn,
    pub q4k_dot_q8k:           Q4kDotQ8kFn,
    pub q4k_dot_q8k_4row:      Q4kDot4RowFn,
    pub q4k_dot_q8k_4row_dual: Q4kDot4RowDualFn,
    pub q5k_dot_q8k:           Q5kDotQ8kFn,
    pub q5k_dot_q8k_4row:      Q5kDot4RowFn,
    pub q6k_dot_q8k:           Q6kDotQ8kFn,
    pub q6k_dot_q8k_4row:      Q6kDot4RowFn,
    pub f32_to_f16:            F32ToF16Fn,
    pub f16_to_f32:            F16ToF32Fn,
    pub softmax_f32:           SoftmaxF32Fn,
    pub gemma4_rmsnorm:        Gemma4RmsnormFn,
    pub gelu_mul:              GeluMulFn,
    pub gemma4_rope:           Gemma4RopeFn,
    pub bf16_dot_f32:          Bf16DotF32Fn,
    pub bf16_dot_f32_4row:     Bf16Dot4RowFn,
    pub vec_add_f32:           VecAddF32Fn,
    pub vec_scale_f32:         VecScaleF32Fn,
    pub vec_fma_f32:           VecFmaF32Fn,
    pub vec_acc_f32:           VecAccF32Fn,
    pub f32_dot:               F32DotFn,
    pub f32_dot_acc:           F32DotAccFn,
    pub bare_rmsnorm_f32:      BareRmsnormF32Fn,
    pub softcap_f32:           SoftcapF32Fn,
    pub q4k_repack_8x8:          Q4kRepack8x8Fn,
    pub q4k_8x8_q8k_matvec:      Q4k8x8MatvecFn,
    pub q4k_8x8_q8k_matvec_dual: Q4k8x8MatvecDualFn,
    pub q8k_repack_4:            Q8kRepack4Fn,
    pub q4k_8x8_q8k_gemm:       Q4k8x8GemmFn,
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

    // Try I8MM variant first on ARM, fall back to base
    #[allow(unused_variables)]
    let load_best = |name: &str| -> Result<Library, String> {
        #[cfg(target_arch = "aarch64")]
        {
            // HWCAP2 = 26, HWCAP2_I8MM = (1 << 13)
            let has_i8mm = unsafe { libc::getauxval(26) & (1 << 13) != 0 };
            if has_i8mm {
                let i8mm_name = format!("{name}_i8mm");
                if let Ok(lib) = load(&i8mm_name) {
                    eprintln!("olorin: {name}=i8mm");
                    return Ok(lib);
                }
            }
        }
        load(name)
    };

    let q4kq = load("q4k_quant")?;
    let q4kd = load_best("q4k_dot")?;
    let q5kd = load("q5k_dot")?;
    let q6kd = load("q6k_dot")?;
    let f16_conv_lib = load("f16_convert")?;
    let softmax_lib  = load("softmax")?;
    let gemma4_rmsnorm_lib = load("gemma4_rmsnorm")?;
    let gemma4_gelu_lib    = load("gemma4_gelu")?;
    let gemma4_rope_lib    = load("gemma4_rope")?;
    let bf16_matvec_lib    = load("bf16_matvec")?;
    let vec_ops_lib        = load("vec_ops")?;
    let attn_ops_lib       = load("attn_ops")?;
    let bare_rmsnorm_lib   = load("bare_rmsnorm")?;
    let softcap_lib        = load("softcap")?;
    let q4k_repack_lib       = load("q4k_repack")?;
    let q4k_dot_8x8_lib      = load("q4k_dot_8x8")?;
    let q4k_dot_8x8_dual_lib = load("q4k_dot_8x8_dual")?;
    let q8k_repack_4_lib     = load("q8k_repack_4")?;
    let q4k_dot_8x8_gemm_lib = load("q4k_dot_8x8_gemm")?;

    unsafe {
        let sym = |lib: &Library, name: &[u8]| -> Result<usize, String> {
            let s: Symbol<*const ()> = lib
                .get(name)
                .map_err(|e| format!("symbol {:?}: {e}",
                    std::str::from_utf8(&name[..name.len()-1]).unwrap_or("?")))?;
            Ok(*s as usize)
        };

        let t = KernelTableInference {
            quant_f32_q8k:         std::mem::transmute(sym(&q4kq, b"quant_f32_q8k\0")?),
            q4k_dot_q8k:           std::mem::transmute(sym(&q4kd, b"q4k_dot_q8k\0")?),
            q4k_dot_q8k_4row:      std::mem::transmute(sym(&q4kd, b"q4k_dot_q8k_4row\0")?),
            q4k_dot_q8k_4row_dual: std::mem::transmute(sym(&q4kd, b"q4k_dot_q8k_4row_dual\0")?),
            q5k_dot_q8k:           std::mem::transmute(sym(&q5kd, b"q5k_dot_q8k\0")?),
            q5k_dot_q8k_4row:      std::mem::transmute(sym(&q5kd, b"q5k_dot_q8k_4row\0")?),
            q6k_dot_q8k:           std::mem::transmute(sym(&q6kd, b"q6k_dot_q8k\0")?),
            q6k_dot_q8k_4row:      std::mem::transmute(sym(&q6kd, b"q6k_dot_q8k_4row\0")?),
            f32_to_f16:     std::mem::transmute(sym(&f16_conv_lib, b"f32_to_f16\0")?),
            f16_to_f32:     std::mem::transmute(sym(&f16_conv_lib, b"f16_to_f32\0")?),
            softmax_f32:    std::mem::transmute(sym(&softmax_lib, b"softmax_f32\0")?),
            gemma4_rmsnorm: std::mem::transmute(sym(&gemma4_rmsnorm_lib, b"gemma4_rmsnorm\0")?),
            gelu_mul:       std::mem::transmute(sym(&gemma4_gelu_lib, b"gelu_mul\0")?),
            gemma4_rope:    std::mem::transmute(sym(&gemma4_rope_lib, b"gemma4_rope\0")?),
            bf16_dot_f32:      std::mem::transmute(sym(&bf16_matvec_lib, b"bf16_dot_f32\0")?),
            bf16_dot_f32_4row: std::mem::transmute(sym(&bf16_matvec_lib, b"bf16_dot_f32_4row\0")?),
            vec_add_f32:       std::mem::transmute(sym(&vec_ops_lib, b"vec_add_f32\0")?),
            vec_scale_f32:     std::mem::transmute(sym(&vec_ops_lib, b"vec_scale_f32\0")?),
            vec_fma_f32:       std::mem::transmute(sym(&vec_ops_lib, b"vec_fma_f32\0")?),
            vec_acc_f32:       std::mem::transmute(sym(&vec_ops_lib, b"vec_acc_f32\0")?),
            f32_dot:           std::mem::transmute(sym(&attn_ops_lib, b"f32_dot\0")?),
            f32_dot_acc:       std::mem::transmute(sym(&attn_ops_lib, b"f32_dot_acc\0")?),
            bare_rmsnorm_f32:  std::mem::transmute(sym(&bare_rmsnorm_lib, b"bare_rmsnorm_f32\0")?),
            softcap_f32:       std::mem::transmute(sym(&softcap_lib, b"softcap_f32\0")?),
            q4k_repack_8x8:          std::mem::transmute(sym(&q4k_repack_lib,       b"q4k_repack_8x8\0")?),
            q4k_8x8_q8k_matvec:      std::mem::transmute(sym(&q4k_dot_8x8_lib,      b"q4k_8x8_q8k_matvec\0")?),
            q4k_8x8_q8k_matvec_dual: std::mem::transmute(sym(&q4k_dot_8x8_dual_lib, b"q4k_8x8_q8k_matvec_dual\0")?),
            q8k_repack_4:            std::mem::transmute(sym(&q8k_repack_4_lib,     b"q8k_repack_4\0")?),
            q4k_8x8_q8k_gemm:       std::mem::transmute(sym(&q4k_dot_8x8_gemm_lib, b"q4k_8x8_q8k_gemm\0")?),
            libs: vec![q4kq, q4kd, q5kd, q6kd, f16_conv_lib, softmax_lib, gemma4_rmsnorm_lib, gemma4_gelu_lib, gemma4_rope_lib, bf16_matvec_lib, vec_ops_lib, attn_ops_lib, bare_rmsnorm_lib, softcap_lib, q4k_repack_lib, q4k_dot_8x8_lib, q4k_dot_8x8_dual_lib, q8k_repack_4_lib, q4k_dot_8x8_gemm_lib],
        };
        Ok(t)
    }
}

// Public wrappers

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

pub unsafe fn q5k_dot_q8k(
    q5: *const u8, q8: *const i8, bsums: *const i16,
    n_blocks: i32, q8_d: *const f32, pow2: *const f32,
) -> f32 {
    (k().q5k_dot_q8k)(q5, q8, bsums, n_blocks, q8_d, pow2)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn q5k_dot_q8k_4row(
    rw0: *const u8, rw1: *const u8, rw2: *const u8, rw3: *const u8,
    q8: *const i8, bsums: *const i16,
    scores: *mut f32, n_blocks: i32, q8_d: *const f32, pow2: *const f32,
) {
    (k().q5k_dot_q8k_4row)(rw0, rw1, rw2, rw3, q8, bsums, scores, n_blocks, q8_d, pow2)
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

pub unsafe fn f16_to_f32(src: *const u16, dst: *mut f32, n: i32) {
    (k().f16_to_f32)(src, dst, n)
}

pub unsafe fn softmax_f32(data: *mut f32, n: i32, scale: f32) {
    (k().softmax_f32)(data, n, scale)
}

pub fn gemma4_rmsnorm(x: *const f32, weight: *const f32, out: *mut f32, n: i32, eps: f32) {
    unsafe { (k().gemma4_rmsnorm)(x, weight, out, n, eps) }
}

pub fn gelu_mul(gate: *const f32, up: *const f32, out: *mut f32, n: i32) {
    unsafe { (k().gelu_mul)(gate, up, out, n) }
}

pub fn gemma4_rope(data: *mut f32, cos_table: *const f32, sin_table: *const f32, head_dim: i32, n_heads: i32) {
    unsafe { (k().gemma4_rope)(data, cos_table, sin_table, head_dim, n_heads) }
}

pub unsafe fn bf16_dot_f32(
    weight: *const u16, input: *const f32, scratch: *mut i32, n_cols: i32,
) -> f32 {
    (k().bf16_dot_f32)(weight, input, scratch, n_cols)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn bf16_dot_f32_4row(
    w0: *const u16, w1: *const u16, w2: *const u16, w3: *const u16,
    input: *const f32, scores: *mut f32, scratch: *mut i32, n_cols: i32,
) {
    (k().bf16_dot_f32_4row)(w0, w1, w2, w3, input, scores, scratch, n_cols)
}

pub fn vec_add_f32(a: *const f32, b: *const f32, out: *mut f32, n: i32) {
    unsafe { (k().vec_add_f32)(a, b, out, n) }
}

pub fn vec_scale_f32(a: *const f32, out: *mut f32, s: f32, n: i32) {
    unsafe { (k().vec_scale_f32)(a, out, s, n) }
}

pub fn vec_fma_f32(a: *const f32, b: *const f32, out: *mut f32, s: f32, n: i32) {
    unsafe { (k().vec_fma_f32)(a, b, out, s, n) }
}

pub fn f32_dot(a: *const f32, b: *const f32, n: i32) -> f32 {
    unsafe { (k().f32_dot)(a, b, n) }
}

pub fn f32_dot_acc(out: *mut f32, a: *const f32, s: f32, n: i32) {
    unsafe { (k().f32_dot_acc)(out, a, s, n) }
}

pub fn bare_rmsnorm_f32(x: *mut f32, n: i32, eps: f32) {
    unsafe { (k().bare_rmsnorm_f32)(x, n, eps) }
}

pub fn softcap_f32(data: *mut f32, n: i32, cap: f32) {
    unsafe { (k().softcap_f32)(data, n, cap) }
}

pub unsafe fn q4k_repack_8x8(
    src: *const u8, dst: *mut u8, n_rows: i32, n_cols: i32,
) {
    (k().q4k_repack_8x8)(src, dst, n_rows, n_cols)
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn q4k_8x8_q8k_matvec(
    packed: *const u8,
    q8_qs: *const i8,
    q8_d: *const f32,
    q8_bsums: *const i16,
    pow2: *const f32,
    scratch: *mut u8,
    out: *mut f32,
    n_rows: i32,
    n_cols: i32,
) {
    (k().q4k_8x8_q8k_matvec)(
        packed, q8_qs, q8_d, q8_bsums, pow2, scratch, out, n_rows, n_cols,
    )
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn q4k_8x8_q8k_matvec_dual(
    packed_a: *const u8,
    packed_b: *const u8,
    q8_qs: *const i8,
    q8_d: *const f32,
    q8_bsums: *const i16,
    pow2: *const f32,
    scratch: *mut u8,
    out_a: *mut f32,
    out_b: *mut f32,
    n_rows: i32,
    n_cols: i32,
) {
    (k().q4k_8x8_q8k_matvec_dual)(
        packed_a, packed_b, q8_qs, q8_d, q8_bsums, pow2, scratch,
        out_a, out_b, n_rows, n_cols,
    )
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn q8k_repack_4(
    row0_qs:    *const i8,
    row1_qs:    *const i8,
    row2_qs:    *const i8,
    row3_qs:    *const i8,
    row_d:      *const f32,
    row0_bsums: *const i16,
    row1_bsums: *const i16,
    row2_bsums: *const i16,
    row3_bsums: *const i16,
    dst:        *mut u8,
    nb:         i32,
) {
    (k().q8k_repack_4)(
        row0_qs, row1_qs, row2_qs, row3_qs,
        row_d,
        row0_bsums, row1_bsums, row2_bsums, row3_bsums,
        dst, nb,
    )
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn q4k_8x8_q8k_gemm(
    packed: *const u8,
    q8_a: *const u8,
    scratch: *mut u8,
    out: *mut f32,
    bs: i32,
    n: i32,
    nr: i32,
    nc: i32,
) {
    (k().q4k_8x8_q8k_gemm)(packed, q8_a, scratch, out, bs, n, nr, nc)
}

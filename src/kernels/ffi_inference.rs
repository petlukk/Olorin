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
    pub q6k_dot_q8k:           Q6kDotQ8kFn,
    pub q6k_dot_q8k_4row:      Q6kDot4RowFn,
    pub f32_to_f16:            F32ToF16Fn,
    pub f16_to_f32:            F16ToF32Fn,
    pub softmax_f32:           SoftmaxF32Fn,
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
    let q6kd = load("q6k_dot")?;
    let f16_conv_lib = load("f16_convert")?;
    let softmax_lib  = load("softmax")?;

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
            q6k_dot_q8k:           std::mem::transmute(sym(&q6kd, b"q6k_dot_q8k\0")?),
            q6k_dot_q8k_4row:      std::mem::transmute(sym(&q6kd, b"q6k_dot_q8k_4row\0")?),
            f32_to_f16:     std::mem::transmute(sym(&f16_conv_lib, b"f32_to_f16\0")?),
            f16_to_f32:     std::mem::transmute(sym(&f16_conv_lib, b"f16_to_f32\0")?),
            softmax_f32:    std::mem::transmute(sym(&softmax_lib, b"softmax_f32\0")?),
            libs: vec![q4kq, q4kd, q6kd, f16_conv_lib, softmax_lib],
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

pub unsafe fn f32_to_f16(src: *const f32, dst: *mut u16, n: i32) {
    (k().f32_to_f16)(src, dst, n)
}

pub unsafe fn f16_to_f32(src: *const u16, dst: *mut f32, n: i32) {
    (k().f16_to_f32)(src, dst, n)
}

pub unsafe fn softmax_f32(data: *mut f32, n: i32, scale: f32) {
    (k().softmax_f32)(data, n, scale)
}

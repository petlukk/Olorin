//! FFI layer for Ea SIMD kernels — inference.

use libloading::{Library, Symbol};
use std::path::Path;
use std::sync::OnceLock;
use crate::kernels::ffi_inference_types::*;

mod wrappers;
pub use wrappers::*;

pub struct KernelTableInference {
    pub libs: Vec<Library>,
    pub quant_f32_q8k:         QuantF32Q8kFn,
    pub q4k_dot_q8k:           Q4kDotQ8kFn,
    pub q4k_dot_q8k_4row:      Q4kDot4RowFn,
    pub q4k_dot_q8k_4row_dual: Q4kDot4RowDualFn,
    pub q3k_dot_q8k:           Q3kDotQ8kFn,
    pub q5k_dot_q8k:           Q5kDotQ8kFn,
    pub q5k_dot_q8k_4row:      Q5kDot4RowFn,
    pub q6k_dot_q8k:           Q6kDotQ8kFn,
    pub q6k_dot_q8k_4row:          Q6kDot4RowFn,
    pub q6k_dot_q8k_4row_repacked: Q6kDot4RowRepackedFn,
    pub f32_to_f16:                F32ToF16Fn,
    pub gemma4_rmsnorm:        Gemma4RmsnormFn,
    pub gelu_mul:              GeluMulFn,
    pub gemma4_rope:           Gemma4RopeFn,
    pub bf16_dot_f32:          Bf16DotF32Fn,
    pub bf16_dot_f32_4row:     Bf16Dot4RowFn,
    pub bf16_dot_multi_input:  Bf16DotMultiInputFn,
    pub vec_add_f32:           VecAddF32Fn,
    pub vec_scale_f32:         VecScaleF32Fn,
    pub vec_fma_f32:           VecFmaF32Fn,
    pub bare_rmsnorm_f32:      BareRmsnormF32Fn,
    pub softcap_f32:           SoftcapF32Fn,
    pub q4k_repack_8x8:          Q4kRepack8x8Fn,
    pub q5k_repack_8x8:          Q5kRepack8x8Fn,
    pub q3k_repack_8x8:          Q3kRepack8x8Fn,
    pub q4k_8x8_q8k_matvec:      Q4k8x8MatvecFn,
    pub q4k_8x8_q8k_matvec_dual: Q4k8x8MatvecDualFn,
    pub q8k_repack_4:            Q8kRepack4Fn,
    pub q4k_8x8_q8k_gemm:       Q4k8x8GemmFn,
    #[cfg(target_arch = "aarch64")]
    pub q6k_gemm:               Q6kGemmFn,
    #[cfg(target_arch = "aarch64")]
    pub q5k_gemm:               Q5kGemmFn,
    #[cfg(target_arch = "aarch64")]
    pub q5k_8x8_q8k_matvec:     Q4k8x8MatvecFn,
    #[cfg(target_arch = "aarch64")]
    pub q5k_8x8_q8k_gemm:       Q4k8x8GemmFn,
    #[cfg(target_arch = "aarch64")]
    pub q3k_8x8_q8k_gemm:       Q4k8x8GemmFn,
    pub attn_fused_batched:      AttnFusedBatchedFn,
}

// SAFETY: KernelTableInference holds function pointers and library handles.
// Function pointers are valid for the lifetime of the libraries.
// Libraries are never unloaded (held in OnceLock for program lifetime).
unsafe impl Send for KernelTableInference {}
unsafe impl Sync for KernelTableInference {}

static KERNELS: OnceLock<KernelTableInference> = OnceLock::new();

pub(super) fn k() -> &'static KernelTableInference {
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
    let q3kd = load("q3k_dot")?;
    let q5kd = load("q5k_dot")?;
    let q6kd = load("q6k_dot")?;
    let q6k_dot_repacked_lib = load("q6k_dot_repacked")?;
    let f16_conv_lib = load("f16_convert")?;
    let gemma4_rmsnorm_lib = load("gemma4_rmsnorm")?;
    let gemma4_gelu_lib    = load("gemma4_gelu")?;
    let gemma4_rope_lib    = load("gemma4_rope")?;
    let bf16_matvec_lib    = load("bf16_matvec")?;
    let vec_ops_lib        = load("vec_ops")?;
    let bare_rmsnorm_lib   = load("bare_rmsnorm")?;
    let softcap_lib        = load("softcap")?;
    let q4k_repack_lib       = load("q4k_repack")?;
    let q5k_repack_lib       = load("q5k_repack")?;
    let q3k_repack_lib       = load("q3k_repack")?;
    let q4k_dot_8x8_lib      = load("q4k_dot_8x8")?;
    let q4k_dot_8x8_dual_lib = load("q4k_dot_8x8_dual")?;
    let q8k_repack_4_lib     = load("q8k_repack_4")?;
    let q4k_dot_8x8_gemm_lib = load("q4k_dot_8x8_gemm")?;
    #[cfg(target_arch = "aarch64")]
    let q6k_gemm_lib         = load("q6k_gemm")?;
    #[cfg(target_arch = "aarch64")]
    let q5k_gemm_lib         = load("q5k_gemm")?;
    #[cfg(target_arch = "aarch64")]
    let q5k_dot_8x8_lib      = load("q5k_dot_8x8")?;
    #[cfg(target_arch = "aarch64")]
    let q5k_dot_8x8_gemm_lib = load("q5k_dot_8x8_gemm")?;
    #[cfg(target_arch = "aarch64")]
    let q3k_dot_8x8_gemm_lib = load("q3k_dot_8x8_gemm")?;
    let attn_fused_batched_lib = load("attn_fused_batched")?;

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
            q3k_dot_q8k:           std::mem::transmute(sym(&q3kd, b"q3k_dot_q8k\0")?),
            q5k_dot_q8k:           std::mem::transmute(sym(&q5kd, b"q5k_dot_q8k\0")?),
            q5k_dot_q8k_4row:      std::mem::transmute(sym(&q5kd, b"q5k_dot_q8k_4row\0")?),
            q6k_dot_q8k:               std::mem::transmute(sym(&q6kd, b"q6k_dot_q8k\0")?),
            q6k_dot_q8k_4row:          std::mem::transmute(sym(&q6kd, b"q6k_dot_q8k_4row\0")?),
            q6k_dot_q8k_4row_repacked: std::mem::transmute(sym(&q6k_dot_repacked_lib, b"q6k_dot_q8k_4row_repacked\0")?),
            f32_to_f16:     std::mem::transmute(sym(&f16_conv_lib, b"f32_to_f16\0")?),
            gemma4_rmsnorm: std::mem::transmute(sym(&gemma4_rmsnorm_lib, b"gemma4_rmsnorm\0")?),
            gelu_mul:       std::mem::transmute(sym(&gemma4_gelu_lib, b"gelu_mul\0")?),
            gemma4_rope:    std::mem::transmute(sym(&gemma4_rope_lib, b"gemma4_rope\0")?),
            bf16_dot_f32:      std::mem::transmute(sym(&bf16_matvec_lib, b"bf16_dot_f32\0")?),
            bf16_dot_f32_4row: std::mem::transmute(sym(&bf16_matvec_lib, b"bf16_dot_f32_4row\0")?),
            bf16_dot_multi_input: std::mem::transmute(sym(&bf16_matvec_lib, b"bf16_dot_multi_input\0")?),
            vec_add_f32:       std::mem::transmute(sym(&vec_ops_lib, b"vec_add_f32\0")?),
            vec_scale_f32:     std::mem::transmute(sym(&vec_ops_lib, b"vec_scale_f32\0")?),
            vec_fma_f32:       std::mem::transmute(sym(&vec_ops_lib, b"vec_fma_f32\0")?),
            bare_rmsnorm_f32:  std::mem::transmute(sym(&bare_rmsnorm_lib, b"bare_rmsnorm_f32\0")?),
            softcap_f32:       std::mem::transmute(sym(&softcap_lib, b"softcap_f32\0")?),
            q4k_repack_8x8:          std::mem::transmute(sym(&q4k_repack_lib,       b"q4k_repack_8x8\0")?),
            q5k_repack_8x8:          std::mem::transmute(sym(&q5k_repack_lib,       b"q5k_repack_8x8\0")?),
            q3k_repack_8x8:          std::mem::transmute(sym(&q3k_repack_lib,       b"q3k_repack_8x8\0")?),
            q4k_8x8_q8k_matvec:      std::mem::transmute(sym(&q4k_dot_8x8_lib,      b"q4k_8x8_q8k_matvec\0")?),
            q4k_8x8_q8k_matvec_dual: std::mem::transmute(sym(&q4k_dot_8x8_dual_lib, b"q4k_8x8_q8k_matvec_dual\0")?),
            q8k_repack_4:            std::mem::transmute(sym(&q8k_repack_4_lib,     b"q8k_repack_4\0")?),
            q4k_8x8_q8k_gemm:       std::mem::transmute(sym(&q4k_dot_8x8_gemm_lib, b"q4k_8x8_q8k_gemm\0")?),
            #[cfg(target_arch = "aarch64")]
            q6k_gemm:               std::mem::transmute(sym(&q6k_gemm_lib, b"q6k_gemm\0")?),
            #[cfg(target_arch = "aarch64")]
            q5k_gemm:               std::mem::transmute(sym(&q5k_gemm_lib, b"q5k_gemm\0")?),
            #[cfg(target_arch = "aarch64")]
            q5k_8x8_q8k_matvec:     std::mem::transmute(sym(&q5k_dot_8x8_lib, b"q5k_8x8_q8k_matvec\0")?),
            #[cfg(target_arch = "aarch64")]
            q5k_8x8_q8k_gemm:       std::mem::transmute(sym(&q5k_dot_8x8_gemm_lib, b"q5k_8x8_q8k_gemm\0")?),
            #[cfg(target_arch = "aarch64")]
            q3k_8x8_q8k_gemm:       std::mem::transmute(sym(&q3k_dot_8x8_gemm_lib, b"q3k_8x8_q8k_gemm\0")?),
            attn_fused_batched:      std::mem::transmute(sym(&attn_fused_batched_lib, b"attn_fused_batched\0")?),
            libs: {
                #[allow(unused_mut)]
                let mut libs = vec![q4kq, q4kd, q3kd, q5kd, q6kd, q6k_dot_repacked_lib, f16_conv_lib, gemma4_rmsnorm_lib, gemma4_gelu_lib, gemma4_rope_lib, bf16_matvec_lib, vec_ops_lib, bare_rmsnorm_lib, softcap_lib, q4k_repack_lib, q5k_repack_lib, q3k_repack_lib, q4k_dot_8x8_lib, q4k_dot_8x8_dual_lib, q8k_repack_4_lib, q4k_dot_8x8_gemm_lib, attn_fused_batched_lib];
                #[cfg(target_arch = "aarch64")]
                libs.push(q6k_gemm_lib);
                #[cfg(target_arch = "aarch64")]
                libs.push(q5k_gemm_lib);
                #[cfg(target_arch = "aarch64")]
                libs.push(q5k_dot_8x8_lib);
                #[cfg(target_arch = "aarch64")]
                libs.push(q5k_dot_8x8_gemm_lib);
                #[cfg(target_arch = "aarch64")]
                libs.push(q3k_dot_8x8_gemm_lib);
                libs
            },
        };
        Ok(t)
    }
}

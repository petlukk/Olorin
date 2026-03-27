//! Runtime kernel loading for eakv SIMD kernels.
//!
//! Loads pre-compiled .so files from ~/.olorin/lib/ via libloading.
//! Kernels are compiled by olorin-cli's unified build.rs and extracted
//! on first run.

use libloading::{Library, Symbol};
use std::path::{Path, PathBuf};

// ── Type aliases for kernel function signatures ──

/// q4_quantize_split_f32(src, weights_out, scales_out, biases_out, n_groups)
pub type QuantizeFn = unsafe extern "C" fn(
    *const f32, *mut i32, *mut f32, *mut f32, i32,
);

/// turbo_rotate(vec, signs, dim)
pub type RotateFn = unsafe extern "C" fn(*mut f32, *const f32, i32);

/// fwht_inplace(vec, dim)
pub type FwhtFn = unsafe extern "C" fn(*mut f32, i32);

/// sign_flip(vec, signs, dim)
pub type SignFlipFn = unsafe extern "C" fn(*mut f32, *const f32, i32);

/// q4_fused_k_score_multi_f32 / q4_fused_k_score_multi_64_f32
/// (q_vecs, k_packed, k_scales, k_biases, all_scores, seq_len, n_heads, groups_per_head)
pub type KScoreMhaFn = unsafe extern "C" fn(
    *const f32, *const u8, *const f32, *const f32,
    *mut f32, i32, i32, i32,
);

/// q4_k_score_gqa_f32 / q4_k_score_gqa_64_f32
/// (q_vecs, k_packed, k_scales, k_biases, all_scores, seq_len, n_q_heads, n_kv_heads, groups_per_head)
pub type KScoreGqaFn = unsafe extern "C" fn(
    *const f32, *const u8, *const f32, *const f32,
    *mut f32, i32, i32, i32, i32,
);

/// q4_fused_v_sum_multi_f32 / q4_fused_v_sum_multi_64_f32
/// (all_weights, v_packed, v_scales, v_biases, all_out, seq_len, n_heads, groups_per_head)
pub type VSumMhaFn = unsafe extern "C" fn(
    *const f32, *const u8, *const f32, *const f32,
    *mut f32, i32, i32, i32,
);

/// q4_v_sum_gqa_f32 / q4_v_sum_gqa_64_f32
/// (all_weights, v_packed, v_scales, v_biases, all_out, seq_len, n_q_heads, n_kv_heads, groups_per_head)
pub type VSumGqaFn = unsafe extern "C" fn(
    *const f32, *const u8, *const f32, *const f32,
    *mut f32, i32, i32, i32, i32,
);

// ── Kernel table ──

pub struct KernelTable {
    pub quantize: QuantizeFn,
    pub rotate: RotateFn,
    pub fwht: FwhtFn,
    pub sign_flip: SignFlipFn,
    pub k_score_mha: KScoreMhaFn,
    pub k_score_mha_64: KScoreMhaFn,
    pub k_score_gqa: KScoreGqaFn,
    pub k_score_gqa_64: KScoreGqaFn,
    pub v_sum_mha: VSumMhaFn,
    pub v_sum_mha_64: VSumMhaFn,
    pub v_sum_gqa: VSumGqaFn,
    pub v_sum_gqa_64: VSumGqaFn,
    /// Keeps libraries alive for the lifetime of the table.
    _libs: Vec<Library>,
}

// SAFETY: KernelTable holds function pointers and library handles.
// The function pointers are valid for the lifetime of the libraries.
// Libraries are never unloaded while the table exists.
unsafe impl Send for KernelTable {}
unsafe impl Sync for KernelTable {}

// ── Discovery ──

/// Search ~/.olorin/lib/ for a directory containing libquantize_simd.so.
pub fn find_kernel_dir() -> Result<PathBuf, String> {
    let home = std::env::var("HOME")
        .map_err(|_| "HOME not set".to_string())?;
    let lib_root = PathBuf::from(&home).join(".olorin").join("lib");

    if !lib_root.is_dir() {
        return Err(format!("{} does not exist", lib_root.display()));
    }

    let entries = std::fs::read_dir(&lib_root)
        .map_err(|e| format!("cannot read {}: {e}", lib_root.display()))?;

    // Sort by name descending so we prefer the latest version
    let mut dirs: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    dirs.reverse();

    for dir in &dirs {
        if dir.join("libquantize_simd.so").exists() {
            return Ok(dir.clone());
        }
    }

    Err(format!(
        "no directory in {} contains libquantize_simd.so",
        lib_root.display()
    ))
}

/// Search all dirs in ~/.olorin/lib/ for a specific .so file.
fn find_lib_in_any_dir(filename: &str) -> Result<PathBuf, String> {
    let home = std::env::var("HOME")
        .map_err(|_| "HOME not set".to_string())?;
    let lib_root = PathBuf::from(&home).join(".olorin").join("lib");

    if !lib_root.is_dir() {
        return Err(format!("{} does not exist", lib_root.display()));
    }

    let entries = std::fs::read_dir(&lib_root)
        .map_err(|e| format!("cannot read {}: {e}", lib_root.display()))?;

    let mut dirs: Vec<PathBuf> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    dirs.reverse();

    for dir in &dirs {
        let candidate = dir.join(filename);
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    Err(format!(
        "cannot find {} in any subdirectory of {}",
        filename,
        lib_root.display()
    ))
}

// ── Loading ──

impl KernelTable {
    /// Load all eakv kernels from the given directory.
    /// For turbo_rotate.so, searches all ~/.olorin/lib/ dirs if not
    /// found in `lib_dir`.
    pub fn load(lib_dir: &Path) -> Result<Self, String> {
        let load_lib = |path: &Path| -> Result<Library, String> {
            unsafe {
                Library::new(path)
                    .map_err(|e| format!("failed to load {}: {e}", path.display()))
            }
        };

        let load_from_dir = |dir: &Path, name: &str| -> Result<Library, String> {
            load_lib(&dir.join(format!("lib{name}.so")))
        };

        // eakv-specific kernels — must be in lib_dir
        let quantize_lib = load_from_dir(lib_dir, "quantize_simd")?;
        let k_score_lib = load_from_dir(lib_dir, "fused_k_score")?;
        let k_score_64_lib = load_from_dir(lib_dir, "fused_k_score_64")?;
        let k_score_gqa_lib = load_from_dir(lib_dir, "fused_k_score_gqa")?;
        let k_score_gqa_64_lib = load_from_dir(lib_dir, "fused_k_score_gqa_64")?;
        let v_sum_lib = load_from_dir(lib_dir, "fused_v_sum")?;
        let v_sum_64_lib = load_from_dir(lib_dir, "fused_v_sum_64")?;

        // turbo_rotate may live in a different directory (olorin-core's kernel dir)
        let turbo_path = {
            let local = lib_dir.join("libturbo_rotate.so");
            if local.exists() {
                local
            } else {
                find_lib_in_any_dir("libturbo_rotate.so")?
            }
        };
        let turbo_lib = load_lib(&turbo_path)?;

        unsafe {
            let sym = |lib: &Library, name: &[u8]| -> Result<usize, String> {
                let s: Symbol<*const ()> = lib
                    .get(name)
                    .map_err(|e| {
                        format!(
                            "symbol {:?}: {e}",
                            std::str::from_utf8(&name[..name.len() - 1])
                                .unwrap_or("?")
                        )
                    })?;
                Ok(*s as usize)
            };

            let table = KernelTable {
                quantize: std::mem::transmute(
                    sym(&quantize_lib, b"q4_quantize_split_f32\0")?,
                ),
                rotate: std::mem::transmute(
                    sym(&turbo_lib, b"turbo_rotate\0")?,
                ),
                fwht: std::mem::transmute(
                    sym(&turbo_lib, b"fwht_inplace\0")?,
                ),
                sign_flip: std::mem::transmute(
                    sym(&turbo_lib, b"sign_flip\0")?,
                ),
                k_score_mha: std::mem::transmute(
                    sym(&k_score_lib, b"q4_fused_k_score_multi_f32\0")?,
                ),
                k_score_mha_64: std::mem::transmute(
                    sym(&k_score_64_lib, b"q4_fused_k_score_multi_64_f32\0")?,
                ),
                k_score_gqa: std::mem::transmute(
                    sym(&k_score_gqa_lib, b"q4_k_score_gqa_f32\0")?,
                ),
                k_score_gqa_64: std::mem::transmute(
                    sym(&k_score_gqa_64_lib, b"q4_k_score_gqa_64_f32\0")?,
                ),
                v_sum_mha: std::mem::transmute(
                    sym(&v_sum_lib, b"q4_fused_v_sum_multi_f32\0")?,
                ),
                v_sum_mha_64: std::mem::transmute(
                    sym(&v_sum_64_lib, b"q4_fused_v_sum_multi_64_f32\0")?,
                ),
                v_sum_gqa: std::mem::transmute(
                    sym(&k_score_gqa_lib, b"q4_v_sum_gqa_f32\0")?,
                ),
                v_sum_gqa_64: std::mem::transmute(
                    sym(&k_score_gqa_64_lib, b"q4_v_sum_gqa_64_f32\0")?,
                ),
                _libs: vec![
                    quantize_lib,
                    k_score_lib,
                    k_score_64_lib,
                    k_score_gqa_lib,
                    k_score_gqa_64_lib,
                    v_sum_lib,
                    v_sum_64_lib,
                    turbo_lib,
                ],
            };
            Ok(table)
        }
    }

    /// Convenience: find the kernel directory and load.
    pub fn init() -> Result<Self, String> {
        let dir = find_kernel_dir()?;
        Self::load(&dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_kernel_dir() {
        // This test verifies find_kernel_dir logic works.
        // It may or may not find eakv kernels depending on whether
        // olorin-cli has been built with eakv kernel compilation.
        match find_kernel_dir() {
            Ok(dir) => {
                assert!(dir.join("libquantize_simd.so").exists());
                eprintln!("found eakv kernels in: {}", dir.display());
            }
            Err(e) => {
                eprintln!("eakv kernels not found (expected if not yet compiled): {e}");
            }
        }
    }

    #[test]
    fn test_load_kernels() {
        // Only run if kernels are available
        let dir = match find_kernel_dir() {
            Ok(d) => d,
            Err(_) => {
                eprintln!("skipping load test — eakv kernels not compiled yet");
                return;
            }
        };

        let table = KernelTable::load(&dir).expect("kernel load failed");
        // Verify we got valid function pointers (non-null)
        assert_ne!(table.quantize as usize, 0);
        assert_ne!(table.rotate as usize, 0);
        assert_ne!(table.fwht as usize, 0);
        assert_ne!(table.sign_flip as usize, 0);
        assert_ne!(table.k_score_mha as usize, 0);
        assert_ne!(table.k_score_mha_64 as usize, 0);
        assert_ne!(table.k_score_gqa as usize, 0);
        assert_ne!(table.k_score_gqa_64 as usize, 0);
        assert_ne!(table.v_sum_mha as usize, 0);
        assert_ne!(table.v_sum_mha_64 as usize, 0);
        assert_ne!(table.v_sum_gqa as usize, 0);
        assert_ne!(table.v_sum_gqa_64 as usize, 0);
        eprintln!("all 12 kernel functions loaded successfully");
    }
}

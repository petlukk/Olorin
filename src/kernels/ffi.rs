//! FFI layer for Eä SIMD kernels — core, safety, search, rotation, chacha20.
//!
//! Kernels are embedded in the binary and extracted to ~/.olorin/lib/{VERSION}/
//! on first run. Call `init()` once at startup before using any kernel function.

use libloading::{Library, Symbol};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub mod embedded {
    include!(concat!(env!("OUT_DIR"), "/embedded_kernels.rs"));
}

pub use embedded::KernelId;
pub use embedded::KERNEL_COUNT;

// ── Type aliases ──────────────────────────────────────────────────────────────

type ClassifyBytesFn    = unsafe extern "C" fn(*const u8, *mut u8, i32);
type PretokenizeFn      = unsafe extern "C" fn(*const u8, *mut u8, *mut u8, i32);
type ScanPrefixesFn     = unsafe extern "C" fn(*const u8, i32, *mut i32, *mut i32);
type MatchCommandFn     = unsafe extern "C" fn(*const u8, i32, *mut i32);
type FusedSafetyFn      = unsafe extern "C" fn(*const u8, i32, *mut i32, *mut i32, *mut i32);
type ClassifyIntentFn   = unsafe extern "C" fn(*const u8, i32, *mut i32, *mut i32, *mut i32);
type EvalExprFn         = unsafe extern "C" fn(*const u8, i32, *mut i64, *mut i32, *mut i64, *mut i32);
type ZeroizeFn          = unsafe extern "C" fn(*mut u8, i32);
type AnsiClassifyFn     = unsafe extern "C" fn(*const u8, *mut u8, i32);
type TerminalDiffFn     = unsafe extern "C" fn(*const u8, *const u8, *mut u8, i32);
type BatchDotFn         = unsafe extern "C" fn(*const f32, *const f32, i32, i32, *mut f32);
type BatchCosineFn      = unsafe extern "C" fn(*const f32, f32, *const f32, i32, i32, *mut f32);
type BatchL2Fn          = unsafe extern "C" fn(*const f32, *const f32, i32, i32, *mut f32);
type NormalizeFn        = unsafe extern "C" fn(*mut f32, i32, i32);
type ThresholdFn        = unsafe extern "C" fn(*const f32, i32, f32, *mut i32, *mut i32);
type TopKFn             = unsafe extern "C" fn(*const f32, i32, i32, *mut i32, *mut f32);
type JlProjectFn        = unsafe extern "C" fn(*const f32, *const f32, i32, i32, *mut f32, *mut f32);
type JlProjectBatchFn   = unsafe extern "C" fn(*const f32, *const f32, i32, i32, *mut f32, *mut f32, i32);
type SignFlipFn         = unsafe extern "C" fn(*mut f32, *const f32, i32);
type FwhtFn             = unsafe extern "C" fn(*mut f32, i32);
type TurboRotateFn      = unsafe extern "C" fn(*mut f32, *const f32, i32);
type Chacha20EncryptFn  = unsafe extern "C" fn(
    *const i32, *const i32, i32,
    *const u8, *mut u8, i32,
    *mut i32, *mut u8,
    *mut i32, *mut i32,
);
type SearchV2Fn = unsafe extern "C" fn(
    *const i32, *const i32, i32,
    *const u8, i32,
    *mut i32, *mut u8,
    *const i32,
    *mut u8, *mut i32,
    *mut u8,
    *const u8, *const i32,
    *const i32, i32,
    *mut u8, i32,
    *mut i32, *mut i32,
    *mut i32, *mut i32,
    i32, i32, i32,
    *mut i32, *mut i32,
);

// ── KernelTable ───────────────────────────────────────────────────────────────

pub struct KernelTable {
    pub libs: Vec<Library>,
    pub classify_bytes:           ClassifyBytesFn,
    pub scan_leak_prefixes:       ScanPrefixesFn,
    pub scan_injection_prefixes:  ScanPrefixesFn,
    pub match_command:            MatchCommandFn,
    pub scan_safety_fused:        FusedSafetyFn,
    pub classify_intent:          ClassifyIntentFn,
    pub eval_expr:                EvalExprFn,
    pub zeroize:                  ZeroizeFn,
    pub batch_dot:                BatchDotFn,
    pub batch_cosine:             BatchCosineFn,
    pub batch_l2:                 BatchL2Fn,
    pub normalize_vectors:        NormalizeFn,
    pub threshold_filter:         ThresholdFn,
    pub top_k:                    TopKFn,
    pub jl_project:               JlProjectFn,
    pub jl_project_batch:         JlProjectBatchFn,
    pub sign_flip:                SignFlipFn,
    pub fwht_inplace:             FwhtFn,
    pub turbo_rotate:             TurboRotateFn,
    pub chacha20_encrypt:         Chacha20EncryptFn,
    pub chacha20_search_v2:       SearchV2Fn,
    pub pretokenize:              PretokenizeFn,
    pub ansi_classify:            AnsiClassifyFn,
    pub terminal_diff:            TerminalDiffFn,
}

// SAFETY: KernelTable holds function pointers and library handles.
// Function pointers are valid for the lifetime of the libraries.
// Libraries are never unloaded (held in OnceLock for program lifetime).
unsafe impl Send for KernelTable {}
unsafe impl Sync for KernelTable {}

static KERNELS: OnceLock<KernelTable> = OnceLock::new();

fn k() -> &'static KernelTable {
    KERNELS.get_or_init(|| {
        let dir = extract_kernels().expect("failed to extract SIMD kernels");
        load_kernels(&dir).expect("failed to load SIMD kernels")
    })
}

// ── init / extract / load ─────────────────────────────────────────────────────

/// Initialize the kernel runtime: extract embedded .so files and load them.
/// Must be called once before any kernel function is used.
/// Safe to call multiple times (only the first call does work).
pub fn init() -> Result<(), String> {
    if KERNELS.get().is_some() {
        return Ok(());
    }
    let dir = extract_kernels()?;
    let table = load_kernels(&dir)?;
    let _ = KERNELS.set(table);
    // Also initialize inference kernels
    crate::kernels::ffi_inference::init_from(&dir)?;
    Ok(())
}

/// Return the versioned kernel directory path.
pub fn kernel_dir() -> Result<PathBuf, String> {
    let home = std::env::var("HOME")
        .map_err(|_| "HOME not set".to_string())?;
    Ok(PathBuf::from(home)
        .join(".olorin")
        .join("lib")
        .join(embedded::VERSION))
}

fn extract_kernels() -> Result<PathBuf, String> {
    let dir = kernel_dir()?;
    let marker = dir.join(".extracted");
    if marker.exists() {
        return Ok(dir);
    }

    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("failed to create {}: {e}", dir.display()))?;

    for (_id, filename, bytes) in embedded::FILES {
        let path = dir.join(filename);
        std::fs::write(&path, bytes)
            .map_err(|e| format!("failed to write {}: {e}", path.display()))?;
    }

    std::fs::write(&marker, embedded::VERSION)
        .map_err(|e| format!("failed to write marker: {e}"))?;

    eprintln!("olorin: extracted kernels to {}", dir.display());
    Ok(dir)
}

fn load_kernels(lib_dir: &Path) -> Result<KernelTable, String> {
    let load = |name: &str| -> Result<Library, String> {
        let path = lib_dir.join(format!("lib{name}.so"));
        unsafe {
            Library::new(&path)
                .map_err(|e| format!("failed to load {}: {e}", path.display()))
        }
    };

    let byte_classifier = load("byte_classifier")?;
    let leak_scanner    = load("leak_scanner")?;
    let sanitizer       = load("sanitizer")?;
    let command_router  = load("command_router")?;
    let fused_safety    = load("fused_safety")?;
    let intent_router   = load("intent_router")?;
    let expr_eval       = load("expr_eval")?;
    let zeroize_lib     = load("zeroize")?;
    let jl_project_lib  = load("jl_project")?;
    let turbo_rotate_lib = load("turbo_rotate")?;
    let chacha20_lib    = load("chacha20")?;
    let chacha20_sv2    = load("chacha20_search_v2")?;
    let pretokenize_lib = load("pretokenize")?;
    let ansi_parser_lib  = load("ansi_parser")?;
    let terminal_diff_lib = load("terminal_diff")?;

    // Runtime CPU detection: prefer AVX-512 search kernel if available
    #[cfg(target_arch = "x86_64")]
    let (search, variant) = if is_x86_feature_detected!("avx512f") {
        match load("search_avx512") {
            Ok(lib) => (lib, "avx512"),
            Err(_)  => (load("search")?, "sse2"),
        }
    } else {
        (load("search")?, "sse2")
    };
    #[cfg(not(target_arch = "x86_64"))]
    let (search, variant) = (load("search")?, "neon");

    eprintln!("olorin: search={variant}");

    unsafe {
        let sym = |lib: &Library, name: &[u8]| -> Result<usize, String> {
            let s: Symbol<*const ()> = lib
                .get(name)
                .map_err(|e| format!("symbol {:?}: {e}",
                    std::str::from_utf8(&name[..name.len()-1]).unwrap_or("?")))?;
            Ok(*s as usize)
        };

        let table = KernelTable {
            classify_bytes: std::mem::transmute(
                sym(&byte_classifier, b"classify_bytes\0")?),
            scan_leak_prefixes: std::mem::transmute(
                sym(&leak_scanner, b"scan_leak_prefixes\0")?),
            scan_injection_prefixes: std::mem::transmute(
                sym(&sanitizer, b"scan_injection_prefixes\0")?),
            match_command: std::mem::transmute(
                sym(&command_router, b"match_command\0")?),
            scan_safety_fused: std::mem::transmute(
                sym(&fused_safety, b"scan_safety_fused\0")?),
            classify_intent: std::mem::transmute(
                sym(&intent_router, b"classify_intent\0")?),
            eval_expr: std::mem::transmute(
                sym(&expr_eval, b"eval_expr\0")?),
            zeroize: std::mem::transmute(
                sym(&zeroize_lib, b"zeroize_simd\0")?),
            batch_dot: std::mem::transmute(
                sym(&search, b"batch_dot\0")?),
            batch_cosine: std::mem::transmute(
                sym(&search, b"batch_cosine\0")?),
            batch_l2: std::mem::transmute(
                sym(&search, b"batch_l2\0")?),
            normalize_vectors: std::mem::transmute(
                sym(&search, b"normalize_vectors\0")?),
            threshold_filter: std::mem::transmute(
                sym(&search, b"threshold_filter\0")?),
            top_k: std::mem::transmute(
                sym(&search, b"top_k\0")?),
            jl_project: std::mem::transmute(
                sym(&jl_project_lib, b"jl_project\0")?),
            jl_project_batch: std::mem::transmute(
                sym(&jl_project_lib, b"jl_project_batch\0")?),
            sign_flip: std::mem::transmute(
                sym(&turbo_rotate_lib, b"sign_flip\0")?),
            fwht_inplace: std::mem::transmute(
                sym(&turbo_rotate_lib, b"fwht_inplace\0")?),
            turbo_rotate: std::mem::transmute(
                sym(&turbo_rotate_lib, b"turbo_rotate\0")?),
            chacha20_encrypt: std::mem::transmute(
                sym(&chacha20_lib, b"chacha20_encrypt\0")?),
            chacha20_search_v2: std::mem::transmute(
                sym(&chacha20_sv2, b"chacha20_search_v2\0")?),
            pretokenize: std::mem::transmute(
                sym(&pretokenize_lib, b"pretokenize\0")?),
            ansi_classify: std::mem::transmute(
                sym(&ansi_parser_lib, b"ansi_classify\0")?),
            terminal_diff: std::mem::transmute(
                sym(&terminal_diff_lib, b"terminal_diff\0")?),
            libs: vec![
                byte_classifier, leak_scanner, sanitizer, command_router,
                fused_safety, intent_router, expr_eval,
                zeroize_lib, search, jl_project_lib, turbo_rotate_lib,
                chacha20_lib, chacha20_sv2, pretokenize_lib,
                ansi_parser_lib, terminal_diff_lib,
            ],
        };
        Ok(table)
    }
}

// ── Public wrappers ───────────────────────────────────────────────────────────

pub unsafe fn classify_bytes(text: *const u8, flags: *mut u8, len: i32) {
    (k().classify_bytes)(text, flags, len);
}

pub unsafe fn scan_leak_prefixes(
    text: *const u8, len: i32, out_masks: *mut i32, out_n_blocks: *mut i32,
) {
    (k().scan_leak_prefixes)(text, len, out_masks, out_n_blocks);
}

pub unsafe fn scan_injection_prefixes(
    text: *const u8, len: i32, out_masks: *mut i32, out_n_blocks: *mut i32,
) {
    (k().scan_injection_prefixes)(text, len, out_masks, out_n_blocks);
}

pub unsafe fn match_command(text: *const u8, len: i32, out_match: *mut i32) {
    (k().match_command)(text, len, out_match);
}

pub unsafe fn scan_safety_fused(
    text: *const u8, len: i32,
    out_inject: *mut i32, out_leak: *mut i32, out_n: *mut i32,
) {
    (k().scan_safety_fused)(text, len, out_inject, out_leak, out_n);
}

pub unsafe fn classify_intent(
    text: *const u8, len: i32,
    out_intent: *mut i32, out_arg_start: *mut i32, out_arg_len: *mut i32,
) {
    (k().classify_intent)(text, len, out_intent, out_arg_start, out_arg_len);
}

pub unsafe fn eval_expr(
    text: *const u8, len: i32,
    out_result: *mut i64, out_error: *mut i32,
    val_stack: *mut i64, op_stack: *mut i32,
) {
    (k().eval_expr)(text, len, out_result, out_error, val_stack, op_stack);
}

/// SIMD-accelerated memory wipe. Zeroes `len` bytes at `ptr`.
/// # Safety
/// `ptr` must be valid for `len` bytes.
pub unsafe fn zeroize(ptr: *mut u8, len: i32) {
    (k().zeroize)(ptr, len);
}

pub unsafe fn batch_dot(
    query: *const f32, vecs: *const f32, dim: i32, n_vecs: i32, out: *mut f32,
) {
    (k().batch_dot)(query, vecs, dim, n_vecs, out);
}

pub unsafe fn batch_cosine(
    query: *const f32, query_norm: f32,
    vecs: *const f32, dim: i32, n_vecs: i32, out: *mut f32,
) {
    (k().batch_cosine)(query, query_norm, vecs, dim, n_vecs, out);
}

pub unsafe fn batch_l2(
    query: *const f32, vecs: *const f32, dim: i32, n_vecs: i32, out: *mut f32,
) {
    (k().batch_l2)(query, vecs, dim, n_vecs, out);
}

pub unsafe fn normalize_vectors(vecs: *mut f32, dim: i32, n_vecs: i32) {
    (k().normalize_vectors)(vecs, dim, n_vecs);
}

pub unsafe fn threshold_filter(
    scores: *const f32, n: i32, threshold: f32,
    out_indices: *mut i32, out_count: *mut i32,
) {
    (k().threshold_filter)(scores, n, threshold, out_indices, out_count);
}

pub unsafe fn top_k(
    scores: *const f32, n: i32, k_val: i32,
    out_indices: *mut i32, out_scores: *mut f32,
) {
    (k().top_k)(scores, n, k_val, out_indices, out_scores);
}

pub unsafe fn jl_project(
    vec: *const f32, signs: *const f32, in_dim: i32, out_dim: i32,
    out: *mut f32, scratch: *mut f32,
) {
    (k().jl_project)(vec, signs, in_dim, out_dim, out, scratch);
}

pub unsafe fn jl_project_batch(
    vecs: *const f32, signs: *const f32, in_dim: i32, out_dim: i32,
    out: *mut f32, scratch: *mut f32, n_vecs: i32,
) {
    (k().jl_project_batch)(vecs, signs, in_dim, out_dim, out, scratch, n_vecs);
}

pub unsafe fn sign_flip(vec: *mut f32, signs: *const f32, dim: i32) {
    (k().sign_flip)(vec, signs, dim);
}

pub unsafe fn fwht_inplace(vec: *mut f32, dim: i32) {
    (k().fwht_inplace)(vec, dim);
}

pub unsafe fn turbo_rotate(vec: *mut f32, signs: *const f32, dim: i32) {
    (k().turbo_rotate)(vec, signs, dim);
}

pub unsafe fn chacha20_encrypt(
    key: *const i32, nonce: *const i32, counter: i32,
    plaintext: *const u8, ciphertext: *mut u8, len: i32,
    ks_i32: *mut i32, ks_u8: *mut u8,
    pt_i32: *mut i32, ct_i32: *mut i32,
) {
    (k().chacha20_encrypt)(
        key, nonce, counter, plaintext, ciphertext, len,
        ks_i32, ks_u8, pt_i32, ct_i32,
    );
}

#[allow(clippy::too_many_arguments)]
pub unsafe fn pretokenize(
    text: *const u8, flags: *mut u8, boundaries: *mut u8, len: i32,
) {
    (k().pretokenize)(text, flags, boundaries, len);
}

pub unsafe fn chacha20_search_v2(
    key: *const i32, nonce: *const i32, ctr_init: i32,
    ct_u8: *const u8, len: i32,
    ks_i32: *mut i32, ks_u8: *mut u8,
    ct_i32: *const i32,
    pt_buf: *mut u8, pt_i32: *mut i32,
    overlap: *mut u8,
    needles: *const u8, needle_offsets: *const i32,
    needle_lens: *const i32, needle_count: i32,
    lines_buf: *mut u8, lines_buf_cap: i32,
    line_offsets: *mut i32, line_lens: *mut i32,
    match_offsets: *mut i32, needle_ids: *mut i32,
    max_matches: i32, max_line_len: i32, window_size: i32,
    match_count: *mut i32, lines_written: *mut i32,
) {
    (k().chacha20_search_v2)(
        key, nonce, ctr_init,
        ct_u8, len,
        ks_i32, ks_u8,
        ct_i32,
        pt_buf, pt_i32,
        overlap,
        needles, needle_offsets,
        needle_lens, needle_count,
        lines_buf, lines_buf_cap,
        line_offsets, line_lens,
        match_offsets, needle_ids,
        max_matches, max_line_len, window_size,
        match_count, lines_written,
    );
}

/// SIMD-accelerated ANSI byte classification.
/// Classifies each byte: 0=printable, 1=ESC, 2=bracket, 3=digit, 4=semicolon,
/// 5=final, 6=control, 7=high-byte.
/// # Safety
/// `data` must be valid for `len` bytes. `classes` must be valid for `len` bytes.
pub unsafe fn ansi_classify(data: *const u8, classes: *mut u8, len: i32) {
    (k().ansi_classify)(data, classes, len);
}

/// SIMD-accelerated terminal cell-grid diff.
/// Compares old_grid vs new_grid (each cell = 16 bytes), writes dirty bitmap.
/// # Safety
/// `old_grid` and `new_grid` must be valid for `n_cells * 16` bytes.
/// `dirty` must be valid for `n_cells` bytes.
pub unsafe fn terminal_diff(
    old_grid: *const u8, new_grid: *const u8, dirty: *mut u8, n_cells: i32,
) {
    (k().terminal_diff)(old_grid, new_grid, dirty, n_cells);
}

//! FFI layer for Eä SIMD kernels.
//!
//! Kernels are embedded in the binary and extracted to ~/.eaclaw/lib/
//! on first run. Functions are loaded at runtime via libloading.
//! Call `init()` once at startup before using any kernel function.

use libloading::{Library, Symbol};
use std::path::PathBuf;
use std::sync::OnceLock;

mod embedded {
    include!(concat!(env!("OUT_DIR"), "/embedded_kernels.rs"));
}

// Type aliases for kernel function signatures
type ClassifyBytesFn = unsafe extern "C" fn(*const u8, *mut u8, i32);
type CountJsonFn = unsafe extern "C" fn(*const u8, i32, *mut i32);
type ExtractJsonFn = unsafe extern "C" fn(*const u8, i32, *mut i32, *mut u8, *mut i32);
type ScanPrefixesFn = unsafe extern "C" fn(*const u8, i32, *mut i32, *mut i32);
type MatchCommandFn = unsafe extern "C" fn(*const u8, i32, *mut i32);
type FusedSafetyFn = unsafe extern "C" fn(*const u8, i32, *mut i32, *mut i32, *mut i32);
type BatchDotFn = unsafe extern "C" fn(*const f32, *const f32, i32, i32, *mut f32);
type BatchCosineFn = unsafe extern "C" fn(*const f32, f32, *const f32, i32, i32, *mut f32);
type BatchL2Fn = unsafe extern "C" fn(*const f32, *const f32, i32, i32, *mut f32);
type NormalizeFn = unsafe extern "C" fn(*mut f32, i32, i32);
type ThresholdFn = unsafe extern "C" fn(*const f32, i32, f32, *mut i32, *mut i32);
type TopKFn = unsafe extern "C" fn(*const f32, i32, i32, *mut i32, *mut f32);

struct KernelTable {
    _libs: Vec<Library>,
    classify_bytes: ClassifyBytesFn,
    count_json_structural: CountJsonFn,
    extract_json_structural: ExtractJsonFn,
    scan_leak_prefixes: ScanPrefixesFn,
    scan_injection_prefixes: ScanPrefixesFn,
    match_command: MatchCommandFn,
    scan_safety_fused: FusedSafetyFn,
    batch_dot: BatchDotFn,
    batch_cosine: BatchCosineFn,
    batch_l2: BatchL2Fn,
    normalize_vectors: NormalizeFn,
    threshold_filter: ThresholdFn,
    top_k: TopKFn,
}

// SAFETY: KernelTable holds function pointers and library handles.
// The function pointers are valid for the lifetime of the libraries.
// Libraries are never unloaded (held in OnceLock for program lifetime).
unsafe impl Send for KernelTable {}
unsafe impl Sync for KernelTable {}

static KERNELS: OnceLock<KernelTable> = OnceLock::new();

fn k() -> &'static KernelTable {
    KERNELS.get_or_init(|| {
        let lib_dir = extract_kernels().expect("failed to extract SIMD kernels");
        load_kernels(&lib_dir).expect("failed to load SIMD kernels")
    })
}

/// Initialize the kernel runtime: extract embedded .so files and load them.
/// Must be called once before any kernel function is used.
/// Safe to call multiple times (only the first call does work).
pub fn init() -> Result<(), String> {
    if KERNELS.get().is_some() {
        return Ok(());
    }
    let lib_dir = extract_kernels()?;
    let table = load_kernels(&lib_dir)?;
    // If another thread raced us, that's fine — we just discard ours
    let _ = KERNELS.set(table);
    Ok(())
}

fn kernel_dir() -> Result<PathBuf, String> {
    let base = home::home_dir()
        .ok_or_else(|| "cannot determine home directory".to_string())?;
    Ok(base.join(".eaclaw").join("lib").join(format!("v{}", embedded::VERSION)))
}

fn extract_kernels() -> Result<PathBuf, String> {
    let dir = kernel_dir()?;

    // Check if already extracted (version-stamped directory exists with marker)
    let marker = dir.join(".extracted");
    if marker.exists() {
        return Ok(dir);
    }

    std::fs::create_dir_all(&dir)
        .map_err(|e| format!("failed to create {}: {e}", dir.display()))?;

    let kernels: &[(&str, &[u8])] = &[
        ("libbyte_classifier.so", embedded::BYTE_CLASSIFIER),
        ("libjson_scanner.so", embedded::JSON_SCANNER),
        ("libcommand_router.so", embedded::COMMAND_ROUTER),
        ("libleak_scanner.so", embedded::LEAK_SCANNER),
        ("libsanitizer.so", embedded::SANITIZER),
        ("libfused_safety.so", embedded::FUSED_SAFETY),
        ("libsearch.so", embedded::SEARCH),
        ("libsearch_avx512.so", embedded::SEARCH_AVX512),
    ];

    for (name, data) in kernels {
        let path = dir.join(name);
        std::fs::write(&path, data)
            .map_err(|e| format!("failed to write {}: {e}", path.display()))?;
    }

    // Write marker so we skip extraction next time
    let _ = std::fs::write(&marker, embedded::VERSION);

    Ok(dir)
}

fn load_kernels(lib_dir: &PathBuf) -> Result<KernelTable, String> {
    let load = |name: &str| -> Result<Library, String> {
        let path = lib_dir.join(format!("lib{name}.so"));
        unsafe {
            Library::new(&path).map_err(|e| format!("failed to load {}: {e}", path.display()))
        }
    };

    let byte_classifier = load("byte_classifier")?;
    let json_scanner = load("json_scanner")?;
    let command_router = load("command_router")?;
    let leak_scanner = load("leak_scanner")?;
    let sanitizer = load("sanitizer")?;
    let fused_safety = load("fused_safety")?;

    // Runtime CPU detection: prefer AVX-512 search kernel if available
    #[cfg(target_arch = "x86_64")]
    let (search, search_variant) = if is_x86_feature_detected!("avx512f") {
        match load("search_avx512") {
            Ok(lib) => (lib, "avx512"),
            Err(_) => (load("search")?, "sse2"),
        }
    } else {
        (load("search")?, "sse2")
    };
    #[cfg(not(target_arch = "x86_64"))]
    let (search, search_variant) = (load("search")?, "neon");

    eprintln!("eaclaw kernels: search={search_variant}");

    unsafe {
        let sym = |lib: &Library, name: &[u8]| -> Result<usize, String> {
            let s: Symbol<*const ()> = lib.get(name)
                .map_err(|e| format!("symbol {:?}: {e}", std::str::from_utf8(name)))?;
            Ok(*s as usize)
        };

        let table = KernelTable {
            classify_bytes: std::mem::transmute(
                sym(&byte_classifier, b"classify_bytes\0")?),
            count_json_structural: std::mem::transmute(
                sym(&json_scanner, b"count_json_structural\0")?),
            extract_json_structural: std::mem::transmute(
                sym(&json_scanner, b"extract_json_structural\0")?),
            scan_leak_prefixes: std::mem::transmute(
                sym(&leak_scanner, b"scan_leak_prefixes\0")?),
            scan_injection_prefixes: std::mem::transmute(
                sym(&sanitizer, b"scan_injection_prefixes\0")?),
            match_command: std::mem::transmute(
                sym(&command_router, b"match_command\0")?),
            scan_safety_fused: std::mem::transmute(
                sym(&fused_safety, b"scan_safety_fused\0")?),
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
            _libs: vec![
                byte_classifier, json_scanner, command_router,
                leak_scanner, sanitizer, fused_safety, search,
            ],
        };
        Ok(table)
    }
}

// --- Public FFI wrappers (same signatures as before) ---

pub unsafe fn classify_bytes(text: *const u8, flags: *mut u8, len: i32) {
    (k().classify_bytes)(text, flags, len);
}

pub unsafe fn count_json_structural(text: *const u8, len: i32, out_count: *mut i32) {
    (k().count_json_structural)(text, len, out_count);
}

pub unsafe fn extract_json_structural(
    text: *const u8, len: i32, out_pos: *mut i32, out_types: *mut u8, out_count: *mut i32,
) {
    (k().extract_json_structural)(text, len, out_pos, out_types, out_count);
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
    text: *const u8, len: i32, out_inject: *mut i32, out_leak: *mut i32, out_n: *mut i32,
) {
    (k().scan_safety_fused)(text, len, out_inject, out_leak, out_n);
}

pub unsafe fn batch_dot(
    query: *const f32, vecs: *const f32, dim: i32, n_vecs: i32, out: *mut f32,
) {
    (k().batch_dot)(query, vecs, dim, n_vecs, out);
}

pub unsafe fn batch_cosine(
    query: *const f32, query_norm: f32, vecs: *const f32, dim: i32, n_vecs: i32, out: *mut f32,
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
    scores: *const f32, n: i32, threshold: f32, out_indices: *mut i32, out_count: *mut i32,
) {
    (k().threshold_filter)(scores, n, threshold, out_indices, out_count);
}

pub unsafe fn top_k(
    scores: *const f32, n: i32, k_val: i32, out_indices: *mut i32, out_scores: *mut f32,
) {
    (k().top_k)(scores, n, k_val, out_indices, out_scores);
}

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

type PretokenizeFn      = unsafe extern "C" fn(*const u8, *mut u8, *mut u8, i32);
type MatchCommandFn     = unsafe extern "C" fn(*const u8, i32, *mut i32);
type FusedSafetyFn      = unsafe extern "C" fn(*const u8, i32, *mut i32, *mut i32, *mut i32);
type ClassifyIntentFn   = unsafe extern "C" fn(*const u8, i32, *mut i32, *mut i32, *mut i32);
type CsvScanFn          = unsafe extern "C" fn(
    *const u8, i32,
    *mut i32, *mut i32,
    *mut i32, *mut i32,
    *mut u8,
);
type JsonlStructFn      = unsafe extern "C" fn(
    *const u8, i32,
    *mut i32, *mut i32, *mut i32, *mut i32, *mut i32,
    *mut i32, *mut i32, *mut i32, *mut i32, *mut i32,
    *mut u8,
);
type LogLevelScanFn     = unsafe extern "C" fn(*const u8, i32, *mut i32);
type F32StatsFn         = unsafe extern "C" fn(
    *const f32, i32,
    *mut i32,
    *mut f32, *mut f32, *mut f32,
);
type F64StatsFn         = unsafe extern "C" fn(
    *const f64, i32,
    *mut i32,
    *mut f64, *mut f64, *mut f64,
);
type EvalExprFn         = unsafe extern "C" fn(*const u8, i32, *mut i64, *mut i32, *mut i64, *mut i32);
type ZeroizeFn          = unsafe extern "C" fn(*mut u8, i32);
type AnsiClassifyFn     = unsafe extern "C" fn(*const u8, *mut u8, i32);
type TerminalDiffFn     = unsafe extern "C" fn(*const u8, *const u8, *mut u8, i32);
type BatchCosineFn      = unsafe extern "C" fn(*const f32, f32, *const f32, i32, i32, *mut f32);
type NormalizeFn        = unsafe extern "C" fn(*mut f32, i32, i32);
type TopKFn             = unsafe extern "C" fn(*const f32, i32, i32, *mut i32, *mut f32);
type JlProjectFn        = unsafe extern "C" fn(*const f32, *const f32, i32, i32, *mut f32, *mut f32);
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
    pub match_command:            MatchCommandFn,
    pub scan_safety_fused:        FusedSafetyFn,
    pub classify_intent:          ClassifyIntentFn,
    pub csv_scan:                 CsvScanFn,
    pub jsonl_struct_scan:        JsonlStructFn,
    pub log_level_scan:           LogLevelScanFn,
    pub f32_stats:                F32StatsFn,
    pub f64_stats:                F64StatsFn,
    pub eval_expr:                EvalExprFn,
    pub zeroize:                  ZeroizeFn,
    pub batch_cosine:             BatchCosineFn,
    pub normalize_vectors:        NormalizeFn,
    pub top_k:                    TopKFn,
    pub jl_project:               JlProjectFn,
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
    // Init inference kernels *before* publishing KERNELS so any concurrent
    // caller that observes KERNELS.is_some() also sees inference initialized.
    crate::kernels::ffi_inference::init_from(&dir)?;
    let _ = KERNELS.set(table);
    Ok(())
}

/// Return the versioned kernel directory path.
pub fn kernel_dir() -> Result<PathBuf, String> {
    let home = crate::home_dir()
        .ok_or_else(|| "home directory not found".to_string())?;
    Ok(home
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
        let path = lib_dir.join(super::dynlib_filename(name));
        unsafe {
            Library::new(&path)
                .map_err(|e| format!("failed to load {}: {e}", path.display()))
        }
    };

    let command_router  = load("command_router")?;
    let fused_safety    = load("fused_safety")?;
    let intent_router   = load("intent_router")?;
    let csv_scan_lib    = load("csv_scan")?;
    let jsonl_struct_lib = load("jsonl_struct")?;
    let log_level_scan_lib = load("log_level_scan")?;
    let f32_stats_lib   = load("f32_stats")?;
    let f64_stats_lib   = load("f64_stats")?;
    let expr_eval       = load("expr_eval")?;
    let zeroize_lib     = load("zeroize")?;
    let jl_project_lib  = load("jl_project")?;
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
            match_command: std::mem::transmute(
                sym(&command_router, b"match_command\0")?),
            scan_safety_fused: std::mem::transmute(
                sym(&fused_safety, b"scan_safety_fused\0")?),
            classify_intent: std::mem::transmute(
                sym(&intent_router, b"classify_intent\0")?),
            csv_scan: std::mem::transmute(
                sym(&csv_scan_lib, b"csv_scan\0")?),
            jsonl_struct_scan: std::mem::transmute(
                sym(&jsonl_struct_lib, b"jsonl_struct_scan\0")?),
            log_level_scan: std::mem::transmute(
                sym(&log_level_scan_lib, b"log_level_scan\0")?),
            f32_stats: std::mem::transmute(
                sym(&f32_stats_lib, b"f32_stats\0")?),
            f64_stats: std::mem::transmute(
                sym(&f64_stats_lib, b"f64_stats\0")?),
            eval_expr: std::mem::transmute(
                sym(&expr_eval, b"eval_expr\0")?),
            zeroize: std::mem::transmute(
                sym(&zeroize_lib, b"zeroize_simd\0")?),
            batch_cosine: std::mem::transmute(
                sym(&search, b"batch_cosine\0")?),
            normalize_vectors: std::mem::transmute(
                sym(&search, b"normalize_vectors\0")?),
            top_k: std::mem::transmute(
                sym(&search, b"top_k\0")?),
            jl_project: std::mem::transmute(
                sym(&jl_project_lib, b"jl_project\0")?),
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
                command_router,
                fused_safety, intent_router, expr_eval,
                zeroize_lib, search, jl_project_lib,
                chacha20_lib, chacha20_sv2, pretokenize_lib,
                ansi_parser_lib, terminal_diff_lib,
                csv_scan_lib, jsonl_struct_lib, log_level_scan_lib,
                f32_stats_lib, f64_stats_lib,
            ],
        };
        Ok(table)
    }
}

// ── Public wrappers ───────────────────────────────────────────────────────────

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

pub unsafe fn batch_cosine(
    query: *const f32, query_norm: f32,
    vecs: *const f32, dim: i32, n_vecs: i32, out: *mut f32,
) {
    (k().batch_cosine)(query, query_norm, vecs, dim, n_vecs, out);
}

pub unsafe fn normalize_vectors(vecs: *mut f32, dim: i32, n_vecs: i32) {
    (k().normalize_vectors)(vecs, dim, n_vecs);
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

/// Scan CSV bytes, writing comma and newline byte-indices into the two
/// output arrays and the counts into `out_n_comma` / `out_n_newline`.
///
/// # Safety
/// - `text` must point to `len` readable bytes.
/// - `out_commas` and `out_newlines` must each be valid for at least `len`
///   writable `i32` elements (worst case is every byte being a delimiter).
/// - `out_n_comma` and `out_n_newline` must each be valid for one writable
///   `i32`. Passing non-null dangling pointers is UB even when `len == 0`.
/// - `scratch` must be valid for 16 writable bytes. The kernel overwrites
///   it once per 16-byte chunk; the contents after the call are unspecified.
pub unsafe fn csv_scan(
    text: *const u8, len: i32,
    out_commas: *mut i32, out_newlines: *mut i32,
    out_n_comma: *mut i32, out_n_newline: *mut i32,
    scratch: *mut u8,
) {
    (k().csv_scan)(text, len, out_commas, out_newlines, out_n_comma, out_n_newline, scratch);
}

/// Single-pass JSONL structural scan: writes positions of newlines, quotes,
/// colons, commas, and backslashes across `text` to five caller-allocated
/// arrays. Backslash positions allow callers to filter out escaped quotes
/// (`\"` inside a JSON string is *not* a structural quote).
///
/// # Safety
/// - `text` must point to `len` readable bytes.
/// - Each `out_*` array must be writable for `len` `i32` elements (worst
///   case: every byte is a structural character of that type).
/// - `out_n_*` must each be valid for one writable `i32`.
/// - `scratch` must be writable for 16 bytes; contents after the call are
///   unspecified (overwritten once per 16-byte chunk).
pub unsafe fn jsonl_struct_scan(
    text: *const u8, len: i32,
    out_newlines: *mut i32, out_quotes: *mut i32,
    out_colons: *mut i32,   out_commas: *mut i32, out_backslashes: *mut i32,
    out_n_newline: *mut i32, out_n_quote: *mut i32,
    out_n_colon: *mut i32,   out_n_comma: *mut i32, out_n_backslash: *mut i32,
    scratch: *mut u8,
) {
    (k().jsonl_struct_scan)(
        text, len,
        out_newlines, out_quotes, out_colons, out_commas, out_backslashes,
        out_n_newline, out_n_quote, out_n_colon, out_n_comma, out_n_backslash,
        scratch,
    );
}

/// Multi-keyword severity scanner. Counts word-bounded occurrences of
/// DEBUG, INFO, WARN, ERROR, FATAL plus newline bytes in `text`. Word
/// boundary = bounded by one of: space, tab, newline, CR, '[', ']',
/// '"', ':'. Start/end of buffer count as implicit delimiters.
///
/// # Safety
/// - `text` must point to `len` readable bytes.
/// - `out_counts` must be valid for **six** writable `i32` elements; the
///   kernel writes [DEBUG, INFO, WARN, ERROR, FATAL, NEWLINES] in that
///   order and always zeroes all six before counting (safe to call with
///   `len == 0`).
pub unsafe fn log_level_scan(text: *const u8, len: i32, out_counts: *mut i32) {
    (k().log_level_scan)(text, len, out_counts);
}

/// Streaming stats over `len` f32 elements. Writes count, sum, min, max.
///
/// # Safety
/// - `data` must point to `len` readable `f32` elements; may be dangling
///   when `len == 0` (the kernel does not dereference it in that case).
/// - `out_count`, `out_sum`, `out_min`, `out_max` must each be valid for
///   one writable element **even when `len == 0`** — the kernel always
///   writes zeros to all four before the early return.
pub unsafe fn f32_stats(
    data: *const f32, len: i32,
    out_count: *mut i32,
    out_sum: *mut f32, out_min: *mut f32, out_max: *mut f32,
) {
    (k().f32_stats)(data, len, out_count, out_sum, out_min, out_max);
}

/// f64 streaming stats — the double-precision counterpart to `f32_stats`.
/// Used by eaparquet for INT64 statistics aggregation where f32's
/// 24-bit mantissa would lose precision on large integer values.
///
/// # Safety
/// Same as `f32_stats`: data must be valid for `len` reads (or len==0,
/// in which case data is not dereferenced); the four out pointers must
/// each be valid for one write.
pub unsafe fn f64_stats(
    data: *const f64, len: i32,
    out_count: *mut i32,
    out_sum: *mut f64, out_min: *mut f64, out_max: *mut f64,
) {
    (k().f64_stats)(data, len, out_count, out_sum, out_min, out_max);
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

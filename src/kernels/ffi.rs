//! FFI layer for Eä SIMD kernels — core, safety, search, rotation, chacha20.
//!
//! Kernels are embedded in the binary and extracted to ~/.olorin/lib/{VERSION}/
//! on first run. Call `init()` once at startup before using any kernel function.

use libloading::{Library, Symbol};
use std::path::Path;
use std::sync::OnceLock;

pub mod embedded {
    include!(concat!(env!("OUT_DIR"), "/embedded_kernels.rs"));
}

pub use embedded::KernelId;
pub use embedded::KERNEL_COUNT;

// The `extern "C"` aliases and the `KernelTable` struct live in `ffi_types`,
// split out so this file holds only the (per-kernel growing) loader and the
// public wrappers. Re-exported to preserve the `kernels::ffi::KernelTable` path.
pub use super::ffi_types::KernelTable;

static KERNELS: OnceLock<KernelTable> = OnceLock::new();

pub(super) fn k() -> &'static KernelTable {
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

pub use super::loader::{extract_kernels, kernel_dir};

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
    let csv_groupby_lib = load("csv_groupby_scan")?;
    let jsonl_struct_lib = load("jsonl_struct")?;
    let log_level_scan_lib = load("log_level_scan")?;
    let sql_scan_lib    = load("sql_scan")?;
    let timestamp_scan_lib = load("timestamp_scan")?;
    let clf_scan_lib    = load("clf_scan")?;
    let clf_status_lib  = load("clf_status_scan")?;
    let json_epoch_lib  = load("json_epoch_scan")?;
    let json_level_lib  = load("json_level_scan")?;
    let syslog_scan_lib = load("syslog_scan")?;
    let apache_error_lib = load("apache_error_scan")?;
    let f32_stats_lib   = load("f32_stats")?;
    let f64_stats_lib   = load("f64_stats")?;
    let col_reduce_lib  = load("col_reduce")?;
    let corr_sweep_lib  = load("corr_sweep")?;
    let expr_eval       = load("expr_eval")?;
    let zeroize_lib     = load("zeroize")?;
    let jl_project_lib  = load("jl_project")?;
    let chacha20_lib    = load("chacha20")?;
    let chacha20_sv2    = load("chacha20_search_v2")?;
    let pretokenize_lib = load("pretokenize")?;
    let ansi_parser_lib  = load("ansi_parser")?;
    let terminal_diff_lib = load("terminal_diff")?;
    let poly1305_lib    = load("poly1305")?;
    let blake2b_lib     = load("blake2b")?;
    let argon2_block_lib = load("argon2_block")?;

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
            csv_groupby_scan: std::mem::transmute(
                sym(&csv_groupby_lib, b"csv_groupby_scan\0")?),
            jsonl_struct_scan: std::mem::transmute(
                sym(&jsonl_struct_lib, b"jsonl_struct_scan\0")?),
            log_level_scan: std::mem::transmute(
                sym(&log_level_scan_lib, b"log_level_scan\0")?),
            sql_scan: std::mem::transmute(
                sym(&sql_scan_lib, b"sql_scan\0")?),
            timestamp_scan: std::mem::transmute(
                sym(&timestamp_scan_lib, b"timestamp_scan\0")?),
            clf_scan: std::mem::transmute(
                sym(&clf_scan_lib, b"clf_scan\0")?),
            clf_status_scan: std::mem::transmute(
                sym(&clf_status_lib, b"clf_status_scan\0")?),
            json_epoch_scan: std::mem::transmute(
                sym(&json_epoch_lib, b"json_epoch_scan\0")?),
            json_level_scan: std::mem::transmute(
                sym(&json_level_lib, b"json_level_scan\0")?),
            syslog_scan: std::mem::transmute(
                sym(&syslog_scan_lib, b"syslog_scan\0")?),
            apache_error_scan: std::mem::transmute(
                sym(&apache_error_lib, b"apache_error_scan\0")?),
            f32_stats: std::mem::transmute(
                sym(&f32_stats_lib, b"f32_stats\0")?),
            f64_stats: std::mem::transmute(
                sym(&f64_stats_lib, b"f64_stats\0")?),
            col_reduce: std::mem::transmute(
                sym(&col_reduce_lib, b"col_reduce\0")?),
            corr_sweep: std::mem::transmute(
                sym(&corr_sweep_lib, b"corr_sweep\0")?),
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
            poly1305_mac_kern: std::mem::transmute(
                sym(&poly1305_lib, b"poly1305_mac\0")?),
            poly1305_verify_kern: std::mem::transmute(
                sym(&poly1305_lib, b"poly1305_verify\0")?),
            blake2b_compress_kern: std::mem::transmute(
                sym(&blake2b_lib, b"blake2b_compress\0")?),
            argon2_block_kern: std::mem::transmute(
                sym(&argon2_block_lib, b"argon2_block_compress\0")?),
            libs: vec![
                command_router,
                fused_safety, intent_router, expr_eval,
                zeroize_lib, search, jl_project_lib,
                chacha20_lib, chacha20_sv2, pretokenize_lib,
                ansi_parser_lib, terminal_diff_lib,
                csv_scan_lib, csv_groupby_lib, jsonl_struct_lib, log_level_scan_lib,
                sql_scan_lib,
                timestamp_scan_lib, clf_scan_lib, clf_status_lib, json_epoch_lib, json_level_lib, syslog_scan_lib,
            apache_error_lib,
                f32_stats_lib, f64_stats_lib, col_reduce_lib,
                corr_sweep_lib,
                poly1305_lib,
                blake2b_lib,
                argon2_block_lib,
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

// Data-plane rune kernels (csv_scan, jsonl_struct_scan, log_level_scan,
// timestamp_scan, f32_stats, f64_stats) live in ffi_data.rs to keep
// this file under the 500-LOC cap. Re-exported here so existing
// `kernels::ffi::<name>` call sites continue to compile unchanged.
pub use super::ffi_data::{
    clf_scan, clf_status_scan, col_reduce, corr_sweep, csv_groupby_scan, csv_scan,
    apache_error_scan, f32_stats, f64_stats, json_epoch_scan, json_level_scan, jsonl_struct_scan,
    log_level_scan, sql_scan, syslog_scan, timestamp_scan,
};

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

// Crypto-primitive wrappers (poly1305, Blake2b, Argon2) live in
// ffi_crypto.rs to keep this file under the 500-LOC cap.
// Re-exported here so existing `kernels::ffi::<name>` call sites
// continue to compile unchanged.
pub use super::ffi_crypto::{
    argon2_block_compress, blake2b_compress, poly1305_mac, poly1305_verify,
};

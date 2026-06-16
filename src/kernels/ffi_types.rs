//! FFI type shape for the core/safety/search/storage kernel table.
//!
//! The `extern "C"` function-pointer aliases and the `KernelTable` struct they
//! populate live here, split out of `ffi.rs` so that file holds only the
//! (per-kernel growing) loader and the public wrappers. Mirrors the
//! `ffi_inference` / `ffi_inference_types` split. The aliases are private —
//! only `KernelTable` names them; `ffi.rs` re-exports `KernelTable`.

use libloading::Library;

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
// csv_groupby_scan: fused column-projection scan for eacrunch GROUP BY.
// (text, len, needed cols, n_needed, out_off, out_len, out_n_rows, scratch)
type CsvGroupbyScanFn   = unsafe extern "C" fn(
    *const u8, i32,
    *const i32, i32,
    *mut i32, *mut i32,
    *mut i32,
    *mut u8,
);
// (text, len, out_newlines/quotes/colons/backslashes, out_n_* ×4, scratch)
type JsonlStructFn      = unsafe extern "C" fn(
    *const u8, i32,
    *mut i32, *mut i32, *mut i32, *mut i32,
    *mut i32, *mut i32, *mut i32, *mut i32,
    *mut u8,
);
type LogLevelScanFn     = unsafe extern "C" fn(
    *const u8, i32,
    *mut i32,
    *mut i32, i32, *mut i32,
    *mut u8,
);
type TimestampScanFn    = unsafe extern "C" fn(
    *const u8, i32,
    *mut i32, i32, *mut i32,
    *mut u8,
);
type F32StatsFn         = unsafe extern "C" fn(
    *const f32, i32,
    *mut i32,
    *mut f32, *mut f32, *mut f32,
);
type CorrSweepFn        = unsafe extern "C" fn(
    *const f32, *const f32, i32, i32,
    *mut f32,
);
type F64StatsFn         = unsafe extern "C" fn(
    *const f64, i32,
    *mut i32,
    *mut f64, *mut f64, *mut f64,
);
type ColReduceFn        = unsafe extern "C" fn(
    *const f32, i32, i32,
    *mut f32, *mut f32, *mut f32,
);
type EvalExprFn         = unsafe extern "C" fn(*const u8, i32, *mut i64, *mut i32, *mut i64, *mut i32);
type ZeroizeFn          = unsafe extern "C" fn(*mut u8, i32);
type AnsiClassifyFn     = unsafe extern "C" fn(*const u8, *mut u8, i32);
type TerminalDiffFn     = unsafe extern "C" fn(*const u8, *const u8, *mut u8, i32);
type BatchCosineFn      = unsafe extern "C" fn(*const f32, f32, *const f32, i32, i32, *mut f32);
type NormalizeFn        = unsafe extern "C" fn(*mut f32, i32, i32);
type TopKFn             = unsafe extern "C" fn(*const f32, i32, i32, *mut i32, *mut f32);
type JlProjectFn        = unsafe extern "C" fn(*const f32, *const f32, i32, i32, *mut f32, *mut f32);
// poly1305_mac kernel: takes an extra *mut i32 scratch (22 words) that the
// Rust wrapper allocates on the stack.  The public `poly1305_mac` exposed
// in ffi_crypto presents the clean 4-parameter interface from the task spec.
type Poly1305MacKernFn  = unsafe extern "C" fn(
    *const u8, *const u8, i32, *mut u8, *mut i32,
);
type Poly1305VerifyKernFn = unsafe extern "C" fn(
    *const u8, *const u8, i32, *const u8, *mut i32,
) -> i32;
// blake2b_compress kernel: caller allocates the IV constants table (9 u64s
// — eight Blake2b IVs and the all-ones finalization mask), the 16-u64
// v_scratch, and the 8-u64 chaining state.  Algorithm constants live on
// the Rust side because Ea's hex parser caps at i64::MAX and four of the
// Blake2b IVs need bit 63 set; passing them in as data also matches the
// pattern used everywhere else in Olorin (weights/configs in Rust,
// algorithm in the kernel).
type Blake2bCompressFn   = unsafe extern "C" fn(
    *mut u64, *const u64, *const u64,
    u64, u64, i32, *mut u64,
);
// argon2_block_compress kernel: G(X, Y) -> Z on three 1024-byte blocks
// (128 u64s each); `scratch` is a 16-u64 column-gather buffer.
type Argon2BlockFn       = unsafe extern "C" fn(
    *const u64, *const u64, *mut u64, *mut u64,
);
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
    pub csv_groupby_scan:         CsvGroupbyScanFn,
    pub jsonl_struct_scan:        JsonlStructFn,
    pub log_level_scan:           LogLevelScanFn,
    pub sql_scan:                 LogLevelScanFn,
    pub timestamp_scan:           TimestampScanFn,
    pub clf_scan:                 TimestampScanFn,
    pub clf_status_scan:          TimestampScanFn,
    pub json_epoch_scan:          TimestampScanFn,
    pub json_level_scan:          TimestampScanFn,
    pub syslog_scan:              TimestampScanFn,
    pub apache_error_scan:        TimestampScanFn,
    pub f32_stats:                F32StatsFn,
    pub f64_stats:                F64StatsFn,
    pub col_reduce:               ColReduceFn,
    pub corr_sweep:               CorrSweepFn,
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
    pub poly1305_mac_kern:        Poly1305MacKernFn,
    pub poly1305_verify_kern:     Poly1305VerifyKernFn,
    pub blake2b_compress_kern:    Blake2bCompressFn,
    pub argon2_block_kern:        Argon2BlockFn,
}

// SAFETY: KernelTable holds function pointers and library handles.
// Function pointers are valid for the lifetime of the libraries.
// Libraries are never unloaded (held in OnceLock for program lifetime).
unsafe impl Send for KernelTable {}
unsafe impl Sync for KernelTable {}

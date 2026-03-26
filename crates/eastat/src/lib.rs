//! Rust FFI wrapper for eastat — SIMD-accelerated CSV column statistics.
//!
//! Runs the full pipeline: mmap → scan → layout → parse → stats.
//! Loads four shared libraries at runtime via libloading:
//!   libcsv_scan.so, libcsv_layout.so, libcsv_parse.so, libcsv_stats.so

use libloading::Library;
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

mod pipeline;

pub use pipeline::process;

#[derive(Error, Debug)]
pub enum ProcessError {
    #[error("file not found: {0}")]
    FileNotFound(String),
    #[error("empty file: {0}")]
    EmptyFile(String),
    #[error("kernel load failed ({lib}): {cause}")]
    KernelLoad { lib: String, cause: String },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ColumnStats {
    Integer {
        name: String,
        col_index: usize,
        rows: usize,
        count: usize,
        nulls: usize,
        min: f64,
        max: f64,
        mean: f64,
        stddev: f64,
        sum: f64,
        p25: f64,
        p50: f64,
        p75: f64,
    },
    Float {
        name: String,
        col_index: usize,
        rows: usize,
        count: usize,
        nulls: usize,
        min: f64,
        max: f64,
        mean: f64,
        stddev: f64,
        sum: f64,
        p25: f64,
        p50: f64,
        p75: f64,
    },
    String {
        name: String,
        col_index: usize,
        rows: usize,
        count: usize,
        nulls: usize,
        min_length: usize,
        max_length: usize,
        mean_length: f64,
    },
}

impl ColumnStats {
    pub fn name(&self) -> &str {
        match self {
            ColumnStats::Integer { name, .. } => name,
            ColumnStats::Float { name, .. } => name,
            ColumnStats::String { name, .. } => name,
        }
    }

    pub fn col_index(&self) -> usize {
        match self {
            ColumnStats::Integer { col_index, .. } => *col_index,
            ColumnStats::Float { col_index, .. } => *col_index,
            ColumnStats::String { col_index, .. } => *col_index,
        }
    }
}

/// Result of processing a CSV file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessResult {
    pub columns: Vec<ColumnStats>,
    pub headers: Vec<std::string::String>,
    pub row_count: usize,
    pub col_count: usize,
}

// Kernel function type aliases matching the .ea.json signatures.
pub(crate) type ScanFastFn = unsafe extern "C" fn(
    text: *const u8,
    len: i32,
    delim: u8,
    out_delim_pos: *mut i32,
    out_lf_pos: *mut i32,
    out_counts: *mut i32,
);

pub(crate) type ScanQuotedFn = unsafe extern "C" fn(
    text: *const u8,
    len: i32,
    delim: u8,
    out_delim_pos: *mut i32,
    out_lf_pos: *mut i32,
    out_counts: *mut i32,
);

pub(crate) type BuildRowArraysFn = unsafe extern "C" fn(
    lf_pos: *const i32,
    n_lfs: i32,
    header_end: i32,
    text_len: i32,
    out_row_starts: *mut i32,
    out_row_ends: *mut i32,
    out_n_rows: *mut i32,
);

pub(crate) type BuildRowDelimIndexFn = unsafe extern "C" fn(
    delim_pos: *const i32,
    n_delims: i32,
    row_ends: *const i32,
    n_rows: i32,
    out_delims_per_row: *mut i32,
    out_row_delim_offset: *mut i32,
);

pub(crate) type ComputeFieldBoundsFn = unsafe extern "C" fn(
    col_idx: i32,
    col_count: i32,
    n_rows: i32,
    row_starts: *const i32,
    row_ends: *const i32,
    delim_pos: *const i32,
    row_delim_offset: *const i32,
    delims_per_row: *const i32,
    out_field_starts: *mut i32,
    out_field_ends: *mut i32,
);

pub(crate) type BatchAtofFn = unsafe extern "C" fn(
    data: *const u8,
    starts: *const i32,
    ends: *const i32,
    n: i32,
    out: *mut f32,
    out_count: *mut i32,
);

pub(crate) type FieldLengthStatsFn = unsafe extern "C" fn(
    starts: *const i32,
    ends: *const i32,
    n: i32,
    out_min_len: *mut i32,
    out_max_len: *mut i32,
    out_total_len: *mut i32,
    out_null_count: *mut i32,
);

pub(crate) type F32ColumnStatsFn = unsafe extern "C" fn(
    data: *const f32,
    len: i32,
    out_sum: *mut f32,
    out_min: *mut f32,
    out_max: *mut f32,
    out_sumsq: *mut f32,
);

pub(crate) type F32PercentilesFn = unsafe extern "C" fn(
    data: *const f32,
    len: i32,
    min_val: f32,
    max_val: f32,
    out_p25: *mut f32,
    out_p50: *mut f32,
    out_p75: *mut f32,
);

pub(crate) fn load_lib(lib_dir: &Path, name: &str) -> Result<Library, ProcessError> {
    let path = lib_dir.join(format!("lib{name}.so"));
    unsafe {
        Library::new(&path).map_err(|e| ProcessError::KernelLoad {
            lib: name.to_string(),
            cause: e.to_string(),
        })
    }
}

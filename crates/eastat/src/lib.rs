//! Rust FFI wrapper for eastat — SIMD-accelerated CSV column statistics.
//!
//! Runs the full pipeline: mmap → scan → layout → parse → stats.
//! Loads four shared libraries at runtime via libloading:
//!   libcsv_scan.so, libcsv_layout.so, libcsv_parse.so, libcsv_stats.so

use libloading::{Library, Symbol};
use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

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
type ScanFastFn = unsafe extern "C" fn(
    text: *const u8,
    len: i32,
    delim: u8,
    out_delim_pos: *mut i32,
    out_lf_pos: *mut i32,
    out_counts: *mut i32,
);

type ScanQuotedFn = unsafe extern "C" fn(
    text: *const u8,
    len: i32,
    delim: u8,
    out_delim_pos: *mut i32,
    out_lf_pos: *mut i32,
    out_counts: *mut i32,
);

type BuildRowArraysFn = unsafe extern "C" fn(
    lf_pos: *const i32,
    n_lfs: i32,
    header_end: i32,
    text_len: i32,
    out_row_starts: *mut i32,
    out_row_ends: *mut i32,
    out_n_rows: *mut i32,
);

type BuildRowDelimIndexFn = unsafe extern "C" fn(
    delim_pos: *const i32,
    n_delims: i32,
    row_ends: *const i32,
    n_rows: i32,
    out_delims_per_row: *mut i32,
    out_row_delim_offset: *mut i32,
);

type ComputeFieldBoundsFn = unsafe extern "C" fn(
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

type BatchAtofFn = unsafe extern "C" fn(
    data: *const u8,
    starts: *const i32,
    ends: *const i32,
    n: i32,
    out: *mut f32,
    out_count: *mut i32,
);

type FieldLengthStatsFn = unsafe extern "C" fn(
    starts: *const i32,
    ends: *const i32,
    n: i32,
    out_min_len: *mut i32,
    out_max_len: *mut i32,
    out_total_len: *mut i32,
    out_null_count: *mut i32,
);

type F32ColumnStatsFn = unsafe extern "C" fn(
    data: *const f32,
    len: i32,
    out_sum: *mut f32,
    out_min: *mut f32,
    out_max: *mut f32,
    out_sumsq: *mut f32,
);

type F32PercentilesFn = unsafe extern "C" fn(
    data: *const f32,
    len: i32,
    min_val: f32,
    max_val: f32,
    out_p25: *mut f32,
    out_p50: *mut f32,
    out_p75: *mut f32,
);

fn load_lib(lib_dir: &Path, name: &str) -> Result<Library, ProcessError> {
    let path = lib_dir.join(format!("lib{name}.so"));
    unsafe {
        Library::new(&path).map_err(|e| ProcessError::KernelLoad {
            lib: name.to_string(),
            cause: e.to_string(),
        })
    }
}

/// Process a CSV file and return per-column statistics.
///
/// `lib_dir` must contain libcsv_scan.so, libcsv_layout.so,
/// libcsv_parse.so, and libcsv_stats.so.
pub fn process(
    filepath: &Path,
    lib_dir: &Path,
    delimiter: u8,
    has_header: bool,
) -> Result<ProcessResult, ProcessError> {
    // 1. Read file into memory.
    if !filepath.exists() {
        return Err(ProcessError::FileNotFound(filepath.display().to_string()));
    }
    let data = std::fs::read(filepath)?;
    let n = data.len();
    if n == 0 {
        return Err(ProcessError::EmptyFile(filepath.display().to_string()));
    }

    // Load libraries.
    let lib_scan = load_lib(lib_dir, "csv_scan")?;
    let lib_layout = load_lib(lib_dir, "csv_layout")?;
    let lib_parse = load_lib(lib_dir, "csv_parse")?;
    let lib_stats = load_lib(lib_dir, "csv_stats")?;

    // 2. Structural scan.
    let sample = &data[..data.len().min(4096)];
    let use_quoted = sample.contains(&b'"');

    let max_delims = n / 4 + 256;
    let max_lfs = n / 20 + 256;
    let mut delim_pos: Vec<i32> = vec![0; max_delims];
    let mut lf_pos: Vec<i32> = vec![0; max_lfs];
    let mut counts: Vec<i32> = vec![0; 3];

    unsafe {
        if use_quoted {
            let f: Symbol<ScanQuotedFn> = lib_scan
                .get(b"scan_positions_quoted\0")
                .map_err(|e| ProcessError::KernelLoad {
                    lib: "csv_scan".into(),
                    cause: e.to_string(),
                })?;
            f(
                data.as_ptr(),
                n as i32,
                delimiter,
                delim_pos.as_mut_ptr(),
                lf_pos.as_mut_ptr(),
                counts.as_mut_ptr(),
            );
        } else {
            let f: Symbol<ScanFastFn> = lib_scan
                .get(b"scan_positions_fast\0")
                .map_err(|e| ProcessError::KernelLoad {
                    lib: "csv_scan".into(),
                    cause: e.to_string(),
                })?;
            f(
                data.as_ptr(),
                n as i32,
                delimiter,
                delim_pos.as_mut_ptr(),
                lf_pos.as_mut_ptr(),
                counts.as_mut_ptr(),
            );
        }
    }

    let n_delims = counts[0] as usize;
    let n_lfs = counts[1] as usize;
    let header_dc = counts[2] as usize;
    delim_pos.truncate(n_delims);
    lf_pos.truncate(n_lfs);

    let col_count = header_dc + 1;

    // 3. Parse header row.
    let (headers, header_end) = if has_header {
        let first_nl = lf_pos.first().copied().unwrap_or(n as i32) as usize;
        let mut hdr = &data[..first_nl];
        if hdr.starts_with(b"\xef\xbb\xbf") {
            hdr = &hdr[3..];
        }
        let hdr = if hdr.ends_with(b"\r") { &hdr[..hdr.len() - 1] } else { hdr };
        let s = std::string::String::from_utf8_lossy(hdr);
        let hdrs: Vec<std::string::String> = s
            .split(delimiter as char)
            .map(|h| h.to_string())
            .collect();
        let he = lf_pos.first().copied().unwrap_or(-1);
        (hdrs, he)
    } else {
        let hdrs: Vec<std::string::String> =
            (0..col_count).map(|i| format!("col_{i}")).collect();
        (hdrs, -1i32)
    };

    // 4. Build row layout.
    let mut row_starts: Vec<i32> = vec![0; n_lfs + 2];
    let mut row_ends: Vec<i32> = vec![0; n_lfs + 2];
    let mut n_rows_out: Vec<i32> = vec![0; 1];

    unsafe {
        let f: Symbol<BuildRowArraysFn> = lib_layout
            .get(b"build_row_arrays\0")
            .map_err(|e| ProcessError::KernelLoad {
                lib: "csv_layout".into(),
                cause: e.to_string(),
            })?;
        f(
            lf_pos.as_ptr(),
            n_lfs as i32,
            header_end,
            n as i32,
            row_starts.as_mut_ptr(),
            row_ends.as_mut_ptr(),
            n_rows_out.as_mut_ptr(),
        );
    }

    let data_rows = n_rows_out[0] as usize;
    row_starts.truncate(data_rows);
    row_ends.truncate(data_rows);

    if data_rows == 0 {
        return Ok(ProcessResult {
            columns: vec![],
            headers,
            row_count: 0,
            col_count,
        });
    }

    // Skip header delimiters when indexing data rows.
    let data_delim_pos = if has_header {
        delim_pos[header_dc..].to_vec()
    } else {
        delim_pos.clone()
    };
    let n_data_delims = data_delim_pos.len();

    let mut delims_per_row: Vec<i32> = vec![0; data_rows];
    let mut row_delim_offset: Vec<i32> = vec![0; data_rows];

    if n_data_delims > 0 {
        unsafe {
            let f: Symbol<BuildRowDelimIndexFn> = lib_layout
                .get(b"build_row_delim_index\0")
                .map_err(|e| ProcessError::KernelLoad {
                    lib: "csv_layout".into(),
                    cause: e.to_string(),
                })?;
            f(
                data_delim_pos.as_ptr(),
                n_data_delims as i32,
                row_ends.as_ptr(),
                data_rows as i32,
                delims_per_row.as_mut_ptr(),
                row_delim_offset.as_mut_ptr(),
            );
        }
    }

    // 5. Per-column statistics.
    let mut fs_buf: Vec<i32> = vec![0; data_rows];
    let mut fe_buf: Vec<i32> = vec![0; data_rows];
    let mut val_buf: Vec<f32> = vec![0.0; data_rows];
    let mut cnt_buf: Vec<i32> = vec![0; 1];

    let mut columns: Vec<ColumnStats> = Vec::with_capacity(col_count);

    for ci in 0..col_count {
        let col_name = headers.get(ci).cloned().unwrap_or_else(|| format!("col_{ci}"));

        unsafe {
            let f: Symbol<ComputeFieldBoundsFn> = lib_layout
                .get(b"compute_field_bounds\0")
                .map_err(|e| ProcessError::KernelLoad {
                    lib: "csv_layout".into(),
                    cause: e.to_string(),
                })?;
            f(
                ci as i32,
                col_count as i32,
                data_rows as i32,
                row_starts.as_ptr(),
                row_ends.as_ptr(),
                data_delim_pos.as_ptr(),
                row_delim_offset.as_ptr(),
                delims_per_row.as_ptr(),
                fs_buf.as_mut_ptr(),
                fe_buf.as_mut_ptr(),
            );
        }

        cnt_buf[0] = 0;
        unsafe {
            let f: Symbol<BatchAtofFn> = lib_parse
                .get(b"batch_atof\0")
                .map_err(|e| ProcessError::KernelLoad {
                    lib: "csv_parse".into(),
                    cause: e.to_string(),
                })?;
            f(
                data.as_ptr(),
                fs_buf.as_ptr(),
                fe_buf.as_ptr(),
                data_rows as i32,
                val_buf.as_mut_ptr(),
                cnt_buf.as_mut_ptr(),
            );
        }

        let count = cnt_buf[0] as usize;
        let nulls = data_rows - count;

        let stats = if count >= data_rows / 2 {
            numeric_stats(
                &lib_stats,
                &val_buf[..count],
                count,
                data_rows,
                ci,
                col_name,
                nulls,
            )?
        } else {
            string_stats(&lib_parse, &fs_buf, &fe_buf, data_rows, ci, col_name)?
        };

        columns.push(stats);
    }

    Ok(ProcessResult {
        columns,
        headers,
        row_count: data_rows,
        col_count,
    })
}

fn numeric_stats(
    lib_stats: &Library,
    values: &[f32],
    count: usize,
    total_rows: usize,
    col_index: usize,
    name: std::string::String,
    nulls: usize,
) -> Result<ColumnStats, ProcessError> {
    let mut out_sum: f32 = 0.0;
    let mut out_min: f32 = 0.0;
    let mut out_max: f32 = 0.0;
    let mut out_sumsq: f32 = 0.0;

    if count >= 16 {
        unsafe {
            let f: Symbol<F32ColumnStatsFn> = lib_stats
                .get(b"f32_column_stats\0")
                .map_err(|e| ProcessError::KernelLoad {
                    lib: "csv_stats".into(),
                    cause: e.to_string(),
                })?;
            f(
                values.as_ptr(),
                count as i32,
                &mut out_sum,
                &mut out_min,
                &mut out_max,
                &mut out_sumsq,
            );
        }
    } else {
        for &v in values {
            let v64 = v as f64;
            out_sum += v;
            out_sumsq += (v64 * v64) as f32;
            if v < out_min || count == 0 {
                out_min = v;
            }
            if v > out_max {
                out_max = v;
            }
        }
        if count > 0 {
            // recompute properly for small counts
            out_min = values.iter().copied().fold(f32::INFINITY, f32::min);
            out_max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            out_sum = values.iter().copied().sum();
            out_sumsq = values.iter().map(|&v| v * v).sum();
        }
    }

    let mean = if count > 0 { out_sum as f64 / count as f64 } else { 0.0 };
    let variance = if count > 0 {
        let sq_mean = (out_sumsq as f64) / count as f64;
        (sq_mean - mean * mean).max(0.0)
    } else {
        0.0
    };
    let stddev = variance.sqrt();

    let (p25, p50, p75) = if count >= 16 {
        let mut p25: f32 = 0.0;
        let mut p50: f32 = 0.0;
        let mut p75: f32 = 0.0;
        unsafe {
            let f: Symbol<F32PercentilesFn> = lib_stats
                .get(b"f32_percentiles\0")
                .map_err(|e| ProcessError::KernelLoad {
                    lib: "csv_stats".into(),
                    cause: e.to_string(),
                })?;
            f(
                values.as_ptr(),
                count as i32,
                out_min,
                out_max,
                &mut p25,
                &mut p50,
                &mut p75,
            );
        }
        (p25 as f64, p50 as f64, p75 as f64)
    } else if count > 0 {
        let mut sorted: Vec<f32> = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let pct = |q: f64| -> f64 {
            let pos = q * (count - 1) as f64;
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(count - 1);
            let frac = pos - lo as f64;
            sorted[lo] as f64 * (1.0 - frac) + sorted[hi] as f64 * frac
        };
        (pct(0.25), pct(0.50), pct(0.75))
    } else {
        (0.0, 0.0, 0.0)
    };

    let mn = out_min as f64;
    let mx = out_max as f64;
    let is_integer = mn == mn.floor() && mx == mx.floor() && mx.abs() < 1e7;

    if is_integer {
        Ok(ColumnStats::Integer {
            name,
            col_index,
            rows: total_rows,
            count,
            nulls,
            min: mn,
            max: mx,
            mean,
            stddev,
            sum: out_sum as f64,
            p25,
            p50,
            p75,
        })
    } else {
        Ok(ColumnStats::Float {
            name,
            col_index,
            rows: total_rows,
            count,
            nulls,
            min: mn,
            max: mx,
            mean,
            stddev,
            sum: out_sum as f64,
            p25,
            p50,
            p75,
        })
    }
}

fn string_stats(
    lib_parse: &Library,
    fs_buf: &[i32],
    fe_buf: &[i32],
    total_rows: usize,
    col_index: usize,
    name: std::string::String,
) -> Result<ColumnStats, ProcessError> {
    let mut out_min_len: i32 = 0;
    let mut out_max_len: i32 = 0;
    let mut out_total_len: i32 = 0;
    let mut out_null_count: i32 = 0;

    unsafe {
        let f: Symbol<FieldLengthStatsFn> = lib_parse
            .get(b"field_length_stats\0")
            .map_err(|e| ProcessError::KernelLoad {
                lib: "csv_parse".into(),
                cause: e.to_string(),
            })?;
        f(
            fs_buf.as_ptr(),
            fe_buf.as_ptr(),
            total_rows as i32,
            &mut out_min_len,
            &mut out_max_len,
            &mut out_total_len,
            &mut out_null_count,
        );
    }

    let nulls = out_null_count as usize;
    let count = total_rows - nulls;
    let mean_length = if count > 0 {
        out_total_len as f64 / count as f64
    } else {
        0.0
    };

    Ok(ColumnStats::String {
        name,
        col_index,
        rows: total_rows,
        count,
        nulls,
        min_length: out_min_len as usize,
        max_length: out_max_len as usize,
        mean_length,
    })
}

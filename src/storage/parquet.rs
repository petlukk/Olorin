//! Parquet footer reader: parses FileMetaData (Thrift compact) and
//! aggregates per-column stats across row groups via `f64_stats`.
//! Footer-only — never decodes column data.

use crate::storage::parquet_meta::{self, FileMetaData};
use crate::storage::thrift_compact::ThriftReader;

const PARQUET_MAGIC: &[u8] = b"PAR1";

/// Physical type of a Parquet column (matches `parquet.thrift::Type`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PhysicalType {
    Boolean,
    Int32,
    Int64,
    Int96,
    Float,
    Double,
    ByteArray,
    FixedLenByteArray,
}

impl PhysicalType {
    pub(super) fn from_i32(v: i32) -> Option<Self> {
        Some(match v {
            0 => Self::Boolean,
            1 => Self::Int32,
            2 => Self::Int64,
            3 => Self::Int96,
            4 => Self::Float,
            5 => Self::Double,
            6 => Self::ByteArray,
            7 => Self::FixedLenByteArray,
            _ => return None,
        })
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Boolean => "bool",
            Self::Int32 | Self::Int64 | Self::Int96 => "number",
            Self::Float | Self::Double => "number",
            Self::ByteArray | Self::FixedLenByteArray => "text",
        }
    }
}

/// Decoded numeric statistic — a single value (the bytes interpreted by
/// the column's physical type). For BYTE_ARRAY / FixedLenByteArray we
/// don't decode and the stat is absent.
#[derive(Debug, Clone, Copy)]
pub enum NumStat { I64(i64), F64(f64), Bool(bool) }

impl NumStat {
    pub fn as_f64(self) -> f64 {
        match self {
            Self::I64(v) => v as f64,
            Self::F64(v) => v,
            Self::Bool(v) => if v { 1.0 } else { 0.0 },
        }
    }
}

/// Resolution of a TIMESTAMP logical type (parquet.thrift `TimeUnit`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeUnit { Millis, Micros, Nanos, Seconds }

impl TimeUnit {
    /// Convert a raw epoch count in this unit to whole epoch-seconds.
    pub fn to_epoch_seconds(self, raw: i64) -> i64 {
        match self {
            Self::Millis  => raw.div_euclid(1_000),
            Self::Micros  => raw.div_euclid(1_000_000),
            Self::Nanos   => raw.div_euclid(1_000_000_000),
            // INT96 is decoded directly to whole epoch-seconds.
            Self::Seconds => raw,
        }
    }
}

/// A column's logical (annotated) type, when it carries one we render
/// specially. Modern Parquet (pyarrow) marks timestamps via `LogicalType`
/// (SchemaElement field 10) only — `ConvertedType` is often absent — so
/// the footer reader must parse the union to recognize them at all.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogicalKind {
    /// A physical INT64 holding an epoch count; `utc` is `isAdjustedToUTC`.
    Timestamp { unit: TimeUnit, utc: bool },
    /// A DECIMAL: the physical value (INT32/INT64 little-endian, or
    /// FIXED_LEN_BYTE_ARRAY/BYTE_ARRAY big-endian two's-complement) is an
    /// unscaled integer; the real value is `unscaled / 10^scale`.
    Decimal { scale: i32 },
}

/// Decode a big-endian two's-complement integer of up to 16 bytes (the
/// FIXED_LEN_BYTE_ARRAY / BYTE_ARRAY encoding of a DECIMAL's unscaled
/// value). Sign-extends from the top bit. Returns `None` for empty or
/// >16-byte values (decimal256 is out of scope).
fn be_twos_complement_i128(bytes: &[u8]) -> Option<i128> {
    if bytes.is_empty() || bytes.len() > 16 {
        return None;
    }
    let mut v: i128 = if bytes[0] & 0x80 != 0 { -1 } else { 0 };
    for &b in bytes {
        v = (v << 8) | (b as i128);
    }
    Some(v)
}

/// Decode a DECIMAL stat value to its real (scaled) f64. The unscaled
/// integer comes from the physical encoding; we divide by `10^scale`.
fn decode_decimal(pt: PhysicalType, bytes: &[u8], scale: i32) -> Option<f64> {
    let unscaled: i128 = match pt {
        PhysicalType::Int32 if bytes.len() == 4 =>
            i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as i128,
        PhysicalType::Int64 if bytes.len() == 8 => {
            let mut b = [0u8; 8]; b.copy_from_slice(bytes);
            i64::from_le_bytes(b) as i128
        }
        PhysicalType::FixedLenByteArray | PhysicalType::ByteArray =>
            be_twos_complement_i128(bytes)?,
        _ => return None,
    };
    Some(unscaled as f64 / 10f64.powi(scale))
}

/// Aggregated per-column summary across all row groups.
#[derive(Debug, Clone)]
pub struct ColumnSummary {
    pub name: String,
    pub physical_type: PhysicalType,
    /// Logical type annotation (e.g. TIMESTAMP), when present.
    pub logical: Option<LogicalKind>,
    /// Total values across all row groups (includes nulls; equals the row
    /// count for a top-level column).
    pub total_values: i64,
    /// Sum of null counts across row groups, when statistics were present.
    pub null_count: Option<i64>,
    /// Min across row groups, when present and decodable.
    pub min: Option<NumStat>,
    /// Max across row groups, when present and decodable.
    pub max: Option<NumStat>,
}

/// Top-level file summary returned by [`read_summary`].
#[derive(Debug, Clone)]
pub struct ParquetSummary {
    pub num_rows: i64,
    pub num_row_groups: usize,
    pub columns: Vec<ColumnSummary>,
}

/// Read and summarize a Parquet file's metadata.
///
/// `bytes` is the entire file mapped into memory (or read into a Vec).
/// Returns the aggregated per-column summary, or an `Err` if the file
/// isn't a valid Parquet file or the footer can't be decoded.
pub fn read_summary(bytes: &[u8]) -> Result<ParquetSummary, String> {
    if bytes.len() < 12 {
        return Err("file too short to be a Parquet file".into());
    }
    if &bytes[..4] != PARQUET_MAGIC {
        return Err("missing PAR1 magic at start".into());
    }
    if &bytes[bytes.len() - 4..] != PARQUET_MAGIC {
        return Err("missing PAR1 magic at end".into());
    }
    // 4 bytes before trailing magic = footer length (i32 LE).
    let len_off = bytes.len() - 8;
    let footer_len = i32::from_le_bytes([
        bytes[len_off], bytes[len_off + 1], bytes[len_off + 2], bytes[len_off + 3],
    ]) as usize;
    if footer_len + 8 > bytes.len() {
        return Err(format!("footer length {footer_len} exceeds file size"));
    }
    let footer_start = bytes.len() - 8 - footer_len;
    let footer = &bytes[footer_start..bytes.len() - 8];

    let mut r = ThriftReader::new(footer);
    let meta = parquet_meta::read_file_metadata(&mut r)?;
    aggregate_summary(meta)
}

// ── Aggregation: row groups → per-column summary ──────────────────────────────

fn decode_stat_value(pt: PhysicalType, bytes: &[u8], unsigned: bool) -> Option<NumStat> {
    match pt {
        PhysicalType::Boolean if bytes.len() == 1 => Some(NumStat::Bool(bytes[0] != 0)),
        PhysicalType::Int32 if bytes.len() == 4 => {
            let raw = [bytes[0], bytes[1], bytes[2], bytes[3]];
            // UINT_8/16/32 share the INT32 physical type; decode unsigned so
            // values above 2^31 don't wrap to negative. u32 fits in i64.
            if unsigned {
                Some(NumStat::I64(u32::from_le_bytes(raw) as i64))
            } else {
                Some(NumStat::I64(i32::from_le_bytes(raw) as i64))
            }
        }
        PhysicalType::Int64 if bytes.len() == 8 => {
            let mut b = [0u8; 8]; b.copy_from_slice(bytes);
            // UINT_64 can exceed i64::MAX; carry it as f64 (the stat pipeline
            // reduces through f64 anyway, so >2^53 is approximate either way —
            // but the value stays correct in sign and magnitude).
            if unsigned {
                Some(NumStat::F64(u64::from_le_bytes(b) as f64))
            } else {
                Some(NumStat::I64(i64::from_le_bytes(b)))
            }
        }
        PhysicalType::Float if bytes.len() == 4 => {
            Some(NumStat::F64(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64))
        }
        PhysicalType::Double if bytes.len() == 8 => {
            let mut b = [0u8; 8]; b.copy_from_slice(bytes);
            Some(NumStat::F64(f64::from_le_bytes(b)))
        }
        PhysicalType::Int96 => int96_to_epoch_seconds(bytes).map(NumStat::I64),
        _ => None,
    }
}

/// Decode a 12-byte INT96 timestamp to whole epoch-seconds.
///
/// The deprecated Impala/Hive/Spark encoding: first 8 bytes = i64
/// nanoseconds within the day (LE), last 4 = i32 Julian day number (LE).
/// `epoch_seconds = (jd − 2440588)·86400 + nanos/1e9` (2440588 = the Julian
/// day of the Unix epoch, 1970-01-01). Decoded to whole seconds: the footer
/// min/max render to second-resolution ISO, and seconds (~1.7e9) stay exact
/// through the f64 reduction pipeline where raw nanos (~1.7e18) would not.
///
/// Returns `None` unless `bytes.len() == 12`. NOTE: pyarrow and many engines
/// omit statistics for INT96 columns (its sort order is undefined), so a
/// real file often has no min/max to decode here — but when stats ARE
/// present, this turns them into instants instead of skipping them.
pub fn int96_to_epoch_seconds(bytes: &[u8]) -> Option<i64> {
    if bytes.len() != 12 {
        return None;
    }
    let mut nb = [0u8; 8];
    nb.copy_from_slice(&bytes[0..8]);
    let nanos = i64::from_le_bytes(nb);
    let jd = i32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as i64;
    Some((jd - 2_440_588) * 86_400 + nanos.div_euclid(1_000_000_000))
}

/// Per-column scratch: collect every row-group's stat into a Vec, then
/// reduce all three vectors via a single SIMD `f64_stats` call each.
/// For files with hundreds of row groups (large parquets), this is
/// genuinely SIMD reduction work; for small files (1-2 row groups) it's
/// equivalent to scalar but stays on the kernel-first dispatch path.
struct ColumnScratch {
    physical_type: PhysicalType,
    logical:     Option<LogicalKind>,
    unsigned:    bool,
    total_values: i64,
    mins:        Vec<f64>,
    maxes:       Vec<f64>,
    null_counts: Vec<f64>,
}

fn aggregate_summary(meta: FileMetaData) -> Result<ParquetSummary, String> {
    if meta.row_groups.is_empty() {
        return Err("no row groups in file".into());
    }
    // Schema: root + flat columns. Skip the root (first) element; remaining
    // are leaf columns. Anything with num_children > 0 (nested group) is
    // skipped from the column list since v1 is flat-only.
    let name_to_type: std::collections::HashMap<&str, PhysicalType> = meta.schema.iter()
        .skip(1)
        .filter(|e| e.num_children == 0)
        .filter_map(|e| e.physical_type.map(|t| (e.name.as_str(), t)))
        .collect();
    // Columns whose ConvertedType is UINT_8/16/32/64 (11..=14) hold unsigned
    // values in a signed INT32/INT64 physical type.
    let name_to_unsigned: std::collections::HashMap<&str, bool> = meta.schema.iter()
        .skip(1)
        .filter(|e| e.num_children == 0)
        .filter_map(|e| e.physical_type.map(|_|
            (e.name.as_str(), matches!(e.converted_type, Some(11..=14)))))
        .collect();
    // LogicalType annotations (TIMESTAMP / DECIMAL) keyed by column name.
    // Falls back to the legacy ConvertedType DECIMAL (5) + SchemaElement
    // scale when no modern LogicalType is present.
    let name_to_logical: std::collections::HashMap<&str, LogicalKind> = meta.schema.iter()
        .skip(1)
        .filter(|e| e.num_children == 0)
        .filter_map(|e| {
            let l = e.logical.or_else(|| {
                (e.converted_type == Some(5))
                    .then(|| LogicalKind::Decimal { scale: e.scale.unwrap_or(0) })
            })?;
            Some((e.name.as_str(), l))
        })
        .collect();

    // Phase 1: walk row groups, collect per-column f64 vectors of mins,
    // maxes, null_counts. No reduction yet — that's the SIMD step.
    let mut scratches: std::collections::HashMap<String, ColumnScratch> =
        std::collections::HashMap::new();
    let mut order: Vec<String> = Vec::new();

    for rg in &meta.row_groups {
        for col in &rg.columns {
            let m = &col.meta;
            let Some(name) = m.path_in_schema.first() else { continue; };
            let pt = match m.physical_type.or_else(|| name_to_type.get(name.as_str()).copied()) {
                Some(t) => t,
                None => continue,
            };
            let unsigned = name_to_unsigned.get(name.as_str()).copied().unwrap_or(false);
            // INT96 is always a timestamp (it carries no LogicalType
            // annotation), so mark it implicitly; otherwise use the schema's
            // annotation. This routes INT96 through the ISO-instant renderer.
            let logical = if pt == PhysicalType::Int96 {
                Some(LogicalKind::Timestamp { unit: TimeUnit::Seconds, utc: false })
            } else {
                name_to_logical.get(name.as_str()).copied()
            };
            let entry = scratches.entry(name.clone()).or_insert_with(|| {
                order.push(name.clone());
                ColumnScratch {
                    physical_type: pt,
                    logical,
                    unsigned,
                    total_values: 0,
                    mins:        Vec::with_capacity(meta.row_groups.len()),
                    maxes:       Vec::with_capacity(meta.row_groups.len()),
                    null_counts: Vec::with_capacity(meta.row_groups.len()),
                }
            });
            entry.total_values += m.num_values;
            // Decimal stats decode through the unscaled-integer path; all
            // others through the physical-type decoder. Read the column's
            // logical/unsigned into locals first to avoid borrowing `entry`
            // while pushing into its vectors.
            let col_logical = entry.logical;
            let col_unsigned = entry.unsigned;
            let decode = |bytes: &[u8]| -> Option<f64> {
                if let Some(LogicalKind::Decimal { scale }) = col_logical {
                    decode_decimal(pt, bytes, scale)
                } else {
                    decode_stat_value(pt, bytes, col_unsigned).map(|v| v.as_f64())
                }
            };
            if let Some(stats) = &m.statistics {
                if let Some(nc) = stats.null_count {
                    entry.null_counts.push(nc as f64);
                }
                if let Some(min_b) = &stats.min_value {
                    if let Some(v) = decode(min_b) { entry.mins.push(v); }
                }
                if let Some(max_b) = &stats.max_value {
                    if let Some(v) = decode(max_b) { entry.maxes.push(v); }
                }
            }
        }
    }

    // Phase 2: SIMD reduction. For each column, run `f64_stats` over
    // mins[], maxes[], and null_counts[]. The min-of-mins comes from
    // the kernel's `out_min`; the max-of-maxes from `out_max`; the
    // total null_count from `out_sum`. One kernel call per axis per
    // column = 3*C kernel calls for C columns. For files with R row
    // groups, each call processes R values — real SIMD reduction at
    // production scale.
    let columns: Vec<ColumnSummary> = order.into_iter()
        .filter_map(|name| {
            let s = scratches.remove(&name)?;
            let pt = s.physical_type;

            let min = simd_reduce_min(&s.mins, pt, s.logical);
            let max = simd_reduce_max(&s.maxes, pt, s.logical);
            let nulls = simd_reduce_sum(&s.null_counts);

            Some(ColumnSummary {
                name,
                physical_type: pt,
                logical: s.logical,
                total_values: s.total_values,
                null_count: nulls,
                min,
                max,
            })
        })
        .collect();

    Ok(ParquetSummary {
        num_rows: meta.num_rows,
        num_row_groups: meta.row_groups.len(),
        columns,
    })
}

/// Reduce a Vec<f64> of per-row-group min values to a single min via
/// the SIMD `f64_stats` kernel. Returns `None` if the vec is empty.
fn simd_reduce_min(vals: &[f64], pt: PhysicalType, logical: Option<LogicalKind>) -> Option<NumStat> {
    if vals.is_empty() { return None; }
    let v = simd_min_max(vals)?.0;
    Some(typed_stat(pt, v, logical))
}

fn simd_reduce_max(vals: &[f64], pt: PhysicalType, logical: Option<LogicalKind>) -> Option<NumStat> {
    if vals.is_empty() { return None; }
    let v = simd_min_max(vals)?.1;
    Some(typed_stat(pt, v, logical))
}

fn simd_reduce_sum(vals: &[f64]) -> Option<i64> {
    if vals.is_empty() { return None; }
    let s = simd_full_stats(vals)?.0;
    Some(s as i64)
}

/// Run `f64_stats` once over `vals`, returning (sum, min, max).
fn simd_full_stats(vals: &[f64]) -> Option<(f64, f64, f64)> {
    use crate::kernels::ffi;
    if vals.is_empty() { return None; }
    let mut count = 0i32;
    let mut sum = 0f64; let mut mn = 0f64; let mut mx = 0f64;
    unsafe {
        ffi::f64_stats(
            vals.as_ptr(), vals.len() as i32,
            &mut count, &mut sum, &mut mn, &mut mx,
        );
    }
    if count == 0 { return None; }
    Some((sum, mn, mx))
}

fn simd_min_max(vals: &[f64]) -> Option<(f64, f64)> {
    let (_sum, mn, mx) = simd_full_stats(vals)?;
    Some((mn, mx))
}

/// Convert a reduced f64 value back into the column's NumStat representation
/// so format_column can print it with the right shape (int vs float vs bool).
fn typed_stat(pt: PhysicalType, v: f64, logical: Option<LogicalKind>) -> NumStat {
    // A DECIMAL's value is the scaled f64 (e.g. 19.99); never coerce it to an
    // integer, even when the physical type is INT32/INT64.
    if let Some(LogicalKind::Decimal { .. }) = logical {
        return NumStat::F64(v);
    }
    match pt {
        PhysicalType::Boolean => NumStat::Bool(v != 0.0),
        PhysicalType::Int32 | PhysicalType::Int64 | PhysicalType::Int96 => {
            // A reduced UINT_64 stat can exceed i64::MAX (decode_stat_value
            // carries it as f64). `v as i64` SATURATES — Rust's float→int cast
            // pins anything above i64::MAX to i64::MAX, so u64 maxima > 2^63
            // would collapse to ~9.22e18. Keep out-of-range values as F64 so
            // the magnitude survives; as_f64() (the only consumer) is identical
            // either way for in-range values.
            if (i64::MIN as f64..=i64::MAX as f64).contains(&v) {
                NumStat::I64(v as i64)
            } else {
                NumStat::F64(v)
            }
        }
        PhysicalType::Float | PhysicalType::Double => NumStat::F64(v),
        _ => NumStat::F64(v),
    }
}

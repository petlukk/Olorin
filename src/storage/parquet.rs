//! Minimal Parquet metadata reader. Parses the file footer (Thrift compact
//! encoded `FileMetaData`) and extracts what `eaparquet` needs to summarize
//! a file: schema, row count, per-column type, and per-column statistics
//! (min/max/null_count) aggregated across row groups.
//!
//! Scope:
//! - Reads the trailing footer only — never decodes column data.
//! - Supports primitive types: BOOLEAN, INT32, INT64, FLOAT, DOUBLE.
//!   BYTE_ARRAY (strings) and INT96 (legacy timestamp) are reported with
//!   type but their stats are left absent.
//! - Flat schemas only — nested groups (LIST/MAP/STRUCT) are skipped.
//!
//! Hand-rolled Thrift compact decoder. The full spec is short and
//! well-defined; pulling in the `thrift` or `parquet` crates would
//! violate the 2-dependency rule.

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
    fn from_i32(v: i32) -> Option<Self> {
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

/// Aggregated per-column summary across all row groups.
#[derive(Debug, Clone)]
pub struct ColumnSummary {
    pub name: String,
    pub physical_type: PhysicalType,
    /// Total non-null values across all row groups, when known.
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
    let meta = read_file_metadata(&mut r)?;
    aggregate_summary(meta)
}

// ── Thrift Compact decoder ────────────────────────────────────────────────────

struct ThriftReader<'a> { bytes: &'a [u8], pos: usize }

#[derive(Debug, Clone, Copy, PartialEq)]
enum CompactType { Stop, True, False, I8, I16, I32, I64, Double, Binary, List, Set, Map, Struct, Other(u8) }

impl CompactType {
    fn from_byte(b: u8) -> Self {
        match b {
            0 => Self::Stop, 1 => Self::True, 2 => Self::False, 3 => Self::I8,
            4 => Self::I16, 5 => Self::I32, 6 => Self::I64, 7 => Self::Double,
            8 => Self::Binary, 9 => Self::List, 10 => Self::Set, 11 => Self::Map,
            12 => Self::Struct,
            o => Self::Other(o),
        }
    }
}

impl<'a> ThriftReader<'a> {
    fn new(bytes: &'a [u8]) -> Self { Self { bytes, pos: 0 } }

    fn read_byte(&mut self) -> Result<u8, String> {
        let b = *self.bytes.get(self.pos).ok_or("unexpected end of footer")?;
        self.pos += 1;
        Ok(b)
    }

    fn read_varint_u64(&mut self) -> Result<u64, String> {
        let mut result: u64 = 0;
        let mut shift = 0;
        for _ in 0..10 {
            let b = self.read_byte()?;
            result |= ((b & 0x7f) as u64) << shift;
            if b & 0x80 == 0 { return Ok(result); }
            shift += 7;
        }
        Err("varint exceeds 10 bytes".into())
    }

    fn read_zigzag_i32(&mut self) -> Result<i32, String> {
        let v = self.read_varint_u64()?;
        Ok(((v >> 1) as i32) ^ -((v & 1) as i32))
    }

    fn read_zigzag_i64(&mut self) -> Result<i64, String> {
        let v = self.read_varint_u64()?;
        Ok(((v >> 1) as i64) ^ -((v & 1) as i64))
    }

    fn read_double(&mut self) -> Result<f64, String> {
        if self.pos + 8 > self.bytes.len() { return Err("short double".into()); }
        let mut buf = [0u8; 8];
        buf.copy_from_slice(&self.bytes[self.pos..self.pos + 8]);
        self.pos += 8;
        Ok(f64::from_le_bytes(buf))
    }

    fn read_binary(&mut self) -> Result<&'a [u8], String> {
        let len = self.read_varint_u64()? as usize;
        if self.pos + len > self.bytes.len() { return Err("short binary".into()); }
        let s = &self.bytes[self.pos..self.pos + len];
        self.pos += len;
        Ok(s)
    }

    /// Read a list header: returns (element_type, length).
    fn read_list_header(&mut self) -> Result<(CompactType, usize), String> {
        let b = self.read_byte()?;
        let elem_type = CompactType::from_byte(b & 0x0f);
        let size_hi = (b >> 4) & 0x0f;
        let size = if size_hi == 0x0f {
            self.read_varint_u64()? as usize
        } else {
            size_hi as usize
        };
        Ok((elem_type, size))
    }

    /// Skip a value of the given compact type. Used to walk past fields we
    /// don't care about so we can find the ones we do.
    fn skip_value(&mut self, ty: CompactType) -> Result<(), String> {
        match ty {
            CompactType::True | CompactType::False | CompactType::Stop => {}
            CompactType::I8 => { self.read_byte()?; }
            CompactType::I16 | CompactType::I32 => { self.read_zigzag_i32()?; }
            CompactType::I64 => { self.read_zigzag_i64()?; }
            CompactType::Double => { self.read_double()?; }
            CompactType::Binary => { self.read_binary()?; }
            CompactType::List | CompactType::Set => {
                let (et, n) = self.read_list_header()?;
                for _ in 0..n { self.skip_value(et)?; }
            }
            CompactType::Map => {
                let n = self.read_varint_u64()? as usize;
                if n > 0 {
                    let kv = self.read_byte()?;
                    let kt = CompactType::from_byte((kv >> 4) & 0x0f);
                    let vt = CompactType::from_byte(kv & 0x0f);
                    for _ in 0..n {
                        self.skip_value(kt)?;
                        self.skip_value(vt)?;
                    }
                }
            }
            CompactType::Struct => { self.skip_struct()?; }
            CompactType::Other(o) => return Err(format!("unknown compact type {o}")),
        }
        Ok(())
    }

    /// Skip until STOP, drainage for unknown nested structs.
    fn skip_struct(&mut self) -> Result<(), String> {
        let mut last_field_id: i16 = 0;
        loop {
            let h = self.read_byte()?;
            if h == 0 { return Ok(()); } // STOP
            let ty = CompactType::from_byte(h & 0x0f);
            let delta = (h >> 4) as i16;
            if delta != 0 {
                last_field_id += delta;
            } else {
                last_field_id = self.read_zigzag_i32()? as i16;
            }
            let _ = last_field_id;
            self.skip_value(ty)?;
        }
    }

    /// Read field header: returns Some((field_id, type)) or None on STOP.
    fn read_field_header(&mut self, last_field_id: i16) -> Result<Option<(i16, CompactType)>, String> {
        let h = self.read_byte()?;
        if h == 0 { return Ok(None); }
        let ty = CompactType::from_byte(h & 0x0f);
        let delta = (h >> 4) as i16;
        let field_id = if delta != 0 {
            last_field_id + delta
        } else {
            self.read_zigzag_i32()? as i16
        };
        Ok(Some((field_id, ty)))
    }

}

// ── Parquet metadata structs (intermediate) ───────────────────────────────────

#[derive(Debug, Default)]
struct FileMetaData {
    num_rows: i64,
    schema: Vec<SchemaElement>,
    row_groups: Vec<RowGroup>,
}

#[derive(Debug, Default, Clone)]
struct SchemaElement {
    name: String,
    physical_type: Option<PhysicalType>,
    num_children: i32,
}

#[derive(Debug, Default)]
struct RowGroup {
    columns: Vec<ColumnChunk>,
    num_rows: i64,
}

#[derive(Debug, Default)]
struct ColumnChunk {
    meta: ColumnMetaData,
}

#[derive(Debug, Default)]
struct ColumnMetaData {
    physical_type: Option<PhysicalType>,
    path_in_schema: Vec<String>,
    num_values: i64,
    statistics: Option<Statistics>,
}

#[derive(Debug, Default)]
struct Statistics {
    null_count: Option<i64>,
    min_value: Option<Vec<u8>>,
    max_value: Option<Vec<u8>>,
}

// ── Parquet struct decoders ───────────────────────────────────────────────────

fn read_file_metadata(r: &mut ThriftReader) -> Result<FileMetaData, String> {
    let mut meta = FileMetaData::default();
    let mut last: i16 = 0;
    while let Some((fid, ty)) = r.read_field_header(last)? {
        last = fid;
        match (fid, ty) {
            (1, CompactType::I32) => { let _ = r.read_zigzag_i32()?; } // version
            (2, CompactType::List) => {
                let (et, n) = r.read_list_header()?;
                if et != CompactType::Struct { return Err("schema list element type not Struct".into()); }
                for _ in 0..n {
                    meta.schema.push(read_schema_element(r)?);
                }
            }
            (3, CompactType::I64) => { meta.num_rows = r.read_zigzag_i64()?; }
            (4, CompactType::List) => {
                let (et, n) = r.read_list_header()?;
                if et != CompactType::Struct { return Err("row_groups list element type not Struct".into()); }
                for _ in 0..n {
                    meta.row_groups.push(read_row_group(r)?);
                }
            }
            _ => r.skip_value(ty)?,
        }
    }
    Ok(meta)
}

fn read_schema_element(r: &mut ThriftReader) -> Result<SchemaElement, String> {
    let mut e = SchemaElement::default();
    let mut last: i16 = 0;
    while let Some((fid, ty)) = r.read_field_header(last)? {
        last = fid;
        match (fid, ty) {
            (1, CompactType::I32) => {
                e.physical_type = PhysicalType::from_i32(r.read_zigzag_i32()?);
            }
            (4, CompactType::Binary) => {
                e.name = String::from_utf8_lossy(r.read_binary()?).into_owned();
            }
            (5, CompactType::I32) => {
                e.num_children = r.read_zigzag_i32()?;
            }
            _ => r.skip_value(ty)?,
        }
    }
    Ok(e)
}

fn read_row_group(r: &mut ThriftReader) -> Result<RowGroup, String> {
    let mut rg = RowGroup::default();
    let mut last: i16 = 0;
    while let Some((fid, ty)) = r.read_field_header(last)? {
        last = fid;
        match (fid, ty) {
            (1, CompactType::List) => {
                let (et, n) = r.read_list_header()?;
                if et != CompactType::Struct { return Err("columns list element type not Struct".into()); }
                for _ in 0..n {
                    rg.columns.push(read_column_chunk(r)?);
                }
            }
            (3, CompactType::I64) => { rg.num_rows = r.read_zigzag_i64()?; }
            _ => r.skip_value(ty)?,
        }
    }
    Ok(rg)
}

fn read_column_chunk(r: &mut ThriftReader) -> Result<ColumnChunk, String> {
    let mut cc = ColumnChunk::default();
    let mut last: i16 = 0;
    while let Some((fid, ty)) = r.read_field_header(last)? {
        last = fid;
        match (fid, ty) {
            (3, CompactType::Struct) => { cc.meta = read_column_metadata(r)?; }
            _ => r.skip_value(ty)?,
        }
    }
    Ok(cc)
}

fn read_column_metadata(r: &mut ThriftReader) -> Result<ColumnMetaData, String> {
    let mut m = ColumnMetaData::default();
    let mut last: i16 = 0;
    while let Some((fid, ty)) = r.read_field_header(last)? {
        last = fid;
        match (fid, ty) {
            (1, CompactType::I32) => {
                m.physical_type = PhysicalType::from_i32(r.read_zigzag_i32()?);
            }
            (3, CompactType::List) => {
                let (et, n) = r.read_list_header()?;
                if et != CompactType::Binary { return Err("path_in_schema element type not Binary".into()); }
                for _ in 0..n {
                    m.path_in_schema.push(String::from_utf8_lossy(r.read_binary()?).into_owned());
                }
            }
            (5, CompactType::I64) => { m.num_values = r.read_zigzag_i64()?; }
            (12, CompactType::Struct) => { m.statistics = Some(read_statistics(r)?); }
            _ => r.skip_value(ty)?,
        }
    }
    Ok(m)
}

fn read_statistics(r: &mut ThriftReader) -> Result<Statistics, String> {
    let mut s = Statistics::default();
    let mut last: i16 = 0;
    while let Some((fid, ty)) = r.read_field_header(last)? {
        last = fid;
        match (fid, ty) {
            // Field 1 is `max` (deprecated), 2 is `min` (deprecated). Skip.
            (3, CompactType::I64) => { s.null_count = Some(r.read_zigzag_i64()?); }
            (5, CompactType::Binary) => { s.max_value = Some(r.read_binary()?.to_vec()); }
            (6, CompactType::Binary) => { s.min_value = Some(r.read_binary()?.to_vec()); }
            _ => r.skip_value(ty)?,
        }
    }
    Ok(s)
}

// ── Aggregation: row groups → per-column summary ──────────────────────────────

fn decode_stat_value(pt: PhysicalType, bytes: &[u8]) -> Option<NumStat> {
    match pt {
        PhysicalType::Boolean if bytes.len() == 1 => Some(NumStat::Bool(bytes[0] != 0)),
        PhysicalType::Int32 if bytes.len() == 4 => {
            Some(NumStat::I64(i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as i64))
        }
        PhysicalType::Int64 if bytes.len() == 8 => {
            let mut b = [0u8; 8]; b.copy_from_slice(bytes);
            Some(NumStat::I64(i64::from_le_bytes(b)))
        }
        PhysicalType::Float if bytes.len() == 4 => {
            Some(NumStat::F64(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as f64))
        }
        PhysicalType::Double if bytes.len() == 8 => {
            let mut b = [0u8; 8]; b.copy_from_slice(bytes);
            Some(NumStat::F64(f64::from_le_bytes(b)))
        }
        _ => None,
    }
}

/// Per-column scratch: collect every row-group's stat into a Vec, then
/// reduce all three vectors via a single SIMD `f64_stats` call each.
/// For files with hundreds of row groups (large parquets), this is
/// genuinely SIMD reduction work; for small files (1-2 row groups) it's
/// equivalent to scalar but stays on the kernel-first dispatch path.
struct ColumnScratch {
    physical_type: PhysicalType,
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
            let entry = scratches.entry(name.clone()).or_insert_with(|| {
                order.push(name.clone());
                ColumnScratch {
                    physical_type: pt,
                    total_values: 0,
                    mins:        Vec::with_capacity(meta.row_groups.len()),
                    maxes:       Vec::with_capacity(meta.row_groups.len()),
                    null_counts: Vec::with_capacity(meta.row_groups.len()),
                }
            });
            entry.total_values += m.num_values;
            if let Some(stats) = &m.statistics {
                if let Some(nc) = stats.null_count {
                    entry.null_counts.push(nc as f64);
                }
                if let Some(min_b) = &stats.min_value {
                    if let Some(v) = decode_stat_value(pt, min_b) {
                        entry.mins.push(v.as_f64());
                    }
                }
                if let Some(max_b) = &stats.max_value {
                    if let Some(v) = decode_stat_value(pt, max_b) {
                        entry.maxes.push(v.as_f64());
                    }
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

            let min = simd_reduce_min(&s.mins, pt);
            let max = simd_reduce_max(&s.maxes, pt);
            let nulls = simd_reduce_sum(&s.null_counts);

            Some(ColumnSummary {
                name,
                physical_type: pt,
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
fn simd_reduce_min(vals: &[f64], pt: PhysicalType) -> Option<NumStat> {
    if vals.is_empty() { return None; }
    let v = simd_min_max(vals)?.0;
    Some(typed_stat(pt, v))
}

fn simd_reduce_max(vals: &[f64], pt: PhysicalType) -> Option<NumStat> {
    if vals.is_empty() { return None; }
    let v = simd_min_max(vals)?.1;
    Some(typed_stat(pt, v))
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
fn typed_stat(pt: PhysicalType, v: f64) -> NumStat {
    match pt {
        PhysicalType::Boolean => NumStat::Bool(v != 0.0),
        PhysicalType::Int32 | PhysicalType::Int64 | PhysicalType::Int96 => NumStat::I64(v as i64),
        PhysicalType::Float | PhysicalType::Double => NumStat::F64(v),
        _ => NumStat::F64(v),
    }
}

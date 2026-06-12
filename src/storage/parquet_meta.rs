//! Parquet FileMetaData decode: the intermediate Thrift-compact structs and
//! their decoders. Split out of `parquet.rs` to keep each file under the
//! 500-LOC cap; `parquet.rs` owns the public types, `read_summary`
//! orchestration, and the row-group → per-column SIMD aggregation that
//! consumes the `FileMetaData` produced here.

use super::parquet::{LogicalKind, PhysicalType, TimeUnit};
use super::thrift_compact::{CompactType, ThriftReader};

// ── Parquet metadata structs (intermediate) ───────────────────────────────────

#[derive(Debug, Default)]
pub(super) struct FileMetaData {
    pub(super) num_rows: i64,
    pub(super) schema: Vec<SchemaElement>,
    pub(super) row_groups: Vec<RowGroup>,
}

#[derive(Debug, Default, Clone)]
pub(super) struct SchemaElement {
    pub(super) name: String,
    pub(super) physical_type: Option<PhysicalType>,
    pub(super) num_children: i32,
    /// Parquet ConvertedType (field 6). UINT_8/16/32/64 = 11..=14 — the
    /// signal that an INT32/INT64 physical column holds unsigned values.
    pub(super) converted_type: Option<i32>,
    /// Parquet LogicalType (field 10), the modern annotation. Parsed for
    /// TIMESTAMP; pyarrow writes it without a ConvertedType, so this is the
    /// only place a timestamp is recognizable.
    pub(super) logical: Option<LogicalKind>,
}

#[derive(Debug, Default)]
pub(super) struct RowGroup {
    pub(super) columns: Vec<ColumnChunk>,
    pub(super) num_rows: i64,
}

#[derive(Debug, Default)]
pub(super) struct ColumnChunk {
    pub(super) meta: ColumnMetaData,
}

#[derive(Debug, Default)]
pub(super) struct ColumnMetaData {
    pub(super) physical_type: Option<PhysicalType>,
    pub(super) path_in_schema: Vec<String>,
    pub(super) num_values: i64,
    pub(super) statistics: Option<Statistics>,
}

#[derive(Debug, Default)]
pub(super) struct Statistics {
    pub(super) null_count: Option<i64>,
    pub(super) min_value: Option<Vec<u8>>,
    pub(super) max_value: Option<Vec<u8>>,
}

// ── Parquet struct decoders ───────────────────────────────────────────────────

pub(super) fn read_file_metadata(r: &mut ThriftReader) -> Result<FileMetaData, String> {
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
            (6, CompactType::I32) => {
                e.converted_type = Some(r.read_zigzag_i32()?);
            }
            (10, CompactType::Struct) => {
                e.logical = read_logical_type(r)?;
            }
            _ => r.skip_value(ty)?,
        }
    }
    Ok(e)
}

/// Parse the `LogicalType` union (parquet.thrift). Only TIMESTAMP (union
/// field 8) is decoded; every other arm is skipped and yields `None`.
fn read_logical_type(r: &mut ThriftReader) -> Result<Option<LogicalKind>, String> {
    let mut logical = None;
    let mut last: i16 = 0;
    while let Some((fid, ty)) = r.read_field_header(last)? {
        last = fid;
        match (fid, ty) {
            (8, CompactType::Struct) => { logical = Some(read_timestamp_type(r)?); }
            _ => r.skip_value(ty)?,
        }
    }
    Ok(logical)
}

/// Parse `TimestampType { 1: bool isAdjustedToUTC, 2: TimeUnit unit }`.
/// Compact-protocol bools live in the field header type (True/False) with
/// no value byte. `unit` defaults to millis if absent (parquet's default).
fn read_timestamp_type(r: &mut ThriftReader) -> Result<LogicalKind, String> {
    let mut utc = false;
    let mut unit = TimeUnit::Millis;
    let mut last: i16 = 0;
    while let Some((fid, ty)) = r.read_field_header(last)? {
        last = fid;
        match (fid, ty) {
            (1, CompactType::True)   => utc = true,
            (1, CompactType::False)  => utc = false,
            (2, CompactType::Struct) => unit = read_time_unit(r)?,
            _ => r.skip_value(ty)?,
        }
    }
    Ok(LogicalKind::Timestamp { unit, utc })
}

/// Parse the `TimeUnit` union: each arm (MILLIS=1, MICROS=2, NANOS=3) is an
/// empty struct, so the present field id alone selects the unit.
fn read_time_unit(r: &mut ThriftReader) -> Result<TimeUnit, String> {
    let mut unit = TimeUnit::Millis;
    let mut last: i16 = 0;
    while let Some((fid, ty)) = r.read_field_header(last)? {
        last = fid;
        unit = match fid {
            1 => TimeUnit::Millis,
            2 => TimeUnit::Micros,
            3 => TimeUnit::Nanos,
            _ => unit,
        };
        r.skip_value(ty)?; // empty struct body
    }
    Ok(unit)
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

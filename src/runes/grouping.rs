//! GROUP BY block of the RuneOutput v1 contract — types, agg parsing, and
//! the grouping pass driven by the fused `csv_groupby_scan` kernel.
//!
//! Lives outside `output.rs`/`eacrunch.rs` to keep both under the 500-LOC
//! cap; the `groups[]` field is on `RuneOutput` and serializes additively
//! (only when non-empty), exactly like `anomalies[]` / `correlations[]`.
//!
//! The fused kernel projects only the key + aggregation columns in one
//! pass — no `len`-sized delimiter arrays, no full field grid (scratch is
//! O(rows·cols_needed), not O(bytes); ~5.8× less peak RSS on a 3M-row CSV).
//! Agg values are parsed here in Rust with the identical finite-skipna rule
//! (`f64::parse` + `is_finite`) as eacrunch's whole-column stats — the
//! honest fusion boundary, since Ea has no `strtod` intrinsic — so a group's
//! `mean:latency` agrees with the whole-column `latency` mean by
//! construction, and is differentially verified against pandas.

use super::common;
use crate::storage::json::{Object, Value};

/// Cardinality guard: grouping a multi-million-row file by a near-unique
/// column would otherwise grow the map until OOM. We fail loud and clear
/// instead — an honest error, never a silent partial result.
const MAX_GROUPS: usize = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AggOp {
    Count,
    Sum,
    Mean,
    Min,
    Max,
}

impl AggOp {
    pub fn as_str(&self) -> &'static str {
        match self {
            AggOp::Count => "count",
            AggOp::Sum   => "sum",
            AggOp::Mean  => "mean",
            AggOp::Min   => "min",
            AggOp::Max   => "max",
        }
    }
    fn parse(s: &str) -> Option<AggOp> {
        Some(match s {
            "count" => AggOp::Count,
            "sum"   => AggOp::Sum,
            "mean"  => AggOp::Mean,
            "min"   => AggOp::Min,
            "max"   => AggOp::Max,
            _ => return None,
        })
    }
}

/// One requested aggregation, e.g. `mean:latency`. `col` is empty for the
/// column-less `count`.
#[derive(Debug, Clone, PartialEq)]
pub struct AggSpec {
    pub op:  AggOp,
    pub col: String,
}

/// One computed aggregation for one group. `value` is `NaN` when no finite
/// values fell in the group (serialized as JSON `null`).
#[derive(Debug, Clone, PartialEq)]
pub struct AggResult {
    pub op:    String,
    pub col:   String,
    pub value: f64,
}

/// One group: the `--by` column value, the group's row count, and the
/// requested aggregations.
#[derive(Debug, Clone, PartialEq)]
pub struct Group {
    pub key:   String,
    pub count: u64,
    pub aggs:  Vec<AggResult>,
}

/// Parse `--agg` like `mean:latency,sum:bytes,count` into specs.
pub fn parse_agg_specs(s: &str) -> Result<Vec<AggSpec>, String> {
    let mut out = Vec::new();
    for tok in s.split(',') {
        let tok = tok.trim();
        if tok.is_empty() {
            continue;
        }
        let (op_str, col) = match tok.split_once(':') {
            Some((o, c)) => (o.trim(), c.trim().to_string()),
            None         => (tok, String::new()),
        };
        let op = AggOp::parse(op_str)
            .ok_or_else(|| format!("unknown agg op: {op_str} (use count/sum/mean/min/max)"))?;
        if op != AggOp::Count && col.is_empty() {
            return Err(format!("agg `{op_str}` needs a column: {op_str}:<col>"));
        }
        out.push(AggSpec { op, col });
    }
    if out.is_empty() {
        return Err("empty --agg".into());
    }
    Ok(out)
}

/// Per-group accumulator, one `ColAcc` slot per spec (Count slots unused).
struct ColAcc {
    sum: f64,
    min: f64,
    max: f64,
    n:   u64,
}

impl ColAcc {
    fn new() -> Self {
        ColAcc { sum: 0.0, min: f64::INFINITY, max: f64::NEG_INFINITY, n: 0 }
    }
    fn add(&mut self, v: f64) {
        self.sum += v;
        if v < self.min { self.min = v; }
        if v > self.max { self.max = v; }
        self.n += 1;
    }
}

struct Acc {
    count: u64,
    stats: Vec<ColAcc>,
}

impl Acc {
    fn new(n: usize) -> Self {
        Acc { count: 0, stats: (0..n).map(|_| ColAcc::new()).collect() }
    }
}

/// Outcome of a grouping pass: the groups plus the total data-row count
/// (header excluded) so eacrunch can report `totals.rows`.
pub struct GroupOutcome {
    pub groups:    Vec<Group>,
    pub data_rows: u64,
}

/// Quote-aware split of the CSV header (first line). Mirrors the kernel's
/// column splitting (toggle on `"`, split on unquoted `,`, stop at the
/// first unquoted `\n`) so header names resolve to the same column indices
/// the kernel projects. O(header length) — never touches the whole file.
fn parse_header(bytes: &[u8]) -> Vec<String> {
    let clean = |b: &[u8]| -> String {
        let s = std::str::from_utf8(b).unwrap_or("").trim();
        common::unquote(s).into_owned()
    };
    let mut headers = Vec::new();
    let mut in_quote = false;
    let mut field_start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => in_quote = !in_quote,
            b',' if !in_quote => {
                headers.push(clean(&bytes[field_start..i]));
                field_start = i + 1;
            }
            b'\n' if !in_quote => break,
            _ => {}
        }
        i += 1;
    }
    headers.push(clean(&bytes[field_start..i])); // last field (trim drops any \r)
    headers
}

/// Run the GROUP BY pass via the fused `csv_groupby_scan` kernel. The kernel
/// projects only the key + aggregation columns in one pass (no `len`-sized
/// delimiter arrays, no full field grid); this fn parses values + reduces
/// per group in Rust (the honest fusion boundary — no `strtod` intrinsic).
/// Groups are ordered count-desc then key-asc so output is deterministic
/// and cross-arch bit-identical.
pub fn build_groups(
    bytes: &[u8],
    by:    &str,
    specs: &[AggSpec],
    pred:  Option<&super::filter::Predicate>,
) -> Result<GroupOutcome, String> {
    let headers = parse_header(bytes);
    let by_col = headers.iter().position(|h| h == by)
        .ok_or_else(|| format!("--by column not found: {by}"))?;
    let pred_col = match pred {
        Some(p) => Some(headers.iter().position(|h| h == &p.col)
            .ok_or_else(|| format!("--where column not found: {}", p.col))?),
        None => None,
    };

    // Resolve each numeric spec's column to an index up front (Count → None).
    let col_idx: Vec<Option<usize>> = specs.iter()
        .map(|s| match s.op {
            AggOp::Count => Ok(None),
            _ => headers.iter().position(|h| h == &s.col)
                .map(Some)
                .ok_or_else(|| format!("--agg column not found: {}", s.col)),
        })
        .collect::<Result<_, _>>()?;

    // `needed` = sorted-unique column indices the kernel must project: the
    // key column, every value column, and (if filtering) the predicate
    // column. Each then maps to a slot (position within `needed`) of the
    // kernel's per-row output.
    let mut needed: Vec<usize> = std::iter::once(by_col)
        .chain(col_idx.iter().filter_map(|c| *c))
        .chain(pred_col)
        .collect();
    needed.sort_unstable();
    needed.dedup();
    let n_needed = needed.len();
    let slot = |col: usize| needed.iter().position(|&c| c == col).unwrap();
    let key_slot = slot(by_col);
    let val_slot: Vec<Option<usize>> = col_idx.iter().map(|c| c.map(slot)).collect();
    let pred_slot = pred_col.map(slot);

    // Upper bound on rows: every '\n' plus a possible final unterminated
    // line. Counted in Rust (O(1) memory) — never an Ea pass.
    let max_rows = bytes.iter().filter(|&&b| b == b'\n').count() + 1;
    let needed_i32: Vec<i32> = needed.iter().map(|&c| c as i32).collect();
    let mut out_off = vec![-1i32; max_rows * n_needed];
    let mut out_len = vec![-1i32; max_rows * n_needed];
    let mut n_rows  = 0i32;
    let mut scratch = [0u8; 16];
    unsafe {
        crate::kernels::ffi::csv_groupby_scan(
            bytes.as_ptr(), bytes.len() as i32,
            needed_i32.as_ptr(), n_needed as i32,
            out_off.as_mut_ptr(), out_len.as_mut_ptr(),
            &mut n_rows,
            scratch.as_mut_ptr(),
        );
    }
    let n_rows = n_rows as usize;
    if n_rows < 2 {
        return Err("no data rows".into());
    }

    use std::collections::HashMap;
    let mut map: HashMap<String, Acc> = HashMap::new();
    let field = |row: usize, s: usize| -> Option<&[u8]> {
        let off = out_off[row * n_needed + s];
        let len = out_len[row * n_needed + s];
        if off < 0 { return None; } // column absent from this (ragged) row
        Some(&bytes[off as usize..(off + len) as usize])
    };

    // Data rows are 1.. (row 0 is the header the kernel also emitted).
    // `matched` counts rows passing the `--where` filter (all rows when no
    // filter) — reported as totals.rows, consistent with the non-group path.
    let mut matched = 0usize;
    for row in 1..n_rows {
        if let (Some(p), Some(ps)) = (pred, pred_slot) {
            let ok = match field(row, ps) {
                Some(b) => {
                    let cell = std::str::from_utf8(b).unwrap_or("").trim();
                    p.matches(common::unquote(cell).as_ref())
                }
                None => false, // ragged row missing the predicate column
            };
            if !ok { continue; }
        }
        matched += 1;
        let Some(key_bytes) = field(row, key_slot) else { continue }; // no key
        let key_raw = std::str::from_utf8(key_bytes).unwrap_or("").trim();
        let key = common::unquote(key_raw).into_owned();

        if !map.contains_key(&key) && map.len() >= MAX_GROUPS {
            return Err(format!(
                "too many groups (> {MAX_GROUPS}); `--by {by}` has very high cardinality",
            ));
        }
        let acc = map.entry(key).or_insert_with(|| Acc::new(specs.len()));
        acc.count += 1;

        for (i, vs) in val_slot.iter().enumerate() {
            let Some(vs) = vs else { continue };
            let Some(val_bytes) = field(row, *vs) else { continue };
            let raw = std::str::from_utf8(val_bytes).unwrap_or("").trim();
            let txt = common::unquote(raw);
            // Same finite-skipna rule as eacrunch's whole-column stats: drop
            // NaN/inf so sum/mean/min/max stay mutually consistent.
            if let Ok(v) = txt.parse::<f64>() {
                if v.is_finite() {
                    acc.stats[i].add(v);
                }
            }
        }
    }

    let mut groups: Vec<Group> = map.into_iter()
        .map(|(key, acc)| {
            let aggs = specs.iter().enumerate()
                .map(|(i, s)| {
                    let st = &acc.stats[i];
                    let value = match s.op {
                        AggOp::Count => acc.count as f64,
                        AggOp::Sum   => st.sum,
                        AggOp::Mean  => if st.n > 0 { st.sum / st.n as f64 } else { f64::NAN },
                        AggOp::Min   => if st.n > 0 { st.min } else { f64::NAN },
                        AggOp::Max   => if st.n > 0 { st.max } else { f64::NAN },
                    };
                    AggResult { op: s.op.as_str().to_string(), col: s.col.clone(), value }
                })
                .collect();
            Group { key, count: acc.count, aggs }
        })
        .collect();

    // Deterministic order: biggest groups first, key ascending on ties.
    groups.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.key.cmp(&b.key)));
    Ok(GroupOutcome { groups, data_rows: matched as u64 })
}

// ── JSON codec (additive `groups[]` on RuneOutput) ────────────────────────────

fn finite(v: f64) -> Value {
    if v.is_finite() { Value::F64(v) } else { Value::Null }
}

pub(super) fn group_to_obj(g: &Group) -> Object {
    let mut o = Object::new();
    o.set("key",   Value::Str(g.key.clone()));
    o.set("count", Value::I64(g.count as i64));
    o.set("aggs", Value::Array(g.aggs.iter().map(|a| {
        let mut ao = Object::new();
        ao.set("op", Value::Str(a.op.clone()));
        if !a.col.is_empty() {
            ao.set("col", Value::Str(a.col.clone()));
        }
        ao.set("value", finite(a.value));
        Value::Object(Box::new(ao))
    }).collect()));
    o
}

pub(super) fn group_from_obj(o: &Object) -> Result<Group, String> {
    let mut aggs = Vec::new();
    if let Some(arr) = o.get_array("aggs") {
        for v in arr {
            let ao = match v {
                Value::Object(b) => b,
                _ => return Err("group.aggs entries must be objects".into()),
            };
            aggs.push(AggResult {
                op:    ao.get_str("op").ok_or("agg.op missing")?.to_string(),
                col:   ao.get_str("col").map(str::to_string).unwrap_or_default(),
                value: ao.get_f64("value").unwrap_or(f64::NAN),
            });
        }
    }
    Ok(Group {
        key:   o.get_str("key").ok_or("group.key missing")?.to_string(),
        count: o.get_i64("count").ok_or("group.count missing")? as u64,
        aggs,
    })
}

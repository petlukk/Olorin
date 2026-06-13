//! GROUP BY block of the RuneOutput v1 contract — types, agg parsing, and
//! the grouping pass over eacrunch's CSV field grid.
//!
//! Lives outside `output.rs`/`eacrunch.rs` to keep both under the 500-LOC
//! cap; the `groups[]` field is on `RuneOutput` and serializes additively
//! (only when non-empty), exactly like `anomalies[]` / `correlations[]`.
//!
//! Phase 1 is correctness-first: grouping reuses the same `csv_scan` field
//! grid eacrunch already builds and parses agg values with the identical
//! finite-skipna rule (`f64::parse` + `is_finite`), so a group's
//! `mean:latency` agrees with eacrunch's whole-column `latency` mean by
//! construction. The fused single-pass kernel is Phase 2.

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

/// Run the grouping pass. `rows_fields[0]` is the header row (skipped);
/// data rows are `1..`. Returns groups ordered count-desc then key-asc so
/// the output is deterministic and cross-arch bit-identical.
pub fn build_groups(
    bytes:       &[u8],
    rows_fields: &[Vec<(usize, usize)>],
    headers:     &[String],
    by_col:      usize,
    specs:       &[AggSpec],
) -> Result<Vec<Group>, String> {
    // Resolve each numeric spec's column to an index up front (Count → None).
    let col_idx: Vec<Option<usize>> = specs.iter()
        .map(|s| match s.op {
            AggOp::Count => Ok(None),
            _ => headers.iter().position(|h| h == &s.col)
                .map(Some)
                .ok_or_else(|| format!("--agg column not found: {}", s.col)),
        })
        .collect::<Result<_, _>>()?;

    use std::collections::HashMap;
    let mut map: HashMap<String, Acc> = HashMap::new();

    for fields in rows_fields.iter().skip(1) {
        if by_col >= fields.len() {
            continue; // ragged row missing the key column
        }
        let (ks, ke) = fields[by_col];
        let key_raw = std::str::from_utf8(&bytes[ks..ke]).unwrap_or("").trim();
        let key = common::unquote(key_raw).into_owned();

        if !map.contains_key(&key) && map.len() >= MAX_GROUPS {
            return Err(format!(
                "too many groups (> {MAX_GROUPS}); `--by {}` has very high cardinality",
                headers.get(by_col).map(String::as_str).unwrap_or("?"),
            ));
        }
        let acc = map.entry(key).or_insert_with(|| Acc::new(specs.len()));
        acc.count += 1;

        for (i, ci) in col_idx.iter().enumerate() {
            let Some(ci) = ci else { continue };
            if *ci >= fields.len() {
                continue;
            }
            let (vs, ve) = fields[*ci];
            let raw = std::str::from_utf8(&bytes[vs..ve]).unwrap_or("").trim();
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
    Ok(groups)
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

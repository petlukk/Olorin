//! RuneOutput v1 — structured contract every rune emits.
//!
//! Today each rune produces a free-form `answer: String` consumed by the
//! LLM. That works for narration but blocks rune composition (a 2-input
//! rune like `eadiff` would need N² string-shape special cases). This
//! module declares the v1 cross-rune shape so composition becomes one
//! generic operation over a stable Rust type plus a JSONL wire format.
//!
//! Migration is incremental: each rune builds a `RuneOutput`, formats it
//! as today's `answer` string for the LLM AND (when a forthcoming
//! `--json` flag is set) serializes it via `to_json()` for downstream
//! runes. 2-input runes call `from_json` to read upstream outputs back.
//!
//! Wire format: one compact JSON object per line, produced via
//! `storage::json` (zero new deps). NaN/±Inf are emitted as `null`
//! because they aren't valid JSON; `f32_stats` returns NaN on empty
//! inputs which a rune will hit.

use crate::storage::json::{self, Object, Value};

pub const SCHEMA_VERSION: i64 = 1;

#[derive(Debug, Clone, PartialEq)]
pub enum FieldKind {
    Number,
    Text,
    Bool,
    Timestamp,
    Mixed,
}

impl FieldKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            FieldKind::Number    => "number",
            FieldKind::Text      => "text",
            FieldKind::Bool      => "bool",
            FieldKind::Timestamp => "timestamp",
            FieldKind::Mixed     => "mixed",
        }
    }
    pub fn parse(s: &str) -> Option<FieldKind> {
        Some(match s {
            "number"    => FieldKind::Number,
            "text"      => FieldKind::Text,
            "bool"      => FieldKind::Bool,
            "timestamp" => FieldKind::Timestamp,
            "mixed"     => FieldKind::Mixed,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct NumericStats {
    pub min:  f64,
    pub max:  f64,
    pub mean: f64,
    pub sum:  f64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TextEntry {
    pub value: String,
    pub count: u64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TextStats {
    pub unique: u64,
    pub top:    Vec<TextEntry>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BoolStats {
    pub true_count:  u64,
    pub false_count: u64,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct TimestampStats {
    pub min:    String,
    pub max:    String,
    pub unique: u64,
}

/// Per-column / per-key / per-attribute stats. The `kind` discriminator
/// picks which of the four sub-structs is populated. eacrunch+eajson
/// emit one `FieldStats` per CSV column or JSON key; eaparquet emits one
/// per Parquet column.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldStats {
    pub name:       String,
    pub kind:       FieldKind,
    pub count:      u64,
    pub null_count: Option<u64>,
    pub numeric:    Option<NumericStats>,
    pub text:       Option<TextStats>,
    pub bool:       Option<BoolStats>,
    pub timestamp:  Option<TimestampStats>,
}

/// Generic "counts by category" — orthogonal to `fields[]`. ealog uses
/// it for severity (DEBUG/INFO/WARN/ERROR/FATAL); eatime will use it for
/// hour-of-day / weekday buckets.
#[derive(Debug, Clone, PartialEq)]
pub struct Category {
    pub name:  String,
    pub count: u64,
}

/// Exemplar record. ealog populates byte_offset+line+text; eatime will
/// populate timestamp+text; eacrunch may leave it empty.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sample {
    pub byte_offset: Option<u64>,
    pub line:        Option<u64>,
    pub timestamp:   Option<String>,
    pub text:        String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Source {
    pub path:   String,
    pub bytes:  u64,
    /// Free-form format tag: "csv", "jsonl", "parquet", "log/plain",
    /// "log/jsonl". Used by downstream runes to know whether the upstream
    /// `fields[]` came from a schema (parquet/csv) or sniffed.
    pub format: String,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Totals {
    /// Logical records — rows for CSV/Parquet, lines for log, objects for JSONL.
    pub rows:    u64,
    pub scan_us: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuneOutput {
    pub schema_version: i64,
    pub rune:           String,
    pub rune_version:   i64,
    pub success:        bool,
    pub source:         Option<Source>,
    pub totals:         Totals,
    pub fields:         Vec<FieldStats>,
    pub categories:     Vec<Category>,
    pub samples:        Vec<Sample>,
    pub error:          Option<String>,
}

impl RuneOutput {
    pub fn new(rune: &str, rune_version: i64) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            rune:           rune.to_string(),
            rune_version,
            success:        true,
            source:         None,
            totals:         Totals::default(),
            fields:         Vec::new(),
            categories:     Vec::new(),
            samples:        Vec::new(),
            error:          None,
        }
    }

    pub fn to_json(&self) -> String {
        let mut o = Object::new();
        o.set("schema_version", Value::I64(self.schema_version));
        o.set("rune",           Value::Str(self.rune.clone()));
        o.set("rune_version",   Value::I64(self.rune_version));
        o.set("success",        Value::Bool(self.success));
        if let Some(src) = &self.source {
            o.set("source", obj_value(source_to_obj(src)));
        }
        o.set("totals",     obj_value(totals_to_obj(&self.totals)));
        o.set("fields",     Value::Array(self.fields.iter().map(|f| obj_value(field_to_obj(f))).collect()));
        o.set("categories", Value::Array(self.categories.iter().map(|c| obj_value(category_to_obj(c))).collect()));
        o.set("samples",    Value::Array(self.samples.iter().map(|s| obj_value(sample_to_obj(s))).collect()));
        if let Some(e) = &self.error {
            o.set("error", Value::Str(e.clone()));
        }
        json::serialize(&o)
    }

    pub fn from_json(bytes: &[u8]) -> Result<RuneOutput, String> {
        let obj = json::parse(bytes).map_err(|e| e.to_string())?;
        let schema_version = obj.get_i64("schema_version").ok_or("missing schema_version")?;
        if schema_version != SCHEMA_VERSION {
            return Err(format!("unsupported schema_version: {schema_version}"));
        }
        Ok(RuneOutput {
            schema_version,
            rune:         obj.get_str("rune").ok_or("missing rune")?.to_string(),
            rune_version: obj.get_i64("rune_version").ok_or("missing rune_version")?,
            success:      obj.get_bool("success").ok_or("missing success")?,
            source:       obj.get_object("source").map(source_from_obj).transpose()?,
            totals:       totals_from_obj(obj.get_object("totals").ok_or("missing totals")?)?,
            fields:       array_of(&obj, "fields",     field_from_obj)?,
            categories:   array_of(&obj, "categories", category_from_obj)?,
            samples:      array_of(&obj, "samples",    sample_from_obj)?,
            error:        obj.get_str("error").map(str::to_string),
        })
    }
}

// ── encode helpers ───────────────────────────────────────────────────────────

fn obj_value(o: Object) -> Value { Value::Object(Box::new(o)) }

fn finite_f64(v: f64) -> Value {
    if v.is_finite() { Value::F64(v) } else { Value::Null }
}

fn u64_value(v: u64) -> Value {
    // Counts above i64::MAX would wrap; realistic rune outputs never approach that.
    Value::I64(v as i64)
}

fn source_to_obj(s: &Source) -> Object {
    let mut o = Object::new();
    o.set("path",   Value::Str(s.path.clone()));
    o.set("bytes",  u64_value(s.bytes));
    o.set("format", Value::Str(s.format.clone()));
    o
}

fn totals_to_obj(t: &Totals) -> Object {
    let mut o = Object::new();
    o.set("rows",    u64_value(t.rows));
    o.set("scan_us", u64_value(t.scan_us));
    o
}

fn field_to_obj(f: &FieldStats) -> Object {
    let mut o = Object::new();
    o.set("name",  Value::Str(f.name.clone()));
    o.set("kind",  Value::Str(f.kind.as_str().to_string()));
    o.set("count", u64_value(f.count));
    if let Some(n) = f.null_count {
        o.set("null_count", u64_value(n));
    }
    if let Some(n) = &f.numeric {
        let mut sub = Object::new();
        sub.set("min",  finite_f64(n.min));
        sub.set("max",  finite_f64(n.max));
        sub.set("mean", finite_f64(n.mean));
        sub.set("sum",  finite_f64(n.sum));
        o.set("numeric", obj_value(sub));
    }
    if let Some(t) = &f.text {
        let mut sub = Object::new();
        sub.set("unique", u64_value(t.unique));
        sub.set("top", Value::Array(t.top.iter().map(|e| {
            let mut eo = Object::new();
            eo.set("value", Value::Str(e.value.clone()));
            eo.set("count", u64_value(e.count));
            obj_value(eo)
        }).collect()));
        o.set("text", obj_value(sub));
    }
    if let Some(b) = &f.bool {
        let mut sub = Object::new();
        sub.set("true_count",  u64_value(b.true_count));
        sub.set("false_count", u64_value(b.false_count));
        o.set("bool", obj_value(sub));
    }
    if let Some(ts) = &f.timestamp {
        let mut sub = Object::new();
        sub.set("min",    Value::Str(ts.min.clone()));
        sub.set("max",    Value::Str(ts.max.clone()));
        sub.set("unique", u64_value(ts.unique));
        o.set("timestamp", obj_value(sub));
    }
    o
}

fn category_to_obj(c: &Category) -> Object {
    let mut o = Object::new();
    o.set("name",  Value::Str(c.name.clone()));
    o.set("count", u64_value(c.count));
    o
}

fn sample_to_obj(s: &Sample) -> Object {
    let mut o = Object::new();
    if let Some(b) = s.byte_offset { o.set("byte_offset", u64_value(b)); }
    if let Some(l) = s.line        { o.set("line",        u64_value(l)); }
    if let Some(t) = &s.timestamp  { o.set("timestamp",   Value::Str(t.clone())); }
    o.set("text", Value::Str(s.text.clone()));
    o
}

// ── decode helpers ───────────────────────────────────────────────────────────

fn u64_from(obj: &Object, key: &str) -> Result<u64, String> {
    obj.get_i64(key)
        .map(|v| v as u64)
        .ok_or_else(|| format!("missing {key}"))
}

fn u64_opt(obj: &Object, key: &str) -> Option<u64> {
    obj.get_i64(key).map(|v| v as u64)
}

fn f64_or_zero(obj: &Object, key: &str) -> f64 {
    // Schema permits null for non-finite stats — treat null/missing as 0.0
    // so callers comparing two outputs don't have to thread Option<f64>.
    obj.get_f64(key).unwrap_or(0.0)
}

fn array_of<T, F>(obj: &Object, key: &str, mut f: F) -> Result<Vec<T>, String>
where
    F: FnMut(&Object) -> Result<T, String>,
{
    let arr = match obj.get_array(key) {
        Some(a) => a,
        None    => return Ok(Vec::new()),
    };
    arr.iter()
        .map(|v| match v {
            Value::Object(o) => f(o),
            _ => Err(format!("{key}[] entries must be objects")),
        })
        .collect()
}

fn source_from_obj(o: &Object) -> Result<Source, String> {
    Ok(Source {
        path:   o.get_str("path").ok_or("source.path missing")?.to_string(),
        bytes:  u64_from(o, "bytes")?,
        format: o.get_str("format").ok_or("source.format missing")?.to_string(),
    })
}

fn totals_from_obj(o: &Object) -> Result<Totals, String> {
    Ok(Totals {
        rows:    u64_from(o, "rows")?,
        scan_us: u64_from(o, "scan_us")?,
    })
}

fn field_from_obj(o: &Object) -> Result<FieldStats, String> {
    let kind_str = o.get_str("kind").ok_or("field.kind missing")?;
    let kind = FieldKind::parse(kind_str)
        .ok_or_else(|| format!("unknown field.kind: {kind_str}"))?;
    Ok(FieldStats {
        name:       o.get_str("name").ok_or("field.name missing")?.to_string(),
        kind,
        count:      u64_from(o, "count")?,
        null_count: u64_opt(o, "null_count"),
        numeric:    o.get_object("numeric").map(|n| NumericStats {
            min:  f64_or_zero(n, "min"),
            max:  f64_or_zero(n, "max"),
            mean: f64_or_zero(n, "mean"),
            sum:  f64_or_zero(n, "sum"),
        }),
        text:       o.get_object("text").map(text_from_obj).transpose()?,
        bool:       o.get_object("bool").map(bool_from_obj).transpose()?,
        timestamp:  o.get_object("timestamp").map(timestamp_from_obj).transpose()?,
    })
}

fn text_from_obj(o: &Object) -> Result<TextStats, String> {
    let mut top: Vec<TextEntry> = Vec::new();
    if let Some(arr) = o.get_array("top") {
        for v in arr {
            let entry_obj = match v {
                Value::Object(eo) => eo,
                _ => return Err("text.top entries must be objects".into()),
            };
            top.push(TextEntry {
                value: entry_obj.get_str("value").ok_or("text.top[].value missing")?.to_string(),
                count: u64_from(entry_obj, "count")?,
            });
        }
    }
    Ok(TextStats {
        unique: u64_from(o, "unique")?,
        top,
    })
}

fn bool_from_obj(o: &Object) -> Result<BoolStats, String> {
    Ok(BoolStats {
        true_count:  u64_from(o, "true_count")?,
        false_count: u64_from(o, "false_count")?,
    })
}

fn timestamp_from_obj(o: &Object) -> Result<TimestampStats, String> {
    Ok(TimestampStats {
        min:    o.get_str("min").ok_or("timestamp.min missing")?.to_string(),
        max:    o.get_str("max").ok_or("timestamp.max missing")?.to_string(),
        unique: u64_from(o, "unique")?,
    })
}

fn category_from_obj(o: &Object) -> Result<Category, String> {
    Ok(Category {
        name:  o.get_str("name").ok_or("category.name missing")?.to_string(),
        count: u64_from(o, "count")?,
    })
}

fn sample_from_obj(o: &Object) -> Result<Sample, String> {
    Ok(Sample {
        byte_offset: u64_opt(o, "byte_offset"),
        line:        u64_opt(o, "line"),
        timestamp:   o.get_str("timestamp").map(str::to_string),
        text:        o.get_str("text").ok_or("sample.text missing")?.to_string(),
    })
}

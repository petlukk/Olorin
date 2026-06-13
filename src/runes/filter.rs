//! Row predicate for `eacrunch --where <col><op><value>` — a single
//! comparison applied per row before aggregation (the WHERE of a query).
//!
//! Operators: `=` `!=` (string compare) and `>` `>=` `<` `<=` (numeric:
//! both sides parsed as f64; a non-numeric cell never satisfies an ordered
//! comparison). One predicate, no AND/OR — kept deliberately small.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CmpOp {
    Eq,
    Ne,
    Gt,
    Ge,
    Lt,
    Le,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Predicate {
    pub col:   String,
    pub op:    CmpOp,
    pub value: String,
}

impl Predicate {
    /// Parse `col<op>value`. Multi-char operators are matched before their
    /// single-char prefixes so `>=` doesn't read as `>` then `=value`.
    pub fn parse(s: &str) -> Result<Predicate, String> {
        let s = s.trim();
        // Order matters: longest operators first.
        for (tok, op) in [
            (">=", CmpOp::Ge), ("<=", CmpOp::Le), ("!=", CmpOp::Ne),
            (">",  CmpOp::Gt), ("<",  CmpOp::Lt), ("=",  CmpOp::Eq),
        ] {
            if let Some(idx) = s.find(tok) {
                let col = s[..idx].trim().to_string();
                let value = s[idx + tok.len()..].trim().to_string();
                if col.is_empty() {
                    return Err(format!("--where needs a column before `{tok}`"));
                }
                if value.is_empty() {
                    return Err(format!("--where needs a value after `{tok}`"));
                }
                return Ok(Predicate { col, op, value });
            }
        }
        Err(format!(
            "--where must look like col=value (ops: = != > >= < <=); got `{s}`"
        ))
    }

    /// Evaluate the predicate against a (already trimmed + unquoted) cell.
    pub fn matches(&self, cell: &str) -> bool {
        match self.op {
            CmpOp::Eq => cell == self.value,
            CmpOp::Ne => cell != self.value,
            CmpOp::Gt | CmpOp::Ge | CmpOp::Lt | CmpOp::Le => {
                // Ordered comparison is numeric; a non-numeric cell or
                // value simply doesn't match (never panics, never silently
                // string-compares with surprising lexical order).
                match (cell.parse::<f64>(), self.value.parse::<f64>()) {
                    (Ok(c), Ok(v)) => match self.op {
                        CmpOp::Gt => c > v,
                        CmpOp::Ge => c >= v,
                        CmpOp::Lt => c < v,
                        CmpOp::Le => c <= v,
                        _ => unreachable!(),
                    },
                    _ => false,
                }
            }
        }
    }
}

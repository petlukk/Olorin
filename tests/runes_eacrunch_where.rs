//! eacrunch `--where <col><op><value>` — single-predicate row filter
//! applied before aggregation (with `--by`) or before whole-column stats
//! (standalone). Operators: = != > >= < <=.

use olorin::runes::filter::{CmpOp, Predicate};
use olorin::runes::output::RuneOutput;

const FIXTURE: &str = "status,latency,bytes\n\
                       200,40,1000\n\
                       200,60,2000\n\
                       500,900,500\n\
                       404,12,300\n\
                       500,1060,700\n";

struct TmpFile(std::path::PathBuf);
impl Drop for TmpFile {
    fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); }
}
fn stage(tag: &str) -> TmpFile {
    let tmp = std::env::temp_dir()
        .join(format!("olorin_where_{tag}_{}.csv", std::process::id()));
    std::fs::write(&tmp, FIXTURE).unwrap();
    TmpFile(tmp)
}
fn run_json(args: &str) -> RuneOutput {
    olorin::kernels::ffi::init().unwrap();
    let r = olorin::runes::run_rune("eacrunch", args).expect("eacrunch exists");
    assert!(r.success, "rune failed: {}", r.answer);
    RuneOutput::from_json(r.answer.as_bytes()).expect("valid RuneOutput JSON")
}

// ── predicate unit tests ──────────────────────────────────────────────────────

#[test]
fn parse_operators() {
    assert_eq!(Predicate::parse("status=500").unwrap(),
        Predicate { col: "status".into(), op: CmpOp::Eq, value: "500".into() });
    assert_eq!(Predicate::parse("latency>=50").unwrap().op, CmpOp::Ge);
    assert_eq!(Predicate::parse("latency<=50").unwrap().op, CmpOp::Le);
    assert_eq!(Predicate::parse("x!=y").unwrap().op, CmpOp::Ne);
    assert_eq!(Predicate::parse("x>1").unwrap().op, CmpOp::Gt);
    assert_eq!(Predicate::parse("x<1").unwrap().op, CmpOp::Lt);
    assert!(Predicate::parse("nocolon").is_err());
    assert!(Predicate::parse("=novalue").is_err());
    assert!(Predicate::parse("nocol=").is_err());
}

#[test]
fn matches_string_and_numeric() {
    let p = Predicate::parse("c=500").unwrap();
    assert!(p.matches("500"));
    assert!(!p.matches("404"));
    let g = Predicate::parse("c>50").unwrap();
    assert!(g.matches("60"));
    assert!(!g.matches("40"));
    // non-numeric cell never satisfies an ordered comparison
    assert!(!g.matches("abc"));
}

// ── end-to-end ─────────────────────────────────────────────────────────────────

#[test]
fn where_string_equality_with_group_by() {
    let f = stage("eqby");
    let p = f.0.to_str().unwrap();
    // SELECT status, count(*) WHERE status=500 GROUP BY status
    let out = run_json(&format!("--json --where status=500 --by status --agg count {p}"));
    assert_eq!(out.groups.len(), 1, "only the 500 group survives");
    assert_eq!(out.groups[0].key, "500");
    assert_eq!(out.groups[0].count, 2);
    assert_eq!(out.totals.rows, 2, "totals.rows = matched rows");
}

#[test]
fn where_numeric_gt_with_group_by() {
    let f = stage("gtby");
    let p = f.0.to_str().unwrap();
    // latency > 50 keeps rows: 60(200), 900(500), 1060(500) → 200:1, 500:2
    let out = run_json(&format!("--json --where latency>50 --by status --agg sum:bytes {p}"));
    let g200 = out.groups.iter().find(|g| g.key == "200").unwrap();
    let g500 = out.groups.iter().find(|g| g.key == "500").unwrap();
    assert_eq!(g200.count, 1);
    assert_eq!(g500.count, 2);
    assert!(out.groups.iter().all(|g| g.key != "404"), "404 row (latency 12) filtered out");
}

#[test]
fn where_standalone_filters_column_stats() {
    let f = stage("standalone");
    let p = f.0.to_str().unwrap();
    // No --by: filtered whole-column stats. status=200 keeps 2 rows.
    let out = run_json(&format!("--json --where status=200 {p}"));
    assert_eq!(out.totals.rows, 2, "two matching rows");
    let latency = out.fields.iter().find(|f| f.name == "latency").expect("latency col");
    let n = latency.numeric.as_ref().unwrap();
    assert!((n.sum - 100.0).abs() < 1e-9, "40+60=100, got {}", n.sum); // only 200-rows
    assert!((n.max - 60.0).abs() < 1e-9, "max latency among 200-rows is 60");
}

#[test]
fn where_unknown_column_errors() {
    olorin::kernels::ffi::init().unwrap();
    let f = stage("badcol");
    let p = f.0.to_str().unwrap();
    let r = olorin::runes::run_rune("eacrunch", &format!("--where nope=1 {p}")).unwrap();
    assert!(!r.success, "unknown --where column must fail: {}", r.answer);
}

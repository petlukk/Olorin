//! eacrunch GROUP BY — `--by <col> --agg <op:col,...>`.
//!
//! Phase 1 (correctness-first): grouping runs over the same `csv_scan`
//! field grid eacrunch already builds; agg values are parsed with the same
//! finite-skipna rule as whole-column stats, so a group's `mean:latency`
//! agrees with eacrunch's column `latency` mean by construction.
//!
//! Correctness anchor: a hand-computed fixture. The differential gate vs
//! pandas `groupby().agg()` lives in benchmarks/robustness/diff_eacrunch.py.

use olorin::runes::output::RuneOutput;

/// status,latency,bytes — hand-computed groups:
///   200: count=2  mean(latency)=50   sum(bytes)=3000
///   500: count=2  mean(latency)=980  sum(bytes)=1200
///   404: count=1  mean(latency)=12   sum(bytes)=300
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

fn stage(tag: &str, body: &str) -> TmpFile {
    let tmp = std::env::temp_dir()
        .join(format!("olorin_groupby_{tag}_{}.csv", std::process::id()));
    std::fs::write(&tmp, body).unwrap();
    TmpFile(tmp)
}

fn run_json(args: &str) -> RuneOutput {
    olorin::kernels::ffi::init().unwrap();
    let r = olorin::runes::run_rune("eacrunch", args).expect("eacrunch exists");
    assert!(r.success, "rune failed: {}", r.answer);
    assert!(r.structured, "expected --json structured output: {}", r.answer);
    RuneOutput::from_json(r.answer.as_bytes()).expect("valid RuneOutput JSON")
}

fn agg(g: &olorin::runes::grouping::Group, op: &str, col: &str) -> f64 {
    g.aggs.iter()
        .find(|a| a.op == op && a.col == col)
        .unwrap_or_else(|| panic!("group {} missing agg {op}:{col}", g.key))
        .value
}

/// Direct kernel test: csv_groupby_scan projects only `needed` columns,
/// quote-aware, with a final line lacking a trailing newline.
#[test]
fn fused_kernel_projects_columns() {
    olorin::kernels::ffi::init().unwrap();
    // cols: a(0) b(1) c(2). Row 2's b has a quoted embedded comma. Last
    // row has no trailing newline. Project columns 0 and 2.
    let csv: &[u8] = b"a,b,c\nx1,y1,z1\nx2,\"q,w\",z2";
    let needed = [0i32, 2i32];
    let n = needed.len();
    let max_rows = csv.iter().filter(|&&b| b == b'\n').count() + 1;
    let mut off = vec![-1i32; max_rows * n];
    let mut len = vec![-1i32; max_rows * n];
    let mut n_rows = 0i32;
    let mut scratch = [0u8; 16];
    unsafe {
        olorin::kernels::ffi::csv_groupby_scan(
            csv.as_ptr(), csv.len() as i32,
            needed.as_ptr(), n as i32,
            off.as_mut_ptr(), len.as_mut_ptr(),
            &mut n_rows, scratch.as_mut_ptr(),
        );
    }
    assert_eq!(n_rows, 3, "header + 2 data rows (last has no trailing \\n)");
    let f = |row: usize, slot: usize| -> &str {
        let o = off[row * n + slot] as usize;
        let l = len[row * n + slot] as usize;
        std::str::from_utf8(&csv[o..o + l]).unwrap()
    };
    assert_eq!((f(0, 0), f(0, 1)), ("a", "c"));   // header
    assert_eq!((f(1, 0), f(1, 1)), ("x1", "z1"));
    // col 2 of row 2 is "z2"; the quoted comma in col 1 must NOT shift it.
    assert_eq!((f(2, 0), f(2, 1)), ("x2", "z2"));
}

/// Ragged row: a data row with fewer columns than the projected index
/// leaves that slot at the caller's -1 sentinel (not a spurious field).
#[test]
fn fused_kernel_ragged_row_marks_absent() {
    olorin::kernels::ffi::init().unwrap();
    let csv: &[u8] = b"a,b,c\n1,2,3\n9\n"; // row 2 has only column 0
    let needed = [0i32, 2i32];
    let n = needed.len();
    let max_rows = csv.iter().filter(|&&b| b == b'\n').count() + 1;
    let mut off = vec![-1i32; max_rows * n];
    let mut len = vec![-1i32; max_rows * n];
    let mut n_rows = 0i32;
    let mut scratch = [0u8; 16];
    unsafe {
        olorin::kernels::ffi::csv_groupby_scan(
            csv.as_ptr(), csv.len() as i32, needed.as_ptr(), n as i32,
            off.as_mut_ptr(), len.as_mut_ptr(), &mut n_rows, scratch.as_mut_ptr(),
        );
    }
    assert_eq!(n_rows, 3);
    // row 2, slot for column 2 is absent → off stays -1.
    assert_eq!(off[2 * n + 1], -1, "missing column 2 must stay -1");
    assert_eq!(off[2 * n + 0], 12, "column 0 of row 2 present");
}

#[test]
fn group_by_with_mean_and_sum() {
    let f = stage("basic", FIXTURE);
    let path = f.0.to_str().unwrap();
    let out = run_json(&format!("--json --by status --agg mean:latency,sum:bytes {path}"));

    // Three groups, ordered count-desc then key-asc: 200(2), 500(2), 404(1).
    assert_eq!(out.groups.len(), 3, "expected 3 groups");
    let keys: Vec<&str> = out.groups.iter().map(|g| g.key.as_str()).collect();
    assert_eq!(keys, vec!["200", "500", "404"], "group ordering");

    let g200 = &out.groups[0];
    assert_eq!(g200.count, 2);
    assert!((agg(g200, "mean", "latency") - 50.0).abs() < 1e-9);
    assert!((agg(g200, "sum", "bytes") - 3000.0).abs() < 1e-9);

    let g500 = &out.groups[1];
    assert_eq!(g500.count, 2);
    assert!((agg(g500, "mean", "latency") - 980.0).abs() < 1e-9);
    assert!((agg(g500, "sum", "bytes") - 1200.0).abs() < 1e-9);

    let g404 = &out.groups[2];
    assert_eq!(g404.count, 1);
    assert!((agg(g404, "mean", "latency") - 12.0).abs() < 1e-9);
    assert!((agg(g404, "sum", "bytes") - 300.0).abs() < 1e-9);
}

#[test]
fn group_by_min_max() {
    let f = stage("minmax", FIXTURE);
    let path = f.0.to_str().unwrap();
    let out = run_json(&format!("--json --by status --agg min:latency,max:latency {path}"));
    let g200 = out.groups.iter().find(|g| g.key == "200").unwrap();
    assert!((agg(g200, "min", "latency") - 40.0).abs() < 1e-9);
    assert!((agg(g200, "max", "latency") - 60.0).abs() < 1e-9);
}

#[test]
fn group_by_defaults_to_count_when_no_agg() {
    // SELECT status, count(*) GROUP BY status — bare --by with no --agg.
    let f = stage("count", FIXTURE);
    let path = f.0.to_str().unwrap();
    let out = run_json(&format!("--json --by status {path}"));
    assert_eq!(out.groups.len(), 3);
    let g200 = out.groups.iter().find(|g| g.key == "200").unwrap();
    assert_eq!(g200.count, 2);
}

#[test]
fn agg_without_by_is_an_error() {
    olorin::kernels::ffi::init().unwrap();
    let f = stage("noby", FIXTURE);
    let path = f.0.to_str().unwrap();
    let r = olorin::runes::run_rune("eacrunch", &format!("--agg sum:bytes {path}")).unwrap();
    assert!(!r.success, "agg without --by must fail, got: {}", r.answer);
}

#[test]
fn unknown_by_column_is_an_error() {
    olorin::kernels::ffi::init().unwrap();
    let f = stage("badcol", FIXTURE);
    let path = f.0.to_str().unwrap();
    let r = olorin::runes::run_rune("eacrunch", &format!("--by nope {path}")).unwrap();
    assert!(!r.success, "unknown --by column must fail, got: {}", r.answer);
}

#[test]
fn explicit_count_agg_not_double_rendered() {
    // `--agg count` must not re-print the group count as a noisy `count=N.00`;
    // the integer group count is always shown once. (sum still renders .00.)
    olorin::kernels::ffi::init().unwrap();
    let f = stage("dblcount", FIXTURE);
    let path = f.0.to_str().unwrap();
    let r = olorin::runes::run_rune("eacrunch",
        &format!("--by status --agg count,sum:bytes {path}")).unwrap();
    assert!(r.success, "{}", r.answer);
    assert!(r.answer.contains("count=2"), "integer group count shown: {}", r.answer);
    assert!(!r.answer.contains("count=2.00") && !r.answer.contains("count=1.00"),
        "redundant decimal count must not appear: {}", r.answer);
    for line in r.answer.lines()
        .filter(|l| l.trim_start().starts_with(|c: char| c.is_ascii_digit()))
    {
        assert_eq!(line.matches("count=").count(), 1, "one count per group line: {line}");
    }
}

#[test]
fn text_mode_renders_group_table() {
    // Non-json answer is human-readable and surfaces the group keys + aggs.
    olorin::kernels::ffi::init().unwrap();
    let f = stage("text", FIXTURE);
    let path = f.0.to_str().unwrap();
    let r = olorin::runes::run_rune("eacrunch", &format!("--by status --agg sum:bytes {path}"))
        .unwrap();
    assert!(r.success, "{}", r.answer);
    assert!(r.answer.contains("status"), "missing group column: {}", r.answer);
    assert!(r.answer.contains("200") && r.answer.contains("500"),
        "missing group keys: {}", r.answer);
}

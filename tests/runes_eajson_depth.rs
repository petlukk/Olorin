//! eajson `--depth N` — flatten nested objects into dotted keys up to N
//! levels (default 4). Previously capped at one level (`http.status`);
//! deeper keys were dropped.

use olorin::runes::output::RuneOutput;

const NESTED: &str =
    "{\"a\":1,\"http\":{\"status\":200,\"req\":{\"method\":\"GET\",\"headers\":{\"ua\":\"curl\"}}}}\n\
     {\"a\":2,\"http\":{\"status\":404,\"req\":{\"method\":\"POST\",\"headers\":{\"ua\":\"wget\"}}}}\n";

struct TmpFile(std::path::PathBuf);
impl Drop for TmpFile {
    fn drop(&mut self) { let _ = std::fs::remove_file(&self.0); }
}
fn stage(tag: &str) -> TmpFile {
    let tmp = std::env::temp_dir()
        .join(format!("olorin_eajson_depth_{tag}_{}.jsonl", std::process::id()));
    std::fs::write(&tmp, NESTED).unwrap();
    TmpFile(tmp)
}
fn run_json(args: &str) -> RuneOutput {
    olorin::kernels::ffi::init().unwrap();
    let r = olorin::runes::run_rune("eajson", args).expect("eajson exists");
    assert!(r.success, "rune failed: {}", r.answer);
    RuneOutput::from_json(r.answer.as_bytes()).expect("valid JSON")
}
fn has_key(out: &RuneOutput, name: &str) -> bool {
    out.fields.iter().any(|f| f.name == name)
}

#[test]
fn default_depth_flattens_multiple_levels() {
    let f = stage("default");
    let p = f.0.to_str().unwrap();
    let out = run_json(&format!("--json {p}"));
    // Default depth 4: every nested level is reachable.
    assert!(has_key(&out, "a"), "top-level key");
    assert!(has_key(&out, "http.status"), "1 level");
    assert!(has_key(&out, "http.req.method"), "2 levels");
    assert!(has_key(&out, "http.req.headers.ua"), "3 levels: {:?}",
        out.fields.iter().map(|f| f.name.as_str()).collect::<Vec<_>>());
}

#[test]
fn depth_one_keeps_old_behavior() {
    let f = stage("d1");
    let p = f.0.to_str().unwrap();
    let out = run_json(&format!("--json --depth 1 {p}"));
    assert!(has_key(&out, "http.status"), "1 level present at depth 1");
    assert!(!has_key(&out, "http.req.method"),
        "deeper keys absent at depth 1: {:?}",
        out.fields.iter().map(|f| f.name.as_str()).collect::<Vec<_>>());
}

#[test]
fn depth_zero_top_level_only() {
    let f = stage("d0");
    let p = f.0.to_str().unwrap();
    let out = run_json(&format!("--json --depth 0 {p}"));
    assert!(has_key(&out, "a"), "top-level scalar present");
    assert!(!has_key(&out, "http.status"), "no nesting flattened at depth 0");
}

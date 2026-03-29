use olorin::kernels::ffi;
use olorin::tools;

#[test]
fn test_calc_tool() {
    ffi::init().unwrap();
    let result = tools::calc::run("2 + 3 * 4");
    assert!(result.output.contains("14"), "expected 14, got: {}", result.output);
    assert!(result.success);
}

#[test]
fn test_time_tool() {
    let result = tools::time::run("");
    assert!(result.output.len() > 10, "time output too short: {}", result.output);
    assert!(result.success);
}

#[test]
fn test_ls_tool() {
    let result = tools::ls::run("/tmp");
    assert!(result.success, "ls failed: {}", result.output);
}

#[test]
fn test_tool_registry() {
    ffi::init().unwrap();
    assert!(tools::run_tool("calc", "1+1").is_some());
    assert!(tools::run_tool("nonexistent", "").is_none());
}

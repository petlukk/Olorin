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

// ── File/Shell tools ─────────────────────────────────────────────────────────

#[test]
fn test_read_file() {
    let tmp = std::env::temp_dir().join("olorin_test_read.txt");
    std::fs::write(&tmp, "hello olorin").unwrap();
    let result = tools::read_file::run(&tmp.to_string_lossy());
    assert!(result.success);
    assert!(result.output.contains("hello olorin"));
    std::fs::remove_file(&tmp).unwrap();
}

#[test]
fn test_write_file() {
    let tmp = std::env::temp_dir().join("olorin_test_write.txt");
    let arg = format!("{} test content", tmp.display());
    let result = tools::write_file::run(&arg);
    assert!(result.success);
    let content = std::fs::read_to_string(&tmp).unwrap();
    assert_eq!(content, "test content");
    std::fs::remove_file(&tmp).unwrap();
}

#[test]
fn test_grep() {
    let tmp = std::env::temp_dir().join("olorin_test_grep.txt");
    std::fs::write(&tmp, "line one\nfind me here\nline three").unwrap();
    let arg = format!("find {}", tmp.display());
    let result = tools::grep::run(&arg);
    assert!(result.success);
    assert!(result.output.contains("find me"));
    std::fs::remove_file(&tmp).unwrap();
}

#[test]
fn test_shell() {
    let result = tools::shell::run("echo hello_olorin_test");
    assert!(result.success);
    assert!(result.output.contains("hello_olorin_test"));
}

#[test]
fn test_shell_empty() {
    let result = tools::shell::run("");
    assert!(!result.success);
}

// ── Info tools ───────────────────────────────────────────────────────────────

#[test]
fn test_cpu() {
    let result = tools::cpu::run("");
    assert!(result.success);
    assert!(result.output.len() > 10);
}

#[test]
fn test_tokens() {
    let result = tools::tokens::run("hello world this is a test");
    assert!(result.success);
    assert!(result.output.contains("token"));
}

#[test]
fn test_tokens_empty() {
    let result = tools::tokens::run("");
    assert!(!result.success);
}

#[test]
fn test_json_tool_keys() {
    let result = tools::json_tool::run(r#"keys {"a":1,"b":2}"#);
    assert!(result.success);
    assert!(result.output.contains("a"));
    assert!(result.output.contains("b"));
}

#[test]
fn test_json_tool_get() {
    let result = tools::json_tool::run(r#"get {"name":"olorin"} name"#);
    assert!(result.success);
    assert!(result.output.contains("olorin"));
}

#[test]
fn test_memory_write_read() {
    let w = tools::memory::run("write test_key_42 test_value_42");
    assert!(w.success);
    let r = tools::memory::run("read test_key_42");
    assert!(r.success);
    assert!(r.output.contains("test_value_42"));
}

#[test]
fn test_memory_list() {
    tools::memory::run("write list_test_key hello");
    let result = tools::memory::run("list");
    assert!(result.success);
}

#[test]
fn test_bench() {
    ffi::init().unwrap();
    let result = tools::bench::run("");
    // Bench may fail if no model loaded, but must not panic
    assert!(!result.output.is_empty());
}

#[test]
fn test_git() {
    let result = tools::git::run("status");
    // Should work in the repo directory or fail gracefully
    assert!(!result.output.is_empty());
}

#[test]
fn test_remind() {
    let result = tools::remind::run("1s test reminder");
    assert!(result.success);
}

// ── Network tools (graceful failure tests) ───────────────────────────────────

#[test]
fn test_http_invalid_url() {
    let result = tools::http::run("http://localhost:1");
    // Should not panic — output may be empty if curl not found
    let _ = result;
}

#[test]
fn test_define_no_panic() {
    let result = tools::define::run("hello");
    // May fail without network, must not panic
    assert!(!result.output.is_empty());
}

#[test]
fn test_weather_no_panic() {
    let result = tools::weather::run("Stockholm");
    assert!(!result.output.is_empty());
}

#[test]
fn test_translate_no_panic() {
    let result = tools::translate::run("hello to swedish");
    assert!(!result.output.is_empty());
}

#[test]
fn test_summarize_no_panic() {
    let result = tools::summarize::run("This is a text that should be summarized into something shorter.");
    assert!(!result.output.is_empty());
}

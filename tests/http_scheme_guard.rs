//! HIGH-1 regression: the `http`/`fetch` tool must only fetch http(s).
//!
//! Before the fix, the tool ran `curl -s -L <url>` with no scheme check, so
//! `http file:///etc/...` read arbitrary local files (bypassing path_guard)
//! and other schemes opened SSRF/exfil channels. These tests assert the
//! tool refuses every non-http(s) scheme and never returns file contents.

use olorin::tools::run_tool;

#[test]
fn blocks_file_scheme_and_leaks_no_contents() {
    // Point at a file that reliably exists on Linux CI and dev machines.
    let target = if std::path::Path::new("/etc/hostname").exists() {
        "file:///etc/hostname"
    } else {
        "file:///etc/hosts"
    };
    let r = run_tool("http", target).expect("http tool exists");
    assert!(!r.success, "file:// fetch must fail");
    assert!(
        r.output.contains("blocked") && r.output.contains("http"),
        "expected scheme-block message, got: {}",
        r.output
    );
}

#[test]
fn blocks_all_non_http_schemes() {
    let cases = [
        "file:///etc/passwd",
        "FILE:///etc/passwd", // case-insensitive scheme check
        "gopher://example.com/",
        "scp://host/path",
        "ftp://example.com/x",
        "dict://localhost:11211/",
        "/etc/passwd",   // no scheme at all
        "etc/passwd",    // bare relative
    ];
    for url in cases {
        let r = run_tool("http", url).expect("http tool exists");
        assert!(!r.success, "must block {url}");
        assert!(
            r.output.contains("blocked"),
            "must report block for {url}, got: {}",
            r.output
        );
    }
}

#[test]
fn empty_url_is_usage_error_not_a_fetch() {
    let r = run_tool("http", "").expect("http tool exists");
    assert!(!r.success);
    assert!(r.output.contains("usage"));
}

#[test]
fn fetch_alias_is_guarded_too() {
    let r = run_tool("fetch", "file:///etc/hostname").expect("fetch alias exists");
    assert!(!r.success, "fetch alias must apply the same scheme guard");
    assert!(r.output.contains("blocked"));
}

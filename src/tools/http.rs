use super::ToolResult;

pub fn run(args: &str) -> ToolResult {
    let url = args.trim();
    if url.is_empty() {
        return ToolResult { output: "usage: http <url>".to_string(), success: false };
    }
    // Scheme allowlist. curl otherwise honors file://, gopher://, scp://,
    // etc. — `http file:///etc/shadow` would read any local file, bypassing
    // path_guard entirely, and other schemes open SSRF/exfil channels. Only
    // http(s) is a legitimate fetch.
    let scheme_ok = {
        let lower = url.to_ascii_lowercase();
        lower.starts_with("http://") || lower.starts_with("https://")
    };
    if !scheme_ok {
        return ToolResult {
            output: "blocked: only http:// and https:// URLs are allowed".to_string(),
            success: false,
        };
    }
    // `--proto`/`--proto-redir` pin curl to http(s) at the transport level too,
    // so a 30x redirect to `file://` (which `-L` would otherwise follow) is
    // refused — defense in depth behind the prefix check above.
    let output = std::process::Command::new("curl")
        .args([
            "-s", "-L",
            "--proto", "=http,https",
            "--proto-redir", "=http,https",
            "--max-time", "10",
            url,
        ])
        .output();
    match output {
        Ok(o) => {
            let mut body = String::from_utf8_lossy(&o.stdout).to_string();
            let max_len = 32 * 1024;
            if body.len() > max_len {
                body.truncate(max_len);
                body.push_str("... (truncated)");
            }
            ToolResult { output: body, success: o.status.success() }
        }
        Err(e) => ToolResult { output: format!("curl failed: {e}"), success: false },
    }
}

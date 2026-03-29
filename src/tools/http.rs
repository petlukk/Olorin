use super::ToolResult;

pub fn run(args: &str) -> ToolResult {
    let url = args.trim();
    if url.is_empty() {
        return ToolResult { output: "usage: http <url>".to_string(), success: false };
    }
    let output = std::process::Command::new("curl")
        .args(["-s", "-L", "--max-time", "10", url])
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

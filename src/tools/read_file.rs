use super::ToolResult;

pub fn run(args: &str) -> ToolResult {
    let path = args.trim();
    if path.is_empty() {
        return ToolResult { output: "usage: read <path>".to_string(), success: false };
    }

    match std::fs::read_to_string(path) {
        Ok(content) => {
            let max_len = 64 * 1024;
            if content.len() > max_len {
                ToolResult {
                    output: format!("{}... (truncated, {} bytes total)", { let mut end = max_len; while end > 0 && !content.is_char_boundary(end) { end -= 1; } &content[..end] }, content.len()),
                    success: true,
                }
            } else {
                ToolResult { output: content, success: true }
            }
        }
        Err(e) => ToolResult { output: format!("{path}: {e}"), success: false },
    }
}

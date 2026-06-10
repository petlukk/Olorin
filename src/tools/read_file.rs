use super::ToolResult;
use crate::core::path_guard::{resolve_safe_path_checked, AccessMode};

pub fn run(args: &str) -> ToolResult {
    let path = args.trim();
    if path.is_empty() {
        return ToolResult { output: "usage: read <path>".to_string(), success: false };
    }

    let resolved = match resolve_safe_path_checked(path, AccessMode::Read) {
        Ok(p) => p,
        Err(e) => return ToolResult { output: e.refusal_message(), success: false },
    };

    match std::fs::read_to_string(&resolved) {
        Ok(content) => {
            let max_len = 64 * 1024;
            if content.len() > max_len {
                let mut end = max_len;
                while end > 0 && !content.is_char_boundary(end) { end -= 1; }
                ToolResult {
                    output: format!("{}... (truncated, {} bytes total)", &content[..end], content.len()),
                    success: true,
                }
            } else {
                ToolResult { output: content, success: true }
            }
        }
        Err(e) => ToolResult { output: format!("{path}: {e}"), success: false },
    }
}

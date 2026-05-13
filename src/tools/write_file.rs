use super::ToolResult;
use crate::core::path_guard::{resolve_safe_path, AccessMode};

pub fn run(args: &str) -> ToolResult {
    let args = args.trim();
    let (path, content) = match args.find(' ') {
        Some(pos) => (&args[..pos], &args[pos + 1..]),
        None => return ToolResult {
            output: "usage: write <path> <content>".to_string(),
            success: false,
        },
    };

    let resolved = match resolve_safe_path(path, AccessMode::Write) {
        Ok(p) => p,
        Err(e) => return ToolResult { output: e.refusal_message(), success: false },
    };

    match std::fs::write(&resolved, content) {
        Ok(()) => ToolResult {
            output: format!("Wrote {} bytes to {path}", content.len()),
            success: true,
        },
        Err(e) => ToolResult { output: format!("{path}: {e}"), success: false },
    }
}

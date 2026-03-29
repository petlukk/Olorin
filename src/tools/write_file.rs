use super::ToolResult;

pub fn run(args: &str) -> ToolResult {
    // Format: "<path> <content>"
    let args = args.trim();
    let (path, content) = match args.find(' ') {
        Some(pos) => (&args[..pos], &args[pos + 1..]),
        None => return ToolResult {
            output: "usage: write <path> <content>".to_string(),
            success: false,
        },
    };

    match std::fs::write(path, content) {
        Ok(()) => ToolResult {
            output: format!("Wrote {} bytes to {path}", content.len()),
            success: true,
        },
        Err(e) => ToolResult { output: format!("{path}: {e}"), success: false },
    }
}

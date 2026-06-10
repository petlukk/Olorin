use super::ToolResult;
use crate::core::path_guard::{resolve_safe_path_checked, AccessMode};

const MAX_OUTPUT: usize = 32 * 1024;

pub fn run(args: &str) -> ToolResult {
    if args.trim().is_empty() {
        return ToolResult {
            output: "usage: grep [-i] <pattern> [path]".to_string(),
            success: false,
        };
    }

    let mut argv: Vec<&str> = vec!["grep", "-rn", "--color=never", "--max-count=200"];
    let mut remaining = args.trim();

    if remaining.starts_with("-i ") {
        argv.push("-i");
        remaining = remaining[3..].trim();
    }

    let (pattern, raw_path) = if let Some(pos) = remaining.find(' ') {
        (&remaining[..pos], remaining[pos + 1..].trim())
    } else {
        (remaining, ".")
    };

    let raw_path = if raw_path.is_empty() { "." } else { raw_path };

    let resolved = match resolve_safe_path_checked(raw_path, AccessMode::Read) {
        Ok(p) => p,
        Err(e) => return ToolResult { output: e.refusal_message(), success: false },
    };
    let path_str = resolved.to_string_lossy().into_owned();

    argv.push("--");
    argv.push(pattern);
    argv.push(&path_str);

    let output = std::process::Command::new(argv[0])
        .args(&argv[1..])
        .output();

    match output {
        Ok(o) => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            if stdout.is_empty() {
                return ToolResult { output: "No matches found.".to_string(), success: true };
            }
            let result = if stdout.len() > MAX_OUTPUT {
                let mut end = MAX_OUTPUT;
                while end > 0 && !stdout.is_char_boundary(end) {
                    end -= 1;
                }
                format!("{}... (truncated, {} bytes total)", &stdout[..end], stdout.len())
            } else {
                stdout.trim_end().to_string()
            };
            ToolResult { output: result, success: true }
        }
        Err(e) => ToolResult { output: format!("grep failed: {e}"), success: false },
    }
}

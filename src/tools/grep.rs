use super::ToolResult;

const MAX_OUTPUT: usize = 32 * 1024;

pub fn run(args: &str) -> ToolResult {
    // Parse: [flags] pattern [path]
    // Simple format: "pattern" or "pattern /path" or "-i pattern /path"
    let parts: Vec<&str> = args.splitn(3, ' ').collect();
    if parts.is_empty() || args.trim().is_empty() {
        return ToolResult {
            output: "usage: grep [-i] <pattern> [path]".to_string(),
            success: false,
        };
    }

    let mut argv: Vec<&str> = vec!["grep", "-rn", "--color=never", "--max-count=200"];
    let mut remaining = args.trim();

    // Check for -i flag
    if remaining.starts_with("-i ") {
        argv.push("-i");
        remaining = remaining[3..].trim();
    }

    // Split on first space: pattern path
    let (pattern, path) = if let Some(pos) = remaining.find(' ') {
        (&remaining[..pos], remaining[pos + 1..].trim())
    } else {
        (remaining, ".")
    };

    let path = if path.is_empty() { "." } else { path };

    argv.push("--");
    argv.push(pattern);
    argv.push(path);

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

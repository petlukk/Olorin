use super::ToolResult;

const ALLOWED_SUBCOMMANDS: &[&str] = &[
    "status", "log", "diff", "branch", "show", "blame", "stash",
];

pub fn run(args: &str) -> ToolResult {
    let args = args.trim();
    let parts: Vec<&str> = args.split_whitespace().collect();

    if parts.is_empty() {
        return ToolResult {
            output: format!("usage: git <subcommand> [args]. Allowed: {}", ALLOWED_SUBCOMMANDS.join(", ")),
            success: false,
        };
    }

    let subcommand = parts[0];
    if !ALLOWED_SUBCOMMANDS.contains(&subcommand) {
        return ToolResult {
            output: format!("subcommand '{subcommand}' not allowed. Allowed: {}", ALLOWED_SUBCOMMANDS.join(", ")),
            success: false,
        };
    }

    let mut argv = vec!["git"];
    argv.extend_from_slice(&parts);

    let output = std::process::Command::new(argv[0])
        .args(&argv[1..])
        .output();

    match output {
        Ok(o) => {
            let stdout = String::from_utf8_lossy(&o.stdout);
            let stderr = String::from_utf8_lossy(&o.stderr);
            let mut result = String::new();
            if !stdout.is_empty() {
                result.push_str(&stdout);
            }
            if !stderr.is_empty() {
                if !result.is_empty() { result.push('\n'); }
                result.push_str(&stderr);
            }
            if result.is_empty() {
                result = format!("(exit code: {})", o.status.code().unwrap_or(-1));
            }
            let max_len = 32 * 1024;
            if result.len() > max_len {
                let mut end = max_len;
                while end > 0 && !result.is_char_boundary(end) { end -= 1; }
                result.truncate(end);
                result.push_str("... (truncated)");
            }
            ToolResult { output: result.trim_end().to_string(), success: o.status.success() }
        }
        Err(e) => ToolResult { output: format!("git failed: {e}"), success: false },
    }
}

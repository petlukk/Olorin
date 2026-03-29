use super::ToolResult;
use crate::core::shell_guard::ShellGuard;

pub fn run(args: &str) -> ToolResult {
    let cmd = args.trim();
    if cmd.is_empty() {
        return ToolResult { output: "usage: shell <command>".to_string(), success: false };
    }

    let policy = crate::core::shell_guard::load_shell_policy();
    let guard = ShellGuard::new(policy);

    if let Err(e) = guard.check(cmd) {
        return ToolResult { output: format!("blocked: {e}"), success: false };
    }

    let output = std::process::Command::new("sh")
        .args(["-c", cmd])
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
                if !result.is_empty() {
                    result.push('\n');
                }
                result.push_str("[stderr] ");
                result.push_str(&stderr);
            }
            if result.is_empty() {
                result = format!("(exit code: {})", o.status.code().unwrap_or(-1));
            }
            ToolResult { output: result, success: o.status.success() }
        }
        Err(e) => ToolResult { output: format!("failed to execute: {e}"), success: false },
    }
}

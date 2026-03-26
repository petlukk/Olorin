use super::Tool;
use async_trait::async_trait;
use tokio::process::Command;

pub struct GitTool;

const ALLOWED_SUBCOMMANDS: &[&str] = &[
    "status", "log", "diff", "branch", "show", "blame", "stash",
];

#[async_trait]
impl Tool for GitTool {
    fn name(&self) -> &str {
        "git"
    }

    fn description(&self) -> &str {
        "Run read-only git commands: status, log, diff, branch, show, blame, stash."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "args": {
                    "type": "string",
                    "description": "Git subcommand and arguments (e.g. 'log --oneline -10', 'diff HEAD~1', 'status')"
                }
            },
            "required": ["args"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> crate::error::Result<String> {
        let args_str = params["args"]
            .as_str()
            .ok_or_else(|| crate::error::Error::Tool("missing 'args' parameter".into()))?;

        let parts: Vec<&str> = args_str.split_whitespace().collect();
        if parts.is_empty() {
            return Err(crate::error::Error::Tool(
                format!("usage: /git <subcommand> [args]. Allowed: {}", ALLOWED_SUBCOMMANDS.join(", "))
            ));
        }

        let subcommand = parts[0];
        if !ALLOWED_SUBCOMMANDS.contains(&subcommand) {
            return Err(crate::error::Error::Tool(
                format!("subcommand '{subcommand}' not allowed. Allowed: {}", ALLOWED_SUBCOMMANDS.join(", "))
            ));
        }

        let output = Command::new("git")
            .args(&parts)
            .output()
            .await
            .map_err(|e| crate::error::Error::Tool(format!("failed to execute git: {e}")))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let mut result = String::new();
        if !stdout.is_empty() {
            result.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str(&stderr);
        }

        if result.is_empty() {
            result.push_str(&format!("(exit code: {})", output.status.code().unwrap_or(-1)));
        }

        // Truncate large output
        let max_len = 32 * 1024;
        if result.len() > max_len {
            let mut end = max_len;
            while end > 0 && !result.is_char_boundary(end) {
                end -= 1;
            }
            result.truncate(end);
            result.push_str("... (truncated)");
        }

        Ok(result.trim_end().to_string())
    }
}

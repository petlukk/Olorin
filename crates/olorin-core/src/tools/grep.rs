use super::Tool;
use async_trait::async_trait;

pub struct GrepTool;

const MAX_OUTPUT: usize = 32 * 1024;

#[async_trait]
impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search files for a pattern. Returns file:line:match format. Supports regex."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search (default: current directory)"
                },
                "glob": {
                    "type": "string",
                    "description": "File glob filter (e.g. '*.rs', '*.py')"
                },
                "case_insensitive": {
                    "type": "boolean",
                    "description": "Case-insensitive search (default: false)"
                }
            },
            "required": ["pattern"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> crate::error::Result<String> {
        let pattern = params["pattern"]
            .as_str()
            .ok_or_else(|| crate::error::Error::Tool("missing 'pattern' parameter".into()))?;

        let path = params["path"].as_str().unwrap_or(".");
        let case_insensitive = params["case_insensitive"].as_bool().unwrap_or(false);

        let mut argv: Vec<&str> = vec!["grep", "-rn", "--color=never", "--max-count=200"];

        if case_insensitive {
            argv.push("-i");
        }

        let glob_flag;
        if let Some(glob) = params["glob"].as_str() {
            glob_flag = format!("--include={}", glob);
            argv.push(&glob_flag);
        }

        argv.push("--");
        argv.push(pattern);
        argv.push(path);

        let output = crate::exec::run(&argv)
            .map_err(|e| crate::error::Error::Tool(format!("failed to execute grep: {e}")))?;

        let stdout = String::from_utf8_lossy(&output.stdout);

        if stdout.is_empty() {
            return Ok("No matches found.".to_string());
        }

        let result = if stdout.len() > MAX_OUTPUT {
            let mut end = MAX_OUTPUT;
            while end > 0 && !stdout.is_char_boundary(end) {
                end -= 1;
            }
            format!("{}... (truncated, {} bytes total)", &stdout[..end], stdout.len())
        } else {
            stdout.into_owned()
        };

        Ok(result.trim_end().to_string())
    }
}

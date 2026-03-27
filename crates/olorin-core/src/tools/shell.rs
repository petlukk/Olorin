use super::Tool;
use crate::safety::shell_guard::ShellGuard;
use async_trait::async_trait;

pub struct ShellTool {
    guard: ShellGuard,
}

impl ShellTool {
    pub fn new(guard: ShellGuard) -> Self {
        Self { guard }
    }
}

#[async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "Execute a shell command and return stdout/stderr."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute"
                }
            },
            "required": ["command"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> crate::error::Result<String> {
        let cmd = params["command"]
            .as_str()
            .ok_or_else(|| crate::error::Error::Tool("missing 'command' parameter".into()))?;

        self.guard.check(cmd).map_err(|e| crate::error::Error::Tool(e))?;

        let output = crate::exec::shell(cmd)
            .map_err(|e| crate::error::Error::Tool(format!("failed to execute: {e}")))?;

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
            result.push_str("[stderr] ");
            result.push_str(&stderr);
        }

        if result.is_empty() {
            result.push_str(&format!("(exit code: {})", output.exit_code));
        }

        Ok(result)
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    async fn execute_stream(
        &self,
        params: serde_json::Value,
        on_chunk: &mut (dyn for<'a> FnMut(&'a str) + Send),
    ) -> crate::error::Result<()> {
        let cmd = params["command"]
            .as_str()
            .ok_or_else(|| crate::error::Error::Tool("missing 'command' parameter".into()))?;

        self.guard.check(cmd).map_err(|e| crate::error::Error::Tool(e))?;

        let output = crate::exec::shell(cmd)
            .map_err(|e| crate::error::Error::Tool(format!("failed to execute: {e}")))?;

        let text = String::from_utf8_lossy(&output.stdout);
        for line in text.lines() {
            on_chunk(line);
            on_chunk("\n");
        }

        if !output.stderr.is_empty() {
            let err = String::from_utf8_lossy(&output.stderr);
            on_chunk("[stderr] ");
            on_chunk(&err);
        }

        if output.exit_code != 0 {
            on_chunk(&format!("(exit code: {})", output.exit_code));
        }

        Ok(())
    }
}

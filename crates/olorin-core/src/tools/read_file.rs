use super::Tool;
use async_trait::async_trait;

pub struct ReadFileTool;

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read"
    }

    fn description(&self) -> &str {
        "Read the contents of a file."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to read"
                }
            },
            "required": ["path"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> crate::error::Result<String> {
        let path = params["path"]
            .as_str()
            .ok_or_else(|| crate::error::Error::Tool("missing 'path' parameter".into()))?;

        let content = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| crate::error::Error::Tool(format!("{path}: {e}")))?;

        let max_len = 64 * 1024;
        if content.len() > max_len {
            Ok(format!(
                "{}... (truncated, {} bytes total)",
                &content[..max_len],
                content.len()
            ))
        } else {
            Ok(content)
        }
    }
}

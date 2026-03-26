use super::Tool;
use async_trait::async_trait;

pub struct WriteFileTool;

#[async_trait]
impl Tool for WriteFileTool {
    fn name(&self) -> &str {
        "write"
    }

    fn description(&self) -> &str {
        "Write content to a file. Creates the file if it doesn't exist."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The file path to write"
                },
                "content": {
                    "type": "string",
                    "description": "The content to write"
                }
            },
            "required": ["path", "content"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> crate::error::Result<String> {
        let path = params["path"]
            .as_str()
            .ok_or_else(|| crate::error::Error::Tool("missing 'path' parameter".into()))?;
        let content = params["content"]
            .as_str()
            .ok_or_else(|| crate::error::Error::Tool("missing 'content' parameter".into()))?;

        tokio::fs::write(path, content)
            .await
            .map_err(|e| crate::error::Error::Tool(format!("{path}: {e}")))?;

        Ok(format!("Wrote {} bytes to {path}", content.len()))
    }
}

use super::Tool;
use async_trait::async_trait;

pub struct LsTool;

#[async_trait]
impl Tool for LsTool {
    fn name(&self) -> &str {
        "ls"
    }

    fn description(&self) -> &str {
        "List files and directories in a path."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "The directory path to list (default: current directory)"
                }
            }
        })
    }

    async fn execute(&self, params: serde_json::Value) -> crate::error::Result<String> {
        let path = params["path"].as_str().unwrap_or(".");

        let mut entries = Vec::new();
        let mut dir = tokio::fs::read_dir(path)
            .await
            .map_err(|e| crate::error::Error::Tool(format!("{path}: {e}")))?;

        while let Some(entry) = dir
            .next_entry()
            .await
            .map_err(|e| crate::error::Error::Tool(format!("read entry: {e}")))?
        {
            let name = entry.file_name().to_string_lossy().into_owned();
            let meta = entry.metadata().await.ok();
            let suffix = if meta.as_ref().map_or(false, |m| m.is_dir()) {
                "/"
            } else {
                ""
            };
            let size = meta.as_ref().map_or(0, |m| m.len());
            if suffix == "/" {
                entries.push(format!("  {name}/"));
            } else {
                entries.push(format!("  {name}  ({size} B)"));
            }
        }

        entries.sort();

        if entries.is_empty() {
            Ok(format!("{path}: empty directory"))
        } else {
            Ok(format!("{path}:\n{}", entries.join("\n")))
        }
    }
}

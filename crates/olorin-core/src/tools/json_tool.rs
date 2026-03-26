use super::Tool;
use async_trait::async_trait;

pub struct JsonTool;

#[async_trait]
impl Tool for JsonTool {
    fn name(&self) -> &str {
        "json"
    }

    fn description(&self) -> &str {
        "Parse JSON. Actions: keys (list top-level keys), get <path> (extract by dot path), pretty (format)."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["keys", "get", "pretty"],
                    "description": "The action to perform"
                },
                "input": {
                    "type": "string",
                    "description": "JSON string or file path"
                },
                "path": {
                    "type": "string",
                    "description": "Dot-separated path for 'get' action (e.g. users.0.name)"
                }
            },
            "required": ["action", "input"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> crate::error::Result<String> {
        let action = params["action"]
            .as_str()
            .ok_or_else(|| crate::error::Error::Tool("missing 'action' parameter".into()))?;
        let input = params["input"]
            .as_str()
            .ok_or_else(|| crate::error::Error::Tool("missing 'input' parameter".into()))?;

        // Try parsing as JSON directly, or read from file
        let value: serde_json::Value = if input.trim_start().starts_with('{')
            || input.trim_start().starts_with('[')
        {
            serde_json::from_str(input)
                .map_err(|e| crate::error::Error::Tool(format!("invalid JSON: {e}")))?
        } else {
            let content = tokio::fs::read_to_string(input)
                .await
                .map_err(|e| crate::error::Error::Tool(format!("{input}: {e}")))?;
            serde_json::from_str(&content)
                .map_err(|e| crate::error::Error::Tool(format!("invalid JSON in {input}: {e}")))?
        };

        match action {
            "keys" => {
                if let Some(obj) = value.as_object() {
                    let keys: Vec<&str> = obj.keys().map(|k| k.as_str()).collect();
                    Ok(keys.join(", "))
                } else if let Some(arr) = value.as_array() {
                    Ok(format!("Array with {} elements", arr.len()))
                } else {
                    Ok(format!("Scalar: {value}"))
                }
            }
            "get" => {
                let path = params["path"]
                    .as_str()
                    .ok_or_else(|| crate::error::Error::Tool("missing 'path' for get".into()))?;
                let result = navigate(&value, path)?;
                Ok(serde_json::to_string_pretty(&result)
                    .unwrap_or_else(|_| format!("{result}")))
            }
            "pretty" => {
                let max_len = 64 * 1024;
                let pretty = serde_json::to_string_pretty(&value)
                    .map_err(|e| crate::error::Error::Tool(format!("format error: {e}")))?;
                if pretty.len() > max_len {
                    Ok(format!(
                        "{}... (truncated, {} bytes)",
                        &pretty[..max_len],
                        pretty.len()
                    ))
                } else {
                    Ok(pretty)
                }
            }
            _ => Err(crate::error::Error::Tool(format!(
                "unknown action '{action}'"
            ))),
        }
    }
}

fn navigate(
    value: &serde_json::Value,
    path: &str,
) -> crate::error::Result<serde_json::Value> {
    let mut current = value;
    for segment in path.split('.') {
        if segment.is_empty() {
            continue;
        }
        if let Ok(idx) = segment.parse::<usize>() {
            current = current
                .get(idx)
                .ok_or_else(|| crate::error::Error::Tool(format!("index {idx} out of bounds")))?;
        } else {
            current = current
                .get(segment)
                .ok_or_else(|| crate::error::Error::Tool(format!("key '{segment}' not found")))?;
        }
    }
    Ok(current.clone())
}

use super::Tool;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Mutex;

pub struct MemoryTool {
    store: Mutex<HashMap<String, String>>,
}

impl MemoryTool {
    pub fn new() -> Self {
        Self {
            store: Mutex::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl Tool for MemoryTool {
    fn name(&self) -> &str {
        "memory"
    }

    fn description(&self) -> &str {
        "In-memory key-value store. Use action 'write' to store, 'read' to retrieve, 'list' to show all keys."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["write", "read", "list"],
                    "description": "The action to perform"
                },
                "key": {
                    "type": "string",
                    "description": "The key (required for read/write)"
                },
                "value": {
                    "type": "string",
                    "description": "The value to store (required for write)"
                }
            },
            "required": ["action"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> crate::error::Result<String> {
        let action = params["action"]
            .as_str()
            .ok_or_else(|| crate::error::Error::Tool("missing 'action' parameter".into()))?;

        let mut store = self.store.lock().map_err(|_| {
            crate::error::Error::Tool("memory store lock poisoned".into())
        })?;

        match action {
            "write" => {
                let key = params["key"]
                    .as_str()
                    .ok_or_else(|| crate::error::Error::Tool("missing 'key' for write".into()))?;
                let value = params["value"]
                    .as_str()
                    .ok_or_else(|| crate::error::Error::Tool("missing 'value' for write".into()))?;
                store.insert(key.to_string(), value.to_string());
                Ok(format!("Stored '{key}'"))
            }
            "read" => {
                let key = params["key"]
                    .as_str()
                    .ok_or_else(|| crate::error::Error::Tool("missing 'key' for read".into()))?;
                match store.get(key) {
                    Some(val) => Ok(val.clone()),
                    None => Ok(format!("Key '{key}' not found")),
                }
            }
            "list" => {
                let keys: Vec<&str> = store.keys().map(|k| k.as_str()).collect();
                if keys.is_empty() {
                    Ok("No keys stored".to_string())
                } else {
                    Ok(keys.join(", "))
                }
            }
            _ => Err(crate::error::Error::Tool(format!(
                "unknown action '{action}'"
            ))),
        }
    }
}

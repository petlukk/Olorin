use super::{Tool, check_host};
use async_trait::async_trait;

pub struct DefineTool {
    allowed_hosts: Vec<String>,
}

impl DefineTool {
    pub fn new(allowed_hosts: Vec<String>) -> Self {
        Self { allowed_hosts }
    }
}

#[async_trait]
impl Tool for DefineTool {
    fn name(&self) -> &str {
        "define"
    }

    fn description(&self) -> &str {
        "Look up a word definition."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "word": {
                    "type": "string",
                    "description": "Word to define"
                }
            },
            "required": ["word"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> crate::error::Result<String> {
        let word = params["word"]
            .as_str()
            .ok_or_else(|| crate::error::Error::Tool("missing 'word' parameter".into()))?;

        let url = format!("https://api.dictionaryapi.dev/api/v2/entries/en/{word}");
        check_host(&self.allowed_hosts, &url)?;
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .map_err(|e| crate::error::Error::Tool(format!("http client error: {e}")))?;
        let resp = client.get(&url).send().await?;

        if !resp.status().is_success() {
            return Err(crate::error::Error::Tool(format!("no definition found for: {word}")));
        }

        let body: serde_json::Value = resp.json().await
            .map_err(|e| crate::error::Error::Tool(format!("parse error: {e}")))?;

        let mut output = String::new();
        if let Some(entries) = body.as_array() {
            for entry in entries.iter().take(1) {
                if let Some(meanings) = entry["meanings"].as_array() {
                    for meaning in meanings.iter().take(3) {
                        let pos = meaning["partOfSpeech"].as_str().unwrap_or("?");
                        if let Some(defs) = meaning["definitions"].as_array() {
                            for def in defs.iter().take(2) {
                                let text = def["definition"].as_str().unwrap_or("");
                                output.push_str(&format!("({pos}) {text}\n"));
                            }
                        }
                    }
                }
            }
        }

        if output.is_empty() {
            return Err(crate::error::Error::Tool(format!("no definition found for: {word}")));
        }

        Ok(output.trim().to_string())
    }
}

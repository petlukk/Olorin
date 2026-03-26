use super::Tool;
use crate::llm::{ContentBlock, LlmProvider, Message, Role};
use async_trait::async_trait;
use std::sync::Arc;

pub struct TranslateTool {
    llm: Arc<dyn LlmProvider>,
}

impl TranslateTool {
    pub fn new(llm: Arc<dyn LlmProvider>) -> Self {
        Self { llm }
    }
}

const SYSTEM: &str = "Translate the text to the target language. Output ONLY the translation, nothing else.";

#[async_trait]
impl Tool for TranslateTool {
    fn name(&self) -> &str {
        "translate"
    }

    fn description(&self) -> &str {
        "Translate text to another language."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "lang": {
                    "type": "string",
                    "description": "Target language (e.g. Spanish, French, Japanese)"
                },
                "text": {
                    "type": "string",
                    "description": "Text to translate"
                }
            },
            "required": ["lang", "text"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> crate::error::Result<String> {
        let lang = params["lang"]
            .as_str()
            .ok_or_else(|| crate::error::Error::Tool("missing 'lang' parameter".into()))?;
        let text = params["text"]
            .as_str()
            .ok_or_else(|| crate::error::Error::Tool("missing 'text' parameter".into()))?;

        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::text(format!("Translate to {lang}: {text}"))],
        }];

        let response = self.llm.chat(&messages, &[], SYSTEM).await?;

        let result = response
            .content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();

        Ok(result.trim().to_string())
    }
}

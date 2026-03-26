pub mod anthropic;
pub mod local;
pub mod tool_parse;
#[cfg(feature = "local-llm")]
pub mod llama_ffi;
#[cfg(feature = "local-llm")]
pub mod eakv_ffi;
#[cfg(feature = "local-llm")]
pub use local::LocalLlmProvider;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// A message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// Content block in a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },

    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },

    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

impl ContentBlock {
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text { text: s.into() }
    }

    pub fn tool_result(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::ToolResult {
            tool_use_id: id.into(),
            content: content.into(),
            is_error: None,
        }
    }

    pub fn tool_error(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::ToolResult {
            tool_use_id: id.into(),
            content: content.into(),
            is_error: Some(true),
        }
    }
}

/// Tool definition for the API.
#[derive(Debug, Clone, Serialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Response from the LLM.
#[derive(Debug)]
pub struct LlmResponse {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
}

#[derive(Debug, PartialEq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
}

/// Callback type for streaming text chunks.
pub type OnTextFn<'a> = &'a mut (dyn FnMut(&str) + Send);

/// LLM provider trait.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
        system: &str,
    ) -> crate::error::Result<LlmResponse>;

    /// Stream text via callback, returning full LlmResponse (needed for tool_use blocks).
    /// Default impl: falls back to `chat()` and invokes callback with collected text.
    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
        system: &str,
        on_text: OnTextFn<'_>,
    ) -> crate::error::Result<LlmResponse> {
        let response = self.chat(messages, tools, system).await?;
        for block in &response.content {
            if let ContentBlock::Text { text } = block {
                on_text(text);
            }
        }
        Ok(response)
    }
}

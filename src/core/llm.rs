//! Message types and LLM provider abstraction.
//!
//! Synchronous — no async/tokio. Uses crate::storage::json instead of serde.
//! Supports Anthropic (cloud via curl) and local inference (cougar engine).

use crate::storage::json::{self, Object, Value};

// ── Message types ────────────────────────────────────────────────────────────

/// A message in the conversation.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentBlock>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Role {
    User,
    Assistant,
}

impl Role {
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// Content block in a message.
#[derive(Debug, Clone)]
pub enum ContentBlock {
    Text { text: String },
    ToolUse { id: String, name: String, input: Object },
    ToolResult { tool_use_id: String, content: String, is_error: bool },
}

impl ContentBlock {
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text { text: s.into() }
    }

    pub fn tool_result(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::ToolResult {
            tool_use_id: id.into(),
            content: content.into(),
            is_error: false,
        }
    }

    pub fn tool_error(id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::ToolResult {
            tool_use_id: id.into(),
            content: content.into(),
            is_error: true,
        }
    }
}

/// Tool definition for LLM context.
#[derive(Debug, Clone)]
pub struct ToolDef {
    pub name:         String,
    pub description:  String,
    pub input_schema: String, // JSON string
}

/// Response from the LLM.
#[derive(Debug)]
pub struct LlmResponse {
    pub content:     Vec<ContentBlock>,
    pub stop_reason: StopReason,
}

#[derive(Debug, PartialEq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
}

// ── Serialization helpers ────────────────────────────────────────────────────

/// Serialize messages to JSON array string (for Anthropic API).
pub fn messages_to_json(messages: &[Message]) -> String {
    let vals: Vec<Value> = messages.iter().map(|m| {
        let mut obj = Object::new();
        obj.set("role", Value::Str(m.role.as_str().to_string()));
        // Simple path: concatenate text blocks into a single content string.
        // For tool_use/tool_result, serialize as content array.
        let has_tool = m.content.iter().any(|b| !matches!(b, ContentBlock::Text { .. }));
        if has_tool {
            let blocks: Vec<Value> = m.content.iter().map(|b| {
                let mut block = Object::new();
                match b {
                    ContentBlock::Text { text } => {
                        block.set("type", Value::Str("text".to_string()));
                        block.set("text", Value::Str(text.clone()));
                    }
                    ContentBlock::ToolUse { id, name, input } => {
                        block.set("type", Value::Str("tool_use".to_string()));
                        block.set("id", Value::Str(id.clone()));
                        block.set("name", Value::Str(name.clone()));
                        // input is already an Object; serialize via raw JSON string
                        let input_json = json::serialize(input);
                        block.set("input", Value::Str(input_json));
                    }
                    ContentBlock::ToolResult { tool_use_id, content, is_error } => {
                        block.set("type", Value::Str("tool_result".to_string()));
                        block.set("tool_use_id", Value::Str(tool_use_id.clone()));
                        block.set("content", Value::Str(content.clone()));
                        if *is_error {
                            block.set("is_error", Value::Bool(true));
                        }
                    }
                }
                Value::Object(Box::new(block))
            }).collect();
            obj.set("content", Value::Array(blocks));
        } else {
            let text: String = m.content.iter().filter_map(|b| {
                if let ContentBlock::Text { text } = b { Some(text.as_str()) } else { None }
            }).collect();
            obj.set("content", Value::Str(text));
        }
        Value::Object(Box::new(obj))
    }).collect();

    let arr = Value::Array(vals);
    json::serialize_value(&arr)
}

/// Format conversation as ChatML prompt string (for local inference).
pub fn format_chatml(messages: &[Message], tools: &[ToolDef], system: &str) -> String {
    let mut out = String::with_capacity(2048);

    let has_system = !system.is_empty() || !tools.is_empty();
    if has_system {
        out.push_str("<|im_start|>system\n");
        if !system.is_empty() {
            out.push_str(system);
        }
        if !tools.is_empty() {
            if !system.is_empty() {
                out.push_str("\n\n");
            }
            out.push_str("# Tools\n\nYou have access to the following tools:\n");
            for tool in tools {
                out.push_str("\n## ");
                out.push_str(&tool.name);
                out.push('\n');
                out.push_str(&tool.description);
                out.push_str("\nParameters: ");
                out.push_str(&tool.input_schema);
                out.push('\n');
            }
            out.push_str(
                "\nWhen you need to call a tool, use this format:\n\
                 <tool_call>{\"name\": \"tool_name\", \"arguments\": {\"key\": \"value\"}}</tool_call>\n",
            );
        }
        out.push_str("<|im_end|>\n");
    }

    for msg in messages {
        out.push_str("<|im_start|>");
        out.push_str(msg.role.as_str());
        out.push('\n');
        for block in &msg.content {
            match block {
                ContentBlock::Text { text } => out.push_str(text),
                ContentBlock::ToolUse { name, input, .. } => {
                    out.push_str("<tool_call>");
                    let mut obj = Object::new();
                    obj.set("name", Value::Str(name.clone()));
                    // Inline the input object fields
                    let input_str = json::serialize(input);
                    obj.set("arguments", Value::Str(input_str));
                    out.push_str(&json::serialize(&obj));
                    out.push_str("</tool_call>");
                }
                ContentBlock::ToolResult { content, .. } => {
                    out.push_str(content);
                }
            }
        }
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

// ── System prompt ────────────────────────────────────────────────────────────

pub const SYSTEM_PROMPT: &str = "\
You are Olorin.

Rules:
- Answer as short as possible. One line.
- Never explain unless explicitly asked.
- Never mention Python, code examples, or step-by-step reasoning.
- If unsure, say: \"unknown\"
- If a tool can answer, output ONLY the tool call.

Tool rules:
- math: <tool_call>{\"name\":\"calc\",\"arguments\":{\"expr\":\"...\"}}</tool_call>
- time: <tool_call>{\"name\":\"time\",\"arguments\":{}}</tool_call>
- system: <tool_call>{\"name\":\"cpu\",\"arguments\":{}}</tool_call>

Examples:
User: 6*7
Assistant: <tool_call>{\"name\":\"calc\",\"arguments\":{\"expr\":\"6*7\"}}</tool_call>

User: what is 2^10
Assistant: <tool_call>{\"name\":\"calc\",\"arguments\":{\"expr\":\"2**10\"}}</tool_call>

User: current time
Assistant: <tool_call>{\"name\":\"time\",\"arguments\":{}}</tool_call>

User: what is SIMD?
Assistant: Single Instruction, Multiple Data \u{2014} one instruction processes multiple values in parallel.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_as_str() {
        assert_eq!(Role::User.as_str(), "user");
        assert_eq!(Role::Assistant.as_str(), "assistant");
    }

    #[test]
    fn content_block_text() {
        let b = ContentBlock::text("hello");
        assert!(matches!(b, ContentBlock::Text { text } if text == "hello"));
    }

    #[test]
    fn content_block_tool_result() {
        let b = ContentBlock::tool_result("id1", "output");
        assert!(matches!(b, ContentBlock::ToolResult { is_error: false, .. }));
    }

    #[test]
    fn content_block_tool_error() {
        let b = ContentBlock::tool_error("id1", "fail");
        assert!(matches!(b, ContentBlock::ToolResult { is_error: true, .. }));
    }

    #[test]
    fn system_prompt_rules() {
        assert!(SYSTEM_PROMPT.contains("One line"));
        assert!(SYSTEM_PROMPT.contains("unknown"));
        assert!(SYSTEM_PROMPT.contains("<tool_call>"));
    }

    #[test]
    fn format_chatml_basic() {
        let msgs = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::text("hello")],
        }];
        let out = format_chatml(&msgs, &[], "Be concise.");
        assert!(out.contains("<|im_start|>system\nBe concise.<|im_end|>"));
        assert!(out.contains("<|im_start|>user\nhello<|im_end|>"));
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn format_chatml_no_system() {
        let msgs = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::text("hi")],
        }];
        let out = format_chatml(&msgs, &[], "");
        assert!(!out.contains("system"));
    }
}

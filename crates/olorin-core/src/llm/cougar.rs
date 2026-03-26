use super::tool_parse::{extract_tool_calls, parse_tool_call_json, StrDetectResult, StringToolCallDetector};
use super::{ContentBlock, LlmProvider, LlmResponse, Message, OnTextFn, Role, StopReason, ToolDef};
use crate::error::{Error, Result};
use async_trait::async_trait;
use cougar_engine::forward::InferenceState;
use cougar_engine::gguf::GgufFile;
use cougar_engine::model::BitNetModel;
use cougar_engine::tokenizer::Tokenizer;
use std::path::Path;
use std::sync::Arc;

struct ModelBundle {
    /// Kept alive so BitNetModel's raw pointers into the mmap remain valid.
    _gguf: GgufFile,
    model: *const BitNetModel,
    tokenizer: Tokenizer,
}

// BitNetModel contains raw pointers but is Send+Sync (see model.rs).
// GgufFile owns the backing data and lives alongside the model.
unsafe impl Send for ModelBundle {}
unsafe impl Sync for ModelBundle {}

impl ModelBundle {
    fn load(path: &Path) -> std::result::Result<Self, String> {
        let path_str = path.to_str().ok_or("non-UTF-8 model path")?;
        let gguf = GgufFile::open(path_str)?;
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        let model = BitNetModel::from_gguf(&gguf)?;
        let model_box = Box::new(model);
        let model_ptr = Box::into_raw(model_box);
        Ok(Self {
            _gguf: gguf,
            model: model_ptr,
            tokenizer,
        })
    }

    fn model(&self) -> &BitNetModel {
        unsafe { &*self.model }
    }
}

impl Drop for ModelBundle {
    fn drop(&mut self) {
        unsafe {
            drop(Box::from_raw(self.model as *mut BitNetModel));
        }
    }
}

pub struct CougarProvider {
    bundle: Arc<ModelBundle>,
    max_tokens: usize,
    max_seq_len: usize,
    temperature: f32,
    repetition_penalty: f32,
}

impl CougarProvider {
    pub fn new(model_path: &Path, max_seq_len: usize) -> std::result::Result<Self, String> {
        let bundle = ModelBundle::load(model_path)?;
        Ok(Self {
            bundle: Arc::new(bundle),
            max_tokens: 256,
            max_seq_len,
            temperature: 0.0,
            repetition_penalty: 1.1,
        })
    }

    pub fn set_max_tokens(&mut self, n: usize) {
        self.max_tokens = n;
    }

    pub fn set_temperature(&mut self, t: f32) {
        self.temperature = t;
    }

    pub fn set_repetition_penalty(&mut self, p: f32) {
        self.repetition_penalty = p;
    }
}

/// Format conversation messages into a Qwen/ChatML-style prompt string.
/// When `tools` is non-empty, appends tool definitions to the system message
/// so the model knows how to emit `<tool_call>` blocks.
fn format_chat_prompt(messages: &[Message], tools: &[ToolDef], system: &str) -> String {
    let mut out = String::with_capacity(2048);

    // System message — always present when tools are provided.
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
                out.push_str(&tool.input_schema.to_string());
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
        let role_str = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        out.push_str("<|im_start|>");
        out.push_str(role_str);
        out.push('\n');
        for block in &msg.content {
            match block {
                ContentBlock::Text { text } => out.push_str(text),
                ContentBlock::ToolUse { name, input, .. } => {
                    // Re-serialize tool calls so the model sees its own format.
                    out.push_str("<tool_call>");
                    let obj = serde_json::json!({"name": name, "arguments": input});
                    out.push_str(&obj.to_string());
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

#[async_trait]
impl LlmProvider for CougarProvider {
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
        system: &str,
    ) -> Result<LlmResponse> {
        let prompt = format_chat_prompt(messages, tools, system);
        let has_tools = !tools.is_empty();
        let bundle = Arc::clone(&self.bundle);
        let max_tokens = self.max_tokens;
        let max_seq_len = self.max_seq_len;
        let temperature = self.temperature;
        let repetition_penalty = self.repetition_penalty;

        let text = tokio::task::spawn_blocking(move || -> std::result::Result<String, String> {
            let model = bundle.model();
            let eos_id = bundle.tokenizer.eos_id;
            let prompt_tokens = bundle.tokenizer.encode(&prompt);
            let (output_tokens, _, _) = InferenceState::generate(
                model,
                &prompt_tokens,
                max_tokens,
                temperature,
                repetition_penalty,
                eos_id,
                max_seq_len,
                |_tok| {},
            );
            let generated = &output_tokens[prompt_tokens.len()..];
            Ok(bundle.tokenizer.decode(generated))
        })
        .await
        .map_err(|e| Error::Llm(format!("inference task panicked: {e}")))?
        .map_err(Error::Llm)?;

        // Post-process: extract tool calls from the generated text.
        if has_tools {
            return Ok(extract_tool_calls(&text));
        }
        Ok(LlmResponse {
            content: vec![ContentBlock::Text { text }],
            stop_reason: StopReason::EndTurn,
        })
    }

    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
        system: &str,
        on_text: OnTextFn<'_>,
    ) -> Result<LlmResponse> {
        let prompt = format_chat_prompt(messages, tools, system);
        let has_tools = !tools.is_empty();
        let bundle = Arc::clone(&self.bundle);
        let max_tokens = self.max_tokens;
        let max_seq_len = self.max_seq_len;
        let temperature = self.temperature;
        let repetition_penalty = self.repetition_penalty;

        // Channel for streaming tokens from the blocking task.
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();

        let handle = tokio::task::spawn_blocking(move || -> std::result::Result<Vec<u32>, String> {
            let model = bundle.model();
            let eos_id = bundle.tokenizer.eos_id;
            let prompt_tokens = bundle.tokenizer.encode(&prompt);
            let prompt_len = prompt_tokens.len();
            let mut generated_tokens: Vec<u32> = Vec::new();
            let (output_tokens, _, _) = InferenceState::generate(
                model,
                &prompt_tokens,
                max_tokens,
                temperature,
                repetition_penalty,
                eos_id,
                max_seq_len,
                |tok| {
                    generated_tokens.push(tok);
                    let piece = bundle.tokenizer.decode(&[tok]);
                    let _ = tx.send(piece);
                },
            );
            let generated = &output_tokens[prompt_len..];
            Ok(generated.to_vec())
        });

        let mut content_blocks: Vec<ContentBlock> = Vec::new();
        let mut streamed_text = String::new();
        let mut detector = StringToolCallDetector::new(8192);

        while let Some(piece) = rx.recv().await {
            if !has_tools {
                on_text(&piece);
                streamed_text.push_str(&piece);
                continue;
            }
            // Feed through tool-call detector.
            match detector.feed(&piece) {
                StrDetectResult::Text(t) => {
                    on_text(&t);
                    streamed_text.push_str(&t);
                }
                StrDetectResult::Buffering => {}
                StrDetectResult::ToolCall(body) => {
                    // Flush any accumulated text as a content block.
                    if !streamed_text.is_empty() {
                        content_blocks.push(ContentBlock::Text {
                            text: std::mem::take(&mut streamed_text),
                        });
                    }
                    if let Some(block) = parse_tool_call_json(&body) {
                        content_blocks.push(block);
                    }
                }
                StrDetectResult::Aborted(buf) => {
                    // Failed capture — stream the raw text.
                    on_text(&buf);
                    streamed_text.push_str(&buf);
                }
            }
        }

        // Flush any leftover from the detector.
        if has_tools {
            if let Some(leftover) = detector.finish() {
                on_text(&leftover);
                streamed_text.push_str(&leftover);
            }
        }

        handle
            .await
            .map_err(|e| Error::Llm(format!("inference task panicked: {e}")))?
            .map_err(Error::Llm)?;

        // Determine stop reason and final content.
        if !streamed_text.is_empty() {
            content_blocks.push(ContentBlock::Text {
                text: streamed_text,
            });
        }
        if content_blocks.is_empty() {
            content_blocks.push(ContentBlock::Text {
                text: String::new(),
            });
        }
        let has_tool_use = content_blocks
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolUse { .. }));
        let stop_reason = if has_tool_use {
            StopReason::ToolUse
        } else {
            StopReason::EndTurn
        };

        Ok(LlmResponse {
            content: content_blocks,
            stop_reason,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_empty_system() {
        let msgs = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::text("hello")],
        }];
        let prompt = format_chat_prompt(&msgs, &[], "");
        assert!(!prompt.contains("system"));
        assert!(prompt.contains("<|im_start|>user\nhello<|im_end|>"));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn format_with_system() {
        let msgs = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::text("hi")],
        }];
        let prompt = format_chat_prompt(&msgs, &[], "You are helpful.");
        assert!(prompt.starts_with("<|im_start|>system\nYou are helpful.<|im_end|>\n"));
        assert!(prompt.contains("<|im_start|>user\nhi<|im_end|>"));
    }

    #[test]
    fn format_multi_turn() {
        let msgs = vec![
            Message {
                role: Role::User,
                content: vec![ContentBlock::text("What is 2+2?")],
            },
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::text("4")],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::text("And 3+3?")],
            },
        ];
        let prompt = format_chat_prompt(&msgs, &[], "Be concise.");
        let expected_parts = [
            "<|im_start|>system\nBe concise.<|im_end|>\n",
            "<|im_start|>user\nWhat is 2+2?<|im_end|>\n",
            "<|im_start|>assistant\n4<|im_end|>\n",
            "<|im_start|>user\nAnd 3+3?<|im_end|>\n",
            "<|im_start|>assistant\n",
        ];
        for part in &expected_parts {
            assert!(prompt.contains(part), "missing: {part}");
        }
    }

    #[test]
    fn format_tool_result_block() {
        let msgs = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::tool_result("id1", "file contents here")],
        }];
        let prompt = format_chat_prompt(&msgs, &[], "");
        assert!(prompt.contains("file contents here"));
    }

    #[test]
    fn format_tool_use_block() {
        let msgs = vec![Message {
            role: Role::Assistant,
            content: vec![ContentBlock::ToolUse {
                id: "t1".into(),
                name: "read_file".into(),
                input: serde_json::json!({"path": "/tmp/x"}),
            }],
        }];
        let prompt = format_chat_prompt(&msgs, &[], "");
        assert!(prompt.contains("<tool_call>"));
        assert!(prompt.contains("read_file"));
        assert!(prompt.contains("/tmp/x"));
        assert!(prompt.contains("</tool_call>"));
    }

    #[test]
    fn format_chat_with_tools() {
        let tools = vec![
            ToolDef {
                name: "read_file".into(),
                description: "Read a file from disk.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    },
                    "required": ["path"]
                }),
            },
            ToolDef {
                name: "write_file".into(),
                description: "Write content to a file.".into(),
                input_schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["path", "content"]
                }),
            },
        ];
        let msgs = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::text("List files")],
        }];
        let prompt = format_chat_prompt(&msgs, &tools, "You are an assistant.");

        // System block should contain both system prompt and tool definitions.
        assert!(prompt.contains("You are an assistant."));
        assert!(prompt.contains("# Tools"));
        assert!(prompt.contains("## read_file"));
        assert!(prompt.contains("Read a file from disk."));
        assert!(prompt.contains("## write_file"));
        assert!(prompt.contains("Write content to a file."));
        assert!(prompt.contains("<tool_call>"));
        assert!(prompt.contains("</tool_call>"));
        // User message is still present.
        assert!(prompt.contains("<|im_start|>user\nList files<|im_end|>"));
    }

    #[test]
    fn format_tools_without_system_prompt() {
        let tools = vec![ToolDef {
            name: "search".into(),
            description: "Search the web.".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }];
        let msgs = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::text("find rust docs")],
        }];
        let prompt = format_chat_prompt(&msgs, &tools, "");
        // Should still have a system block with tools.
        assert!(prompt.contains("<|im_start|>system\n# Tools"));
        assert!(prompt.contains("## search"));
    }

}

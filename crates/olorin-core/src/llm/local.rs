use crate::llm::{ContentBlock, Message, Role, ToolDef};

#[cfg(feature = "local-llm")]
use std::sync::Mutex;
#[cfg(feature = "local-llm")]
use super::llama_ffi::LlamaEngine;
#[cfg(feature = "local-llm")]
use super::eakv_ffi::EakvCache;
#[cfg(feature = "local-llm")]
use super::tool_parse::{ToolCallDetector, DetectResult};
#[cfg(feature = "local-llm")]
use super::{LlmProvider, LlmResponse, StopReason, OnTextFn};

/// Format a conversation into the Qwen2.5 `<|im_start|>` / `<|im_end|>` chat template.
///
/// When `tools` is non-empty, tool instructions and definitions are appended to the
/// system message. The returned string ends with `<|im_start|>assistant\n` to prompt
/// the model to generate its next turn.
///
/// INVARIANT: The system block (system prompt + tool definitions) MUST always be
/// the first tokens in the output. This enables KV cache prefix reuse across turns
/// via `common_prefix_len()`. Do not inject dynamic content (timestamps, turn IDs)
/// before or within the system block.
pub fn format_chat_template(system: &str, messages: &[Message], tools: &[ToolDef]) -> String {
    let mut out = String::new();

    // --- system block ---
    out.push_str("<|im_start|>system\n");
    out.push_str(system);

    if !tools.is_empty() {
        out.push_str("\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n");
        out.push_str("You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n");
        for tool in tools {
            let tool_json = serde_json::json!({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                }
            });
            out.push_str(&tool_json.to_string());
            out.push('\n');
        }
        out.push_str("</tools>\n\n");
        out.push_str("For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n");
        out.push_str("<tool_call>\n");
        out.push_str("{\"name\": <function-name>, \"arguments\": <args-json-object>}\n");
        out.push_str("</tool_call>");
    }

    out.push_str("\n<|im_end|>\n");

    // --- conversation turns ---
    for msg in messages {
        let role_str = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        out.push_str(&format!("<|im_start|>{}\n", role_str));

        for block in &msg.content {
            match block {
                ContentBlock::Text { text } => {
                    out.push_str(text);
                }
                ContentBlock::ToolUse { id: _, name, input } => {
                    out.push_str("<tool_call>\n");
                    let obj = serde_json::json!({ "name": name, "arguments": input });
                    out.push_str(&obj.to_string());
                    out.push('\n');
                    out.push_str("</tool_call>");
                }
                ContentBlock::ToolResult {
                    tool_use_id: _,
                    content,
                    ..
                } => {
                    out.push_str("<tool_result>\n");
                    out.push_str(content);
                    out.push('\n');
                    out.push_str("</tool_result>");
                }
            }
        }

        out.push_str("\n<|im_end|>\n");
    }

    // --- prompt the model ---
    out.push_str("<|im_start|>assistant\n");

    out
}

/// Compute the length of the common prefix between two token sequences.
///
/// Used for incremental prefill: find how many tokens from the previous context
/// can be reused before the first point of divergence.
pub fn common_prefix_len(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

#[cfg(feature = "local-llm")]
pub struct LocalLlmProvider {
    inner: Mutex<LocalLlmInner>,
}

#[cfg(feature = "local-llm")]
struct LocalLlmInner {
    engine: LlamaEngine,
    kv_cache: EakvCache,
    prefilled_tokens: Vec<i32>,
    tool_id_counter: u64,
    open_pattern: Vec<i32>,
    close_pattern: Vec<i32>,
    eakv_seq_len: i32,
    n_ctx: u32,
    /// Pre-built token ID → string table. Allows text conversion inside the
    /// generation callback without borrowing `engine`.
    vocab_table: Vec<String>,
}

#[cfg(feature = "local-llm")]
impl LocalLlmProvider {
    pub fn new(model_path: &str, n_ctx: u32, n_batch: u32, n_threads: u32, mlock: bool,
               n_layers: i32, n_kv_heads: i32, head_dim: i32) -> crate::error::Result<Self> {
        let engine = LlamaEngine::new(model_path, n_ctx, n_batch, n_threads, mlock)
            .map_err(|e| crate::error::Error::Llm(e))?;

        let kv_cache = EakvCache::new(n_layers, n_kv_heads, head_dim, n_ctx as i32)
            .ok_or_else(|| crate::error::Error::Llm("failed to create eakv cache".into()))?;

        let open_pattern = engine.tokenize("<tool_call>", false);
        let close_pattern = engine.tokenize("</tool_call>", false);
        let vocab_table = engine.build_vocab_table();

        Ok(Self {
            inner: Mutex::new(LocalLlmInner {
                engine,
                kv_cache,
                prefilled_tokens: Vec::new(),
                tool_id_counter: 0,
                open_pattern,
                close_pattern,
                eakv_seq_len: 0,
                n_ctx,
                vocab_table,
            }),
        })
    }

    fn generate_tool_id(inner: &mut LocalLlmInner) -> String {
        inner.tool_id_counter += 1;
        format!("local_{}", inner.tool_id_counter)
    }
}

#[cfg(feature = "local-llm")]
#[async_trait::async_trait]
impl LlmProvider for LocalLlmProvider {
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
        system: &str,
    ) -> crate::error::Result<LlmResponse> {
        let noop: OnTextFn<'_> = &mut |_: &str| {};
        self.chat_stream(messages, tools, system, noop).await
    }

    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
        system: &str,
        on_text: OnTextFn<'_>,
    ) -> crate::error::Result<LlmResponse> {
        let formatted = format_chat_template(system, messages, tools);

        let mut inner = self.inner.lock()
            .map_err(|e| crate::error::Error::Llm(format!("lock poisoned: {e}")))?;

        let t_tok = std::time::Instant::now();
        let new_tokens = inner.engine.tokenize(&formatted, true);
        let tok_ms = t_tok.elapsed().as_secs_f64() * 1000.0;
        let prefix_len = common_prefix_len(&inner.prefilled_tokens, &new_tokens);

        // If prefix diverges, truncate llama.cpp KV cache (and eakv if synced)
        if prefix_len < inner.prefilled_tokens.len() {
            inner.engine.kv_cache_truncate(prefix_len as i32);
            // eakv restore is best-effort — the bridge format may not be active yet
            let _ = inner.kv_cache.restore(prefix_len as i32);
            inner.eakv_seq_len = prefix_len as i32;
        }

        // Prefill only the new suffix
        let suffix = &new_tokens[prefix_len..];
        let suffix_len = suffix.len();
        let t_decode = std::time::Instant::now();
        if !suffix.is_empty() {
            inner.engine.decode(suffix, prefix_len as i32)
                .map_err(|e| crate::error::Error::Llm(e))?;
        }
        let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;

        let t_eakv = std::time::Instant::now();
        if suffix_len > 0 {
            // Export KV state from llama.cpp -> eakv (Approach A, best-effort)
            let kv_state = inner.engine.export_kv_state();
            let seq_len = inner.eakv_seq_len;
            if let Err(e) = inner.kv_cache.import_llama_state(&kv_state, seq_len) {
                tracing::debug!("eakv KV import skipped: {e}");
            } else {
                inner.eakv_seq_len = inner.kv_cache.seq_len();
            }
        }
        let eakv_ms = t_eakv.elapsed().as_secs_f64() * 1000.0;

        inner.prefilled_tokens = new_tokens.clone();

        // Checkpoint eakv before generation (best-effort)
        let _checkpoint = inner.kv_cache.checkpoint();

        eprintln!("eaclaw prefill: {suffix_len} tokens ({prefix_len} reused) — tokenize {tok_ms:.1} ms, decode {decode_ms:.1} ms, eakv sync {eakv_ms:.1} ms [n_ctx={}]", inner.n_ctx);

        // Generation loop — runs in C++ for minimal per-token overhead.
        // Single-pass: streaming, tool detection, and content block building all
        // happen inside the callback using the pre-built vocab_table, avoiding
        // the previous replay pass entirely.
        let gen_pos = inner.prefilled_tokens.len() as i32;
        let max_gen = 2048i32;

        let mut detector = ToolCallDetector::new(
            inner.open_pattern.clone(),
            inner.close_pattern.clone(),
            512,
        );
        let mut content_blocks: Vec<ContentBlock> = Vec::new();
        let mut text_buf = String::new();
        let mut stop_reason = StopReason::EndTurn;
        let mut tool_call_body: Option<Vec<i32>> = None;
        // Take vocab_table out of inner to avoid borrow conflict with engine.
        // It's put back after generate_stream returns.
        let vocab_table = std::mem::take(&mut inner.vocab_table);

        let result = inner.engine.generate_stream_timed(gen_pos, max_gen, |token| {
            match detector.feed(token) {
                DetectResult::Text(t) => {
                    if let Some(piece) = vocab_table.get(t as usize) {
                        text_buf.push_str(piece);
                        on_text(piece);
                    }
                }
                DetectResult::TagOpen | DetectResult::Captured => {
                    // Suppressed from output
                }
                DetectResult::ToolCall(body_tokens) => {
                    tool_call_body = Some(body_tokens);
                    return true; // stop generation
                }
                DetectResult::Aborted(tokens) => {
                    let text: String = tokens.iter()
                        .filter_map(|&t| vocab_table.get(t as usize))
                        .cloned()
                        .collect();
                    text_buf.push_str(&text);
                    on_text(&text);
                }
            }
            false // continue
        });
        let generated = result.tokens;
        let n_gen = generated.len();
        if n_gen > 0 && result.decode_ms > 0.0 {
            let tps = n_gen as f64 / (result.decode_ms / 1000.0);
            eprintln!("eaclaw decode: {n_gen} tokens in {:.1} ms ({tps:.1} tok/s)", result.decode_ms);
        }

        // Update prefilled_tokens with generated tokens
        inner.prefilled_tokens.extend_from_slice(&generated);
        // Restore vocab_table into inner
        inner.vocab_table = vocab_table;

        // Process tool call if detected
        if let Some(body_tokens) = tool_call_body {
            if !text_buf.is_empty() {
                content_blocks.push(ContentBlock::Text {
                    text: std::mem::take(&mut text_buf),
                });
            }

            let json_str: String = body_tokens.iter()
                .filter_map(|&t| inner.vocab_table.get(t as usize))
                .cloned()
                .collect();
            match serde_json::from_str::<serde_json::Value>(&json_str) {
                Ok(val) => {
                    let name = val["name"].as_str().unwrap_or("").to_string();
                    let arguments = val["arguments"].clone();
                    let id = Self::generate_tool_id(&mut inner);
                    content_blocks.push(ContentBlock::ToolUse {
                        id,
                        name,
                        input: arguments,
                    });
                    stop_reason = StopReason::ToolUse;
                }
                Err(_e) => {
                    text_buf.push_str(&format!("<tool_call>{json_str}</tool_call>"));
                    on_text(&format!("<tool_call>{json_str}</tool_call>"));
                }
            }
        }

        if !text_buf.is_empty() {
            content_blocks.push(ContentBlock::Text { text: text_buf });
        }

        // Fallback: if tools were provided but model output raw JSON without
        // <tool_call> tags, try to parse accumulated text as a tool call.
        if !tools.is_empty() && stop_reason != StopReason::ToolUse {
            if let Some(idx) = content_blocks.iter().position(|b| {
                matches!(b, ContentBlock::Text { text } if text.trim_start().starts_with('{'))
            }) {
                if let ContentBlock::Text { text } = &content_blocks[idx] {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(text.trim()) {
                        if let Some(name_str) = val.get("name").and_then(|n| n.as_str()) {
                            let name = name_str.to_string();
                            let arguments = val.get("arguments")
                                .cloned()
                                .unwrap_or(serde_json::Value::Object(Default::default()));
                            let id = Self::generate_tool_id(&mut inner);
                            content_blocks[idx] = ContentBlock::ToolUse {
                                id, name, input: arguments,
                            };
                            stop_reason = StopReason::ToolUse;
                        }
                    }
                }
            }
        }

        if content_blocks.is_empty() {
            content_blocks.push(ContentBlock::Text { text: String::new() });
        }

        Ok(LlmResponse {
            content: content_blocks,
            stop_reason,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ContentBlock, Message, Role, ToolDef};
    use serde_json::json;

    fn user_msg(text: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::text(text)],
        }
    }

    fn assistant_msg(text: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::text(text)],
        }
    }

    #[test]
    fn format_messages_basic() {
        let messages = vec![user_msg("Hello!"), assistant_msg("Hi there.")];
        let result = format_chat_template("You are helpful.", &messages, &[]);

        assert!(result.starts_with("<|im_start|>system\nYou are helpful."));
        assert!(result.contains("<|im_end|>\n<|im_start|>user\nHello!"));
        assert!(result.contains("<|im_end|>\n<|im_start|>assistant\nHi there."));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn format_messages_with_tools() {
        let tools = vec![ToolDef {
            name: "calc".to_string(),
            description: "Calculate math".to_string(),
            input_schema: json!({"type":"object","properties":{"expr":{"type":"string"}}}),
        }];
        let messages = vec![user_msg("What is 2+2?")];
        let result = format_chat_template("You are helpful.", &messages, &tools);

        assert!(result.contains("You may call one or more functions"));
        assert!(result.contains("<tools>"));
        assert!(result.contains("</tools>"));
        assert!(result.contains("\"name\":\"calc\""));
        assert!(result.contains("\"description\":\"Calculate math\""));
        assert!(result.contains("<tool_call>"));
        assert!(result.contains("</tool_call>"));
        assert!(result.contains("expr"));
        assert!(result.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn common_prefix_length_identical() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![1, 2, 3, 4, 5];
        assert_eq!(common_prefix_len(&a, &b), 5);
    }

    #[test]
    fn common_prefix_length_partial() {
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![1, 2, 3, 6, 7, 8];
        assert_eq!(common_prefix_len(&a, &b), 3);
    }

    #[test]
    fn common_prefix_length_empty() {
        let a: Vec<i32> = vec![];
        let b = vec![1, 2, 3];
        assert_eq!(common_prefix_len(&a, &b), 0);
    }

    #[test]
    fn common_prefix_length_diverge_at_start() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        assert_eq!(common_prefix_len(&a, &b), 0);
    }
}

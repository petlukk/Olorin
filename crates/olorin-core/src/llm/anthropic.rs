use super::{ContentBlock, LlmProvider, LlmResponse, Message, OnTextFn, StopReason, ToolDef};
use crate::config::Config;
use crate::error::{Error, Result};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";
const MAX_TOKENS: u32 = 4096;

pub struct AnthropicProvider {
    client: Client,
    api_key: String,
    model: String,
}

impl AnthropicProvider {
    pub fn new(config: &Config) -> Self {
        Self {
            client: Client::new(),
            api_key: config.api_key.clone(),
            model: config.model.clone(),
        }
    }
}

#[derive(Serialize)]
struct ApiRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    system: &'a str,
    messages: Vec<ApiMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<ApiTool>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    stream: bool,
}

#[derive(Serialize)]
struct ApiMessage {
    role: String,
    content: serde_json::Value,
}

#[derive(Serialize)]
struct ApiTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Deserialize)]
struct ApiResponse {
    content: Vec<ApiContentBlock>,
    stop_reason: String,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ApiContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Deserialize)]
struct ApiError {
    error: ApiErrorDetail,
}

#[derive(Deserialize)]
struct ApiErrorDetail {
    message: String,
}

fn convert_messages(messages: &[Message]) -> Vec<ApiMessage> {
    messages
        .iter()
        .map(|msg| {
            let role = match msg.role {
                super::Role::User => "user",
                super::Role::Assistant => "assistant",
            };

            let content: Vec<serde_json::Value> = msg
                .content
                .iter()
                .map(|block| match block {
                    ContentBlock::Text { text } => {
                        serde_json::json!({"type": "text", "text": text})
                    }
                    ContentBlock::ToolUse { id, name, input } => {
                        serde_json::json!({
                            "type": "tool_use",
                            "id": id,
                            "name": name,
                            "input": input
                        })
                    }
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } => {
                        let mut v = serde_json::json!({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": content
                        });
                        if let Some(true) = is_error {
                            v["is_error"] = serde_json::json!(true);
                        }
                        v
                    }
                })
                .collect();

            ApiMessage {
                role: role.to_string(),
                content: serde_json::Value::Array(content),
            }
        })
        .collect()
}

fn make_api_request<'a>(
    model: &'a str,
    system: &'a str,
    messages: &[Message],
    tools: &[ToolDef],
    stream: bool,
) -> ApiRequest<'a> {
    let api_tools: Vec<ApiTool> = tools
        .iter()
        .map(|t| ApiTool {
            name: t.name.clone(),
            description: t.description.clone(),
            input_schema: t.input_schema.clone(),
        })
        .collect();

    ApiRequest {
        model,
        max_tokens: MAX_TOKENS,
        system,
        messages: convert_messages(messages),
        tools: api_tools,
        stream,
    }
}

/// SSE event types from Anthropic streaming API.
#[derive(Deserialize)]
struct SseContentBlockStart {
    content_block: SseContentBlock,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum SseContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String },
}

#[derive(Deserialize)]
struct SseDelta {
    delta: SseDeltaInner,
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum SseDeltaInner {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
}

#[derive(Deserialize)]
struct SseMessageDelta {
    delta: SseMessageDeltaInner,
}

#[derive(Deserialize)]
struct SseMessageDeltaInner {
    stop_reason: Option<String>,
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
        system: &str,
    ) -> Result<LlmResponse> {
        let request = make_api_request(&self.model, system, messages, tools, false);

        let response = self
            .client
            .post(API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            let err_msg = serde_json::from_str::<ApiError>(&body)
                .map(|e| e.error.message)
                .unwrap_or_else(|_| format!("HTTP {status}: {body}"));
            return Err(Error::Llm(err_msg));
        }

        let api_response: ApiResponse =
            serde_json::from_str(&body).map_err(|e| Error::Llm(format!("parse error: {e}")))?;

        let content = api_response
            .content
            .into_iter()
            .map(|block| match block {
                ApiContentBlock::Text { text } => ContentBlock::Text { text },
                ApiContentBlock::ToolUse { id, name, input } => {
                    ContentBlock::ToolUse { id, name, input }
                }
            })
            .collect();

        let stop_reason = match api_response.stop_reason.as_str() {
            "tool_use" => StopReason::ToolUse,
            "max_tokens" => StopReason::MaxTokens,
            _ => StopReason::EndTurn,
        };

        Ok(LlmResponse {
            content,
            stop_reason,
        })
    }

    async fn chat_stream(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
        system: &str,
        on_text: OnTextFn<'_>,
    ) -> Result<LlmResponse> {
        let request = make_api_request(&self.model, system, messages, tools, true);

        let response = self
            .client
            .post(API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await?;
            let err_msg = serde_json::from_str::<ApiError>(&body)
                .map(|e| e.error.message)
                .unwrap_or_else(|_| format!("HTTP {status}: {body}"));
            return Err(Error::Llm(err_msg));
        }

        // Parse SSE stream
        let mut content_blocks: Vec<ContentBlock> = Vec::new();
        let mut stop_reason = StopReason::EndTurn;

        // Track current block state for accumulation
        let mut current_text = String::new();
        let mut current_tool_id = String::new();
        let mut current_tool_name = String::new();
        let mut current_tool_json = String::new();
        let mut in_tool_block = false;
        let mut done = false;

        let mut stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            if done {
                break;
            }
            let chunk = chunk.map_err(|e| Error::Llm(format!("stream error: {e}")))?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Process complete lines from buffer
            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].trim_end().to_string();
                buffer = buffer[newline_pos + 1..].to_string();

                if !line.starts_with("data: ") {
                    continue;
                }
                let data = &line[6..];
                if data == "[DONE]" {
                    continue;
                }

                // Parse the event type from the JSON
                let json: serde_json::Value = match serde_json::from_str(data) {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                let event_type = match json.get("type").and_then(|t| t.as_str()) {
                    Some(t) => t.to_string(),
                    None => continue,
                };

                match event_type.as_str() {
                    "content_block_start" => {
                        if let Ok(cbs) =
                            serde_json::from_value::<SseContentBlockStart>(json.clone())
                        {
                            match cbs.content_block {
                                SseContentBlock::Text { text } => {
                                    in_tool_block = false;
                                    current_text = text;
                                }
                                SseContentBlock::ToolUse { id, name } => {
                                    // Flush any accumulated text
                                    if !current_text.is_empty() {
                                        content_blocks.push(ContentBlock::Text {
                                            text: std::mem::take(&mut current_text),
                                        });
                                    }
                                    in_tool_block = true;
                                    current_tool_id = id;
                                    current_tool_name = name;
                                    current_tool_json.clear();
                                }
                            }
                        }
                    }
                    "content_block_delta" => {
                        if let Ok(delta) = serde_json::from_value::<SseDelta>(json.clone()) {
                            match delta.delta {
                                SseDeltaInner::TextDelta { text } => {
                                    on_text(&text);
                                    current_text.push_str(&text);
                                }
                                SseDeltaInner::InputJsonDelta { partial_json } => {
                                    current_tool_json.push_str(&partial_json);
                                }
                            }
                        }
                    }
                    "content_block_stop" => {
                        if in_tool_block {
                            let input: serde_json::Value =
                                serde_json::from_str(&current_tool_json)
                                    .unwrap_or_else(|_| serde_json::json!({}));
                            content_blocks.push(ContentBlock::ToolUse {
                                id: std::mem::take(&mut current_tool_id),
                                name: std::mem::take(&mut current_tool_name),
                                input,
                            });
                            current_tool_json.clear();
                            in_tool_block = false;
                        } else if !current_text.is_empty() {
                            content_blocks.push(ContentBlock::Text {
                                text: std::mem::take(&mut current_text),
                            });
                        }
                    }
                    "message_delta" => {
                        if let Ok(md) = serde_json::from_value::<SseMessageDelta>(json.clone()) {
                            if let Some(reason) = md.delta.stop_reason {
                                stop_reason = match reason.as_str() {
                                    "tool_use" => StopReason::ToolUse,
                                    "max_tokens" => StopReason::MaxTokens,
                                    _ => StopReason::EndTurn,
                                };
                            }
                        }
                    }
                    "message_stop" => {
                        done = true;
                        break;
                    }
                    _ => {}
                }
            }
        }

        // Flush any remaining text
        if !current_text.is_empty() {
            content_blocks.push(ContentBlock::Text {
                text: current_text,
            });
        }

        Ok(LlmResponse {
            content: content_blocks,
            stop_reason,
        })
    }
}

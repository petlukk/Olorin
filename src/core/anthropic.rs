//! Anthropic API client via curl subprocess.
//!
//! Uses std::process::Command::new("curl") and crate::storage::json for
//! request building and response parsing.

use std::process::Command;
use crate::error::{Error, Result};
use crate::storage::json::{self, Object, Value};

const API_URL:     &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";
const DEFAULT_CLOUD_MAX_TOKENS: i64 = 4096;

pub struct AnthropicClient {
    api_key: String,
    model:   String,
    max_tokens: i64,
}

impl AnthropicClient {
    pub fn new(api_key: String) -> Self {
        Self { api_key, model: "claude-3-5-haiku-latest".to_string(), max_tokens: DEFAULT_CLOUD_MAX_TOKENS }
    }

    pub fn with_model(api_key: String, model: String) -> Self {
        Self { api_key, model, max_tokens: DEFAULT_CLOUD_MAX_TOKENS }
    }

    pub fn set_api_key(&mut self, key: String) { self.api_key = key; }
    pub fn set_model(&mut self, model: String) { self.model = model; }
    pub fn model(&self) -> &str { &self.model }
    pub fn has_key(&self) -> bool { !self.api_key.is_empty() }
    pub fn set_max_tokens(&mut self, n: i64) { self.max_tokens = n; }
    pub fn max_tokens(&self) -> i64 { self.max_tokens }

    /// Generate a response from the Anthropic API.
    ///
    /// `messages` is a slice of `(role, content)` pairs.
    /// `role` must be `"user"` or `"assistant"`.
    ///
    /// Blocks until the response is complete.
    pub fn generate(&self, system: &str, messages: &[(&str, &str)]) -> Result<String> {
        let body = build_request(&self.model, self.max_tokens, system, messages);
        let output = Command::new("curl")
            .arg("--silent")
            .arg("--fail-with-body")
            .arg("--connect-timeout").arg("15")
            .arg("--max-time").arg("120")
            .arg("-X").arg("POST")
            .arg(API_URL)
            .arg("-H").arg(format!("x-api-key: {}", self.api_key))
            .arg("-H").arg(format!("anthropic-version: {API_VERSION}"))
            .arg("-H").arg("content-type: application/json")
            .arg("-d").arg(&body)
            .output()
            .map_err(|e| Error::Config(format!("curl not found: {e}")))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            // Try to extract API error message from response body
            if let Ok(obj) = json::parse(&output.stdout) {
                if let Some(err_obj) = obj.get_object("error") {
                    if let Some(msg) = err_obj.get_str("message") {
                        return Err(Error::Config(format!("Anthropic API error: {msg}")));
                    }
                }
            }
            return Err(Error::Config(format!(
                "curl failed (exit {}): {stderr} | {stdout}",
                output.status.code().unwrap_or(-1)
            )));
        }

        parse_response(&output.stdout)
    }
}

/// Build JSON request body using crate::storage::json.
fn build_request(model: &str, max_tokens: i64, system: &str, messages: &[(&str, &str)]) -> String {
    let mut req = Object::new();
    req.set("model",      Value::Str(model.to_string()));
    req.set("max_tokens", Value::I64(max_tokens));
    req.set("system",     Value::Str(system.to_string()));

    let msgs: Vec<Value> = messages.iter().map(|(role, content)| {
        let mut msg = Object::new();
        msg.set("role",    Value::Str(role.to_string()));
        msg.set("content", Value::Str(content.to_string()));
        Value::Object(Box::new(msg))
    }).collect();

    req.set("messages", Value::Array(msgs));
    json::serialize(&req)
}

/// Extract text content from Anthropic API response.
fn parse_response(body: &[u8]) -> Result<String> {
    let obj = json::parse(body)
        .map_err(|e| Error::Config(format!("JSON parse error: {e}")))?;

    // Check for API-level error
    if let Some(err_obj) = obj.get_object("error") {
        let msg = err_obj.get_str("message").unwrap_or("unknown API error");
        return Err(Error::Config(format!("Anthropic error: {msg}")));
    }

    let content = obj.get_array("content")
        .ok_or_else(|| Error::Config("missing 'content' array in response".to_string()))?;

    let mut text = String::new();
    for block in content {
        if let Value::Object(obj) = block {
            if obj.get_str("type") == Some("text") {
                if let Some(t) = obj.get_str("text") {
                    text.push_str(t);
                }
            }
        }
    }

    if text.is_empty() {
        return Err(Error::Config("empty response from Anthropic API".to_string()));
    }

    Ok(text)
}

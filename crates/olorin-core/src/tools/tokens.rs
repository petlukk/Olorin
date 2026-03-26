use super::Tool;
use async_trait::async_trait;

pub struct TokensTool;

#[async_trait]
impl Tool for TokensTool {
    fn name(&self) -> &str {
        "tokens"
    }

    fn description(&self) -> &str {
        "Estimate token count for text or a file. Uses byte-pair encoding heuristics."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text to tokenize (or file path)"
                }
            },
            "required": ["text"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> crate::error::Result<String> {
        let input = params["text"]
            .as_str()
            .ok_or_else(|| crate::error::Error::Tool("missing 'text' parameter".into()))?;

        // If it looks like a file path, read it
        let text = if !input.contains(' ') && (input.contains('/') || input.contains('.')) {
            match tokio::fs::read_to_string(input).await {
                Ok(content) => content,
                Err(_) => input.to_string(),
            }
        } else {
            input.to_string()
        };

        let bytes = text.len();
        let chars = text.chars().count();
        let words = text.split_whitespace().count();

        // BPE heuristic: ~4 chars per token for English, ~3 for code
        // Count "code-like" indicators
        let code_chars = text
            .bytes()
            .filter(|b| matches!(b, b'{' | b'}' | b'(' | b')' | b';' | b'=' | b'<' | b'>'))
            .count();
        let is_code = code_chars > chars / 20;
        let chars_per_token: f64 = if is_code { 3.2 } else { 3.8 };
        let estimated_tokens = (chars as f64 / chars_per_token).ceil() as usize;

        Ok(format!(
            "Bytes: {bytes}\nChars: {chars}\nWords: {words}\nEstimated tokens: ~{estimated_tokens} ({})",
            if is_code { "code" } else { "text" }
        ))
    }
}

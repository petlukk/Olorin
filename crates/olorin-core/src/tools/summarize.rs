use super::{Tool, check_host};
use crate::llm::{ContentBlock, LlmProvider, Message, Role};
use async_trait::async_trait;
use std::sync::Arc;

pub struct SummarizeTool {
    llm: Arc<dyn LlmProvider>,
    allowed_hosts: Vec<String>,
}

impl SummarizeTool {
    pub fn new(llm: Arc<dyn LlmProvider>, allowed_hosts: Vec<String>) -> Self {
        Self { llm, allowed_hosts }
    }
}

const SYSTEM: &str = "Summarize the following content in 2-3 concise sentences. Output ONLY the summary.";
const MAX_CONTENT: usize = 16 * 1024;

#[async_trait]
impl Tool for SummarizeTool {
    fn name(&self) -> &str {
        "summarize"
    }

    fn description(&self) -> &str {
        "Fetch a URL and summarize its content."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch and summarize"
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> crate::error::Result<String> {
        let url = params["url"]
            .as_str()
            .ok_or_else(|| crate::error::Error::Tool("missing 'url' parameter".into()))?;

        check_host(&self.allowed_hosts, url)?;

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .map_err(|e| crate::error::Error::Tool(format!("http client error: {e}")))?;
        let body = client.get(url).send().await?.text().await?;

        // Strip HTML tags (cheap, no regex)
        let text = strip_tags(&body);

        // Truncate for LLM context
        let text = if text.len() > MAX_CONTENT {
            let mut end = MAX_CONTENT;
            while !text.is_char_boundary(end) {
                end -= 1;
            }
            &text[..end]
        } else {
            &text
        };

        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::text(text)],
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

/// Strip HTML tags. No regex, no deps.
fn strip_tags(html: &str) -> String {
    let mut out = String::with_capacity(html.len() / 2);
    let mut in_tag = false;
    for c in html.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out
}

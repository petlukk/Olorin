use super::Tool;
use async_trait::async_trait;

pub struct TimeTool;

#[async_trait]
impl Tool for TimeTool {
    fn name(&self) -> &str {
        "time"
    }

    fn description(&self) -> &str {
        "Get the current UTC date and time."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(&self, _params: serde_json::Value) -> crate::error::Result<String> {
        let now = chrono::Utc::now();
        Ok(now.format("%a %b %-d, %Y — %H:%M UTC").to_string())
    }
}

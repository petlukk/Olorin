use super::{Tool, check_host};
use async_trait::async_trait;

pub struct WeatherTool {
    allowed_hosts: Vec<String>,
}

impl WeatherTool {
    pub fn new(allowed_hosts: Vec<String>) -> Self {
        Self { allowed_hosts }
    }
}

#[async_trait]
impl Tool for WeatherTool {
    fn name(&self) -> &str {
        "weather"
    }

    fn description(&self) -> &str {
        "Get current weather for a city."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["city"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> crate::error::Result<String> {
        let city = params["city"]
            .as_str()
            .ok_or_else(|| crate::error::Error::Tool("missing 'city' parameter".into()))?;

        let url = format!("https://wttr.in/{}?format=%l:+%C+%t+%h+%w", urlencode(city));
        check_host(&self.allowed_hosts, &url)?;
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(10))
            .build()
            .map_err(|e| crate::error::Error::Tool(format!("http client error: {e}")))?;
        let body = client.get(&url).send().await?.text().await?;
        let body = body.trim();

        if body.contains("Unknown location") || body.contains("ERROR") {
            return Err(crate::error::Error::Tool(format!("unknown city: {city}")));
        }

        Ok(body.to_string())
    }
}

fn urlencode(s: &str) -> String {
    s.replace(' ', "+")
}

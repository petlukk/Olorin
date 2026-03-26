use super::Tool;
use async_trait::async_trait;
use std::time::Duration;

pub struct RemindTool;

#[async_trait]
impl Tool for RemindTool {
    fn name(&self) -> &str {
        "remind"
    }

    fn description(&self) -> &str {
        "Set a reminder. Sleeps for the given duration, then returns the message. Use with & for background execution."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "time": {
                    "type": "string",
                    "description": "Duration (e.g. '30s', '5m', '2h', '1h30m')"
                },
                "message": {
                    "type": "string",
                    "description": "Reminder message"
                }
            },
            "required": ["time", "message"]
        })
    }

    async fn execute(&self, params: serde_json::Value) -> crate::error::Result<String> {
        let time_str = params["time"]
            .as_str()
            .ok_or_else(|| crate::error::Error::Tool("missing 'time' parameter".into()))?;
        let message = params["message"]
            .as_str()
            .ok_or_else(|| crate::error::Error::Tool("missing 'message' parameter".into()))?;

        let duration = parse_duration(time_str)?;

        if duration.as_secs() > 24 * 60 * 60 {
            return Err(crate::error::Error::Tool("maximum reminder duration is 24 hours".into()));
        }

        tokio::time::sleep(duration).await;

        Ok(format!("Reminder: {message}"))
    }
}

/// Parse a human-friendly duration string like "30s", "5m", "2h", "1h30m", "90m".
fn parse_duration(s: &str) -> crate::error::Result<Duration> {
    let s = s.trim().to_lowercase();
    if s.is_empty() {
        return Err(crate::error::Error::Tool("empty duration".into()));
    }

    let mut total_secs: u64 = 0;
    let mut num_buf = String::new();

    for c in s.chars() {
        if c.is_ascii_digit() {
            num_buf.push(c);
        } else {
            if num_buf.is_empty() {
                return Err(crate::error::Error::Tool(
                    format!("invalid duration: '{s}'. Use e.g. '30s', '5m', '2h', '1h30m'")
                ));
            }
            let n: u64 = num_buf.parse().map_err(|_| {
                crate::error::Error::Tool(format!("invalid number in duration: '{s}'"))
            })?;
            num_buf.clear();
            match c {
                's' => total_secs += n,
                'm' => total_secs += n * 60,
                'h' => total_secs += n * 3600,
                _ => {
                    return Err(crate::error::Error::Tool(
                        format!("unknown unit '{c}' in duration. Use s/m/h.")
                    ));
                }
            }
        }
    }

    // Handle bare number (default to minutes)
    if !num_buf.is_empty() {
        let n: u64 = num_buf.parse().map_err(|_| {
            crate::error::Error::Tool(format!("invalid duration: '{s}'"))
        })?;
        total_secs += n * 60;
    }

    if total_secs == 0 {
        return Err(crate::error::Error::Tool("duration must be > 0".into()));
    }

    Ok(Duration::from_secs(total_secs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_seconds() {
        assert_eq!(parse_duration("30s").unwrap(), Duration::from_secs(30));
    }

    #[test]
    fn test_parse_minutes() {
        assert_eq!(parse_duration("5m").unwrap(), Duration::from_secs(300));
    }

    #[test]
    fn test_parse_hours() {
        assert_eq!(parse_duration("2h").unwrap(), Duration::from_secs(7200));
    }

    #[test]
    fn test_parse_combined() {
        assert_eq!(parse_duration("1h30m").unwrap(), Duration::from_secs(5400));
    }

    #[test]
    fn test_parse_bare_number_defaults_to_minutes() {
        assert_eq!(parse_duration("10").unwrap(), Duration::from_secs(600));
    }

    #[test]
    fn test_parse_empty_fails() {
        assert!(parse_duration("").is_err());
    }

    #[test]
    fn test_parse_zero_fails() {
        assert!(parse_duration("0s").is_err());
    }
}

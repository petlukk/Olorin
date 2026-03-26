use super::Tool;
use async_trait::async_trait;

pub struct CpuTool;

#[async_trait]
impl Tool for CpuTool {
    fn name(&self) -> &str {
        "cpu"
    }

    fn description(&self) -> &str {
        "Show system info: CPU, memory, disk, uptime."
    }

    fn parameters_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(&self, _params: serde_json::Value) -> crate::error::Result<String> {
        let mut lines = Vec::new();

        // CPU info
        if let Ok(info) = tokio::fs::read_to_string("/proc/cpuinfo").await {
            let model = info
                .lines()
                .find(|l| l.starts_with("model name"))
                .and_then(|l| l.split(':').nth(1))
                .map(|s| s.trim().to_string());
            let cores = info
                .lines()
                .filter(|l| l.starts_with("processor"))
                .count();
            if let Some(model) = model {
                lines.push(format!("CPU: {model} ({cores} cores)"));
            }
        }

        // Memory
        if let Ok(info) = tokio::fs::read_to_string("/proc/meminfo").await {
            let parse_kb = |prefix: &str| -> Option<u64> {
                info.lines()
                    .find(|l| l.starts_with(prefix))
                    .and_then(|l| {
                        l.split_whitespace()
                            .nth(1)
                            .and_then(|v| v.parse::<u64>().ok())
                    })
            };
            if let (Some(total), Some(avail)) = (parse_kb("MemTotal:"), parse_kb("MemAvailable:"))
            {
                let used = total.saturating_sub(avail);
                lines.push(format!(
                    "Memory: {} MB used / {} MB total",
                    used / 1024,
                    total / 1024
                ));
            }
        }

        // Uptime
        if let Ok(info) = tokio::fs::read_to_string("/proc/uptime").await {
            if let Some(secs) = info.split_whitespace().next().and_then(|s| s.parse::<f64>().ok())
            {
                let hours = (secs / 3600.0) as u64;
                let mins = ((secs % 3600.0) / 60.0) as u64;
                lines.push(format!("Uptime: {}h {}m", hours, mins));
            }
        }

        // Load average
        if let Ok(info) = tokio::fs::read_to_string("/proc/loadavg").await {
            let parts: Vec<&str> = info.split_whitespace().take(3).collect();
            if parts.len() == 3 {
                lines.push(format!("Load: {} {} {}", parts[0], parts[1], parts[2]));
            }
        }

        if lines.is_empty() {
            Ok("System info unavailable".to_string())
        } else {
            Ok(lines.join("\n"))
        }
    }
}

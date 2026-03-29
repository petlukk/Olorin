use super::ToolResult;

pub fn run(args: &str) -> ToolResult {
    let city = args.trim();
    if city.is_empty() {
        return ToolResult { output: "usage: weather <city>".to_string(), success: false };
    }

    let encoded = city.replace(' ', "+");
    let url = format!("https://wttr.in/{}?format=%l:+%C+%t+%h+%w", encoded);

    let output = std::process::Command::new("curl")
        .args(["-s", "-L", "--max-time", "10", "-A", "curl/7.0", &url])
        .output();

    match output {
        Ok(o) => {
            let body = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if body.contains("Unknown location") || body.contains("ERROR") || body.is_empty() {
                ToolResult { output: format!("unknown city or service unavailable: {city}"), success: false }
            } else {
                ToolResult { output: body, success: true }
            }
        }
        Err(e) => ToolResult { output: format!("curl failed: {e}"), success: false },
    }
}

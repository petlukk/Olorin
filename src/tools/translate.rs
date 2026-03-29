use super::ToolResult;

pub fn run(args: &str) -> ToolResult {
    // Format: "<lang> <text>"
    let args = args.trim();
    let (lang, text) = match args.find(' ') {
        Some(pos) => (&args[..pos], args[pos + 1..].trim()),
        None => return ToolResult {
            output: "usage: translate <lang> <text>".to_string(),
            success: false,
        },
    };

    if text.is_empty() {
        return ToolResult { output: "usage: translate <lang> <text>".to_string(), success: false };
    }

    // Use LibreTranslate or a free API via curl if available; otherwise fall back to stub.
    // Try lingva.ml (no API key needed)
    let url = format!(
        "https://lingva.ml/api/v1/en/{}/{}", 
        lang.to_lowercase(),
        urlenc(text)
    );

    let output = std::process::Command::new("curl")
        .args(["-s", "--max-time", "10", &url])
        .output();

    match output {
        Ok(o) if o.status.success() => {
            let body = String::from_utf8_lossy(&o.stdout);
            // Try to extract "translation" field from JSON
            if let Some(start) = body.find("\"translation\":\"") {
                let rest = &body[start + 15..];
                if let Some(end) = rest.find('"') {
                    let translation = &rest[..end];
                    return ToolResult { output: translation.to_string(), success: true };
                }
            }
            ToolResult { output: format!("Could not parse translation response for '{text}' to {lang}"), success: false }
        }
        Ok(_) => ToolResult {
            output: format!("Translation service unavailable. Text: '{text}'"),
            success: false,
        },
        Err(e) => ToolResult { output: format!("curl failed: {e}"), success: false },
    }
}

fn urlenc(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            b' ' => out.push('+'),
            b => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

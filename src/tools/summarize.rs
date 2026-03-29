use super::ToolResult;

const MAX_CONTENT: usize = 16 * 1024;

pub fn run(args: &str) -> ToolResult {
    let url = args.trim();
    if url.is_empty() {
        return ToolResult { output: "usage: summarize <url>".to_string(), success: false };
    }

    // Fetch content via curl
    let output = std::process::Command::new("curl")
        .args(["-s", "-L", "--max-time", "10", url])
        .output();

    let body = match output {
        Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
        Err(e) => return ToolResult { output: format!("curl failed: {e}"), success: false },
    };

    // Strip HTML tags
    let text = strip_tags(&body);
    let text = if text.len() > MAX_CONTENT {
        let mut end = MAX_CONTENT;
        while end > 0 && !text.is_char_boundary(end) { end -= 1; }
        text[..end].to_string()
    } else {
        text
    };

    // Word count as a simple summary proxy (no LLM available in sync context)
    let words = text.split_whitespace().count();
    let first_200: String = text.split_whitespace().take(200).collect::<Vec<_>>().join(" ");

    ToolResult {
        output: format!("Content from {url} ({words} words):\n{first_200}{}",
            if words > 200 { "..." } else { "" }),
        success: true,
    }
}

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

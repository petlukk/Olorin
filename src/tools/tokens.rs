use super::ToolResult;

pub fn run(args: &str) -> ToolResult {
    let input = args.trim();
    if input.is_empty() {
        return ToolResult { output: "usage: tokens <text or file path>".to_string(), success: false };
    }

    // If looks like a file path, try to read it
    let text = if !input.contains(' ') && (input.contains('/') || input.contains('.')) {
        match std::fs::read_to_string(input) {
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
    let code_chars = text
        .bytes()
        .filter(|b| matches!(b, b'{' | b'}' | b'(' | b')' | b';' | b'=' | b'<' | b'>'))
        .count();
    let is_code = code_chars > chars / 20;
    let chars_per_token: f64 = if is_code { 3.2 } else { 3.8 };
    let estimated_tokens = (chars as f64 / chars_per_token).ceil() as usize;

    ToolResult {
        output: format!(
            "Bytes: {bytes}\nChars: {chars}\nWords: {words}\nEstimated tokens: ~{estimated_tokens} ({})",
            if is_code { "code" } else { "text" }
        ),
        success: true,
    }
}

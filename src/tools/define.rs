use super::ToolResult;

pub fn run(args: &str) -> ToolResult {
    let word = args.trim();
    if word.is_empty() {
        return ToolResult { output: "usage: define <word>".to_string(), success: false };
    }

    let url = format!("https://api.dictionaryapi.dev/api/v2/entries/en/{word}");

    let output = std::process::Command::new("curl")
        .args(["-s", "-L", "--max-time", "10", &url])
        .output();

    match output {
        Ok(o) if o.status.success() => {
            let body = String::from_utf8_lossy(&o.stdout);
            if body.contains("\"title\"") && body.contains("No Definitions Found") {
                return ToolResult { output: format!("no definition found for: {word}"), success: false };
            }
            match parse_definitions(&body) {
                Some(defs) => ToolResult { output: defs, success: true },
                None => ToolResult { output: format!("no definition found for: {word}"), success: false },
            }
        }
        Ok(o) => ToolResult {
            output: format!("API error (exit {}): {}", o.status.code().unwrap_or(-1),
                String::from_utf8_lossy(&o.stderr)),
            success: false,
        },
        Err(e) => ToolResult { output: format!("curl failed: {e}"), success: false },
    }
}

/// Extract definitions from the dictionaryapi.dev JSON response without serde_json.
fn parse_definitions(body: &str) -> Option<String> {

    // Response is an array — find the first object
    let start = body.find('{')?;
    let first_obj = &body[start..];
    // Find closing brace for the first entry (rough extraction)
    // We'll parse the meanings array manually
    let mut out = String::new();

    // Find "meanings" array
    let meanings_start = first_obj.find("\"meanings\":")?;
    let arr_start = first_obj[meanings_start..].find('[')? + meanings_start;
    // Walk through partOfSpeech and definitions
    let meanings_body = &first_obj[arr_start..];

    let mut count = 0;
    let mut pos = 0;
    while pos < meanings_body.len() && count < 3 {
        // Find next partOfSpeech
        let pos_key = match meanings_body[pos..].find("\"partOfSpeech\":\"") {
            Some(p) => p + pos,
            None => break,
        };
        let pos_val = pos_key + 16;
        let pos_end = match meanings_body[pos_val..].find('"') {
            Some(e) => e + pos_val,
            None => break,
        };
        let part_of_speech = &meanings_body[pos_val..pos_end];

        // Find definitions array after this point
        let def_search_start = pos_end;
        let def_key = match meanings_body[def_search_start..].find("\"definitions\":") {
            Some(p) => p + def_search_start,
            None => break,
        };
        let def_arr_start = match meanings_body[def_key..].find('[') {
            Some(p) => p + def_key,
            None => break,
        };

        // Extract up to 2 definitions from this meanings entry
        let mut def_pos = def_arr_start + 1;
        let mut def_count = 0;
        while def_pos < meanings_body.len() && def_count < 2 {
            let def_key_pos = match meanings_body[def_pos..].find("\"definition\":\"") {
                Some(p) => p + def_pos,
                None => break,
            };
            let def_val_start = def_key_pos + 14;
            // Find end of string (unescaped quote)
            let mut end = def_val_start;
            let bytes = meanings_body.as_bytes();
            while end < bytes.len() {
                if bytes[end] == b'"' && (end == 0 || bytes[end - 1] != b'\\') {
                    break;
                }
                end += 1;
            }
            let definition = &meanings_body[def_val_start..end];
            out.push_str(&format!("({part_of_speech}) {definition}\n"));
            def_count += 1;
            def_pos = end + 1;
        }

        pos = def_arr_start + 1;
        count += 1;
    }

    if out.is_empty() { None } else { Some(out.trim().to_string()) }
}

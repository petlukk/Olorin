use super::ToolResult;
use crate::storage::json::{self, Value};

pub fn run(args: &str) -> ToolResult {
    // Format: "<action> <input> [path]"
    let args = args.trim();
    let mut parts = args.splitn(3, ' ');
    let action = match parts.next() {
        Some(a) if !a.is_empty() => a,
        _ => return ToolResult { output: "usage: json <keys|get|pretty> <input> [path]".to_string(), success: false },
    };
    let input = match parts.next() {
        Some(i) if !i.is_empty() => i,
        _ => return ToolResult { output: "usage: json <keys|get|pretty> <input> [path]".to_string(), success: false },
    };
    let extra = parts.next().unwrap_or("").trim();

    // Try parsing as JSON directly, or read from file
    let raw = if input.trim_start().starts_with('{') || input.trim_start().starts_with('[') {
        input.to_string()
    } else {
        match std::fs::read_to_string(input) {
            Ok(content) => content,
            Err(e) => return ToolResult { output: format!("{input}: {e}"), success: false },
        }
    };

    match action {
        "keys" => {
            match json::parse(raw.as_bytes()) {
                Ok(obj) => {
                    let keys: Vec<&str> = obj.key_names().collect();
                    if keys.is_empty() {
                        ToolResult { output: "(empty object)".to_string(), success: true }
                    } else {
                        ToolResult { output: keys.join(", "), success: true }
                    }
                }
                Err(e) => ToolResult { output: format!("invalid JSON: {e}"), success: false },
            }
        }
        "get" => {
            if extra.is_empty() {
                return ToolResult { output: "usage: json get <input> <dot.path>".to_string(), success: false };
            }
            match json::parse(raw.as_bytes()) {
                Ok(obj) => {
                    match navigate_obj(&obj, extra) {
                        Some(val) => ToolResult { output: json::serialize_value(&val), success: true },
                        None => ToolResult { output: format!("path '{extra}' not found"), success: false },
                    }
                }
                Err(e) => ToolResult { output: format!("invalid JSON: {e}"), success: false },
            }
        }
        "pretty" => {
            match json::parse(raw.as_bytes()) {
                Ok(obj) => {
                    let pretty = pretty_print_obj(&obj, 0);
                    let max_len = 64 * 1024;
                    if pretty.len() > max_len {
                        ToolResult {
                            output: format!("{}... (truncated, {} bytes)", &pretty[..max_len], pretty.len()),
                            success: true,
                        }
                    } else {
                        ToolResult { output: pretty, success: true }
                    }
                }
                Err(e) => ToolResult { output: format!("invalid JSON: {e}"), success: false },
            }
        }
        _ => ToolResult {
            output: format!("unknown action '{action}'. Use: keys, get, pretty"),
            success: false,
        },
    }
}

fn navigate_obj(obj: &json::Object, path: &str) -> Option<Value> {
    let mut segments = path.split('.');
    let first = segments.next()?;
    let val = obj.get(first)?.clone();
    navigate_value(val, &mut segments)
}

fn navigate_value<'a>(val: Value, segments: &mut impl Iterator<Item = &'a str>) -> Option<Value> {
    match segments.next() {
        None => Some(val),
        Some(seg) if seg.is_empty() => Some(val),
        Some(seg) => {
            match val {
                Value::Object(o) => {
                    let child = o.get(seg)?.clone();
                    navigate_value(child, segments)
                }
                Value::Array(arr) => {
                    let idx: usize = seg.parse().ok()?;
                    let child = arr.get(idx)?.clone();
                    navigate_value(child, segments)
                }
                _ => None,
            }
        }
    }
}

fn pretty_print_obj(obj: &json::Object, indent: usize) -> String {
    let pad = "  ".repeat(indent);
    let inner_pad = "  ".repeat(indent + 1);
    let pairs: Vec<(&str, &Value)> = obj.iter().collect();
    let mut out = String::from("{\n");
    for (i, (key, val)) in pairs.iter().enumerate() {
        out.push_str(&format!("{inner_pad}\"{key}\": {}", pretty_print_val(val, indent + 1)));
        if i + 1 < pairs.len() { out.push(','); }
        out.push('\n');
    }
    out.push_str(&format!("{pad}}}"));
    out
}

fn pretty_print_val(val: &Value, indent: usize) -> String {
    match val {
        Value::Object(o) => pretty_print_obj(o, indent),
        Value::Array(arr) => {
            if arr.is_empty() { return "[]".to_string(); }
            let inner_pad = "  ".repeat(indent + 1);
            let pad = "  ".repeat(indent);
            let mut out = String::from("[\n");
            for (i, v) in arr.iter().enumerate() {
                out.push_str(&format!("{inner_pad}{}", pretty_print_val(v, indent + 1)));
                if i + 1 < arr.len() { out.push(','); }
                out.push('\n');
            }
            out.push_str(&format!("{pad}]"));
            out
        }
        _ => json::serialize_value(val),
    }
}

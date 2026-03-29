use super::ToolResult;
use std::collections::HashMap;
use std::sync::Mutex;

static STORE: Mutex<Option<HashMap<String, String>>> = Mutex::new(None);

fn with_store<F: FnOnce(&mut HashMap<String, String>) -> ToolResult>(f: F) -> ToolResult {
    let mut guard = match STORE.lock() {
        Ok(g) => g,
        Err(_) => return ToolResult { output: "memory store lock poisoned".to_string(), success: false },
    };
    if guard.is_none() {
        *guard = Some(HashMap::new());
    }
    f(guard.as_mut().unwrap())
}

pub fn run(args: &str) -> ToolResult {
    let args = args.trim();
    // Format: "write key value", "read key", "list"
    let mut parts = args.splitn(3, ' ');
    let action = parts.next().unwrap_or("").trim();
    let key = parts.next().unwrap_or("").trim();
    let value = parts.next().unwrap_or("").trim();

    match action {
        "write" => {
            if key.is_empty() {
                return ToolResult { output: "usage: memory write <key> <value>".to_string(), success: false };
            }
            if value.is_empty() {
                return ToolResult { output: "usage: memory write <key> <value>".to_string(), success: false };
            }
            with_store(|store| {
                store.insert(key.to_string(), value.to_string());
                ToolResult { output: format!("Stored '{key}'"), success: true }
            })
        }
        "read" => {
            if key.is_empty() {
                return ToolResult { output: "usage: memory read <key>".to_string(), success: false };
            }
            with_store(|store| {
                match store.get(key) {
                    Some(val) => ToolResult { output: val.clone(), success: true },
                    None => ToolResult { output: format!("Key '{key}' not found"), success: false },
                }
            })
        }
        "list" => {
            with_store(|store| {
                if store.is_empty() {
                    ToolResult { output: "No keys stored".to_string(), success: true }
                } else {
                    let mut keys: Vec<&str> = store.keys().map(|k| k.as_str()).collect();
                    keys.sort_unstable();
                    ToolResult { output: keys.join(", "), success: true }
                }
            })
        }
        _ => ToolResult {
            output: "usage: memory <write|read|list> [key] [value]".to_string(),
            success: false,
        },
    }
}

use super::ToolResult;
use crate::core::path_guard::{resolve_safe_path_checked, AccessMode};

pub fn run(args: &str) -> ToolResult {
    let raw = if args.trim().is_empty() { "." } else { args.trim() };

    let resolved = match resolve_safe_path_checked(raw, AccessMode::Read) {
        Ok(p) => p,
        Err(e) => return ToolResult { output: e.refusal_message(), success: false },
    };

    let dir = match std::fs::read_dir(&resolved) {
        Ok(d) => d,
        Err(e) => return ToolResult { output: format!("{raw}: {e}"), success: false },
    };

    let mut entries = Vec::new();
    for entry in dir {
        match entry {
            Ok(e) => {
                let name = e.file_name().to_string_lossy().into_owned();
                let meta = e.metadata().ok();
                let is_dir = meta.as_ref().map_or(false, |m| m.is_dir());
                let size = meta.as_ref().map_or(0, |m| m.len());
                if is_dir {
                    entries.push(format!("  {name}/"));
                } else {
                    entries.push(format!("  {name}  ({size} B)"));
                }
            }
            Err(e) => entries.push(format!("  <error: {e}>")),
        }
    }

    entries.sort();

    if entries.is_empty() {
        ToolResult { output: format!("{raw}: empty directory"), success: true }
    } else {
        ToolResult { output: format!("{raw}:\n{}", entries.join("\n")), success: true }
    }
}

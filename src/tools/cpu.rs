use super::ToolResult;
use crate::platform::sysinfo;

pub fn run(_args: &str) -> ToolResult {
    let mut lines = Vec::new();

    let cores = sysinfo::cpu_cores();
    match sysinfo::cpu_model() {
        Some(m) => lines.push(format!("CPU: {m} ({cores} cores)")),
        None    => lines.push(format!("CPU: {cores} cores")),
    }

    if let Some((used, total)) = sysinfo::memory_usage_mb() {
        lines.push(format!("Memory: {used} MB used / {total} MB total"));
    }

    if let Some(secs) = sysinfo::uptime_seconds() {
        let hours = secs / 3600;
        let mins  = (secs % 3600) / 60;
        lines.push(format!("Uptime: {hours}h {mins}m"));
    }

    if let Some((a, b, c)) = sysinfo::load_average() {
        lines.push(format!("Load: {a} {b} {c}"));
    }

    if lines.is_empty() {
        ToolResult { output: "System info unavailable".to_string(), success: false }
    } else {
        ToolResult { output: lines.join("\n"), success: true }
    }
}

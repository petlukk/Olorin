pub mod calc;
pub mod shell;
pub mod http;
pub mod grep;
pub mod git;
pub mod time;
pub mod cpu;
pub mod weather;
pub mod memory;
pub mod tokens;
pub mod summarize;
pub mod translate;
pub mod define;
pub mod remind;
pub mod read_file;
pub mod write_file;
pub mod ls;
pub mod json_tool;
pub mod bench;

pub struct ToolResult {
    pub output: String,
    pub success: bool,
}

/// Run a tool by name. Returns None if tool not found.
pub fn run_tool(name: &str, args: &str) -> Option<ToolResult> {
    match name {
        "calc" => Some(calc::run(args)),
        "shell" => Some(shell::run(args)),
        "http" | "fetch" => Some(http::run(args)),
        "grep" => Some(grep::run(args)),
        "git" => Some(git::run(args)),
        "time" => Some(time::run(args)),
        "cpu" => Some(cpu::run(args)),
        "weather" => Some(weather::run(args)),
        "memory" | "mem" => Some(memory::run(args)),
        "tokens" => Some(tokens::run(args)),
        "summarize" => Some(summarize::run(args)),
        "translate" => Some(translate::run(args)),
        "define" => Some(define::run(args)),
        "remind" => Some(remind::run(args)),
        "read" | "read_file" => Some(read_file::run(args)),
        "write" | "write_file" => Some(write_file::run(args)),
        "ls" => Some(ls::run(args)),
        "json" => Some(json_tool::run(args)),
        "bench" => Some(bench::run(args)),
        _ => None,
    }
}

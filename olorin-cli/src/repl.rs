use std::io::{self, Write};
use std::sync::Arc;

use olorin_core::kernels::command_router as cmd_router;

use crate::CougarEngine;

pub enum ReplAction {
    Quit,
    Print(String),
    Generate(String),
}

pub struct OlorinRepl {
    pub engine: Option<Arc<CougarEngine>>,
}

impl OlorinRepl {
    pub fn process(&self, input: &str) -> ReplAction {
        let (cmd_id, cmd_arg) = cmd_router::match_command_verified(input.as_bytes());
        let arg = std::str::from_utf8(cmd_arg).unwrap_or("").trim();

        match cmd_id {
            cmd_router::CMD_QUIT => ReplAction::Quit,
            cmd_router::CMD_HELP => ReplAction::Print(self.help_text()),
            cmd_router::CMD_MODEL => ReplAction::Print(self.handle_model(arg)),
            cmd_router::CMD_TELEPORT => ReplAction::Print(self.handle_teleport(arg)),
            cmd_router::CMD_TIME => ReplAction::Print(self.handle_time()),
            cmd_router::CMD_SHELL => ReplAction::Print(self.handle_shell(arg)),
            cmd_router::CMD_TOOLS => ReplAction::Print(self.handle_tools()),
            cmd_router::CMD_CLEAR => ReplAction::Print("Context cleared.".into()),
            cmd_router::CMD_PROFILE => ReplAction::Print("No timing data yet.".into()),
            cmd_router::CMD_TASKS => ReplAction::Print("No background tasks.".into()),
            cmd_router::CMD_RECALL => ReplAction::Print(self.handle_recall(arg)),
            cmd_router::CMD_CALC => ReplAction::Print(self.handle_calc(arg)),
            cmd_router::CMD_CPU => ReplAction::Print(self.handle_cpu()),
            cmd_router::CMD_TOKENS => ReplAction::Print(self.handle_tokens(arg)),
            cmd_router::CMD_LS => ReplAction::Print(self.handle_ls(arg)),
            cmd_router::CMD_READ => ReplAction::Print(self.handle_read(arg)),
            cmd_router::CMD_WRITE => ReplAction::Print(self.handle_write(arg)),
            cmd_router::CMD_GREP => ReplAction::Print(self.handle_grep(arg)),
            cmd_router::CMD_GIT => ReplAction::Print(self.handle_git(arg)),
            cmd_router::CMD_HTTP => ReplAction::Print(self.handle_http(arg)),
            cmd_router::CMD_JSON => ReplAction::Print(self.handle_json(arg)),
            cmd_router::CMD_BENCH => ReplAction::Print(self.handle_bench(arg)),
            cmd_router::CMD_MEMORY => ReplAction::Print(self.handle_memory(arg)),
            cmd_router::CMD_WEATHER => ReplAction::Print(self.handle_weather(arg)),
            cmd_router::CMD_TRANSLATE => ReplAction::Print(self.handle_translate(arg)),
            cmd_router::CMD_DEFINE => ReplAction::Print(self.handle_define(arg)),
            cmd_router::CMD_SUMMARIZE => ReplAction::Print(self.handle_summarize(arg)),
            cmd_router::CMD_REMIND => ReplAction::Print(self.handle_remind(arg)),
            cmd_router::CMD_NONE if input.starts_with('/') => {
                ReplAction::Print(format!(
                    "Unknown command: {}. Type /help for available commands.",
                    input
                ))
            }
            cmd_router::CMD_NONE => ReplAction::Generate(input.to_string()),
            _ => {
                let name = cmd_router::command_name(cmd_id).unwrap_or("unknown");
                ReplAction::Print(format!("/{} — coming soon", name))
            }
        }
    }

    pub fn generate_streaming(&self, prompt: &str) {
        let stdout = io::stdout();
        match &self.engine {
            Some(eng) => {
                eng.generate_text(prompt, &|tok| {
                    print!("{}", tok);
                    stdout.lock().flush().ok();
                });
            }
            None => print!("No local model loaded."),
        }
    }

    fn help_text(&self) -> String {
        let model_status = if self.engine.is_some() {
            "Cougar BitNet 2B (loaded)"
        } else {
            "no model loaded"
        };
        format!(
            "Commands:\n\
             \x20 /help    /quit    /tools   /clear   /model [local|cloud|auto]   /profile\n\
             \x20 /tasks             — List background tasks\n\
             \x20 /recall <query>    — Search conversation history\n\
             \n\
             Tools:\n\
             \x20 /time  /calc <expr>  /http <url>  /shell <cmd>  /cpu\n\
             \x20 /memory <action> [key] [value]   /read <path>   /write <path> <content>\n\
             \x20 /ls [path]  /json <action> <input> [path]  /tokens <text>  /bench <target>\n\
             \x20 /weather <city>  /translate <lang> <text>  /define <word>  /summarize <url>\n\
             \x20 /grep <pattern> [path]  /git <subcommand> [args]  /remind <time> <message>\n\
             \x20 /teleport <whatsapp|web>\n\
             \n\
             Agent: olorin | Model: {}",
            model_status
        )
    }

    fn handle_model(&self, arg: &str) -> String {
        match arg {
            "local" => "Backend: local (Cougar)".into(),
            "cloud" => "Backend: cloud (Anthropic)".into(),
            "auto" => "Backend: auto".into(),
            "" => {
                let local = if self.engine.is_some() {
                    "Cougar BitNet 2B (loaded)"
                } else {
                    "no model loaded"
                };
                format!("Backend: auto\n  Local: {}", local)
            }
            other => format!("Unknown backend '{}'. Use: local|cloud|auto", other),
        }
    }

    fn handle_teleport(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /teleport <whatsapp|web>".into();
        }
        let olorin_dir = match home::home_dir() {
            Some(h) => h.join(".olorin"),
            None => return "Cannot determine home directory.".into(),
        };
        if let Err(e) = std::fs::create_dir_all(&olorin_dir) {
            return format!("Failed to create ~/.olorin: {e}");
        }
        let vault_id = format!("{}_olorin", arg);
        let token =
            olorin_core::session::SessionToken::new(arg, &vault_id, "cougar-bitnet-2b");
        let session_path = olorin_dir.join("session.json");
        match token.save(&session_path) {
            Ok(()) => format!("Session token saved. Ready to continue on {arg}."),
            Err(e) => format!("Failed to save session token: {e}"),
        }
    }

    fn handle_time(&self) -> String {
        chrono::Local::now()
            .format("%Y-%m-%d %H:%M:%S")
            .to_string()
    }

    fn handle_shell(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /shell <command>".into();
        }
        match std::process::Command::new("sh")
            .arg("-c")
            .arg(arg)
            .output()
        {
            Ok(out) => {
                let mut result = String::new();
                let text = String::from_utf8_lossy(&out.stdout);
                let err = String::from_utf8_lossy(&out.stderr);
                if !text.is_empty() {
                    result.push_str(&text);
                }
                if !err.is_empty() {
                    result.push_str(&err);
                }
                if result.is_empty() {
                    "(no output)".into()
                } else {
                    result.trim_end().to_string()
                }
            }
            Err(e) => format!("Shell error: {}", e),
        }
    }

    fn handle_tools(&self) -> String {
        "Available tools: time, calc, http, shell, memory, read, write, ls, json, \
         cpu, tokens, bench, weather, translate, define, summarize, grep, git, remind"
            .into()
    }

    fn handle_recall(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /recall <query>".into();
        }
        "Recall not yet wired in REPL. Coming in Wire Task 3.".into()
    }

    fn handle_calc(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /calc <expression>".into();
        }
        match std::process::Command::new("python3")
            .arg("-c")
            .arg(format!("print({})", arg))
            .output()
        {
            Ok(out) => {
                let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if text.is_empty() {
                    let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
                    if err.is_empty() {
                        "(no result)".into()
                    } else {
                        err
                    }
                } else {
                    text
                }
            }
            Err(_) => "python3 not available for /calc".into(),
        }
    }

    fn handle_cpu(&self) -> String {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        format!("CPU cores: {}", cores)
    }

    fn handle_tokens(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /tokens <text>".into();
        }
        let words = arg.split_whitespace().count();
        let chars = arg.len();
        format!(
            "~{} tokens (est.) | {} words | {} chars",
            (chars + 3) / 4,
            words,
            chars
        )
    }

    fn handle_ls(&self, arg: &str) -> String {
        let path = if arg.is_empty() { "." } else { arg };
        match std::fs::read_dir(path) {
            Ok(entries) => {
                let mut names: Vec<String> = entries
                    .filter_map(|e| e.ok())
                    .map(|e| {
                        let name = e.file_name().to_string_lossy().into_owned();
                        if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                            format!("{}/", name)
                        } else {
                            name
                        }
                    })
                    .collect();
                names.sort();
                if names.is_empty() {
                    "(empty directory)".into()
                } else {
                    names.join("\n")
                }
            }
            Err(e) => format!("ls error: {}", e),
        }
    }

    fn handle_read(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /read <path>".into();
        }
        match std::fs::read_to_string(arg) {
            Ok(content) => {
                if content.len() > 4096 {
                    format!(
                        "{}...\n[truncated, {} bytes total]",
                        &content[..4096],
                        content.len()
                    )
                } else {
                    content
                }
            }
            Err(e) => format!("Read error: {}", e),
        }
    }

    fn handle_write(&self, arg: &str) -> String {
        let parts: Vec<&str> = arg.splitn(2, char::is_whitespace).collect();
        if parts.len() < 2 {
            return "Usage: /write <path> <content>".into();
        }
        match std::fs::write(parts[0], parts[1]) {
            Ok(()) => format!("Wrote {} bytes to {}", parts[1].len(), parts[0]),
            Err(e) => format!("Write error: {}", e),
        }
    }

    fn handle_grep(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /grep <pattern> [path]".into();
        }
        let parts: Vec<&str> = arg.splitn(2, char::is_whitespace).collect();
        let pattern = parts[0];
        let path = if parts.len() > 1 { parts[1] } else { "." };
        match std::process::Command::new("grep")
            .args(["-rn", "--color=never", pattern, path])
            .output()
        {
            Ok(out) => {
                let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if text.is_empty() {
                    "No matches found.".into()
                } else if text.len() > 4096 {
                    format!("{}...\n[truncated]", &text[..4096])
                } else {
                    text
                }
            }
            Err(e) => format!("Grep error: {}", e),
        }
    }

    fn handle_git(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /git <subcommand> [args]".into();
        }
        match std::process::Command::new("git")
            .args(arg.split_whitespace())
            .output()
        {
            Ok(out) => {
                let mut result = String::new();
                let text = String::from_utf8_lossy(&out.stdout);
                let err = String::from_utf8_lossy(&out.stderr);
                if !text.is_empty() {
                    result.push_str(&text);
                }
                if !err.is_empty() {
                    result.push_str(&err);
                }
                if result.is_empty() {
                    "(no output)".into()
                } else {
                    result.trim_end().to_string()
                }
            }
            Err(e) => format!("Git error: {}", e),
        }
    }

    fn handle_http(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /http <url>".into();
        }
        match std::process::Command::new("curl")
            .args(["-sS", "-L", "--max-time", "10", arg])
            .output()
        {
            Ok(out) => {
                let text = String::from_utf8_lossy(&out.stdout).to_string();
                let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
                if !err.is_empty() && text.is_empty() {
                    return format!("HTTP error: {}", err);
                }
                if text.len() > 4096 {
                    format!(
                        "{}...\n[truncated, {} bytes total]",
                        &text[..4096],
                        text.len()
                    )
                } else {
                    text
                }
            }
            Err(e) => format!("HTTP error: {}", e),
        }
    }

    fn handle_json(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /json <action> <input> [path]".into();
        }
        let parts: Vec<&str> = arg.splitn(2, char::is_whitespace).collect();
        match parts[0] {
            "parse" | "fmt" | "format" => {
                let input = if parts.len() > 1 { parts[1] } else { "" };
                match serde_json::from_str::<serde_json::Value>(input) {
                    Ok(v) => serde_json::to_string_pretty(&v)
                        .unwrap_or_else(|e| format!("JSON error: {e}")),
                    Err(e) => format!("JSON parse error: {e}"),
                }
            }
            "validate" => {
                let input = if parts.len() > 1 { parts[1] } else { "" };
                match serde_json::from_str::<serde_json::Value>(input) {
                    Ok(_) => "Valid JSON.".into(),
                    Err(e) => format!("Invalid JSON: {e}"),
                }
            }
            other => format!(
                "Unknown json action '{}'. Use: parse|format|validate",
                other
            ),
        }
    }

    fn handle_bench(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /bench <target>".into();
        }
        match arg {
            "router" => {
                let input = b"/help";
                let start = std::time::Instant::now();
                for _ in 0..10000 {
                    let _ = cmd_router::match_command_verified(input);
                }
                let elapsed = start.elapsed();
                format!(
                    "SIMD router: 10k iterations in {:?} ({:.0} ns/call)",
                    elapsed,
                    elapsed.as_nanos() as f64 / 10000.0
                )
            }
            other => format!("Unknown bench target '{}'. Available: router", other),
        }
    }

    fn handle_memory(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /memory <get|set|list> [key] [value]".into();
        }
        "Memory store not yet wired in REPL. Coming in Wire Task 3.".into()
    }

    fn handle_weather(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /weather <city>".into();
        }
        let url = format!("https://wttr.in/{}?format=3", arg.replace(' ', "+"));
        match std::process::Command::new("curl")
            .args(["-sS", "--max-time", "5", &url])
            .output()
        {
            Ok(out) => String::from_utf8_lossy(&out.stdout).trim().to_string(),
            Err(e) => format!("Weather error: {}", e),
        }
    }

    fn handle_translate(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /translate <lang> <text>".into();
        }
        "Translation requires cloud LLM. Coming in Wire Task 5.".into()
    }

    fn handle_define(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /define <word>".into();
        }
        "Dictionary lookup requires cloud LLM. Coming in Wire Task 5.".into()
    }

    fn handle_summarize(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /summarize <url>".into();
        }
        "Summarization requires cloud LLM. Coming in Wire Task 5.".into()
    }

    fn handle_remind(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /remind <time> <message>".into();
        }
        "Reminders not yet wired in REPL.".into()
    }
}

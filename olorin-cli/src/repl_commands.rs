use super::repl::OlorinRepl;

impl OlorinRepl {
    pub(crate) fn help_text(&self) -> String {
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

    pub(crate) fn handle_model(&mut self, arg: &str) -> String {
        match arg {
            "local" => {
                if self.engine.is_none() {
                    return "Cannot switch to local: no model loaded.".into();
                }
                self.backend_mode = "local".to_string();
                "Backend: local (Cougar)".into()
            }
            "cloud" => {
                if std::env::var("ANTHROPIC_API_KEY").is_err() {
                    return "Cannot switch to cloud: ANTHROPIC_API_KEY not set.".into();
                }
                self.backend_mode = "cloud".to_string();
                "Backend: cloud (Anthropic)".into()
            }
            "auto" => {
                self.backend_mode = "auto".to_string();
                "Backend: auto".into()
            }
            "" => {
                let local = if self.engine.is_some() {
                    "Cougar BitNet 2B (loaded)"
                } else {
                    "no model loaded"
                };
                format!("Backend: {}\n  Local: {}", self.backend_mode, local)
            }
            other => format!("Unknown backend '{}'. Use: local|cloud|auto", other),
        }
    }

    pub(crate) fn handle_teleport(&mut self, arg: &str) -> String {
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

        let last_hash = if let Some(ref mut vault) = self.vault {
            vault.flush().ok();
            vault
                .last_block_hash()
                .map(|h| format!("{:016x}", h))
                .unwrap_or_default()
        } else {
            String::new()
        };

        let vault_id = format!("{}_olorin", arg);
        let mut token =
            olorin_core::session::SessionToken::new(arg, &vault_id, "cougar-bitnet-2b");
        token.last_msg_hash = last_hash;

        let session_path = olorin_dir.join("session.json");
        match token.save(&session_path) {
            Ok(()) => format!("Teleporting to {arg}. Vault flushed, session saved."),
            Err(e) => format!("Failed to save session token: {e}"),
        }
    }

    pub(crate) fn handle_time(&self) -> String {
        chrono::Local::now()
            .format("%Y-%m-%d %H:%M:%S")
            .to_string()
    }

    pub(crate) fn handle_shell(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /shell <command>".into();
        }
        match olorin_core::exec::shell(arg) {
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

    pub(crate) fn handle_tools(&self) -> String {
        "Available tools: time, calc, http, shell, memory, read, write, ls, json, \
         cpu, tokens, bench, weather, translate, define, summarize, grep, git, remind"
            .into()
    }

    pub(crate) fn handle_recall(&mut self, arg: &str) -> String {
        if arg.is_empty() {
            return format!(
                "Recall level: {} (top_k={}, min_score={:.2})\n\
                 Usage: /recall <0-10> to set level, /recall <query> to search",
                self.recall.level, self.recall.top_k, self.recall.min_score
            );
        }
        // If arg is a number 0-10, set recall level
        if let Ok(level) = arg.parse::<u8>() {
            if level <= 10 {
                self.recall = super::repl::RecallConfig::from_level(level);
                return format!(
                    "Recall level set to {} (top_k={}, min_score={:.2})",
                    self.recall.level, self.recall.top_k, self.recall.min_score
                );
            }
        }
        match self.vault {
            Some(ref mut vault) => {
                let results = match vault.search(arg, 5) {
                    Ok(r) => r,
                    Err(e) => return format!("Recall error: {e}"),
                };
                if results.is_empty() {
                    return "No matching conversation history found.".into();
                }
                let mut out = String::from("Recall results:\n");
                for (i, r) in results.iter().enumerate() {
                    let text = String::from_utf8_lossy(&r.text);
                    let preview: String = text.chars().take(120).collect();
                    out.push_str(&format!(
                        "  [{}] (score {:.2}) {}\n",
                        i + 1,
                        r.score,
                        preview
                    ));
                }
                out.trim_end().to_string()
            }
            None => "Vault not available.".into(),
        }
    }

    pub(crate) fn handle_calc(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /calc <expression>".into();
        }
        let expr = format!("print({})", arg);
        match olorin_core::exec::run(&["python3", "-c", &expr]) {
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

    pub(crate) fn handle_cpu(&self) -> String {
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        format!("CPU cores: {}", cores)
    }

    pub(crate) fn handle_tokens(&self, arg: &str) -> String {
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

    pub(crate) fn handle_ls(&self, arg: &str) -> String {
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

    pub(crate) fn handle_read(&self, arg: &str) -> String {
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

    pub(crate) fn handle_write(&self, arg: &str) -> String {
        let parts: Vec<&str> = arg.splitn(2, char::is_whitespace).collect();
        if parts.len() < 2 {
            return "Usage: /write <path> <content>".into();
        }
        match std::fs::write(parts[0], parts[1]) {
            Ok(()) => format!("Wrote {} bytes to {}", parts[1].len(), parts[0]),
            Err(e) => format!("Write error: {}", e),
        }
    }

    pub(crate) fn handle_grep(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /grep <pattern> [path]".into();
        }
        let parts: Vec<&str> = arg.splitn(2, char::is_whitespace).collect();
        let pattern = parts[0];
        let path = if parts.len() > 1 { parts[1] } else { "." };
        match olorin_core::exec::run(&["grep", "-rn", "--color=never", pattern, path]) {
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

    pub(crate) fn handle_git(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /git <subcommand> [args]".into();
        }
        let mut argv: Vec<&str> = vec!["git"];
        argv.extend(arg.split_whitespace());
        match olorin_core::exec::run(&argv) {
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

    pub(crate) fn handle_http(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /http <url>".into();
        }
        match olorin_core::exec::run(&["curl", "-sS", "-L", "--max-time", "10", arg]) {
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

    pub(crate) fn handle_json(&self, arg: &str) -> String {
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

    pub(crate) fn handle_bench(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /bench <target>\nTargets: safety, router, recall".into();
        }
        use olorin_core::tools::bench_tool;
        match arg.trim() {
            "safety" => bench_tool::bench_safety().unwrap_or_else(|e| format!("error: {e}")),
            "router" => bench_tool::bench_router().unwrap_or_else(|e| format!("error: {e}")),
            "recall" => bench_tool::bench_recall().unwrap_or_else(|e| format!("error: {e}")),
            other => format!("Unknown target '{other}'. Use: safety, router, recall"),
        }
    }

    pub(crate) fn handle_memory(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /memory <get|set|list> [key] [value]".into();
        }
        "Memory store not yet wired in REPL. Coming in Wire Task 3.".into()
    }

    pub(crate) fn handle_weather(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /weather <city>".into();
        }
        let url = format!("https://wttr.in/{}?format=3", arg.replace(' ', "+"));
        match olorin_core::exec::run(&["curl", "-sS", "--max-time", "5", &url]) {
            Ok(out) => String::from_utf8_lossy(&out.stdout).trim().to_string(),
            Err(e) => format!("Weather error: {}", e),
        }
    }

    pub(crate) fn handle_translate(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /translate <lang> <text>".into();
        }
        "Translation requires cloud LLM. Coming in Wire Task 5.".into()
    }

    pub(crate) fn handle_define(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /define <word>".into();
        }
        "Dictionary lookup requires cloud LLM. Coming in Wire Task 5.".into()
    }

    pub(crate) fn handle_summarize(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /summarize <url>".into();
        }
        "Summarization requires cloud LLM. Coming in Wire Task 5.".into()
    }

    pub(crate) fn handle_remind(&self, arg: &str) -> String {
        if arg.is_empty() {
            return "Usage: /remind <time> <message>".into();
        }
        "Reminders not yet wired in REPL.".into()
    }
}

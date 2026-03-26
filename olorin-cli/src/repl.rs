use std::io::{self, Write};
use std::sync::Arc;

use olorin_core::kernels::command_router as cmd_router;
use olorin_core::safety::SafetyLayer;

use crate::CougarEngine;

pub enum ReplAction {
    Quit,
    Print(String),
    Generate(String),
}

pub struct OlorinRepl {
    pub engine: Option<Arc<CougarEngine>>,
    safety: SafetyLayer,
}

impl OlorinRepl {
    pub fn new(engine: Option<Arc<CougarEngine>>) -> Self {
        let safety = SafetyLayer::with_capacity(8192);
        Self { engine, safety }
    }

    pub fn process(&mut self, input: &str) -> ReplAction {
        // Safety scan on all input before routing
        let scan = self.safety.scan_input(input);
        if scan.is_blocked() {
            return ReplAction::Print(format!(
                "Blocked: {}",
                scan.block_reason().unwrap_or("safety")
            ));
        }

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

    pub fn generate_streaming(&mut self, prompt: &str) {
        let stdout = io::stdout();
        match &self.engine {
            Some(eng) => {
                let response = eng.generate_text(prompt, &|tok| {
                    print!("{}", tok);
                    stdout.lock().flush().ok();
                });

                // Safety scan on LLM output (warn only — already streamed)
                let scan = self.safety.scan_output(&response);
                if scan.is_blocked() {
                    eprintln!(
                        "\n[Safety] Warning: output {}",
                        scan.block_reason().unwrap_or("flagged")
                    );
                }
            }
            None => print!("No local model loaded."),
        }
    }
}

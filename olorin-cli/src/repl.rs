use std::io::{self, Write};
use std::sync::Arc;

use olorin_core::kernels::command_router as cmd_router;
use olorin_core::safety::SafetyLayer;
use olorin_core::vault::{Vault, XorCrypto};

use crate::CougarEngine;

pub enum ReplAction {
    Quit,
    Print(String),
    Generate(String),
}

/// Recall level controls how much vault context is injected into prompts.
/// 0 = no recall (pure inference), 10 = deep recall (full history search).
/// Auto-detected at startup based on model capability.
pub struct RecallConfig {
    pub level: u8,        // 0-10
    pub top_k: usize,     // how many vault blocks to retrieve
    pub min_score: f32,   // minimum similarity threshold
}

impl RecallConfig {
    /// Auto-configure based on model type.
    /// Small/base models get minimal recall; large instruct models get full.
    pub fn auto_detect(quant_type: &str) -> Self {
        match quant_type {
            "I2S" => Self { level: 0, top_k: 0, min_score: 1.0 },    // BitNet: no recall
            "Q4K" => Self { level: 3, top_k: 2, min_score: 0.3 },    // Llama 3B: light recall
            _     => Self { level: 5, top_k: 3, min_score: 0.2 },    // default
        }
    }

    pub fn from_level(level: u8) -> Self {
        let level = level.min(10);
        if level == 0 {
            return Self { level: 0, top_k: 0, min_score: 1.0 };
        }
        Self {
            level,
            top_k: (level as usize + 1) / 2,           // 1→1, 3→2, 5→3, 7→4, 10→5
            min_score: 0.5 - (level as f32 * 0.04),     // 1→0.46, 5→0.30, 10→0.10
        }
    }
}

pub struct OlorinRepl {
    pub engine: Option<Arc<CougarEngine>>,
    safety: SafetyLayer,
    pub(crate) vault: Option<Vault>,
    pub(crate) backend_mode: String,
    pub(crate) recall: RecallConfig,
}

impl OlorinRepl {
    pub fn new(engine: Option<Arc<CougarEngine>>, model_quant: &str) -> Self {
        let safety = SafetyLayer::with_capacity(8192);
        let vault = Self::open_vault();
        let recall = RecallConfig::auto_detect(model_quant);
        eprintln!("[Olorin] Recall level: {} (top_k={}, min_score={:.2})",
            recall.level, recall.top_k, recall.min_score);
        Self {
            engine,
            safety,
            vault,
            backend_mode: "auto".to_string(),
            recall,
        }
    }

    fn open_vault() -> Option<Vault> {
        let home = home::home_dir()?;
        let vault_dir = home.join(".olorin/vault");
        std::fs::create_dir_all(&vault_dir).ok()?;
        let vault_path = vault_dir.join("default.vault");

        // Deterministic key derived from a fixed seed.
        // Real key management comes in a later wire task.
        let seed = b"olorin-vault-seed-v0.5-default!!";
        let mut key = [0u8; 32];
        key.copy_from_slice(seed);

        let crypto = Box::new(XorCrypto);
        if vault_path.exists() {
            match Vault::open(&vault_path, &key, crypto) {
                Ok(v) => {
                    eprintln!(
                        "[Olorin] Vault opened ({} blocks)",
                        v.block_count()
                    );
                    Some(v)
                }
                Err(e) => {
                    eprintln!("[Olorin] Vault open failed: {e} — creating new");
                    Vault::create(&vault_path, &key, Box::new(XorCrypto)).ok()
                }
            }
        } else {
            match Vault::create(&vault_path, &key, crypto) {
                Ok(v) => {
                    eprintln!("[Olorin] Vault created at {}", vault_path.display());
                    Some(v)
                }
                Err(e) => {
                    eprintln!("[Olorin] Vault creation failed: {e}");
                    None
                }
            }
        }
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

    /// Build recall context based on current recall level.
    fn recall_context(&mut self, prompt: &str) -> String {
        if self.recall.level == 0 || self.recall.top_k == 0 {
            return String::new();
        }
        let vault = match self.vault {
            Some(ref mut v) => v,
            None => return String::new(),
        };
        let results = vault.search(prompt, self.recall.top_k).unwrap_or_default();
        let filtered: Vec<_> = results
            .iter()
            .filter(|r| r.score >= self.recall.min_score)
            .collect();
        if filtered.is_empty() {
            return String::new();
        }
        let ctx: Vec<String> = filtered
            .iter()
            .map(|r| String::from_utf8_lossy(&r.text).to_string())
            .collect();
        format!("\n[Recall context]\n{}\n", ctx.join("\n---\n"))
    }

    pub fn generate_streaming(&mut self, prompt: &str) {
        let stdout = io::stdout();
        let context = self.recall_context(prompt);

        let full_prompt = if context.is_empty() {
            prompt.to_string()
        } else {
            format!("{}{}", context, prompt)
        };

        let response = if self.backend_mode == "cloud" {
            let msg = "Cloud backend requires ANTHROPIC_API_KEY. Set it and restart.";
            print!("{}", msg);
            msg.to_string()
        } else {
            match &self.engine {
                Some(eng) => {
                    let resp = eng.generate_text(&full_prompt, &|tok| {
                        print!("{}", tok);
                        stdout.lock().flush().ok();
                    });

                    let scan = self.safety.scan_output(&resp);
                    if scan.is_blocked() {
                        eprintln!(
                            "\n[Safety] Warning: output {}",
                            scan.block_reason().unwrap_or("flagged")
                        );
                    }

                    resp
                }
                None => {
                    let msg = "No local model loaded.";
                    print!("{}", msg);
                    msg.to_string()
                }
            }
        };

        // Save conversation turn to vault
        if let Some(ref mut vault) = self.vault {
            vault
                .append_message(&format!("User: {}\n", prompt))
                .ok();
            vault
                .append_message(&format!("Olorin: {}\n", response))
                .ok();
            vault.flush().ok();
        }
    }

    /// Like `generate_streaming` but sends tokens via callback and returns the full response.
    /// Used by the web channel so every request gets safety + recall + vault persistence.
    pub fn generate_for_web(&mut self, prompt: &str, on_token: &dyn Fn(&str)) -> String {
        let context = self.recall_context(prompt);

        let full_prompt = if context.is_empty() {
            prompt.to_string()
        } else {
            format!("{}{}", context, prompt)
        };

        let response = if self.backend_mode == "cloud" {
            let msg = "Cloud backend requires ANTHROPIC_API_KEY. Set it and restart.";
            on_token(msg);
            msg.to_string()
        } else {
            match &self.engine {
                Some(eng) => {
                    let resp = eng.generate_text(&full_prompt, on_token);

                    let scan = self.safety.scan_output(&resp);
                    if scan.is_blocked() {
                        let warn = format!(
                            "\n[Safety] Warning: output {}",
                            scan.block_reason().unwrap_or("flagged")
                        );
                        on_token(&warn);
                    }

                    resp
                }
                None => {
                    let msg = "No local model loaded.";
                    on_token(msg);
                    msg.to_string()
                }
            }
        };

        if let Some(ref mut vault) = self.vault {
            vault
                .append_message(&format!("User: {}\n", prompt))
                .ok();
            vault
                .append_message(&format!("Olorin: {}\n", response))
                .ok();
            vault.flush().ok();
        }

        response
    }
}

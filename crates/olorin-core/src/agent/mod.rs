pub mod background;
mod handlers;
pub mod router;
pub mod tool_dispatch;

use crate::channel::Channel;
use crate::config::Config;
use crate::error::Result;
use crate::kernels::arg_tokenizer::ArgTokenizer;
use crate::kernels::command_router as cmd_router;
use crate::recall::VectorStore;
use crate::llm::{LlmProvider, Message, ToolDef};
use crate::safety::SafetyLayer;
use crate::tools::ToolRegistry;
use std::sync::Arc;

const BASE_SYSTEM_PROMPT: &str = "\
You are Olorin, a high-performance AI assistant. \
You have access to tools that you can use to help the user. \
Be concise and helpful. Use tools when they would help answer the user's question.";

/// Timing data for a single agent turn.
pub struct TurnTiming {
    pub safety_scan_us: u64,
    pub llm_call_ms: u64,
    pub tool_execs: Vec<(String, u64)>,
}

impl TurnTiming {
    fn total_ms(&self) -> u64 {
        let tool_ms: u64 = self.tool_execs.iter().map(|(_, ms)| ms).sum();
        let safety_ms = (self.safety_scan_us + 999) / 1000;
        safety_ms + self.llm_call_ms + tool_ms
    }

    fn format(&self) -> String {
        let mut lines = vec![
            "Last turn timing:".to_string(),
            format!("  Safety scan:    {} µs", self.safety_scan_us),
            format!("  LLM call:       {} ms", self.llm_call_ms),
        ];
        for (name, ms) in &self.tool_execs {
            lines.push(format!("  Tool: {:<10}{} ms", name, ms));
        }
        lines.push(format!("  Total:          {} ms", self.total_ms()));
        lines.join("\n")
    }
}

pub struct Agent {
    config: Config,
    llm: Arc<dyn LlmProvider>,
    pub(crate) tools: ToolRegistry,
    pub(crate) safety: SafetyLayer,
    messages: Vec<Message>,
    last_timing: Option<TurnTiming>,
    bg_tasks: background::TaskTable,
    pub(crate) tokenizer: ArgTokenizer,
    system_prompt: String,
    recall_store: VectorStore,
}

impl Agent {
    pub fn new(
        config: Config,
        llm: Arc<dyn LlmProvider>,
        tools: ToolRegistry,
        safety: SafetyLayer,
    ) -> Self {
        let system_prompt = match &config.identity {
            Some(identity) => format!("{BASE_SYSTEM_PROMPT}\n\n{identity}"),
            None => BASE_SYSTEM_PROMPT.to_string(),
        };
        Self {
            config,
            llm,
            tools,
            safety,
            messages: Vec::new(),
            last_timing: None,
            bg_tasks: background::TaskTable::new(),
            tokenizer: ArgTokenizer::with_capacity(256),
            system_prompt,
            recall_store: VectorStore::with_capacity(1024),
        }
    }

    /// Run the agent loop on the given channel.
    pub async fn run(&mut self, channel: &dyn Channel) -> Result<()> {
        channel
            .send(&format!(
                "Welcome to {}! Type /help for commands, /quit to exit.",
                self.config.agent_name
            ))
            .await;

        let tool_defs: Vec<ToolDef> = self.tools.tool_defs();

        loop {
            let msg = match channel.recv().await {
                Some(m) => m,
                None => break,
            };

            // Notify about background tasks that completed since last prompt
            for task in self.bg_tasks.take_new_completions() {
                let note = match &task.status {
                    crate::agent::background::TaskStatus::Done(output) => {
                        let preview = if output.len() > 200 {
                            format!("{}...", &output[..200])
                        } else {
                            output.clone()
                        };
                        format!("[{}] {} done: {preview}", task.id, task.name)
                    }
                    crate::agent::background::TaskStatus::Failed(err) => {
                        format!("[{}] {} failed: {err}", task.id, task.name)
                    }
                    _ => continue,
                };
                channel.send(&note).await;
            }

            // Pipeline detection: split on " | /" before routing
            if msg.starts_with(&self.config.command_prefix) && msg.contains(" | /") {
                match self.execute_pipeline(&msg, channel).await {
                    Ok(()) => {}
                    Err(e) => channel.send(&format!("Pipeline error: {e}")).await,
                }
                continue;
            }

            // Two-stage SIMD command routing (hash + verify)
            let (cmd_id, cmd_arg) = cmd_router::match_command_verified(msg.as_bytes());

            // Handle /tasks meta command
            if cmd_id == cmd_router::CMD_TASKS {
                let list = self.bg_tasks.format_list();
                let scan = self.safety.scan_output(&list);
                if let Some(reason) = scan.block_reason() {
                    channel.send(&format!("Task output blocked: {reason}. Check tasks individually.")).await;
                } else {
                    channel.send(&list).await;
                }
                continue;
            }

            // Handle /recall <query>
            if cmd_id == cmd_router::CMD_RECALL {
                let query = String::from_utf8_lossy(cmd_arg);
                channel.send(&self.recall_store.recall_formatted(&query, 5)).await;
                continue;
            }

            // Handle /model <backend> with argument support
            if cmd_id == cmd_router::CMD_MODEL {
                let arg = String::from_utf8_lossy(cmd_arg);
                channel.send(&self.handle_model_command(&arg)).await;
                continue;
            }

            // Handle meta commands
            if cmd_id >= cmd_router::CMD_HELP && cmd_id <= cmd_router::CMD_PROFILE {
                if self.handle_meta(cmd_id, channel).await? {
                    continue;
                } else {
                    break; // /quit
                }
            }

            // Handle direct tool commands — bypass the LLM
            if cmd_id >= cmd_router::CMD_TOOL_FIRST && cmd_id <= cmd_router::CMD_TOOL_LAST {
                self.handle_tool_command(cmd_id, cmd_arg, channel).await;
                continue;
            }

            // Handle /teleport <target>
            if cmd_id == cmd_router::CMD_TELEPORT {
                let target = String::from_utf8_lossy(cmd_arg);
                let target = target.trim().to_string();
                if target.is_empty() {
                    channel.send("[Olorin] Usage: /teleport <whatsapp|web>").await;
                } else {
                    self.handle_teleport(&target, channel).await;
                }
                continue;
            }

            // Check for unknown slash commands
            if msg.starts_with(&self.config.command_prefix) && cmd_id == cmd_router::CMD_NONE {
                channel
                    .send(&format!(
                        "Unknown command: {msg}. Type /help for available commands."
                    ))
                    .await;
                continue;
            }

            // LLM conversation turn
            self.handle_llm_turn(&msg, &tool_defs, channel).await?;
        }

        Ok(())
    }

    pub(crate) fn help_text(&self) -> String {
        format!("\
Commands:
  /help    /quit    /tools   /clear   /model [local|cloud|auto]   /profile
  /tasks             — List background tasks
  /recall <query>    — Search conversation history

Tools:
  /time  /calc <expr>  /http <url>  /shell <cmd>  /cpu
  /memory <action> [key] [value]   /read <path>   /write <path> <content>
  /ls [path]  /json <action> <input> [path]  /tokens <text>  /bench <target>
  /weather <city>  /translate <lang> <text>  /define <word>  /summarize <url>
  /grep <pattern> [path]  /git <subcommand> [args]  /remind <time> <message>
  /teleport <whatsapp|web>

Background: append & (e.g. /shell sleep 5 &)
Pipelines: /shell ls | /tokens

Agent: {} | Model: {}", self.config.agent_name, self.config.model)
    }
}

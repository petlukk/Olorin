pub mod background;
pub mod router;
pub mod tool_dispatch;

use crate::channel::Channel;
use crate::config::Config;
use crate::error::Result;
use crate::kernels::arg_tokenizer::ArgTokenizer;
use crate::kernels::command_router as cmd_router;
use crate::recall::VectorStore;
use crate::llm::{
    ContentBlock, LlmProvider, Message, Role, StopReason, ToolDef,
};
use crate::safety::SafetyLayer;
use crate::tools::ToolRegistry;
use std::sync::Arc;
use std::time::Instant;

const BASE_SYSTEM_PROMPT: &str = "\
You are eaclaw, a high-performance AI assistant. \
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

    /// Handle a meta command. Returns true to continue the loop, false to quit.
    async fn handle_meta(&mut self, cmd_id: i32, channel: &dyn Channel) -> Result<bool> {
        match cmd_id {
            cmd_router::CMD_QUIT => { channel.send("Goodbye!").await; Ok(false) }
            cmd_router::CMD_HELP => { channel.send(&self.help_text()).await; Ok(true) }
            cmd_router::CMD_TOOLS => {
                channel.send(&format!("Available tools: {}", self.tools.list_names().join(", "))).await;
                Ok(true)
            }
            cmd_router::CMD_CLEAR => {
                self.messages.clear();
                self.recall_store.clear();
                channel.send("Context cleared.").await;
                Ok(true)
            }
            cmd_router::CMD_MODEL => {
                channel.send(&format!("Model: {}", self.config.model)).await;
                Ok(true)
            }
            cmd_router::CMD_PROFILE => {
                let msg = match &self.last_timing {
                    Some(t) => t.format(),
                    None => "No timing data yet.".to_string(),
                };
                channel.send(&msg).await;
                Ok(true)
            }
            _ => Ok(true),
        }
    }

    /// Handle a direct tool command (foreground or background).
    async fn handle_tool_command(
        &mut self,
        cmd_id: i32,
        cmd_arg: &[u8],
        channel: &dyn Channel,
    ) {
        let arg_str = String::from_utf8_lossy(cmd_arg).into_owned();
        let tool_name = cmd_router::command_name(cmd_id).unwrap_or("unknown");

        // Detect background execution: trailing " &"
        let (arg_str, is_background) = if arg_str.ends_with(" &") {
            (arg_str[..arg_str.len() - 2].to_string(), true)
        } else if arg_str == "&" {
            (String::new(), true)
        } else {
            (arg_str, false)
        };

        if is_background {
            self.spawn_background(cmd_id, tool_name, &arg_str, channel)
                .await;
            return;
        }

        // Foreground execution
        let tool_start = Instant::now();

        let tool_streams = self
            .tools
            .get(tool_name)
            .map_or(false, |t| t.supports_streaming());

        if tool_streams {
            if let Err(e) = self
                .execute_direct_tool_stream(cmd_id, &arg_str, channel)
                .await
            {
                channel.send(&format!("Tool error: {e}")).await;
            }
        } else {
            match self.execute_direct_tool(cmd_id, &arg_str).await {
                Ok(output) => {
                    let scan = self.safety.scan_output(&output);
                    if let Some(reason) = scan.block_reason() {
                        channel
                            .send(&format!("Tool output blocked: {reason}."))
                            .await;
                    } else {
                        channel.send(&output).await;
                    }
                }
                Err(e) => {
                    channel.send(&format!("Tool error: {e}")).await;
                }
            }
        }

        let tool_ms = tool_start.elapsed().as_millis() as u64;
        self.last_timing = Some(TurnTiming {
            safety_scan_us: 0,
            llm_call_ms: 0,
            tool_execs: vec![(tool_name.to_string(), tool_ms)],
        });
    }

    /// Spawn a tool as a background task.
    async fn spawn_background(
        &mut self,
        cmd_id: i32,
        tool_name: &str,
        arg_str: &str,
        channel: &dyn Channel,
    ) {
        match self.build_tool_params(cmd_id, arg_str) {
            Ok((name, params)) => {
                let tool = self.tools.get(name).cloned();
                if let Some(tool) = tool {
                    let task_id = self
                        .bg_tasks
                        .register(tool_name, &format!("/{tool_name} {arg_str}"));
                    let bg_tasks = self.bg_tasks.clone();
                    tokio::spawn(async move {
                        match tool.execute(params).await {
                            Ok(output) => bg_tasks.complete(task_id, output),
                            Err(e) => bg_tasks.fail(task_id, e.to_string()),
                        }
                    });
                    channel
                        .send(&format!(
                            "[{task_id}] Started in background: /{tool_name} {arg_str}"
                        ))
                        .await;
                } else {
                    channel
                        .send(&format!("Tool error: {tool_name} not registered"))
                        .await;
                }
            }
            Err(e) => {
                channel.send(&format!("Tool error: {e}")).await;
            }
        }
    }

    /// Run a full LLM conversation turn with safety scanning and tool loop.
    async fn handle_llm_turn(
        &mut self,
        msg: &str,
        tool_defs: &[ToolDef],
        channel: &dyn Channel,
    ) -> Result<()> {
        // Safety scan on input (timed, reuses SIMD buffers)
        let safety_start = Instant::now();
        let scan = self.safety.scan_input(msg);
        let safety_scan_us = safety_start.elapsed().as_micros() as u64;

        if scan.injection_found {
            let details: Vec<String> = scan
                .details
                .iter()
                .map(|w| format!("  - {} at position {}", w.pattern, w.position))
                .collect();
            channel
                .send(&format!(
                    "Warning: potential injection detected:\n{}",
                    details.join("\n")
                ))
                .await;
            return Ok(());
        }
        if scan.leaks_found {
            channel
                .send("Warning: your message appears to contain secrets. Message not sent.")
                .await;
            return Ok(());
        }

        // Add user message and index for recall
        self.messages.push(Message {
            role: Role::User,
            content: vec![ContentBlock::text(msg)],
        });
        self.recall_store.insert(msg);

        // Agentic tool loop
        let mut turns = 0;
        let mut total_llm_ms: u64 = 0;
        let mut tool_execs: Vec<(String, u64)> = Vec::new();

        loop {
            if turns >= self.config.max_turns {
                channel.send("Max tool turns reached. Stopping.").await;
                break;
            }
            turns += 1;

            // Stream LLM response (timed)
            let llm_start = Instant::now();
            let mut streamed_any_text = false;

            let response = {
                let streamed_flag = &mut streamed_any_text;
                let prefix = channel.response_prefix();
                let mut on_text = |chunk: &str| {
                    if !chunk.is_empty() {
                        if !*streamed_flag {
                            print!("\r\x1b[2K{prefix} ");
                        }
                        *streamed_flag = true;
                        print!("{chunk}");
                        let _ = std::io::Write::flush(&mut std::io::stdout());
                    }
                };

                match self
                    .llm
                    .chat_stream(&self.messages, tool_defs, &self.system_prompt, &mut on_text)
                    .await
                {
                    Ok(r) => r,
                    Err(e) => {
                        if streamed_any_text {
                            channel.flush().await;
                        }
                        channel.send(&format!("LLM error: {e}")).await;
                        break;
                    }
                }
            };
            total_llm_ms += llm_start.elapsed().as_millis() as u64;

            // Collect tool uses and text
            let mut tool_uses = Vec::new();
            let mut text_parts = Vec::new();
            let mut assistant_blocks = Vec::new();

            for block in &response.content {
                match block {
                    ContentBlock::Text { text } => {
                        text_parts.push(text.as_str());
                        assistant_blocks.push(block.clone());
                    }
                    ContentBlock::ToolUse { id, name, input } => {
                        tool_uses.push((id.clone(), name.clone(), input.clone()));
                        assistant_blocks.push(block.clone());
                    }
                    _ => {
                        assistant_blocks.push(block.clone());
                    }
                }
            }

            // Push assistant message
            self.messages.push(Message {
                role: Role::Assistant,
                content: assistant_blocks,
            });

            if tool_uses.is_empty() || response.stop_reason != StopReason::ToolUse {
                // Safety scan LLM response text before displaying
                if !text_parts.is_empty() {
                    let full_text = text_parts.join("");
                    let scan = self.safety.scan_output(&full_text);
                    if let Some(reason) = scan.block_reason() {
                        channel
                            .send(&format!("LLM response blocked: {reason}."))
                            .await;
                        break;
                    }
                    // Index assistant response for recall
                    if !full_text.trim().is_empty() {
                        self.recall_store.insert(&full_text);
                    }
                }
                if streamed_any_text {
                    channel.flush().await;
                } else if !text_parts.is_empty() {
                    channel.send(&text_parts.join("")).await;
                }
                break;
            }

            // Flush any leading streamed text before tool execution
            if streamed_any_text {
                channel.flush().await;
            }

            // Execute tools (timed)
            let mut result_blocks = Vec::new();
            for (id, name, input) in &tool_uses {
                let tool_start = Instant::now();
                let result = match self.tools.get(name) {
                    Some(tool) => match tool.execute(input.clone()).await {
                        Ok(output) => {
                            let scan = self.safety.scan_output(&output);
                            if let Some(reason) = scan.block_reason() {
                                ContentBlock::tool_error(
                                    id,
                                    format!("Tool output blocked: {reason}"),
                                )
                            } else {
                                ContentBlock::tool_result(id, &output)
                            }
                        }
                        Err(e) => ContentBlock::tool_error(id, e.to_string()),
                    },
                    None => ContentBlock::tool_error(id, format!("Unknown tool: {name}")),
                };
                let tool_ms = tool_start.elapsed().as_millis() as u64;
                tool_execs.push((name.clone(), tool_ms));
                result_blocks.push(result);
            }

            self.messages.push(Message {
                role: Role::User,
                content: result_blocks,
            });
        }

        // Store timing for /profile
        self.last_timing = Some(TurnTiming {
            safety_scan_us,
            llm_call_ms: total_llm_ms,
            tool_execs,
        });

        Ok(())
    }

    pub(crate) fn help_text(&self) -> String {
        format!("\
Commands:
  /help    /quit    /tools   /clear   /model   /profile
  /tasks             — List background tasks
  /recall <query>    — Search conversation history

Tools:
  /time  /calc <expr>  /http <url>  /shell <cmd>  /cpu
  /memory <action> [key] [value]   /read <path>   /write <path> <content>
  /ls [path]  /json <action> <input> [path]  /tokens <text>  /bench <target>
  /weather <city>  /translate <lang> <text>  /define <word>  /summarize <url>
  /grep <pattern> [path]  /git <subcommand> [args]  /remind <time> <message>

Background: append & (e.g. /shell sleep 5 &)
Pipelines: /shell ls | /tokens

Agent: {} | Model: {}", self.config.agent_name, self.config.model)
    }
}

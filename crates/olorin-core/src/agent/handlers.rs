use crate::channel::Channel;
use crate::error::Result;
use crate::kernels::command_router as cmd_router;
use crate::llm::{ContentBlock, Message, Role, StopReason, ToolDef};
use std::time::Instant;

use super::{Agent, TurnTiming};

impl Agent {
    /// Handle a meta command. Returns true to continue the loop, false to quit.
    pub(crate) async fn handle_meta(
        &mut self,
        cmd_id: i32,
        channel: &dyn Channel,
    ) -> Result<bool> {
        match cmd_id {
            cmd_router::CMD_QUIT => {
                channel.send("Goodbye!").await;
                Ok(false)
            }
            cmd_router::CMD_HELP => {
                channel.send(&self.help_text()).await;
                Ok(true)
            }
            cmd_router::CMD_TOOLS => {
                channel
                    .send(&format!(
                        "Available tools: {}",
                        self.tools.list_names().join(", ")
                    ))
                    .await;
                Ok(true)
            }
            cmd_router::CMD_CLEAR => {
                self.messages.clear();
                self.recall_store.clear();
                channel.send("Context cleared.").await;
                Ok(true)
            }
            cmd_router::CMD_MODEL => {
                channel
                    .send(&format!("Model: {}", self.config.model))
                    .await;
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
    pub(crate) async fn handle_tool_command(
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
    pub(crate) async fn spawn_background(
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
    pub(crate) async fn handle_llm_turn(
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
                    .chat_stream(
                        &self.messages,
                        tool_defs,
                        &self.system_prompt,
                        &mut on_text,
                    )
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
}

//! WhatsApp agent loop: gateway + LLM integration.
//!
//! Receives messages from WhatsAppChannel, routes through Gateway,
//! calls LLM for forwarded messages, and sends responses back.

use crate::agent::tool_dispatch::build_tool_params;
use crate::channel::gateway::{Action, Gateway};
use crate::channel::types::GroupChannel;
use crate::channel::whatsapp::WhatsAppChannel;
use crate::config::Config;
use crate::error::Result;
use crate::kernels::arg_tokenizer::ArgTokenizer;
use crate::kernels::command_router as cmd_router;
use crate::llm::{ContentBlock, LlmProvider, Message, Role, StopReason, ToolDef};
use crate::safety::SafetyLayer;
use crate::tools::ToolRegistry;
use std::sync::Arc;

const WA_SYSTEM_PROMPT: &str = "\
You are eaclaw, an AI assistant in a WhatsApp group. \
Be concise — WhatsApp messages should be short and readable. \
You have access to tools. Use them when they help answer the question. \
Reply in the same language the user writes in.";

/// Print a status line to stderr (visible in the terminal).
fn status(msg: &str) {
    eprintln!("[eaclaw] {msg}");
}

/// Run the WhatsApp agent loop.
pub async fn run(
    bridge_path: &str,
    session_dir: &str,
    config: &Config,
    llm: Arc<dyn LlmProvider>,
    tools: &ToolRegistry,
) -> Result<()> {
    let channel = WhatsAppChannel::start(bridge_path, session_dir).await?;
    let mut gateway = Gateway::new(config);
    let mut safety = SafetyLayer::new();
    let mut tokenizer = ArgTokenizer::with_capacity(256);
    let tool_defs: Vec<ToolDef> = tools.tool_defs();

    let system_prompt = match &config.identity {
        Some(identity) => format!("{WA_SYSTEM_PROMPT}\n\n{identity}"),
        None => WA_SYSTEM_PROMPT.to_string(),
    };

    status("Waiting for WhatsApp connection...");

    // Wait for connection
    while !channel.is_connected() {
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    }

    status("Connected to WhatsApp!");
    status(&format!(
        "Listening for messages mentioning @{} or !{}",
        config.agent_name, config.agent_name
    ));
    status("Press Ctrl+C to stop.");
    eprintln!();

    loop {
        let msg = match channel.recv().await {
            Some(m) => m,
            None => {
                status("WhatsApp channel closed.");
                break;
            }
        };

        let processed = match gateway.process_inbound(&msg) {
            Some(p) => p,
            None => continue,
        };

        match processed.action {
            Action::Blocked(reason) => {
                eprintln!("  ⚠ Blocked: {reason}");
                channel
                    .send(&processed.jid, &format!("Message blocked: {reason}"))
                    .await;
            }
            Action::Forward {
                text,
                sender_name,
                context: raw_context,
            } => {
                // Strip trigger prefix to get the actual command/message
                let stripped = strip_trigger(&text, &config.agent_name);

                // Try direct slash command (no LLM needed)
                if stripped.starts_with('/') {
                    let (cmd_id, cmd_arg) =
                        cmd_router::match_command_verified(stripped.as_bytes());
                    if cmd_id >= cmd_router::CMD_TOOL_FIRST
                        && cmd_id <= cmd_router::CMD_TOOL_LAST
                    {
                        let arg_str =
                            String::from_utf8_lossy(cmd_arg).into_owned();
                        let tool_name =
                            cmd_router::command_name(cmd_id).unwrap_or("unknown");
                        eprintln!(
                            "  ⚡ Triggered by {sender_name} — /{tool_name} (direct)"
                        );
                        let response_text =
                            match build_tool_params(cmd_id, &arg_str, &mut tokenizer)
                            {
                                Ok((name, params)) => match tools.get(name) {
                                    Some(tool) => match tool.execute(params).await {
                                        Ok(output) => {
                                            let scan = safety.scan_output(&output);
                                            if let Some(reason) = scan.block_reason() {
                                                format!("Output blocked: {reason}.")
                                            } else {
                                                output
                                            }
                                        }
                                        Err(e) => format!("Tool error: {e}"),
                                    },
                                    None => format!("Unknown tool: {name}"),
                                },
                                Err(e) => format!("{e}"),
                            };
                        channel.send(&processed.jid, &response_text).await;
                        gateway.record_response(&processed.jid, &response_text);
                        eprintln!(
                            "  → [{jid}] eaclaw: {text}",
                            jid = short_jid(&processed.jid),
                            text = truncate(&response_text, 120),
                        );
                        continue;
                    }
                }

                eprintln!(
                    "  ⚡ Triggered by {sender_name} — calling LLM..."
                );

                // Truncate recall context to fit within ctx budget.
                // Reserve tokens for: system prompt (~200), user message (~100),
                // response generation (512). Rough estimate: 1 token ≈ 4 chars.
                let mut context = raw_context;
                let ctx_budget = config.ctx_size.saturating_sub(812);
                let char_budget = ctx_budget * 4;
                let mut total_chars: usize = 0;
                let mut keep_from = 0;
                for (i, line) in context.iter().enumerate().rev() {
                    total_chars += line.len();
                    if total_chars > char_budget {
                        keep_from = i + 1;
                        break;
                    }
                }
                if keep_from > 0 {
                    context.drain(..keep_from);
                }

                // Build conversation with context
                let mut messages = Vec::new();

                // Add recall context as a system-like user message
                if !context.is_empty() {
                    let context_text = context.join("\n");
                    messages.push(Message {
                        role: Role::User,
                        content: vec![ContentBlock::text(format!(
                            "[Recent conversation context]\n{context_text}"
                        ))],
                    });
                    messages.push(Message {
                        role: Role::Assistant,
                        content: vec![ContentBlock::text(
                            "I have the conversation context. What can I help with?",
                        )],
                    });
                }

                // Add the actual user message
                messages.push(Message {
                    role: Role::User,
                    content: vec![ContentBlock::text(format!("{sender_name}: {text}"))],
                });

                // LLM call with tool loop
                let response_text = match run_llm_turn(
                    &llm,
                    &mut messages,
                    &tool_defs,
                    &system_prompt,
                    tools,
                    &mut safety,
                    config.max_turns,
                )
                .await
                {
                    Ok(text) => text,
                    Err(e) => {
                        eprintln!("  ✗ LLM error: {e}");
                        format!("Sorry, I encountered an error: {e}")
                    }
                };

                // Send response and record it
                channel.send(&processed.jid, &response_text).await;
                gateway.record_response(&processed.jid, &response_text);

                eprintln!(
                    "  → [{jid}] eaclaw: {text}",
                    jid = short_jid(&processed.jid),
                    text = truncate(&response_text, 120),
                );
            }
        }
    }

    Ok(())
}

/// Strip trigger prefix (@eaclaw, !eaclaw, or "eaclaw ") from message text.
fn strip_trigger<'a>(text: &'a str, trigger: &str) -> &'a str {
    let lower = text.to_lowercase();
    let trig = trigger.to_lowercase();

    // Try @trigger or !trigger
    for prefix in [format!("@{trig}"), format!("!{trig}")] {
        if let Some(pos) = lower.find(&prefix) {
            let after = pos + prefix.len();
            return text[after..].trim_start();
        }
    }
    // Try "trigger " at start
    if lower.starts_with(&trig) {
        return text[trig.len()..].trim_start();
    }
    text
}

/// Shorten a JID for display: "1234567890@s.whatsapp.net" → "1234567890"
fn short_jid(jid: &str) -> &str {
    jid.split('@').next().unwrap_or(jid)
}

/// Truncate text for terminal display (char-boundary safe).
fn truncate(s: &str, max: usize) -> String {
    let s = s.replace('\n', " ");
    if s.len() <= max {
        s
    } else {
        let mut end = max;
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &s[..end])
    }
}

/// Run an LLM turn with tool loop, returning the final text response.
async fn run_llm_turn(
    llm: &Arc<dyn LlmProvider>,
    messages: &mut Vec<Message>,
    tool_defs: &[ToolDef],
    system: &str,
    tools: &ToolRegistry,
    safety: &mut SafetyLayer,
    max_turns: usize,
) -> Result<String> {
    let mut turns = 0;

    loop {
        if turns >= max_turns {
            return Ok("I've reached my tool use limit for this message.".into());
        }
        turns += 1;

        let response = llm.chat(messages, tool_defs, system).await?;

        let mut tool_uses = Vec::new();
        let mut text_parts = Vec::new();
        let mut assistant_blocks = Vec::new();

        for block in &response.content {
            match block {
                ContentBlock::Text { text } => {
                    text_parts.push(text.clone());
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

        messages.push(Message {
            role: Role::Assistant,
            content: assistant_blocks,
        });

        // Only return text when we're done with tools — intermediate text
        // (e.g. "Let me check that") is thinking, not the real answer.
        if tool_uses.is_empty() || response.stop_reason != StopReason::ToolUse {
            let full_text = text_parts.join("");
            let scan = safety.scan_output(&full_text);
            if let Some(reason) = scan.block_reason() {
                return Ok(format!("LLM response blocked: {reason}."));
            }
            return Ok(full_text);
        }

        // Execute tools
        let mut result_blocks = Vec::new();
        for (id, name, input) in &tool_uses {
            eprintln!("  ↳ Tool: {name} (local)");
            let result = match tools.get(name) {
                Some(tool) => match tool.execute(input.clone()).await {
                    Ok(output) => {
                        let scan = safety.scan_output(&output);
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
            result_blocks.push(result);
        }

        messages.push(Message {
            role: Role::User,
            content: result_blocks,
        });
    }
}

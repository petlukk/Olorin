//! The Olorin Pipe — central dispatch system.
//!
//! Single entry/exit point for all messages. Implements a 6-step pipeline:
//!   1. Safety scan → block if dangerous
//!   2. Slash command → tools direct
//!   3. Intent router → kernel (calc/time/cpu/weather)
//!   4. Recall → vault context
//!   5. Inference → generate tokens
//!   6. Output guard → truncate/block
//!
//! Synchronous — no async/tokio.

use crate::core::anthropic::AnthropicClient;
use crate::core::dispatch;
use crate::core::handlers;
use crate::core::llm::{self, ContentBlock, Message};
use crate::core::safety;
use crate::recall::VectorStore;
use std::time::Instant;

// ── Public types ─────────────────────────────────────────────────────────────

/// Response from the dispatch pipeline.
pub struct Response {
    pub text:    String,
    pub blocked: bool,
}

impl Response {
    fn text(s: impl Into<String>) -> Self {
        Self { text: s.into(), blocked: false }
    }

    fn blocked(reason: impl Into<String>) -> Self {
        Self { text: reason.into(), blocked: true }
    }
}

/// Central dispatch context — holds all state needed for the pipeline.
pub struct DispatchContext {
    /// Conversation history
    messages:     Vec<Message>,
    /// Recall store for conversation search
    recall:       VectorStore,
    /// Anthropic cloud client (optional — requires API key)
    anthropic:    Option<AnthropicClient>,
    /// Last turn timing
    last_timing:  Option<handlers::TurnTiming>,
    /// System prompt
    system_prompt: String,
    /// Max tool-loop turns before stopping
    _max_turns:   usize,
}

impl DispatchContext {
    /// Create a new dispatch context.
    /// `api_key`: optional Anthropic API key for cloud inference.
    pub fn new(api_key: Option<String>) -> Self {
        let anthropic = api_key.map(AnthropicClient::new);
        Self {
            messages:      Vec::new(),
            recall:        VectorStore::new(1024),
            anthropic,
            last_timing:   None,
            system_prompt: llm::SYSTEM_PROMPT.to_string(),
            _max_turns:    8,
        }
    }

    /// Create with a custom system prompt.
    pub fn with_system_prompt(mut self, prompt: &str) -> Self {
        self.system_prompt = format!("{}\n\n{prompt}", llm::SYSTEM_PROMPT);
        self
    }

    /// The Olorin Pipe — process a single input through the 6-step pipeline.
    ///
    /// ```text
    /// raw input
    ///     |
    ///     +- 1. Safety Scan → BLOCK if dangerous
    ///     +- 2. Slash Command? → tools direct
    ///     +- 3. Intent Router → kernel (calc/time/cpu/weather)
    ///     +- 4. Recall → vault context
    ///     +- 5. Inference → generate tokens
    ///     +- 6. Output Guard → truncate/block
    ///     |
    ///     v
    /// Response
    /// ```
    pub fn dispatch(&mut self, input: &str) -> Response {
        let input = input.trim();
        if input.is_empty() {
            return Response::text("");
        }

        let safety_start = Instant::now();

        // ── Step 1: Safety Scan ──────────────────────────────────────────
        let scan = safety::scan(input.as_bytes());
        let safety_us = safety_start.elapsed().as_micros() as u64;

        if scan.blocked {
            let details: Vec<String> = scan.details.iter().map(|w| {
                format!("  - {} at position {}", w.pattern, w.position)
            }).collect();
            return Response::blocked(format!(
                "Input blocked:\n{}", details.join("\n")
            ));
        }

        // ── Step 2: Slash Command ────────────────────────────────────────
        let input_bytes = input.as_bytes();
        let (cmd_id, cmd_arg) = dispatch::match_command(input_bytes);

        // Meta commands
        if cmd_id >= dispatch::CMD_HELP && cmd_id <= dispatch::CMD_PROFILE {
            return self.handle_meta(cmd_id);
        }

        if cmd_id == dispatch::CMD_TASKS {
            return Response::text("No background tasks. (Task system not yet available.)");
        }

        if cmd_id == dispatch::CMD_RECALL {
            let query = String::from_utf8_lossy(cmd_arg);
            return Response::text(self.recall.recall_formatted(&query, 5));
        }

        // Tool commands
        if cmd_id >= dispatch::CMD_TOOL_FIRST && cmd_id <= dispatch::CMD_TOOL_LAST {
            return self.handle_tool_command(cmd_id, cmd_arg);
        }

        // Unknown slash command
        if input.starts_with('/') && cmd_id == dispatch::CMD_NONE {
            return Response::text(format!(
                "Unknown command: {input}. Type /help for available commands."
            ));
        }

        // ── Step 3: Intent Router ────────────────────────────────────────
        let (intent, arg_start, arg_len) = dispatch::classify_intent(input_bytes);
        if intent != dispatch::INTENT_NONE {
            if let Some(tool_name) = dispatch::intent_to_tool_name(intent) {
                let arg_bytes = if arg_start + arg_len <= input_bytes.len() {
                    &input_bytes[arg_start..arg_start + arg_len]
                } else {
                    &[]
                };
                return self.execute_intent(tool_name, intent, arg_bytes);
            }
        }

        // ── Step 4: Recall ───────────────────────────────────────────────
        // Index user message and search for context.
        self.recall.add(input);
        let recall_context = self.recall.synthesize_context(input, 3);

        // ── Step 5: Inference ────────────────────────────────────────────
        self.messages.push(handlers::user_message(input));
        let response = self.run_inference(&recall_context);

        let llm_ms = match &response {
            Ok(_) => 0, // timing would be set inside run_inference
            Err(_) => 0,
        };

        match response {
            Ok(text) => {
                // ── Step 6: Output Guard ─────────────────────────────────
                let guarded = handlers::apply_guard(&text);

                // Safety scan on output
                let output_scan = safety::scan(guarded.as_bytes());
                if output_scan.blocked {
                    return Response::blocked("LLM response blocked by safety scan.");
                }

                // Index assistant response for recall
                if !guarded.trim().is_empty() {
                    self.recall.add(&guarded);
                }

                self.messages.push(handlers::assistant_message(&guarded));
                self.last_timing = Some(handlers::TurnTiming {
                    safety_scan_us: safety_us,
                    llm_call_ms: llm_ms,
                    tool_execs: vec![],
                });
                Response::text(guarded)
            }
            Err(e) => Response::text(format!("LLM error: {e}")),
        }
    }

    /// Clear conversation history.
    pub fn clear(&mut self) {
        self.messages.clear();
        self.recall.clear();
    }

    /// Get last turn timing data.
    pub fn last_timing(&self) -> Option<&handlers::TurnTiming> {
        self.last_timing.as_ref()
    }

    // ── Meta command handling ────────────────────────────────────────────────

    fn handle_meta(&mut self, cmd_id: i32) -> Response {
        match cmd_id {
            dispatch::CMD_QUIT => Response::text("Goodbye!"),
            dispatch::CMD_HELP => Response::text(self.help_text()),
            dispatch::CMD_TOOLS => Response::text(
                "Available tools: time, calc, cpu, shell, http, memory, read, write, \
                 ls, json, tokens, bench, weather, translate, define, summarize, \
                 grep, git, remind"
            ),
            dispatch::CMD_CLEAR => {
                self.clear();
                Response::text("Context cleared.")
            }
            dispatch::CMD_MODEL => {
                let backend = if self.anthropic.is_some() { "cloud (Anthropic)" } else { "none" };
                Response::text(format!("[Olorin] Current backend: {backend}"))
            }
            dispatch::CMD_PROFILE => {
                let msg = match &self.last_timing {
                    Some(t) => t.format(),
                    None => "No timing data yet.".to_string(),
                };
                Response::text(msg)
            }
            _ => Response::text(""),
        }
    }

    // ── Tool command handling ────────────────────────────────────────────────

    fn handle_tool_command(&mut self, cmd_id: i32, cmd_arg: &[u8]) -> Response {
        let arg_str = String::from_utf8_lossy(cmd_arg);
        let tool_start = Instant::now();

        match dispatch::build_tool_params(cmd_id, &arg_str) {
            Ok((name, params)) => {
                let result = self.execute_tool(name, &params);
                let tool_ms = tool_start.elapsed().as_millis() as u64;
                self.last_timing = Some(handlers::TurnTiming {
                    safety_scan_us: 0,
                    llm_call_ms: 0,
                    tool_execs: vec![(name.to_string(), tool_ms)],
                });
                match result {
                    Ok(output) => {
                        // Safety scan tool output
                        let scan = safety::scan(output.as_bytes());
                        if scan.blocked {
                            Response::blocked("Tool output blocked by safety scan.")
                        } else {
                            Response::text(output)
                        }
                    }
                    Err(e) => Response::text(format!("Tool error: {e}")),
                }
            }
            Err(e) => Response::text(format!("{e}")),
        }
    }

    // ── Intent execution ─────────────────────────────────────────────────────

    fn execute_intent(&mut self, tool_name: &str, intent: i32, arg_bytes: &[u8]) -> Response {
        let params = dispatch::intent_to_params(intent, arg_bytes);
        match self.execute_tool(tool_name, &params) {
            Ok(output) => Response::text(output),
            Err(e) => Response::text(format!("Tool error: {e}")),
        }
    }

    // ── Tool execution ───────────────────────────────────────────────────────

    /// Execute a tool by name with parameters.
    fn execute_tool(
        &self,
        name: &str,
        params: &[(&str, String)],
    ) -> Result<String, String> {
        // Calc uses SIMD kernel directly with fixed-point formatting
        if name == "calc" {
            let expr = params.iter()
                .find(|(k, _)| *k == "expr")
                .map(|(_, v)| v.as_str())
                .unwrap_or("");
            return match dispatch::eval_expr(expr) {
                Ok(result) => Ok(result),
                Err(e) => Err(format!("{e}")),
            };
        }

        // All other tools go through the tool registry
        let args = params.iter()
            .map(|(_, v)| v.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        match crate::tools::run_tool(name, &args) {
            Some(r) => if r.success { Ok(r.output) } else { Err(r.output) },
            None => Err(format!("unknown tool: {name}")),
        }
    }

    // ── Inference ────────────────────────────────────────────────────────────

    fn run_inference(
        &self,
        recall_context: &Option<String>,
    ) -> Result<String, String> {
        // Build system prompt with recall context
        let system = match recall_context {
            Some(ctx) => format!("{}\n\n{ctx}", self.system_prompt),
            None => self.system_prompt.clone(),
        };

        // Try Anthropic cloud first
        if let Some(client) = &self.anthropic {
            let msg_pairs: Vec<(&str, &str)> = self.messages.iter().map(|m| {
                let role = m.role.as_str();
                let text: &str = m.content.iter().find_map(|b| {
                    if let ContentBlock::Text { text } = b { Some(text.as_str()) } else { None }
                }).unwrap_or("");
                (role, text)
            }).collect();

            match client.generate(&system, &msg_pairs) {
                Ok(text) => return Ok(text),
                Err(e) => {
                    eprintln!("[olorin] cloud inference failed: {e}");
                }
            }
        }

        // No inference backend available
        Err("No LLM backend available. Set ANTHROPIC_API_KEY for cloud inference.".to_string())
    }

    // ── Help text ────────────────────────────────────────────────────────────

    fn help_text(&self) -> String {
        "\
Commands:
  /help    /quit    /tools   /clear   /model   /profile

Tools:
  /time  /calc <expr>  /http <url>  /shell <cmd>  /cpu
  /memory <action> [key] [value]   /read <path>   /write <path> <content>
  /ls [path]  /json <action> <input>  /tokens <text>  /bench <target>
  /weather <city>  /translate <lang> <text>  /define <word>  /summarize <url>
  /grep <pattern> [path]  /git <subcommand> [args]  /remind <time> <message>
  /recall <query>

Agent: Olorin".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_empty_input() {
        let mut ctx = DispatchContext::new(None);
        let resp = ctx.dispatch("");
        assert_eq!(resp.text, "");
        assert!(!resp.blocked);
    }

    #[test]
    fn dispatch_help_command() {
        let mut ctx = DispatchContext::new(None);
        let resp = ctx.dispatch("/help");
        assert!(resp.text.contains("Commands:"));
        assert!(!resp.blocked);
    }

    #[test]
    fn dispatch_clear_command() {
        let mut ctx = DispatchContext::new(None);
        let resp = ctx.dispatch("/clear");
        assert_eq!(resp.text, "Context cleared.");
    }

    #[test]
    fn dispatch_unknown_command() {
        let mut ctx = DispatchContext::new(None);
        let resp = ctx.dispatch("/foobar");
        assert!(resp.text.contains("Unknown command"));
    }

    #[test]
    fn response_text_not_blocked() {
        let r = Response::text("hello");
        assert_eq!(r.text, "hello");
        assert!(!r.blocked);
    }

    #[test]
    fn response_blocked() {
        let r = Response::blocked("reason");
        assert!(r.blocked);
    }
}

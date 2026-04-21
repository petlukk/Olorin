//! Tool and command handling for the Olorin Pipe.
//!
//! Handles slash commands (/help, /tools, /clear, etc.), tool execution,
//! intent-based tool dispatch, and post-inference tool-call wiring.
//! Split from router.rs for the 500-line rule.

use crate::core::dispatch;
use crate::core::handlers;
use crate::core::safety;
use crate::core::tool_parse;
use crate::core::llm::ContentBlock;
use crate::core::router::{DispatchContext, Response, StreamEvent};
use crate::storage::json;
use std::time::Instant;

impl DispatchContext {
    // ── The Olorin Pipe (non-streaming) ──────────────────────────────────────

    /// The Olorin Pipe — process a single input through the pipeline.
    ///
    /// ```text
    /// raw input
    ///     |
    ///     +- 1. Safety Scan (inbound) → BLOCK if dangerous
    ///     +- 2. Slash Command? → tools direct
    ///     +- 3. Intent Router → kernel (calc/time/cpu/weather)
    ///     +- 4. Recall → vault context
    ///     +- 5. Inference → generate tokens
    ///     +- 6. Leak Scan (outbound) + ChatML Trim
    ///     |
    ///     v
    /// Response → vault save
    /// ```
    pub fn dispatch(&mut self, input: &str) -> Response {
        let input = input.trim();
        if input.is_empty() {
            return Response::text("");
        }

        let safety_start = std::time::Instant::now();
        let recall_context = match self.pre_inference(input) {
            Err(early) => return early,
            Ok(ctx) => ctx,
        };
        let safety_us = safety_start.elapsed().as_micros() as u64;

        self.messages.push(handlers::user_message(input));
        let response = self.run_inference(&recall_context);

        match response {
            Ok(text) => {
                if safety::scan_outbound(text.as_bytes()).blocked {
                    return Response::blocked("Response blocked: potential secret leak.");
                }
                self.finalize_response(input, &text);
                self.last_timing = Some(handlers::TurnTiming {
                    safety_scan_us: safety_us,
                    llm_call_ms: 0,
                    tool_execs: vec![],
                });
                Response::text(text)
            }
            Err(e) => Response::text(format!("LLM error: {e}")),
        }
    }

    // ── Meta command handling ────────────────────────────────────────────────

    pub(crate) fn handle_meta(&mut self, cmd_id: i32) -> Response {
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
                let local = self.engine.as_ref().map(|e| e.quant_type_str()).unwrap_or("none");
                let cloud = if self.anthropic.is_some() { "Anthropic" } else { "none" };
                Response::text(format!("Local: {local}, Cloud: {cloud}"))
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

    // ── Teleport handling ────────────────────────────────────────────────────

    pub(crate) fn handle_teleport(&mut self) -> Response {
        crate::interface::whatsapp::teleport_loop(self)
    }

    // ── Rune handling ────────────────────────────────────────────────────────

    pub(crate) fn handle_rune(&mut self, cmd_arg: &[u8]) -> Response {
        let full = std::str::from_utf8(cmd_arg).unwrap_or("").trim();
        let (name, args) = match full.split_once(char::is_whitespace) {
            Some((n, a)) => (n, a.trim()),
            None => (full, ""),
        };
        if name.is_empty() {
            return Response::text(
                "usage: /rune <name> [args] — try `/rune eacrunch <path.csv>`"
            );
        }
        match crate::runes::run_rune(name, args) {
            Some(result) => {
                let mut body = result.answer;
                if let Some(d) = result.details {
                    body.push_str("\n\n---\n");
                    body.push_str(&d);
                }
                body.push_str(&format!("\n[timing: {}µs]", result.timing_us));
                // Spec requires inbound safety::scan on rune output before it
                // can reach the LLM turn. Runes classified UntrustedQuoted
                // (e.g. eacrunch, eacount) echo file-derived bytes, so this
                // is the last defense before the content enters context.
                // Runs on every rune path regardless of OutputSafety — the
                // extra few µs on Trusted output is cheap defense-in-depth.
                let scan = safety::scan(body.as_bytes());
                if scan.blocked {
                    return Response::blocked("Rune output blocked by safety scan.");
                }
                self.vault_save(b"user", full.as_bytes());
                self.vault_save(b"tool", body.as_bytes());
                Response::text(body)
            }
            None => Response::text(format!("unknown rune: {name}")),
        }
    }

    // ── Tool command handling ────────────────────────────────────────────────

    pub(crate) fn handle_tool_command(&mut self, cmd_id: i32, cmd_arg: &[u8]) -> Response {
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
                        let scan = safety::scan(output.as_bytes());
                        if scan.blocked {
                            Response::blocked("Tool output blocked by safety scan.")
                        } else {
                            self.vault_save(b"user", arg_str.as_bytes());
                            self.vault_save(b"tool", output.as_bytes());
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

    pub(crate) fn execute_intent(&mut self, tool_name: &str, intent: i32, arg_bytes: &[u8]) -> Response {
        let params = dispatch::intent_to_params(intent, arg_bytes);
        match self.execute_tool(tool_name, &params) {
            Ok(output) => {
                self.vault_save(b"user", arg_bytes);
                self.vault_save(b"tool", output.as_bytes());
                Response::text(output)
            }
            Err(e) => Response::text(format!("Tool error: {e}")),
        }
    }

    // ── Tool execution ───────────────────────────────────────────────────────

    pub(crate) fn execute_tool(
        &self,
        name: &str,
        params: &[(&str, String)],
    ) -> Result<String, String> {
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

        let args = params.iter()
            .map(|(_, v)| v.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        match crate::tools::run_tool(name, &args) {
            Some(r) => if r.success { Ok(r.output) } else { Err(r.output) },
            None => Err(format!("unknown tool: {name}")),
        }
    }

    // ── Help text ────────────────────────────────────────────────────────────

    pub(crate) fn help_text(&self) -> String {
        "\
Commands:
  /help    /quit    /tools   /clear   /model   /profile

Tools:
  /time  /calc <expr>  /http <url>  /shell <cmd>  /cpu
  /memory <action> [key] [value]   /read <path>   /write <path> <content>
  /ls [path]  /json <action> <input>  /tokens <text>  /bench <target>
  /weather <city>  /translate <lang> <text>  /define <word>  /summarize <url>
  /grep <pattern> [path]  /git <subcommand> [args]  /remind <time> <message>
  /recall <query>  /teleport

Runes (SIMD tool calls):
  /rune eacrunch <csv>   — summarize a CSV via SIMD

Agent: Olorin".to_string()
    }

    // ── Post-inference tool-call wiring ──────────────────────────────────────

    /// After local `engine.generate()` returns `first_output`, detect any
    /// `<tool_call>` and run the dispatch + synthetic follow-up generate.
    ///
    /// Returns `Some(final_text)` if a tool was called and follow-up succeeded
    /// (the caller should use this as the text to finalize + send Done).
    /// Returns `None` if there was no tool call in `first_output` (caller
    /// falls through to its normal path).
    ///
    /// TODO(runes-v2): multi-iteration tool loop, WhatsApp source gating,
    /// per-rune timeout, concurrency mutex.
    pub(crate) fn run_local_followup_if_tool_call(
        &mut self,
        user_input: &str,
        system: &str,
        first_output: &str,
        tx: &std::sync::mpsc::Sender<StreamEvent>,
    ) -> Option<String> {
        // Step 1: parse tool calls from the first output.
        let parsed = tool_parse::extract_tool_calls(first_output);
        let (tool_name, tool_input) = parsed.content.iter().find_map(|b| match b {
            ContentBlock::ToolUse { name, input, .. } => Some((name.clone(), input.clone())),
            _ => None,
        })?;

        // Step 2: dispatch the tool call.
        let tool_result = match handlers::dispatch_tool_call(&tool_name, &tool_input) {
            Ok(wrapped) => wrapped,
            Err(msg) => format!("<tool_error>{msg}</tool_error>"),
        };

        // Step 3: build synthetic follow-up prompt.
        let args_json = json::serialize(&tool_input);
        let followup_prompt = format!(
            "{user_input}\n\n\
             [earlier, you chose to call a tool]\n\
             <tool_call>{{\"name\": \"{tool_name}\", \"arguments\": {args_json}}}</tool_call>\n\
             [the tool returned]\n\
             {tool_result}\n\n\
             Now answer my original question using the tool result above. \
             Do NOT call another tool."
        );

        // Step 4: second generate — stream tokens through tx.
        let tx_ref = tx.clone();
        let on_event = move |ev: crate::inference::generate::GenEvent| match ev {
            crate::inference::generate::GenEvent::Token(tok) => {
                if !safety::is_chatml_hallucination(tok) {
                    let _ = tx_ref.send(StreamEvent::Token(tok.to_string()));
                }
            }
            crate::inference::generate::GenEvent::Thinking(active) => {
                let _ = tx_ref.send(StreamEvent::Thinking(active));
            }
        };

        let gen_result = if let Some(engine) = &mut self.engine {
            Some(engine.generate(&followup_prompt, system, &on_event))
        } else {
            None
        };

        // Step 5: apply outbound scan; surface tool_result on generation failure.
        let final_text = match gen_result {
            Some(Ok(text)) => text,
            Some(Err(e)) => {
                eprintln!("[olorin] follow-up inference failed: {e}");
                tool_result
            }
            None => tool_result,
        };

        if safety::scan_outbound(final_text.as_bytes()).blocked {
            let _ = tx.send(StreamEvent::Error(
                "Response blocked: potential secret leak.".to_string(),
            ));
            return Some(String::new());
        }

        Some(final_text)
    }

    /// After cloud `client.generate()` returns `first_text`, detect any
    /// `<tool_call>` and run dispatch + a second cloud call with the tool result.
    ///
    /// Returns the final text to stream to the user. If there is no tool call,
    /// `first_text` is returned unchanged.
    ///
    /// TODO(runes-v2): multi-iteration tool loop, WhatsApp source gating,
    /// per-rune timeout, concurrency mutex.
    pub(crate) fn maybe_handle_tool_call_cloud(
        &mut self,
        user_input: &str,
        first_text: String,
    ) -> String {
        // Step 1: parse tool calls.
        let parsed = tool_parse::extract_tool_calls(&first_text);
        let (tool_name, tool_input) = match parsed.content.iter().find_map(|b| match b {
            ContentBlock::ToolUse { name, input, .. } => Some((name.clone(), input.clone())),
            _ => None,
        }) {
            Some(pair) => pair,
            None => return first_text,
        };

        // Step 2: dispatch.
        let tool_result = match handlers::dispatch_tool_call(&tool_name, &tool_input) {
            Ok(wrapped) => wrapped,
            Err(msg) => format!("<tool_error>{msg}</tool_error>"),
        };

        // Step 3: rebuild cloud messages with tool exchange appended.
        let mut msg_pairs_owned = self.build_cloud_messages();
        msg_pairs_owned.push(("assistant".to_string(), first_text));
        msg_pairs_owned.push(("user".to_string(), format!(
            "[the tool you called returned]\n{tool_result}\n\n\
             Now answer my original question using the tool result above. \
             Do NOT call another tool."
        )));

        // Step 4: second cloud call.
        let system = self.system_prompt.clone();
        if let Some(client) = &self.anthropic {
            let pairs: Vec<(&str, &str)> = msg_pairs_owned.iter()
                .map(|(r, t)| (r.as_str(), t.as_str()))
                .collect();
            match client.generate(&system, &pairs) {
                Ok(second_text) => return second_text,
                Err(e) => eprintln!("[olorin] cloud follow-up inference failed: {e}"),
            }
        }

        // Fallback: surface the tool result so the user sees something.
        eprintln!("[olorin] cloud follow-up unavailable for '{user_input}', surfacing tool result");
        tool_result
    }
}

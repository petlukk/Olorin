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

/// Shared closer for the tool-call follow-up prompt. Both the local and
/// cloud tool-call paths append this after the tool result so the model
/// is nudged to answer the original question without chaining tool calls.
/// Kept as a const to prevent drift between the two paths.
const FOLLOWUP_CLOSER: &str =
    "Now answer my original question using the tool result above. Do NOT call another tool.";

/// Decode-token cap for a rune-narration call. The model's max_seq_len is
/// 2048 positions total (prefill + decode).
///
/// Gemma 4 has thinking-mode enabled by default — the model emits a
/// `<|channel>...<channel|>` block of reasoning tokens BEFORE the answer.
/// Those tokens count against decode but are filtered out of the returned
/// string in `Engine::generate`. For complex inputs the thinking block
/// alone can run 400-600 tokens; we set the cap at 768 = ~600 thinking +
/// ~120 answer + margin to leave headroom for both.
pub(crate) const NARRATION_DECODE_TOKEN_CAP: usize = 768;

/// System prompt for narration calls. Narration is a focused
/// analyze-and-summarize task, not a tool-dispatch turn — using the full
/// `runes_prompt_block` (with tools-block + "only call a tool when..."
/// framing) made Gemma 4 emit EOS immediately after seeing the rune
/// output (model interprets "tool already called → conversation done").
///
/// This narration-specific role grounds the model in "data analyst
/// summarizing tool output" without the tool-dispatch baggage. Clean
/// instruction-data shape with this system prompt empirically reproduces
/// reliable narrations on both x86 and Pi 5.
pub(crate) const NARRATION_SYSTEM_PROMPT: &str =
    "You are a helpful data analyst. Read the user's tool output and respond \
     with 1-2 plain-English sentences highlighting what stands out. \
     Avoid repeating raw numbers verbatim.";

/// Total position budget the rune-narration prompt must fit within: the
/// model's max_seq_len minus the decode cap, with a small safety margin
/// for chat-template tokens that get added during `generate` formatting.
/// Anything over this skips narration with a notice — better than panicking
/// on `pos < max_seq_len` deep in `rope_slices`.
pub(crate) const NARRATION_MAX_PROMPT_TOKENS: usize = 2048 - NARRATION_DECODE_TOKEN_CAP - 32;

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

        let audit_turn = self.audit_input(input);
        let audit_start = std::time::Instant::now();

        let safety_start = std::time::Instant::now();
        let recall_context = match self.pre_inference(input) {
            Err(early) => {
                if let Some(prompt) = early.followup.clone() {
                    if self.strict {
                        // No narration in strict mode — kernel output only.
                        self.audit_result(audit_turn, audit_start, "rune_strict_no_narration", &[]);
                        return Response::text(early.text);
                    }
                    self.audit_result(audit_turn, audit_start, "rune_with_narration",
                        &[("narration", crate::core::audit::AuditValue::Bool(true))]);
                    return self.run_followup_sync(input, early.text, &prompt);
                }
                let phase = if early.blocked { "blocked" } else { "command" };
                self.audit_result(audit_turn, audit_start, phase, &[]);
                return early;
            }
            Ok(ctx) => ctx,
        };
        let safety_us = safety_start.elapsed().as_micros() as u64;

        // ── Strict mode: refuse LLM fallback ─────────────────────────
        if self.strict {
            self.audit_result(audit_turn, audit_start, "strict_refused", &[]);
            return Response::text(crate::core::router::STRICT_REFUSAL);
        }

        self.audit_result(audit_turn, audit_start, "llm_start", &[]);
        self.messages.push(handlers::user_message(input));
        let response = self.run_inference(&recall_context);

        match response {
            Ok(first_text) => {
                // ── Tool-call follow-up (sync path) ─────────────────────────
                // run_local_followup_if_tool_call applies its own outbound scan
                // internally and returns Some("") on block. Only fall through to
                // the dispatch-level outbound scan when no tool call was detected
                // (followup returns None), i.e. for plain LLM responses.
                let system = self.system_prompt.clone();
                if let Some(followup_text) = self.run_local_followup_if_tool_call(
                    input, &system, &first_text, None,
                ) {
                    if followup_text.is_empty() {
                        // Outbound scan blocked the follow-up inside the helper.
                        return Response::blocked("Response blocked: potential secret leak.");
                    }
                    self.finalize_response(input, &followup_text);
                    self.last_timing = Some(handlers::TurnTiming {
                        safety_scan_us: safety_us,
                        llm_call_ms: 0,
                        tool_execs: vec![],
                    });
                    return Response::text(followup_text);
                }

                // No tool call — apply outbound scan on the plain LLM response.
                if safety::scan_outbound(first_text.as_bytes()).blocked {
                    return Response::blocked("Response blocked: potential secret leak.");
                }
                self.finalize_response(input, &first_text);
                self.last_timing = Some(handlers::TurnTiming {
                    safety_scan_us: safety_us,
                    llm_call_ms: 0,
                    tool_execs: vec![],
                });
                Response::text(first_text)
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

    /// Sync follow-up: run the model with `prompt`, append narration after
    /// `head` (the displayed kernel output). Used by the non-streaming
    /// `dispatch` path so REPL/test callers also see narration.
    ///
    /// On successful narration, persists the turn so it shows up in next-turn
    /// context: pushes the rune command + narration into `self.messages` and
    /// vault-saves the narration as `assistant`. (The kernel output is already
    /// vault-saved as `tool` by `handle_rune`.)
    fn run_followup_sync(&mut self, input: &str, head: String, prompt: &str) -> Response {
        let Some(engine) = self.engine.as_mut() else {
            return Response::text(head);
        };
        // Narration-specific system prompt (analyst role, no tools-block).
        // The full runes prompt block conditions Gemma 4 to "decide whether
        // to call a tool, then if no tool needed, emit EOS" — which is the
        // wrong framing for a follow-up summarization. See the const doc.
        let system = NARRATION_SYSTEM_PROMPT;
        let prompt_tokens = engine.count_prompt_tokens(prompt, system);
        if prompt_tokens > NARRATION_MAX_PROMPT_TOKENS {
            return Response::text(format!(
                "{head}\n\n[narration skipped: prompt is {prompt_tokens} tokens, over the {NARRATION_MAX_PROMPT_TOKENS}-token narration budget]"
            ));
        }
        let prior_max = engine.max_tokens;
        engine.max_tokens = NARRATION_DECODE_TOKEN_CAP;
        let no_op = |_ev: crate::inference::generate::GenEvent| {};
        let result = engine.generate(prompt, system, &no_op);
        engine.max_tokens = prior_max;
        match result {
            Ok(narr) => {
                let trimmed = narr.trim();
                if trimmed.is_empty() {
                    return Response::text(head);
                }
                self.messages.push(handlers::user_message(input));
                self.messages.push(handlers::assistant_message(trimmed));
                self.vault_save(b"assistant", trimmed.as_bytes());
                Response::text(format!("{head}\n\n{trimmed}"))
            }
            Err(e) => {
                eprintln!("[rune] narration failed: {e}");
                Response::text(head)
            }
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
        let Some(rune) = crate::runes::RUNES.iter().find(|r| r.name() == name) else {
            return Response::text(format!("unknown rune: {name}"));
        };
        let result = rune.run(args);
        let safety_class = rune.output_safety();
        let answer = result.answer.clone();
        let timing_us = result.timing_us;

        // Display body for the user — kernel summary + details + timing.
        let mut body = result.answer;
        if let Some(d) = result.details {
            body.push_str("\n\n---\n");
            body.push_str(&d);
        }
        body.push_str(&format!("\n[timing: {timing_us}µs]"));
        // Inbound safety scan: file-derived bytes are echoed verbatim, so this
        // is the last defense before content reaches the user (or the LLM via
        // the followup turn). Runs regardless of OutputSafety — defense-in-depth.
        let scan = safety::scan(body.as_bytes());
        if scan.blocked {
            return Response::blocked("Rune output blocked by safety scan.");
        }
        self.vault_save(b"user", full.as_bytes());
        self.vault_save(b"tool", body.as_bytes());

        // Followup: feed the (wrapped, safety-scanned) rune answer back to the
        // model for a 1-2 sentence narration. Always built; consumer sites in
        // dispatch / dispatch_streaming gate on engine presence.
        let scratch = crate::runes::RuneResult {
            answer,
            details: None,
            success: result.success,
            timing_us,
        };
        let followup = crate::runes::build_narration_prompt(name, safety_class, scratch);

        let resp = Response::text(body);
        match followup {
            Some(p) => resp.with_followup(p),
            None    => resp,
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
        let mode_line = if self.strict {
            "\nMode: strict (LLM disabled — only deterministic paths fire)\n"
        } else {
            ""
        };
        format!("\
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
  /rune eacrunch <csv>     — summarize a CSV via SIMD
  /rune eajson <jsonl>     — summarize a JSON Lines file via SIMD
{mode_line}
Agent: Olorin")
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
    /// When `tx` is `Some`, follow-up tokens are streamed through the channel
    /// (Web UI / WhatsApp path). When `tx` is `None`, the follow-up runs
    /// synchronously and the final text is returned inline (terminal REPL path).
    /// In both cases the outbound scan is applied exactly once inside this helper;
    /// callers must NOT apply a second scan on the returned text.
    /// On outbound-block the helper returns `Some(String::new())` — the caller
    /// is responsible for surfacing this as a blocked response to the user.
    pub(crate) fn run_local_followup_if_tool_call(
        &mut self,
        user_input: &str,
        system: &str,
        first_output: &str,
        tx: Option<&std::sync::mpsc::Sender<StreamEvent>>,
    ) -> Option<String> {
        // Step 1: parse tool calls from the first output.
        let parsed = tool_parse::extract_tool_calls(first_output);
        let tool_use = parsed.content.iter().find_map(|b| match b {
            ContentBlock::ToolUse { name, input, .. } => Some((name.clone(), input.clone())),
            _ => None,
        });

        let (tool_name, tool_input) = tool_use?;

        // Step 2: dispatch the tool call.
        let dispatch_result = handlers::dispatch_tool_call(&tool_name, &tool_input);
        let tool_result = match &dispatch_result {
            Ok(wrapped) => wrapped.clone(),
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
             {FOLLOWUP_CLOSER}"
        );

        // Step 4: second generate.
        // Streaming path: emit tokens through tx as they arrive.
        // Sync path (tx = None): use a no-op on_event; block until complete.
        let gen_result = if let Some(engine) = &mut self.engine {
            let result = if let Some(tx_ref) = tx {
                let tx_clone = tx_ref.clone();
                let on_event = move |ev: crate::inference::generate::GenEvent| match ev {
                    crate::inference::generate::GenEvent::Token(tok) => {
                        if !safety::is_chatml_hallucination(tok) {
                            let _ = tx_clone.send(StreamEvent::Token(tok.to_string()));
                        }
                    }
                    crate::inference::generate::GenEvent::Thinking(active) => {
                        let _ = tx_clone.send(StreamEvent::Thinking(active));
                    }
                };
                engine.generate(&followup_prompt, system, &on_event)
            } else {
                engine.generate(&followup_prompt, system, &|_ev| {})
            };
            Some(result)
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

        let outbound_scan = safety::scan_outbound(final_text.as_bytes());
        if outbound_scan.blocked {
            if let Some(tx_ref) = tx {
                let _ = tx_ref.send(StreamEvent::Error(
                    "Response blocked: potential secret leak.".to_string(),
                ));
            }
            // Return Some(String::new()) as the block sentinel.
            // Streaming callers use the Done event; sync callers check is_empty().
            return Some(String::new());
        }

        Some(final_text)
    }

    /// After cloud `client.generate()` returns `first_text`, detect any
    /// `<tool_call>` and run dispatch + a second cloud call with the tool result.
    ///
    /// Returns the final text to stream to the user. If there is no tool call,
    /// `first_text` is returned unchanged.
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
             {FOLLOWUP_CLOSER}"
        )));

        // Step 4: second cloud call.
        let system = self.system_prompt.clone();
        if let Some(client) = &self.anthropic {
            let pairs: Vec<(&str, &str)> = msg_pairs_owned.iter()
                .map(|(r, t)| (r.as_str(), t.as_str()))
                .collect();
            match client.generate(&system, &pairs) {
                Ok(second_text) => {
                    if safety::scan_outbound(second_text.as_bytes()).blocked {
                        // Outbound scan flagged the follow-up response. Fall back to the
                        // raw tool result so the user still sees something, and stays
                        // consistent with the local path's behavior.
                        return tool_result;
                    }
                    return second_text;
                }
                Err(e) => eprintln!("[olorin] cloud follow-up inference failed: {e}"),
            }
        }

        // Fallback: surface the tool result so the user sees something.
        eprintln!("[olorin] cloud follow-up unavailable for '{user_input}', surfacing tool result");
        tool_result
    }
}

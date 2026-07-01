//! Slash commands, tools, and runes. Post-inference tool-call wiring lives in
//! router_toolcall.rs.

use crate::core::dispatch;
use crate::core::handlers;
use crate::core::safety;
use crate::core::router::{DispatchContext, Response};
use std::time::Instant;

/// Safety ceiling for narration decode. Narration runs thinking-off (the rune
/// did the analysis; the model only restates it), so this is pure headroom over
/// the ~120-token answer — EOS hits long before the cap. Pi: ~82s → ~18s.
pub(crate) const NARRATION_DECODE_TOKEN_CAP: usize = 768;

/// Analyst-role system. The full runes_prompt_block (with tools framing)
/// makes Gemma 4 emit EOS immediately for narration follow-ups.
///
/// The instruction is deliberately concrete. The previous "avoid repeating raw
/// numbers verbatim" made the model hedge — it would say "a significant peak
/// around a certain time" while the exact date and magnitude sat in front of it.
/// We now ask it to NAME the headline figure (date/category + magnitude) and
/// only forbid reproducing the whole table; `is_grid_continuation` still catches
/// the one real failure mode (the model continuing the numeric grid) after the
/// fact, so the prompt no longer has to suppress all numbers to prevent it.
pub(crate) const NARRATION_SYSTEM_PROMPT: &str =
    "You are a helpful data analyst. Read the user's tool output and reply with \
     1-2 plain-English sentences naming the single most important finding. Be \
     concrete: give the actual date/time, category, or value and the magnitude \
     of any peak, spike, or anomaly — e.g. 'X peaked on <date> at about N× the \
     baseline' — not vague phrasing like 'a significant peak around a certain \
     time'. State the headline finding; do not reproduce the table or list \
     every row.";

/// max_seq_len(2048) − decode_cap(768) − chat-template margin(32).
pub(crate) const NARRATION_MAX_PROMPT_TOKENS: usize = 2048 - NARRATION_DECODE_TOKEN_CAP - 32;

impl DispatchContext {
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

    /// Toggle (or set) the local model's chain-of-thought. `/think` flips it,
    /// `/think on|off` sets it. Off is faster; on helps genuinely hard
    /// reasoning. Default is off on aarch64 (thinking buys little there and
    /// costs latency) and on elsewhere — this is the per-session override.
    pub(crate) fn handle_think(&mut self, arg: &[u8]) -> Response {
        let Some(engine) = self.engine.as_mut() else {
            return Response::text("No local model loaded — nothing to toggle.");
        };
        let arg = String::from_utf8_lossy(arg);
        let want = match arg.trim().to_ascii_lowercase().as_str() {
            "on"  | "true"  => true,
            "off" | "false" => false,
            ""              => !engine.thinking,
            other => return Response::text(format!(
                "Usage: /think [on|off] (currently {}). Unrecognized: '{other}'",
                if engine.thinking { "on" } else { "off" },
            )),
        };
        engine.thinking = want;
        Response::text(if want {
            "Thinking mode ON — slower, but better on hard reasoning.".to_string()
        } else {
            "Thinking mode OFF — faster replies.".to_string()
        })
    }

    /// Sync narration after a rune's kernel output; persists the turn on success.
    fn run_followup_sync(&mut self, input: &str, head: String, prompt: &str) -> Response {
        let Some(engine) = self.engine.as_mut() else {
            // No local model — narrate via the cloud client if one is configured.
            return self.run_followup_cloud_sync(input, head, prompt);
        };
        let system = NARRATION_SYSTEM_PROMPT;
        let prompt_tokens = engine.count_prompt_tokens(prompt, system);
        if prompt_tokens > NARRATION_MAX_PROMPT_TOKENS {
            return Response::text(format!(
                "{head}\n\n[narration skipped: prompt is {prompt_tokens} tokens, over the {NARRATION_MAX_PROMPT_TOKENS}-token narration budget]"
            ));
        }
        let prior_max = engine.max_tokens;
        let prior_thinking = engine.thinking;
        engine.max_tokens = NARRATION_DECODE_TOKEN_CAP;
        engine.thinking = false; // no chain-of-thought for restating a rune result
        let no_op = |_ev: crate::inference::generate::GenEvent| {};
        let result = engine.generate(prompt, system, &no_op);
        engine.max_tokens = prior_max;
        engine.thinking = prior_thinking;
        match result {
            Ok(narr) => self.finalize_narration_sync(input, head, prompt, &narr),
            Err(e) => {
                eprintln!("[rune] narration failed: {e}");
                Response::text(head)
            }
        }
    }

    /// Narrate a rune's kernel output via the Anthropic cloud client when no
    /// local engine is loaded. Returns the bare kernel output if no client is
    /// configured or the cloud call fails.
    fn run_followup_cloud_sync(&mut self, input: &str, head: String, prompt: &str) -> Response {
        let Some(client) = &self.anthropic else {
            return Response::text(head);
        };
        match client.generate(NARRATION_SYSTEM_PROMPT, &[("user", prompt)]) {
            Ok(narr) => self.finalize_narration_sync(input, head, prompt, &narr),
            Err(e) => {
                eprintln!("[rune] cloud narration failed: {e}");
                Response::text(head)
            }
        }
    }

    /// Apply the narration discard filters and, on a usable narration, persist
    /// the rune turn. Shared by the local-engine and cloud sync paths.
    fn finalize_narration_sync(
        &mut self,
        input: &str,
        head: String,
        prompt: &str,
        narration: &str,
    ) -> Response {
        let trimmed = narration.trim();
        if trimmed.is_empty()
            || crate::runes::narration::is_grid_continuation(prompt, trimmed)
            || crate::runes::narration::looks_like_data_dump(trimmed)
        {
            return Response::text(head);
        }
        self.messages.push(handlers::user_message(input));
        self.messages.push(handlers::assistant_message(trimmed));
        self.vault_save(b"assistant", trimmed.as_bytes());
        Response::text(format!("{head}\n\n{trimmed}"))
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
            return Response::text(format!(
                "usage: /rune <name> [args] — e.g. /rune eacrunch <path.csv>\n\navailable runes:\n{}",
                crate::runes::rune_list()
            ));
        }
        let Some(rune) = crate::runes::RUNES.iter().find(|r| r.name() == name) else {
            return Response::text(format!(
                "unknown rune: {name}\n\navailable runes:\n{}",
                crate::runes::rune_list()
            ));
        };
        let result = rune.run(args);
        let safety_class = rune.output_safety();
        let answer = result.answer.clone();
        let timing_us = result.timing_us;
        let structured = result.structured;

        // Structured (JSON) output: emit verbatim — no `<rune_output>`
        // wrap, no `[timing: …]` footer (both would break JSONL), no
        // narration (defeats the user's --json intent). Safety scan
        // still runs on the raw bytes.
        let body = if structured {
            result.answer
        } else {
            let mut b = result.answer;
            if let Some(d) = result.details {
                b.push_str("\n\n---\n");
                b.push_str(&d);
            }
            b.push_str(&format!("\n[timing: {timing_us}µs]"));
            b
        };
        if safety::scan(body.as_bytes()).blocked {
            return Response::blocked("Rune output blocked by safety scan.");
        }

        // A chronological series renders as a color block-bar chart above
        // the text body in the REPL. Never for `--json` (structured) runs —
        // that output must stay clean JSONL. The chart is presentation only:
        // it is not vault-saved or fed to narration (block glyphs would just
        // confuse the model), so `body` below is unchanged for both.
        let display = if structured {
            body.clone()
        } else {
            match crate::core::router_streaming_analyze::chart_for(name, args, None, true) {
                Some(chart) if !safety::scan(chart.as_bytes()).blocked => {
                    format!("{chart}\n{body}")
                }
                _ => body.clone(),
            }
        };

        self.vault_save(b"user", full.as_bytes());
        self.vault_save(b"tool", body.as_bytes());

        let scratch = crate::runes::RuneResult {
            answer, details: None,
            success: result.success, timing_us, structured,
        };
        let followup = crate::runes::build_narration_prompt(name, safety_class, scratch);
        let resp = Response::text(display);
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
        let formats = crate::runes::select::supported_formats();
        format!("\
Commands:
  /help  /quit  /tools  /clear  /model  /profile  /think [on|off]

Tools:
  /time  /calc <expr>  /http <url>  /shell <cmd>  /cpu
  /memory <action> [key] [value]   /read <path>   /write <path> <content>
  /ls [path]  /json <action> <input>  /tokens <text>  /bench <target>
  /weather <city>  /translate <lang> <text>  /define <word>  /summarize <url>
  /grep <pattern> [path]  /git <subcommand> [args]  /remind <time> <message>
  /recall <query>  /teleport

Runes (SIMD file analysis) — drop a file in the web UI, or run one directly:
  /rune <name> [args] <path>   — in the REPL (run /rune alone to list all)
  olorin rune <name> …         — one-shot CLI, clean stdout for piping

Supported files:
{formats}
{mode_line}
Agent: Olorin")
    }

}

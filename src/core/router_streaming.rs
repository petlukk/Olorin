//! Streaming variant of the Olorin Pipe — extracted from `router.rs`
//! to keep both files under the 500-line cap.

use crate::core::handlers;
use crate::core::router::{DispatchContext, StreamEvent, STRICT_REFUSAL};
use crate::core::safety;

impl DispatchContext {
    /// Streaming dispatch — tokens flow via `tx` as generated.
    pub fn dispatch_streaming(
        &mut self,
        input: &str,
        tx: std::sync::mpsc::Sender<StreamEvent>,
    ) {
        let input = input.trim();
        if input.is_empty() {
            let _ = tx.send(StreamEvent::Done { full_text: String::new() });
            return;
        }

        let audit_turn = self.audit_input(input);
        let audit_start = std::time::Instant::now();

        let (cmd_id, _) = crate::core::dispatch::match_command(input.as_bytes());
        if cmd_id == crate::core::dispatch::CMD_TELEPORT {
            self.audit_result(audit_turn, audit_start, "teleport", &[]);
            crate::interface::whatsapp::teleport_loop_streaming(self, tx);
            return;
        }

        let recall_context = match self.pre_inference(input) {
            Err(resp) => {
                if resp.blocked {
                    self.audit_result(audit_turn, audit_start, "blocked", &[]);
                    let _ = tx.send(StreamEvent::Error(resp.text.clone()));
                    let _ = tx.send(StreamEvent::Done { full_text: resp.text });
                    return;
                }
                let _ = tx.send(StreamEvent::Token(resp.text.clone()));
                if let Some(followup) = resp.followup {
                    if self.strict {
                        self.audit_result(audit_turn, audit_start, "rune_strict_no_narration", &[]);
                        let _ = tx.send(StreamEvent::Done { full_text: resp.text });
                        return;
                    }
                    self.audit_result(audit_turn, audit_start, "rune_with_narration",
                        &[("narration", crate::core::audit::AuditValue::Bool(true))]);
                    self.run_followup_streaming(input, &resp.text, &followup, &tx);
                    return;
                }
                self.audit_result(audit_turn, audit_start, "command", &[]);
                let _ = tx.send(StreamEvent::Done { full_text: resp.text });
                return;
            }
            Ok(ctx) => ctx,
        };

        if self.strict {
            self.audit_result(audit_turn, audit_start, "strict_refused", &[]);
            let msg = STRICT_REFUSAL.to_string();
            let _ = tx.send(StreamEvent::Token(msg.clone()));
            let _ = tx.send(StreamEvent::Done { full_text: msg });
            return;
        }

        self.audit_result(audit_turn, audit_start, "llm_start", &[]);
        self.messages.push(handlers::user_message(input));

        let system = self.system_prompt.clone();
        let prompt = match &recall_context {
            Some(ctx) => format!("{ctx}\n\n{}", self.last_user_text()),
            None => self.last_user_text(),
        };

        if let Some(engine) = &mut self.engine {
            let tx_ref = tx.clone();

            let on_event = move |ev: crate::inference::generate::GenEvent| match ev {
                crate::inference::generate::GenEvent::Token(token_text) => {
                    if safety::is_chatml_hallucination(token_text) {
                        let _ = tx_ref.send(StreamEvent::Error(format!(
                            "[safety: dropped prompt-header token '{}']",
                            token_text.escape_debug()
                        )));
                        return;
                    }
                    let _ = tx_ref.send(StreamEvent::Token(token_text.to_string()));
                }
                crate::inference::generate::GenEvent::Thinking(active) => {
                    let _ = tx_ref.send(StreamEvent::Thinking(active));
                }
            };

            match engine.generate(&prompt, &system, &on_event) {
                Ok(first_text) => {
                    if safety::scan_outbound(first_text.as_bytes()).blocked {
                        let _ = tx.send(StreamEvent::Error(
                            "Response blocked: potential secret leak.".to_string(),
                        ));
                        let _ = tx.send(StreamEvent::Done { full_text: String::new() });
                        return;
                    }
                    let final_text = self.run_local_followup_if_tool_call(
                        input, &system, &first_text, Some(&tx),
                    ).unwrap_or(first_text);
                    self.finalize_response(input, &final_text);
                    let _ = tx.send(StreamEvent::Done { full_text: final_text });
                    return;
                }
                Err(e) => eprintln!("[olorin] local inference failed: {e}"),
            }
        }

        if let Some(client) = &self.anthropic {
            let system = self.system_prompt.clone();
            let owned = self.build_cloud_messages();
            let msg_pairs: Vec<(&str, &str)> = owned.iter()
                .map(|(r, t)| (r.as_str(), t.as_str())).collect();

            match client.generate(&system, &msg_pairs) {
                Ok(first_text) => {
                    if safety::scan_outbound(first_text.as_bytes()).blocked {
                        let _ = tx.send(StreamEvent::Error(
                            "Response blocked: potential secret leak.".to_string(),
                        ));
                        let _ = tx.send(StreamEvent::Done { full_text: String::new() });
                        return;
                    }
                    let final_text = self.maybe_handle_tool_call_cloud(input, first_text);
                    let _ = tx.send(StreamEvent::Token(final_text.clone()));
                    self.finalize_response(input, &final_text);
                    let _ = tx.send(StreamEvent::Done { full_text: final_text });
                    return;
                }
                Err(e) => {
                    // Backend IS configured — surface the real failure instead of
                    // masking it as the generic "no backend" message below.
                    eprintln!("[olorin] cloud inference failed: {e}");
                    let msg = format!("Cloud inference failed: {e}");
                    let _ = tx.send(StreamEvent::Error(msg));
                    let _ = tx.send(StreamEvent::Done { full_text: String::new() });
                    return;
                }
            }
        }

        let msg = "No LLM backend available. Load a model or set ANTHROPIC_API_KEY.".to_string();
        let _ = tx.send(StreamEvent::Error(msg));
        let _ = tx.send(StreamEvent::Done { full_text: String::new() });
    }

    /// Stream a model narration after a rune's kernel output. Persists
    /// the rune turn + narration to messages/vault on success.
    pub(crate) fn run_followup_streaming(
        &mut self,
        input: &str,
        rune_text: &str,
        prompt: &str,
        tx: &std::sync::mpsc::Sender<StreamEvent>,
    ) {
        let Some(engine) = self.engine.as_mut() else {
            // No local model — narrate via the cloud client if one is configured.
            self.run_followup_cloud(input, rune_text, prompt, tx);
            return;
        };
        // The narration system prompt is load-bearing: A/B on the Pi
        // (2026-06-11) with it removed produced ~90s of free association
        // that the discard filters ate, every run. Unlike the chat case,
        // there is no fat to trim here.
        let system = crate::core::router_tools::NARRATION_SYSTEM_PROMPT;
        let prompt_tokens = engine.count_prompt_tokens(prompt, system);
        let cap = crate::core::router_tools::NARRATION_MAX_PROMPT_TOKENS;
        if prompt_tokens > cap {
            let notice = format!(
                "\n\n[narration skipped: prompt is {prompt_tokens} tokens, over the {cap}-token narration budget]"
            );
            let _ = tx.send(StreamEvent::Token(notice.clone()));
            let _ = tx.send(StreamEvent::Done { full_text: format!("{rune_text}{notice}") });
            return;
        }

        // Buffer the narration instead of streaming it live: a grid-continuation
        // (the model emitting a data row instead of a summary) must be suppressed
        // before any of it reaches the user, and a streamed token can't be
        // recalled. Narration runs thinking-off (see below), so there's no
        // chain-of-thought phase to surface — the Thinking handler is a no-op here.
        let buf = std::cell::RefCell::new(String::new());
        let on_event = |ev: crate::inference::generate::GenEvent| match ev {
            crate::inference::generate::GenEvent::Token(t) => {
                if !safety::is_chatml_hallucination(t) {
                    buf.borrow_mut().push_str(t);
                }
            }
            crate::inference::generate::GenEvent::Thinking(active) => {
                let _ = tx.send(StreamEvent::Thinking(active));
            }
        };

        let prior_max = engine.max_tokens;
        let prior_thinking = engine.thinking;
        engine.max_tokens = crate::core::router_tools::NARRATION_DECODE_TOKEN_CAP;
        engine.thinking = false; // no chain-of-thought for restating a rune result
        let _ = engine.generate(prompt, system, &on_event);
        engine.max_tokens = prior_max;
        engine.thinking = prior_thinking;
        let narration = buf.into_inner();
        self.finalize_narration(input, rune_text, prompt, &narration, tx);
    }

    /// Narrate a rune's kernel output via the Anthropic cloud client when no
    /// local engine is loaded. Falls back to the bare kernel output if no
    /// client is configured or the cloud call fails — the rune output has
    /// already streamed, so a missing narration just leaves it standing alone.
    fn run_followup_cloud(
        &mut self,
        input: &str,
        rune_text: &str,
        prompt: &str,
        tx: &std::sync::mpsc::Sender<StreamEvent>,
    ) {
        let system = crate::core::router_tools::NARRATION_SYSTEM_PROMPT;
        let narration = match &self.anthropic {
            Some(client) => match client.generate(system, &[("user", prompt)]) {
                Ok(text) => text,
                Err(e) => {
                    eprintln!("[olorin] cloud narration failed: {e}");
                    let _ = tx.send(StreamEvent::Done { full_text: rune_text.to_string() });
                    return;
                }
            },
            None => {
                let _ = tx.send(StreamEvent::Done { full_text: rune_text.to_string() });
                return;
            }
        };
        self.finalize_narration(input, rune_text, prompt, &narration, tx);
    }

    /// Apply the narration discard filters and, on a usable narration, stream
    /// it and persist the rune turn. Shared by the local-engine and cloud
    /// narration paths so suppression stays identical across backends.
    fn finalize_narration(
        &mut self,
        input: &str,
        rune_text: &str,
        prompt: &str,
        narration: &str,
        tx: &std::sync::mpsc::Sender<StreamEvent>,
    ) {
        let trimmed = narration.trim();
        // Empty, grid-continuation, or a reformatted data dump (the multi-file
        // failure mode) → discard the narration; the kernel output stands alone.
        let empty = trimmed.is_empty();
        let grid = !empty && crate::runes::narration::is_grid_continuation(prompt, trimmed);
        let dump = !empty && !grid && crate::runes::narration::looks_like_data_dump(trimmed);
        if empty || grid || dump {
            if std::env::var_os("OLORIN_DEBUG_NARRATION").is_some() {
                eprintln!(
                    "[narration] DISCARDED (empty={empty} grid={grid} dump={dump}), {} bytes:\n{trimmed}",
                    trimmed.len(),
                );
            }
            let _ = tx.send(StreamEvent::Done { full_text: rune_text.to_string() });
            return;
        }

        let _ = tx.send(StreamEvent::Token("\n\n".to_string()));
        let _ = tx.send(StreamEvent::Token(trimmed.to_string()));
        self.messages.push(handlers::user_message(input));
        self.messages.push(handlers::assistant_message(trimmed));
        self.vault_save(b"assistant", trimmed.as_bytes());
        let _ = tx.send(StreamEvent::Done { full_text: format!("{rune_text}\n\n{trimmed}") });
    }
}

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
                Err(e) => eprintln!("[olorin] cloud inference failed: {e}"),
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
            let _ = tx.send(StreamEvent::Done { full_text: rune_text.to_string() });
            return;
        };
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

        let _ = tx.send(StreamEvent::Token("\n\n".to_string()));

        let tx_ref = tx.clone();
        let on_event = move |ev: crate::inference::generate::GenEvent| match ev {
            crate::inference::generate::GenEvent::Token(t) => {
                if safety::is_chatml_hallucination(t) { return; }
                let _ = tx_ref.send(StreamEvent::Token(t.to_string()));
            }
            crate::inference::generate::GenEvent::Thinking(active) => {
                let _ = tx_ref.send(StreamEvent::Thinking(active));
            }
        };

        let prior_max = engine.max_tokens;
        engine.max_tokens = crate::core::router_tools::NARRATION_DECODE_TOKEN_CAP;
        let narration = engine.generate(prompt, system, &on_event).unwrap_or_default();
        engine.max_tokens = prior_max;
        let trimmed = narration.trim();
        if !trimmed.is_empty() {
            self.messages.push(handlers::user_message(input));
            self.messages.push(handlers::assistant_message(trimmed));
            self.vault_save(b"assistant", trimmed.as_bytes());
        }
        let mut full = String::with_capacity(rune_text.len() + trimmed.len() + 2);
        full.push_str(rune_text);
        full.push_str("\n\n");
        full.push_str(trimmed);
        let _ = tx.send(StreamEvent::Done { full_text: full });
    }
}

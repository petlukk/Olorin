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

    /// File-drop analyst (single file): the user dropped a file, so the
    /// "analyze this" decision is already made — pick the rune deterministically
    /// (no autonomous model tool-call, which is what makes this reliable on the
    /// Pi), run it, stream the kernel output, then narrate via the proven
    /// narration path. `tmp_path` MUST be under the rune path allowlist (~ or
    /// /tmp); the rune enforces it on read.
    pub fn analyze_file_streaming(
        &mut self,
        display_name: &str,
        tmp_path: &str,
        tx: &std::sync::mpsc::Sender<StreamEvent>,
    ) {
        // Sniff a bounded prefix to choose the rune (extension + timestamp).
        let prefix = read_prefix(tmp_path, 8192);
        let Some(rune) = crate::runes::select::pick_rune(display_name, &prefix) else {
            let msg = format!(
                "I don't have a rune that analyzes '{display_name}'. I can summarize \
                 CSVs, JSON Lines, Parquet, and log files."
            );
            let _ = tx.send(StreamEvent::Token(msg.clone()));
            let _ = tx.send(StreamEvent::Done { full_text: msg });
            return;
        };

        let name = rune.name();
        let flags = crate::runes::select::default_args(name);
        let args = if flags.is_empty() {
            tmp_path.to_string()
        } else {
            format!("{flags} {tmp_path}")
        };
        let result = rune.run(&args);
        let safety_class = rune.output_safety();
        let answer = result.answer.clone();
        let timing_us = result.timing_us;
        let structured = result.structured;

        // User-visible kernel output (mirrors handle_rune's body shape).
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
            let _ = tx.send(StreamEvent::Error(
                "Analysis output blocked by safety scan.".to_string(),
            ));
            let _ = tx.send(StreamEvent::Done { full_text: String::new() });
            return;
        }

        let header = format!("📎 ran `{name}` on {display_name}\n\n");
        let descriptor = format!("analyze {display_name} ({name})");
        self.vault_save(b"user", descriptor.as_bytes());
        self.vault_save(b"tool", body.as_bytes());

        // Stream the kernel output, then narrate it.
        let _ = tx.send(StreamEvent::Token(header.clone()));
        let _ = tx.send(StreamEvent::Token(body.clone()));

        let rune_text = format!("{header}{body}");
        let scratch = crate::runes::RuneResult {
            answer, details: None, success: result.success, timing_us, structured,
        };
        match crate::runes::build_narration_prompt(name, safety_class, scratch) {
            Some(prompt) => self.run_followup_streaming(&descriptor, &rune_text, &prompt, tx),
            None => { let _ = tx.send(StreamEvent::Done { full_text: rune_text }); }
        }
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

        // Buffer the narration instead of streaming it live: a grid-continuation
        // (the model emitting a data row instead of a summary) must be suppressed
        // before any of it reaches the user, and a streamed token can't be
        // recalled. Thinking state still flows so the UI shows progress.
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
        engine.max_tokens = crate::core::router_tools::NARRATION_DECODE_TOKEN_CAP;
        let _ = engine.generate(prompt, system, &on_event);
        engine.max_tokens = prior_max;
        let narration = buf.into_inner();
        let trimmed = narration.trim();

        // Empty or grid-continuation → discard the narration; kernel output
        // stands alone, exactly as it did under the old length skip.
        if trimmed.is_empty()
            || crate::runes::narration::is_grid_continuation(prompt, trimmed)
        {
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

/// Read up to `max` bytes from a file for content sniffing. Returns an empty
/// vec on any error — `pick_rune` then falls back to extension-only routing.
fn read_prefix(path: &str, max: usize) -> Vec<u8> {
    use std::io::Read;
    let Ok(mut f) = std::fs::File::open(path) else { return Vec::new() };
    let mut buf = vec![0u8; max];
    match f.read(&mut buf) {
        Ok(n) => { buf.truncate(n); buf }
        Err(_) => Vec::new(),
    }
}

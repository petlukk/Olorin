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

    /// File-drop analyst (single file). Thin wrapper over the multi-file path.
    /// The user dropped a file, so the "analyze this" decision is already made —
    /// no autonomous model tool-call, which is what makes this reliable on the
    /// Pi. `tmp_path` MUST be under the rune path allowlist (~ or /tmp).
    pub fn analyze_file_streaming(
        &mut self,
        display_name: &str,
        tmp_path: &str,
        tx: &std::sync::mpsc::Sender<StreamEvent>,
    ) {
        self.analyze_files_streaming(&[(display_name.to_string(), tmp_path.to_string())], tx);
    }

    /// File-drop analyst (one or more files): pick + run each file's rune
    /// deterministically, stream every kernel output, then narrate — a single
    /// summary for one file, or one correlation pass across the combined
    /// compact answers for several. The correlation step is reasoning over
    /// already-computed results (i.e. narration), which works on the Pi.
    pub fn analyze_files_streaming(
        &mut self,
        files: &[(String, String)],
        tx: &std::sync::mpsc::Sender<StreamEvent>,
    ) {
        if files.is_empty() {
            let _ = tx.send(StreamEvent::Done { full_text: String::new() });
            return;
        }
        let descriptor = if files.len() == 1 {
            format!("analyze {}", files[0].0)
        } else {
            format!("analyze {} files", files.len())
        };
        self.vault_save(b"user", descriptor.as_bytes());

        let mut runs: Vec<FileRun> = Vec::new();
        let mut rune_text = String::new();
        for (i, (name, path)) in files.iter().enumerate() {
            if i > 0 {
                let _ = tx.send(StreamEvent::Token("\n\n".to_string()));
                rune_text.push_str("\n\n");
            }
            if let Some(run) = self.run_file_rune(name, path, tx) {
                rune_text.push_str(&run.streamed);
                runs.push(run);
            }
        }

        // No rune produced output (all skipped/blocked) — nothing to narrate.
        if runs.is_empty() {
            let _ = tx.send(StreamEvent::Done { full_text: rune_text });
            return;
        }

        // One file → single-rune narration; several → one correlation pass.
        if runs.len() == 1 {
            let r = runs.pop().unwrap();
            let scratch = crate::runes::RuneResult {
                answer: r.answer, details: None,
                success: r.success, timing_us: r.timing_us, structured: r.structured,
            };
            match crate::runes::build_narration_prompt(r.rune, r.safety, scratch) {
                Some(prompt) => self.run_followup_streaming(&descriptor, &rune_text, &prompt, tx),
                None => { let _ = tx.send(StreamEvent::Done { full_text: rune_text }); }
            }
        } else {
            let prompt = correlation_prompt(&runs);
            self.run_followup_streaming(&descriptor, &rune_text, &prompt, tx);
        }
    }

    /// Pick + run the rune for one staged file and stream its kernel output.
    /// Returns the result for the narration step, or None if no rune matched or
    /// the output was blocked (a notice is streamed in those cases).
    fn run_file_rune(
        &mut self,
        display_name: &str,
        tmp_path: &str,
        tx: &std::sync::mpsc::Sender<StreamEvent>,
    ) -> Option<FileRun> {
        let prefix = read_prefix(tmp_path, 8192);
        let Some(rune) = crate::runes::select::pick_rune(display_name, &prefix) else {
            let _ = tx.send(StreamEvent::Token(format!(
                "(no rune matched {display_name} — I can analyze CSV, JSON Lines, \
                 Parquet, and log files.)"
            )));
            return None;
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
            return None;
        }

        self.vault_save(b"tool", body.as_bytes());
        let header = format!("📎 ran `{name}` on {display_name}\n\n");
        let _ = tx.send(StreamEvent::Token(header.clone()));
        let _ = tx.send(StreamEvent::Token(body.clone()));

        Some(FileRun {
            display: display_name.to_string(),
            rune: name,
            safety: safety_class,
            answer,
            success: result.success,
            timing_us,
            structured,
            streamed: format!("{header}{body}"),
        })
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

/// One file's analysis, carried from the per-file rune run to the narration
/// step. `streamed` is the header+body already sent to the client (for the
/// final `Done` full_text); `answer` is the compact summary fed to the model.
struct FileRun {
    display: String,
    rune: &'static str,
    safety: crate::runes::OutputSafety,
    answer: String,
    success: bool,
    timing_us: u64,
    structured: bool,
    streamed: String,
}

/// Build the cross-file correlation narration prompt from the compact answers.
/// Data-then-question shape (what Gemma 4 narrates best), kept tiny so several
/// files still fit the narration token budget.
fn correlation_prompt(runs: &[FileRun]) -> String {
    let mut p = format!("I ran SIMD analysis tools on {} files:\n\n", runs.len());
    for r in runs {
        p.push_str(&format!("{} (via {}):\n{}\n\n", r.display, r.rune, r.answer));
    }
    p.push_str(
        "In 1-2 plain sentences, tell me what stands out across these files — \
         which one looks anomalous and roughly when.",
    );
    p
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

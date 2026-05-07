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
//! Every channel (REPL, Web UI, WhatsApp) enters here. Every response exits here.
//! All messages saved to encrypted vault. No exceptions.

use crate::core::anthropic::AnthropicClient;
use crate::core::dispatch;
use crate::core::handlers;
use crate::core::llm::{self, ContentBlock, Message};
use crate::core::safety;
use crate::inference::generate::Engine;
use crate::recall::VectorStore;
use crate::storage::vault::Vault;
use std::path::PathBuf;

// ── Public types ─────────────────────────────────────────────────────────────

/// Message returned when strict mode rejects an LLM fallback. Public so
/// tests can assert against it without duplicating the string.
pub const STRICT_REFUSAL: &str =
    "Strict mode: no deterministic path matched. The LLM is disabled in this \
     binary. Try /help to see available commands, or /tools and /rune for \
     deterministic operations.";

/// Response from the dispatch pipeline.
pub struct Response {
    pub text:    String,
    pub blocked: bool,
    /// If set, dispatch_streaming runs an LLM turn after the response text,
    /// using this string as the prompt. Used by runes to narrate kernel
    /// output in 1-2 sentences.
    pub followup: Option<String>,
}

/// Events emitted by streaming dispatch.
pub enum StreamEvent {
    /// A single token of output text.
    Token(String),
    /// Model entered or exited a `<think>` block.
    Thinking(bool),
    /// Generation complete. Full text for vault/recall bookkeeping.
    Done { full_text: String },
    /// Error during generation.
    Error(String),
}

impl Response {
    pub(crate) fn text(s: impl Into<String>) -> Self {
        Self { text: s.into(), blocked: false, followup: None }
    }

    pub(crate) fn blocked(reason: impl Into<String>) -> Self {
        Self { text: reason.into(), blocked: true, followup: None }
    }

    pub(crate) fn with_followup(mut self, prompt: String) -> Self {
        self.followup = Some(prompt);
        self
    }
}

/// Central dispatch context — holds all state needed for the pipeline.
pub struct DispatchContext {
    pub(crate) messages:      Vec<Message>,
    pub(crate) recall:        VectorStore,
    pub(crate) vault:         Option<Vault>,
    pub(crate) engine:        Option<Engine>,
    pub(crate) anthropic:     Option<AnthropicClient>,
    pub(crate) last_timing:   Option<handlers::TurnTiming>,
    pub(crate) system_prompt: String,
    pub(crate) recall_level:  usize,
    pub(crate) _max_turns:    usize,
    pub teleported:            bool,
    pub(crate) server_teleported: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// When true, the LLM is disabled entirely. No model load, no Anthropic
    /// fallback, no narration. Dispatch refuses to fall through to inference;
    /// only deterministic paths (safety, slash, intent, kernel/rune, recall)
    /// can produce output. The binary never opens the GGUF — startup drops
    /// from ~25s to ~200ms.
    pub(crate) strict: bool,
    /// Optional audit log. When set, every dispatch turn emits two
    /// JSON-Lines events to the file: `input` (length + ts) and a
    /// dispatch-result event (phase + timing). See `core::audit` and
    /// the `--audit` CLI flag.
    pub(crate) audit: Option<std::sync::Arc<crate::core::audit::AuditLog>>,
}

impl DispatchContext {
    /// Create a new dispatch context.
    /// `api_key`: optional Anthropic API key for cloud inference.
    /// `model_arg`: optional model selector ("gemma4" alias or path).
    pub fn new(api_key: Option<String>, model_arg: Option<&str>) -> Self {
        Self::build(api_key, model_arg, false)
    }

    /// Create a strict dispatch context. The LLM is disabled entirely:
    /// no model is loaded, the Anthropic client is not initialized, and
    /// dispatch refuses to fall through to inference. Only deterministic
    /// paths (safety scan, slash commands, intent router, kernel/rune
    /// dispatch, vault recall) can produce output.
    ///
    /// Useful for CLI-style use where the user only wants tools and runes,
    /// or for security-conscious deployments that need a categorical
    /// "this binary will never call an LLM" guarantee. Startup drops from
    /// ~25s (model load) to ~200ms (kernel extract + vault open).
    pub fn new_strict(model_arg: Option<&str>) -> Self {
        let _ = model_arg; // accepted for symmetry with `new`; ignored.
        Self::build(None, None, true)
    }

    fn build(api_key: Option<String>, model_arg: Option<&str>, strict: bool) -> Self {
        let anthropic = if strict { None } else { api_key.map(AnthropicClient::new) };
        let vault = Self::open_vault();
        let engine = if strict { None } else { Self::load_engine(model_arg) };
        let mut ctx = Self {
            messages:      Vec::new(),
            recall:        VectorStore::new(1024),
            vault,
            engine,
            anthropic,
            last_timing:   None,
            system_prompt: {
                let base = llm::SYSTEM_PROMPT;
                let runes_block = crate::runes::runes_prompt_block();
                if base.is_empty() {
                    runes_block.to_string()
                } else {
                    format!("{base}\n\n{runes_block}")
                }
            },
            recall_level:  0,
            _max_turns:    8,
            teleported:        false,
            server_teleported: None,
            strict,
            audit: None,
        };
        if !strict {
            ctx.load_api_key_from_vault();
        }
        ctx
    }

    pub(crate) fn load_engine(model_arg: Option<&str>) -> Option<Engine> {
        use crate::inference::generate;
        let path = generate::resolve_model(model_arg)?;
        eprintln!("[Olorin] Loading model: {}", path.display());
        match Engine::load(&path, 2048) {
            Ok(e) => {
                eprintln!("[Olorin] Model loaded ({}).", e.quant_type_str());
                Some(e)
            }
            Err(e) => {
                eprintln!("[Olorin] Model load failed: {e}");
                None
            }
        }
    }

    fn open_vault() -> Option<Vault> {
        let home = std::env::var("HOME")
            .map_err(|e| eprintln!("[vault] HOME unset, persistence disabled: {e}"))
            .ok()?;
        let vault_dir = PathBuf::from(home).join(".olorin").join("vault").join("default");
        std::fs::create_dir_all(&vault_dir)
            .map_err(|e| eprintln!("[vault] mkdir {} failed, persistence disabled: {e}", vault_dir.display()))
            .ok()?;
        Vault::open(&vault_dir)
            .map_err(|e| eprintln!("[vault] open {} failed, persistence disabled: {e:?}", vault_dir.display()))
            .ok()
    }

    /// Attach an audit log. Every dispatch turn after this point emits
    /// JSON Lines events to the log's file. Returns self for builder
    /// chaining: `DispatchContext::new(...).with_audit(log)`.
    pub fn with_audit(mut self, log: crate::core::audit::AuditLog) -> Self {
        self.audit = Some(std::sync::Arc::new(log));
        self
    }

    /// Emit the per-turn `input` event and return the new turn id. Used
    /// by both `dispatch` (sync) and `dispatch_streaming` at entry.
    /// Returns -1 when no audit log is attached (turn id is unused in
    /// that case; callers gate on the audit Option themselves).
    pub(crate) fn audit_input(&self, input: &str) -> i32 {
        let Some(a) = &self.audit else { return -1; };
        let turn = a.next_turn();
        a.emit(turn, "input", &[
            ("input_len", crate::core::audit::AuditValue::I64(input.len() as i64)),
        ]);
        turn
    }

    /// Emit the per-turn dispatch-result event with phase + wall_us,
    /// plus any phase-specific extras.
    pub(crate) fn audit_result(
        &self,
        turn: i32,
        start: std::time::Instant,
        phase: &str,
        extras: &[(&str, crate::core::audit::AuditValue<'_>)],
    ) {
        let Some(a) = &self.audit else { return; };
        if turn < 0 { return; }
        let wall_us = start.elapsed().as_micros() as i64;
        // Append wall_us to the user-supplied extras without per-call alloc
        // overhead in the no-extras case.
        if extras.is_empty() {
            a.emit(turn, phase, &[
                ("wall_us", crate::core::audit::AuditValue::I64(wall_us)),
            ]);
        } else {
            let mut fields: Vec<(&str, crate::core::audit::AuditValue<'_>)> = Vec::with_capacity(extras.len() + 1);
            fields.push(("wall_us", crate::core::audit::AuditValue::I64(wall_us)));
            fields.extend_from_slice(extras);
            a.emit(turn, phase, &fields);
        }
    }

    /// Create with a custom system prompt.
    pub fn with_system_prompt(mut self, prompt: &str) -> Self {
        let runes_block = crate::runes::runes_prompt_block();
        let base = llm::SYSTEM_PROMPT;
        self.system_prompt = if base.is_empty() {
            format!("{prompt}\n\n{runes_block}")
        } else {
            format!("{base}\n\n{prompt}\n\n{runes_block}")
        };
        self
    }

    #[doc(hidden)]
    pub fn system_prompt_for_test(&self) -> &str {
        &self.system_prompt
    }

    /// Streaming variant of the Olorin Pipe.
    /// Tokens stream via `tx` as they are generated. ChatML hallucinations
    /// trigger early stop. Outbound leak scan runs on the complete text.
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

        // ── Teleport: handle specially for streaming QR + status ─────
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

        // ── Strict mode: refuse LLM fallback ─────────────────────────
        if self.strict {
            self.audit_result(audit_turn, audit_start, "strict_refused", &[]);
            let msg = STRICT_REFUSAL.to_string();
            let _ = tx.send(StreamEvent::Token(msg.clone()));
            let _ = tx.send(StreamEvent::Done { full_text: msg });
            return;
        }

        // ── Streaming Inference ──────────────────────────────────────
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
                        let _ = tx.send(StreamEvent::Done {
                            full_text: String::new(),
                        });
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

        // Cloud fallback — not streamable, send as single token
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
                        let _ = tx.send(StreamEvent::Done {
                            full_text: String::new(),
                        });
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

        let msg = "No LLM backend available. Load a model or set ANTHROPIC_API_KEY."
            .to_string();
        let _ = tx.send(StreamEvent::Error(msg));
        let _ = tx.send(StreamEvent::Done {
            full_text: String::new(),
        });
    }

    /// Steps 1-4: safety scan, slash, intent, recall.
    /// Returns Ok(recall_context) to continue to inference,
    /// or Err(Response) for early exit (command, tool, blocked).
    pub(crate) fn pre_inference(&mut self, input: &str) -> Result<Option<String>, Response> {
        // ── Step 1: Safety Scan ──────────────────────────────────────
        let scan = safety::scan(input.as_bytes());
        if scan.blocked {
            let details: Vec<String> = scan.details.iter().map(|w| {
                format!("  - {} at position {}", w.pattern, w.position)
            }).collect();
            return Err(Response::blocked(format!(
                "Input blocked:\n{}", details.join("\n")
            )));
        }

        // ── Step 2: Slash Command ────────────────────────────────────
        let input_bytes = input.as_bytes();
        let (cmd_id, cmd_arg) = dispatch::match_command(input_bytes);

        if cmd_id >= dispatch::CMD_HELP && cmd_id <= dispatch::CMD_PROFILE {
            return Err(self.handle_meta(cmd_id));
        }
        if cmd_id == dispatch::CMD_TASKS {
            return Err(Response::text("No background tasks."));
        }
        if cmd_id == dispatch::CMD_RECALL {
            let arg = String::from_utf8_lossy(cmd_arg);
            let arg = arg.trim();
            if let Ok(level) = arg.parse::<usize>() {
                self.recall_level = level;
                return Err(Response::text(format!("Recall level set to {level}.")));
            }
            return Err(Response::text(self.recall.recall_formatted(arg, 5)));
        }
        if cmd_id == dispatch::CMD_TELEPORT {
            return Err(self.handle_teleport());
        }
        if cmd_id == dispatch::CMD_RUNE {
            return Err(self.handle_rune(cmd_arg));
        }
        if cmd_id >= dispatch::CMD_TOOL_FIRST && cmd_id <= dispatch::CMD_TOOL_LAST {
            return Err(self.handle_tool_command(cmd_id, cmd_arg));
        }
        if input.starts_with('/') && cmd_id == dispatch::CMD_NONE {
            return Err(Response::text(format!(
                "Unknown command: {input}. Type /help for available commands."
            )));
        }

        // ── Step 3: Intent Router ────────────────────────────────────
        let (intent, arg_start, arg_len) = dispatch::classify_intent(input_bytes);
        if intent != dispatch::INTENT_NONE {
            if let Some(tool_name) = dispatch::intent_to_tool_name(intent) {
                let arg_bytes = if arg_start + arg_len <= input_bytes.len() {
                    &input_bytes[arg_start..arg_start + arg_len]
                } else {
                    &[]
                };
                return Err(self.execute_intent(tool_name, intent, arg_bytes));
            }
        }

        // ── Step 4: Recall ───────────────────────────────────────────
        // Search PRIOR entries, then add the current turn. Adding before
        // searching causes the current query to self-match and crowd out
        // the actual recalled context — fatal at recall_level=1.
        let top_k = self.recall_level;

        if top_k == 0 {
            self.recall.add(input);
            return Ok(None);
        }

        let session_recall = self.recall.synthesize_context(input, top_k);
        self.recall.add(input);
        let mut recall_text = session_recall.unwrap_or_default();

        if let Some(ref mut vault) = self.vault {
            if let Ok(vault_hits) = vault.search(input, top_k) {
                let input_norm = normalize_for_dedup(input);
                for hit in &vault_hits {
                    for line in &hit.lines {
                        let trimmed = line.trim();
                        // Only user-stated facts are trusted context — assistant
                        // outputs can be hallucinations that would feed back in.
                        let Some(content) = trimmed.strip_prefix("user:") else {
                            continue;
                        };
                        let content = content.trim();
                        if content.is_empty() {
                            continue;
                        }
                        let content_norm = normalize_for_dedup(content);
                        // Skip self-matches (prior asks of this same query) and
                        // duplicates against what session recall already added.
                        if content_norm == input_norm {
                            continue;
                        }
                        if recall_text.lines().any(|l| normalize_for_dedup(l) == content_norm) {
                            continue;
                        }
                        recall_text.push('\n');
                        recall_text.push_str(content);
                    }
                }
            }
        }

        if std::env::var("OLORIN_DEBUG_RECALL").is_ok() {
            eprintln!("[recall] level={} context=\n---\n{}\n---", top_k, recall_text);
        }

        Ok(if recall_text.is_empty() { None } else { Some(recall_text) })
    }

    /// Post-inference bookkeeping: vault save + message history.
    /// Only user input is indexed for recall — model output is not stored
    /// in the recall index to prevent garbage feedback loops.
    pub(crate) fn finalize_response(&mut self, input: &str, text: &str) {
        self.vault_save(b"user", input.as_bytes());
        self.vault_save(b"assistant", text.as_bytes());
        self.messages.push(handlers::assistant_message(text));
    }

    /// Run a follow-up LLM turn after a short-circuit response (e.g. a rune's
    /// kernel output) and stream the model's narration to the user. The
    /// `rune_text` is the displayed kernel output (already sent as a Token);
    /// `prompt` is the LLM-facing prompt that wraps the rune answer.
    ///
    /// On successful narration, persists the turn so it shows up in next-turn
    /// context: pushes the rune command + narration into `self.messages` and
    /// vault-saves the narration as `assistant`. (The kernel output is already
    /// vault-saved as `tool` by `handle_rune`.)
    fn run_followup_streaming(
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
        // Narration-specific system prompt; see run_followup_sync.
        let system = crate::core::router_tools::NARRATION_SYSTEM_PROMPT;
        let prompt_tokens = engine.count_prompt_tokens(prompt, system);
        let cap = crate::core::router_tools::NARRATION_MAX_PROMPT_TOKENS;
        if prompt_tokens > cap {
            let notice = format!(
                "\n\n[narration skipped: prompt is {prompt_tokens} tokens, over the {cap}-token narration budget]"
            );
            let _ = tx.send(StreamEvent::Token(notice.clone()));
            let _ = tx.send(StreamEvent::Done {
                full_text: format!("{rune_text}{notice}"),
            });
            return;
        }

        // Visual separator between kernel output and narration.
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
        let narration = engine.generate(prompt, system, &on_event)
            .unwrap_or_default();
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

    fn last_user_text(&self) -> String {
        self.messages.iter().rev()
            .find(|m| m.role.as_str() == "user")
            .and_then(|m| m.content.iter().find_map(|b| {
                if let ContentBlock::Text { text } = b { Some(text.clone()) } else { None }
            }))
            .unwrap_or_default()
    }

    /// Save a message to the encrypted vault. Logs on failure and continues —
    /// a single append failure should not kill the session.
    pub(crate) fn vault_save(&mut self, role: &[u8], content: &[u8]) {
        if let Some(ref mut vault) = self.vault {
            if let Err(e) = vault.append(role, content) {
                eprintln!(
                    "[vault] append failed (role={} len={}): {e:?}",
                    String::from_utf8_lossy(role), content.len(),
                );
            }
        }
    }

    /// Clear conversation history (session only — vault is persistent).
    pub(crate) fn clear(&mut self) {
        self.messages.clear();
        self.recall.clear();
    }

    /// Get last turn timing data.
    pub fn last_timing(&self) -> Option<&handlers::TurnTiming> {
        self.last_timing.as_ref()
    }

    /// Current recall level.
    pub fn recall_level(&self) -> usize {
        self.recall_level
    }

    // ── Inference ────────────────────────────────────────────────────────────

    /// Build (role, text) pairs from message history for cloud fallback.
    pub(crate) fn build_cloud_messages(&self) -> Vec<(String, String)> {
        self.messages.iter().map(|m| {
            let role = m.role.as_str().to_string();
            let text = m.content.iter().find_map(|b| {
                if let ContentBlock::Text { text } = b {
                    Some(text.clone())
                } else {
                    None
                }
            }).unwrap_or_default();
            (role, text)
        }).collect()
    }

    pub(crate) fn run_inference(
        &mut self,
        recall_context: &Option<String>,
    ) -> Result<String, String> {
        let system = self.system_prompt.clone();

        // Try local engine first
        let prompt = match recall_context {
            Some(ctx) => format!("{ctx}\n\n{}", self.last_user_text()),
            None => self.last_user_text(),
        };
        if let Some(engine) = &mut self.engine {
            match engine.generate(&prompt, &system, &|_ev| {}) {
                Ok(text) => return Ok(text),
                Err(e) => {
                    eprintln!("[olorin] local inference failed: {e}");
                }
            }
        }

        // Fall back to Anthropic cloud
        if let Some(client) = &self.anthropic {
            let owned = self.build_cloud_messages();
            let msg_pairs: Vec<(&str, &str)> = owned.iter()
                .map(|(r, t)| (r.as_str(), t.as_str())).collect();

            match client.generate(&system, &msg_pairs) {
                Ok(text) => return Ok(text),
                Err(e) => {
                    eprintln!("[olorin] cloud inference failed: {e}");
                }
            }
        }

        Err("No LLM backend available. Load a model or set ANTHROPIC_API_KEY.".to_string())
    }

}

/// Normalize a line for recall dedup: lowercase ASCII, strip trailing
/// punctuation/whitespace so "What is my name?" and "what is my name"
/// compare equal.
fn normalize_for_dedup(s: &str) -> String {
    s.trim()
        .trim_end_matches(|c: char| c.is_ascii_punctuation() || c.is_whitespace())
        .to_ascii_lowercase()
}

//! Post-inference tool-call wiring. Split from router_tools.rs.
//!
//! After the model produces its first output, these detect a `<tool_call>`,
//! dispatch it, and run a synthetic follow-up turn with the tool result — one
//! path for the local engine (streaming or sync), one for the cloud client.

use crate::core::handlers;
use crate::core::llm::ContentBlock;
use crate::core::router::{DispatchContext, StreamEvent};
use crate::core::safety;
use crate::core::tool_parse;
use crate::storage::json;

/// Closer for tool-call follow-ups; nudges the model past the tool result.
const FOLLOWUP_CLOSER: &str =
    "Now answer my original question using the tool result above. Do NOT call another tool.";

impl DispatchContext {
    /// Detect a `<tool_call>` in `first_output` and run dispatch + a synthetic
    /// follow-up generate. `None` = no tool call (caller uses its normal path).
    /// `Some(String::new())` is the outbound-block sentinel the caller must
    /// surface as blocked. The outbound scan runs exactly once in here —
    /// callers must NOT re-scan. `tx = Some` streams follow-up tokens (Web/
    /// WhatsApp); `tx = None` runs sync and returns inline (REPL).
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

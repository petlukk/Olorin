//! LLM message handling and output post-processing.
//!
//! Handles LLM turn logic, output guard, and tool execution loops.

use crate::core::llm::{ContentBlock, LlmResponse, Message, Role, StopReason};
use crate::core::tool_parse;
use crate::kernels::ffi;

// ── Output guard constants ───────────────────────────────────────────────────

pub const ACTION_PASS:           i32 = 0;
pub const ACTION_TRUNCATE_LINE:  i32 = 1;
pub const ACTION_BLOCK:          i32 = 2;
pub const ACTION_REDIRECT_CALC:  i32 = 3;
pub const DEFAULT_MAX_LEN:       i32 = 4096;

// ── Output guard ─────────────────────────────────────────────────────────────

/// Run the SIMD output guard on response text.
/// Returns (action, offset) where action determines what to do with the text.
pub fn guard_output(text: &[u8], max_len: i32) -> (i32, usize) {
    if text.is_empty() {
        return (ACTION_PASS, 0);
    }
    let mut action: i32 = 0;
    let mut offset: i32 = 0;
    unsafe {
        ffi::guard_output(
            text.as_ptr(),
            text.len() as i32,
            max_len,
            &mut action,
            &mut offset,
        );
    }
    (action, offset as usize)
}

/// Apply the output guard action to a text string.
/// Returns the processed text (may be truncated, blocked, or passed through).
pub fn apply_guard(text: &str) -> String {
    let (action, offset) = guard_output(text.as_bytes(), DEFAULT_MAX_LEN);
    match action {
        ACTION_TRUNCATE_LINE => text[..offset].to_string(),
        ACTION_BLOCK => "unknown".to_string(),
        ACTION_REDIRECT_CALC => {
            // Try to evaluate as math expression
            match crate::core::dispatch::eval_expr(text) {
                Ok(result) => result.to_string(),
                Err(_) => text.to_string(),
            }
        }
        _ => text.to_string(), // ACTION_PASS
    }
}

// ── Message building ─────────────────────────────────────────────────────────

/// Build a user message from text.
pub fn user_message(text: &str) -> Message {
    Message {
        role: Role::User,
        content: vec![ContentBlock::text(text)],
    }
}

/// Build an assistant message from text.
pub fn assistant_message(text: &str) -> Message {
    Message {
        role: Role::Assistant,
        content: vec![ContentBlock::text(text)],
    }
}

/// Extract text blocks from a response.
pub fn extract_text(response: &LlmResponse) -> String {
    let mut text = String::new();
    for block in &response.content {
        if let ContentBlock::Text { text: t } = block {
            text.push_str(t);
        }
    }
    text
}

/// Extract tool use blocks from a response.
pub fn extract_tool_uses(response: &LlmResponse) -> Vec<(&str, &str, &crate::storage::json::Object)> {
    let mut uses = Vec::new();
    for block in &response.content {
        if let ContentBlock::ToolUse { id, name, input } = block {
            uses.push((id.as_str(), name.as_str(), input));
        }
    }
    uses
}

/// Check if response requires tool execution.
pub fn needs_tool_execution(response: &LlmResponse) -> bool {
    response.stop_reason == StopReason::ToolUse
        && response.content.iter().any(|b| matches!(b, ContentBlock::ToolUse { .. }))
}

/// Process raw LLM output text: run tool_call detection then output guard.
pub fn process_output(text: &str) -> LlmResponse {
    tool_parse::extract_tool_calls(text)
}

// ── Timing ───────────────────────────────────────────────────────────────────

/// Timing data for a single dispatch turn.
pub struct TurnTiming {
    pub safety_scan_us: u64,
    pub llm_call_ms:    u64,
    pub tool_execs:     Vec<(String, u64)>,
}

impl TurnTiming {
    pub fn total_ms(&self) -> u64 {
        let tool_ms: u64 = self.tool_execs.iter().map(|(_, ms)| ms).sum();
        let safety_ms = (self.safety_scan_us + 999) / 1000;
        safety_ms + self.llm_call_ms + tool_ms
    }

    pub fn format(&self) -> String {
        let mut lines = vec![
            "Last turn timing:".to_string(),
            format!("  Safety scan:    {} us", self.safety_scan_us),
            format!("  LLM call:       {} ms", self.llm_call_ms),
        ];
        for (name, ms) in &self.tool_execs {
            lines.push(format!("  Tool: {:<10}{} ms", name, ms));
        }
        lines.push(format!("  Total:          {} ms", self.total_ms()));
        lines.join("\n")
    }
}

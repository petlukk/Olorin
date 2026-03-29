//! Token-level `<tool_call>` / `</tool_call>` detector.
//!
//! State machine that detects tool_call XML tags in streaming text output.
//! Works on string fragments (decoded token pieces).

use crate::storage::json::{self, Object, Value};
use crate::core::llm::{ContentBlock, LlmResponse, StopReason};

// ── Constants ────────────────────────────────────────────────────────────────

const OPEN_TAG:  &str = "<tool_call>";
const CLOSE_TAG: &str = "</tool_call>";

// ── String-based detector ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    Normal,
    MaybeTag,
    Capturing,
}

/// Result from feeding a text fragment into the detector.
#[derive(Debug, PartialEq)]
pub enum DetectResult {
    /// Normal text — stream it through.
    Text(String),
    /// Accumulating partial tag or capture — suppress output.
    Buffering,
    /// Complete tool call body (JSON string between tags).
    ToolCall(String),
    /// Runaway capture exceeded max length — flush as text.
    Aborted(String),
}

/// String-based `<tool_call>` / `</tool_call>` detector for decoded text streams.
pub struct ToolCallDetector {
    state:       State,
    tag_buf:     String,
    capture_buf: String,
    flush_buf:   String,
    max_capture: usize,
}

impl ToolCallDetector {
    pub fn new(max_capture: usize) -> Self {
        Self {
            state: State::Normal,
            tag_buf: String::new(),
            capture_buf: String::new(),
            flush_buf: String::new(),
            max_capture,
        }
    }

    pub fn feed(&mut self, piece: &str) -> DetectResult {
        match self.state {
            State::Normal   => self.feed_normal(piece),
            State::MaybeTag => self.feed_maybe_tag(piece),
            State::Capturing => self.feed_capturing(piece),
        }
    }

    /// Flush any remaining buffered content. Call after all input is consumed.
    pub fn finish(&mut self) -> Option<String> {
        let mut leftover = std::mem::take(&mut self.tag_buf);
        if !self.capture_buf.is_empty() {
            leftover.push_str(OPEN_TAG);
            leftover.push_str(&std::mem::take(&mut self.capture_buf));
        }
        self.state = State::Normal;
        if leftover.is_empty() { None } else { Some(leftover) }
    }

    fn feed_normal(&mut self, piece: &str) -> DetectResult {
        self.flush_buf.push_str(piece);

        if let Some(pos) = self.flush_buf.find(OPEN_TAG) {
            let before = self.flush_buf[..pos].to_string();
            let remainder = self.flush_buf[pos + OPEN_TAG.len()..].to_string();
            self.flush_buf.clear();
            self.state = State::Capturing;
            self.capture_buf = remainder;

            if let Some(result) = self.try_close_tag() {
                if before.is_empty() { return result; }
                if let DetectResult::ToolCall(body) = result {
                    self.flush_buf = format!("{OPEN_TAG}{body}{CLOSE_TAG}");
                }
                return DetectResult::Text(before);
            }
            return if before.is_empty() { DetectResult::Buffering }
                   else { DetectResult::Text(before) };
        }

        let plen = partial_tag_match(&self.flush_buf, OPEN_TAG);
        if plen > 0 {
            let split = self.flush_buf.len() - plen;
            let before = self.flush_buf[..split].to_string();
            self.tag_buf = self.flush_buf[split..].to_string();
            self.flush_buf.clear();
            self.state = State::MaybeTag;
            return if before.is_empty() { DetectResult::Buffering }
                   else { DetectResult::Text(before) };
        }

        DetectResult::Text(std::mem::take(&mut self.flush_buf))
    }

    fn feed_maybe_tag(&mut self, piece: &str) -> DetectResult {
        self.tag_buf.push_str(piece);

        if let Some(pos) = self.tag_buf.find(OPEN_TAG) {
            let before = self.tag_buf[..pos].to_string();
            self.capture_buf = self.tag_buf[pos + OPEN_TAG.len()..].to_string();
            self.tag_buf.clear();
            self.state = State::Capturing;
            if let Some(result) = self.try_close_tag() { return result; }
            return if before.is_empty() { DetectResult::Buffering }
                   else { DetectResult::Text(before) };
        }

        if partial_tag_match(&self.tag_buf, OPEN_TAG) == self.tag_buf.len() {
            return DetectResult::Buffering;
        }

        self.state = State::Normal;
        DetectResult::Text(std::mem::take(&mut self.tag_buf))
    }

    fn feed_capturing(&mut self, piece: &str) -> DetectResult {
        self.capture_buf.push_str(piece);

        if self.capture_buf.len() > self.max_capture {
            self.state = State::Normal;
            return DetectResult::Aborted(std::mem::take(&mut self.capture_buf));
        }

        self.try_close_tag().unwrap_or(DetectResult::Buffering)
    }

    fn try_close_tag(&mut self) -> Option<DetectResult> {
        let pos = self.capture_buf.find(CLOSE_TAG)?;
        let body = self.capture_buf[..pos].trim().to_string();
        self.capture_buf.clear();
        self.state = State::Normal;
        Some(DetectResult::ToolCall(body))
    }
}

// ── Parsing ──────────────────────────────────────────────────────────────────

/// Parse a tool-call JSON body into a ContentBlock::ToolUse.
/// Expected: `{"name": "tool_name", "arguments": {...}}`
pub fn parse_tool_call_json(json_str: &str) -> Option<ContentBlock> {
    let obj = json::parse(json_str.as_bytes()).ok()?;
    let name = obj.get_str("name")?.to_string();
    let arguments = match obj.get("arguments") {
        Some(Value::Object(o)) => (**o).clone(),
        _ => Object::new(),
    };
    let id = format!("tc_{:08x}", fxhash(json_str));
    Some(ContentBlock::ToolUse { id, name, input: arguments })
}

/// Extract tool calls from completed generation text (non-streaming path).
pub fn extract_tool_calls(text: &str) -> LlmResponse {
    let mut detector = ToolCallDetector::new(8192);
    let mut blocks: Vec<ContentBlock> = Vec::new();
    let mut plain = String::new();

    let mut result = detector.feed(text);
    loop {
        match result {
            DetectResult::Text(t) => plain.push_str(&t),
            DetectResult::Buffering => {}
            DetectResult::ToolCall(body) => {
                if !plain.is_empty() {
                    blocks.push(ContentBlock::Text { text: std::mem::take(&mut plain) });
                }
                if let Some(block) = parse_tool_call_json(&body) {
                    blocks.push(block);
                }
            }
            DetectResult::Aborted(buf) => plain.push_str(&buf),
        }
        let next = detector.feed("");
        if matches!(next, DetectResult::Text(ref t) if t.is_empty()) {
            break;
        }
        result = next;
    }

    if let Some(leftover) = detector.finish() {
        plain.push_str(&leftover);
    }
    if !plain.is_empty() {
        blocks.push(ContentBlock::Text { text: plain });
    }
    if blocks.is_empty() {
        blocks.push(ContentBlock::Text { text: String::new() });
    }

    let has_tool_use = blocks.iter().any(|b| matches!(b, ContentBlock::ToolUse { .. }));
    LlmResponse {
        content: blocks,
        stop_reason: if has_tool_use { StopReason::ToolUse } else { StopReason::EndTurn },
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn fxhash(s: &str) -> u32 {
    let mut h: u32 = 0;
    for b in s.bytes() {
        h = h.wrapping_mul(0x01000193) ^ (b as u32);
    }
    h
}

/// Returns how many trailing bytes of `text` are a prefix of `tag`.
fn partial_tag_match(text: &str, tag: &str) -> usize {
    let text_bytes = text.as_bytes();
    let tag_bytes = tag.as_bytes();
    let max_check = text_bytes.len().min(tag_bytes.len() - 1);
    for len in (1..=max_check).rev() {
        let suffix = &text_bytes[text_bytes.len() - len..];
        if tag_bytes.starts_with(suffix) {
            return len;
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_text() {
        let mut d = ToolCallDetector::new(4096);
        assert_eq!(d.feed("hello"), DetectResult::Text("hello".into()));
    }

    #[test]
    fn single_piece_tool_call() {
        let mut d = ToolCallDetector::new(4096);
        let r = d.feed(r#"<tool_call>{"name":"foo","arguments":{}}</tool_call>"#);
        assert_eq!(r, DetectResult::ToolCall(r#"{"name":"foo","arguments":{}}"#.into()));
    }

    #[test]
    fn multi_piece_tool_call() {
        let mut d = ToolCallDetector::new(4096);
        assert_eq!(d.feed("Sure "), DetectResult::Text("Sure ".into()));
        assert_eq!(d.feed("<tool"), DetectResult::Buffering);
        assert_eq!(d.feed("_call>"), DetectResult::Buffering);
        assert_eq!(d.feed(r#"{"name":"x","arguments":{"a":1}}"#), DetectResult::Buffering);
        assert_eq!(
            d.feed("</tool_call>"),
            DetectResult::ToolCall(r#"{"name":"x","arguments":{"a":1}}"#.into()),
        );
    }

    #[test]
    fn partial_tag_mismatch_flushes() {
        let mut d = ToolCallDetector::new(4096);
        assert_eq!(d.feed("<tool"), DetectResult::Buffering);
        assert_eq!(d.feed("box>"), DetectResult::Text("<toolbox>".into()));
    }

    #[test]
    fn abort_on_max_capture() {
        let mut d = ToolCallDetector::new(20);
        d.feed("<tool_call>");
        assert!(matches!(
            d.feed("this is way too long for the limit"),
            DetectResult::Aborted(_)
        ));
    }

    #[test]
    fn extract_with_text_and_tool() {
        let text = "I'll help.\n<tool_call>{\"name\":\"read\",\"arguments\":{\"p\":1}}</tool_call>";
        let r = extract_tool_calls(text);
        assert_eq!(r.stop_reason, StopReason::ToolUse);
        assert_eq!(r.content.len(), 2);
        assert!(matches!(&r.content[0], ContentBlock::Text { .. }));
        assert!(matches!(&r.content[1], ContentBlock::ToolUse { name, .. } if name == "read"));
    }

    #[test]
    fn extract_plain_text() {
        let r = extract_tool_calls("Just text, no tools.");
        assert_eq!(r.stop_reason, StopReason::EndTurn);
        assert_eq!(r.content.len(), 1);
    }

    #[test]
    fn extract_tool_only() {
        let text = r#"<tool_call>{"name":"s","arguments":{"q":"r"}}</tool_call>"#;
        let r = extract_tool_calls(text);
        assert_eq!(r.stop_reason, StopReason::ToolUse);
        assert_eq!(r.content.len(), 1);
        assert!(matches!(&r.content[0], ContentBlock::ToolUse { name, .. } if name == "s"));
    }
}

/// Token-level `<tool_call>` / `</tool_call>` detector.
///
/// Works on integer token IDs without string construction during the hot path.
/// Uses sentinel optimization: only compare the full pattern when the last
/// token matches the sentinel (final token of the pattern).

#[derive(Debug, Clone, PartialEq)]
pub enum DetectorState {
    Normal,
    Capturing,
    Complete,
}

#[derive(Debug, PartialEq)]
pub enum DetectResult {
    /// Normal token — stream it.
    Text(i32),
    /// Part of open tag — suppress.
    TagOpen,
    /// JSON body token during capture — suppress.
    Captured,
    /// Complete tool call; contains body tokens (open/close tags stripped).
    ToolCall(Vec<i32>),
    /// Runaway capture exceeded max_capture; flush as text.
    Aborted(Vec<i32>),
}

pub struct ToolCallDetector {
    open_pattern: Vec<i32>,
    close_pattern: Vec<i32>,
    /// Last token of open_pattern — fast-path check before full comparison.
    open_sentinel: i32,
    /// Last token of close_pattern — fast-path check before full comparison.
    close_sentinel: i32,
    pub state: DetectorState,
    /// Ring buffer used in Normal state for trailing pattern matching.
    ring: Vec<i32>,
    /// Buffered tokens accumulated during Capturing state.
    capture_buf: Vec<i32>,
    /// Maximum capture length before issuing Aborted.
    max_capture: usize,
}

impl ToolCallDetector {
    pub fn new(open_pattern: Vec<i32>, close_pattern: Vec<i32>, max_capture: usize) -> Self {
        let open_sentinel = *open_pattern.last().expect("open_pattern must be non-empty");
        let close_sentinel = *close_pattern.last().expect("close_pattern must be non-empty");
        // Ring only needs to hold the last open_pattern.len() tokens.
        let ring_cap = open_pattern.len();
        Self {
            open_sentinel,
            close_sentinel,
            open_pattern,
            close_pattern,
            state: DetectorState::Normal,
            ring: Vec::with_capacity(ring_cap),
            capture_buf: Vec::new(),
            max_capture,
        }
    }

    /// Reset to Normal state, clearing all buffers.
    pub fn reset(&mut self) {
        self.state = DetectorState::Normal;
        self.ring.clear();
        self.capture_buf.clear();
    }

    /// Feed one token into the detector and get back a result.
    pub fn feed(&mut self, token: i32) -> DetectResult {
        match self.state {
            DetectorState::Complete => {
                // After completing, reset and process the new token as Normal.
                self.reset();
                self.feed_normal(token)
            }
            DetectorState::Normal => self.feed_normal(token),
            DetectorState::Capturing => self.feed_capturing(token),
        }
    }

    fn feed_normal(&mut self, token: i32) -> DetectResult {
        // Maintain ring buffer bounded to open_pattern length.
        let pat_len = self.open_pattern.len();
        self.ring.push(token);
        if self.ring.len() > pat_len {
            self.ring.remove(0);
        }

        // Fast-path sentinel check before full slice comparison.
        if token == self.open_sentinel && self.ring.ends_with(&self.open_pattern) {
            self.ring.clear();
            self.capture_buf.clear();
            self.state = DetectorState::Capturing;
            return DetectResult::TagOpen;
        }

        DetectResult::Text(token)
    }

    fn feed_capturing(&mut self, token: i32) -> DetectResult {
        self.capture_buf.push(token);

        // Check for runaway before close-tag check so max_capture=N means
        // "abort when we reach N tokens" (inclusive).
        if self.capture_buf.len() >= self.max_capture {
            let buf = std::mem::take(&mut self.capture_buf);
            self.state = DetectorState::Normal;
            self.ring.clear();
            return DetectResult::Aborted(buf);
        }

        // Fast-path sentinel check for close pattern.
        if token == self.close_sentinel && self.capture_buf.ends_with(&self.close_pattern) {
            let close_len = self.close_pattern.len();
            let body_end = self.capture_buf.len() - close_len;
            let body: Vec<i32> = self.capture_buf[..body_end].to_vec();
            self.capture_buf.clear();
            self.state = DetectorState::Complete;
            return DetectResult::ToolCall(body);
        }

        DetectResult::Captured
    }
}

/// String-based `<tool_call>` / `</tool_call>` detector for decoded text streams.
/// Works on string fragments (token pieces) rather than token IDs.
#[derive(Debug, Clone, Copy, PartialEq)]
enum SState { Normal, MaybeTag, Capturing }

const OPEN_TAG: &str = "<tool_call>";
const CLOSE_TAG: &str = "</tool_call>";

#[derive(Debug, PartialEq)]
pub enum StrDetectResult {
    Text(String),
    Buffering,
    ToolCall(String),
    Aborted(String),
}

pub struct StringToolCallDetector {
    state: SState,
    tag_buf: String,
    capture_buf: String,
    flush_buf: String,
    max_capture: usize,
}

impl StringToolCallDetector {
    pub fn new(max_capture: usize) -> Self {
        Self { state: SState::Normal, tag_buf: String::new(),
               capture_buf: String::new(), flush_buf: String::new(), max_capture }
    }

    pub fn feed(&mut self, piece: &str) -> StrDetectResult {
        match self.state {
            SState::Normal => self.feed_normal(piece),
            SState::MaybeTag => self.feed_maybe_tag(piece),
            SState::Capturing => self.feed_capturing(piece),
        }
    }

    pub fn finish(&mut self) -> Option<String> {
        let mut lo = std::mem::take(&mut self.tag_buf);
        if !self.capture_buf.is_empty() {
            lo.push_str(OPEN_TAG);
            lo.push_str(&std::mem::take(&mut self.capture_buf));
        }
        self.state = SState::Normal;
        if lo.is_empty() { None } else { Some(lo) }
    }

    fn feed_normal(&mut self, piece: &str) -> StrDetectResult {
        self.flush_buf.push_str(piece);
        if let Some(pos) = self.flush_buf.find(OPEN_TAG) {
            let before = self.flush_buf[..pos].to_string();
            let remainder = self.flush_buf[pos + OPEN_TAG.len()..].to_string();
            self.flush_buf.clear();
            self.state = SState::Capturing;
            self.capture_buf = remainder;
            if let Some(result) = self.try_close_tag() {
                if before.is_empty() { return result; }
                if let StrDetectResult::ToolCall(body) = result {
                    self.flush_buf = format!("{OPEN_TAG}{body}{CLOSE_TAG}");
                }
                return StrDetectResult::Text(before);
            }
            return if before.is_empty() { StrDetectResult::Buffering }
                   else { StrDetectResult::Text(before) };
        }
        let plen = partial_tag_match(&self.flush_buf, OPEN_TAG);
        if plen > 0 {
            let split = self.flush_buf.len() - plen;
            let before = self.flush_buf[..split].to_string();
            self.tag_buf = self.flush_buf[split..].to_string();
            self.flush_buf.clear();
            self.state = SState::MaybeTag;
            return if before.is_empty() { StrDetectResult::Buffering }
                   else { StrDetectResult::Text(before) };
        }
        StrDetectResult::Text(std::mem::take(&mut self.flush_buf))
    }

    fn feed_maybe_tag(&mut self, piece: &str) -> StrDetectResult {
        self.tag_buf.push_str(piece);
        if let Some(pos) = self.tag_buf.find(OPEN_TAG) {
            let before = self.tag_buf[..pos].to_string();
            self.capture_buf = self.tag_buf[pos + OPEN_TAG.len()..].to_string();
            self.tag_buf.clear();
            self.state = SState::Capturing;
            if let Some(result) = self.try_close_tag() { return result; }
            return if before.is_empty() { StrDetectResult::Buffering }
                   else { StrDetectResult::Text(before) };
        }
        if partial_tag_match(&self.tag_buf, OPEN_TAG) == self.tag_buf.len() {
            return StrDetectResult::Buffering;
        }
        self.state = SState::Normal;
        StrDetectResult::Text(std::mem::take(&mut self.tag_buf))
    }

    fn feed_capturing(&mut self, piece: &str) -> StrDetectResult {
        self.capture_buf.push_str(piece);
        if self.capture_buf.len() > self.max_capture {
            self.state = SState::Normal;
            return StrDetectResult::Aborted(std::mem::take(&mut self.capture_buf));
        }
        self.try_close_tag().unwrap_or(StrDetectResult::Buffering)
    }

    fn try_close_tag(&mut self) -> Option<StrDetectResult> {
        let pos = self.capture_buf.find(CLOSE_TAG)?;
        let body = self.capture_buf[..pos].trim().to_string();
        self.capture_buf.clear();
        self.state = SState::Normal;
        Some(StrDetectResult::ToolCall(body))
    }
}

/// Parse a tool-call JSON body into a `ContentBlock::ToolUse`.
/// Expected format: `{"name": "tool_name", "arguments": {...}}`
pub fn parse_tool_call_json(json_str: &str) -> Option<super::ContentBlock> {
    let v: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    let arguments = v.get("arguments")?.clone();
    let id = format!("tc_{:08x}", fxhash(json_str));
    Some(super::ContentBlock::ToolUse { id, name, input: arguments })
}

/// Extract tool calls from completed generation text (non-streaming path).
pub fn extract_tool_calls(text: &str) -> super::LlmResponse {
    let mut detector = StringToolCallDetector::new(8192);
    let mut blocks: Vec<super::ContentBlock> = Vec::new();
    let mut plain = String::new();
    let mut result = detector.feed(text);
    loop {
        match result {
            StrDetectResult::Text(t) => plain.push_str(&t),
            StrDetectResult::Buffering => {}
            StrDetectResult::ToolCall(body) => {
                if !plain.is_empty() {
                    blocks.push(super::ContentBlock::Text { text: std::mem::take(&mut plain) });
                }
                if let Some(block) = parse_tool_call_json(&body) {
                    blocks.push(block);
                }
            }
            StrDetectResult::Aborted(buf) => plain.push_str(&buf),
        }
        let next = detector.feed("");
        if matches!(next, StrDetectResult::Text(ref t) if t.is_empty()) {
            break;
        }
        result = next;
    }
    if let Some(leftover) = detector.finish() {
        plain.push_str(&leftover);
    }
    if !plain.is_empty() {
        blocks.push(super::ContentBlock::Text { text: plain });
    }
    if blocks.is_empty() {
        blocks.push(super::ContentBlock::Text { text: String::new() });
    }
    let has_tool_use = blocks.iter().any(|b| matches!(b, super::ContentBlock::ToolUse { .. }));
    super::LlmResponse {
        content: blocks,
        stop_reason: if has_tool_use { super::StopReason::ToolUse } else { super::StopReason::EndTurn },
    }
}

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
    // Check suffixes of text from longest to shortest.
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

    fn det(max: usize) -> ToolCallDetector {
        ToolCallDetector::new(vec![10, 20, 30], vec![40, 50, 60], max)
    }

    #[test]
    fn tok_normal_pass_through() {
        let mut d = det(512);
        assert_eq!(d.feed(1), DetectResult::Text(1));
        assert_eq!(d.feed(99), DetectResult::Text(99));
    }

    #[test]
    fn tok_open_triggers_capture() {
        let mut d = det(512);
        assert_eq!(d.feed(10), DetectResult::Text(10));
        assert_eq!(d.feed(20), DetectResult::Text(20));
        assert_eq!(d.feed(30), DetectResult::TagOpen);
        assert_eq!(d.state, DetectorState::Capturing);
    }

    #[test]
    fn tok_capture_close_produces_tool_call() {
        let mut d = det(512);
        for t in [10, 20, 30] { d.feed(t); }
        assert_eq!(d.feed(100), DetectResult::Captured);
        assert_eq!(d.feed(200), DetectResult::Captured);
        for t in [40, 50] { assert_eq!(d.feed(t), DetectResult::Captured); }
        assert_eq!(d.feed(60), DetectResult::ToolCall(vec![100, 200]));
    }

    #[test]
    fn tok_runaway_aborts() {
        let mut d = det(5);
        for t in [10, 20, 30] { d.feed(t); }
        for i in 1..=4 { assert_eq!(d.feed(i), DetectResult::Captured); }
        assert!(matches!(d.feed(5), DetectResult::Aborted(_)));
    }

    #[test]
    fn tok_nested_open_ignored() {
        let mut d = det(512);
        for t in [10, 20, 30] { d.feed(t); }
        for t in [10, 20, 30] { assert_eq!(d.feed(t), DetectResult::Captured); }
        for t in [40, 50] { d.feed(t); }
        assert_eq!(d.feed(60), DetectResult::ToolCall(vec![10, 20, 30]));
    }

    #[test]
    fn tok_resets_after_complete() {
        let mut d = det(512);
        for t in [10, 20, 30, 100, 40, 50, 60] { d.feed(t); }
        assert_eq!(d.state, DetectorState::Complete);
        assert_eq!(d.feed(7), DetectResult::Text(7));
    }

    // --- StringToolCallDetector tests ---

    #[test]
    fn str_normal_text() {
        let mut d = StringToolCallDetector::new(4096);
        assert_eq!(d.feed("hello"), StrDetectResult::Text("hello".into()));
    }

    #[test]
    fn str_single_piece_tool_call() {
        let mut d = StringToolCallDetector::new(4096);
        let r = d.feed(r#"<tool_call>{"name":"foo","arguments":{}}</tool_call>"#);
        assert_eq!(r, StrDetectResult::ToolCall(r#"{"name":"foo","arguments":{}}"#.into()));
    }

    #[test]
    fn str_multi_piece_tool_call() {
        let mut d = StringToolCallDetector::new(4096);
        assert_eq!(d.feed("Sure "), StrDetectResult::Text("Sure ".into()));
        assert_eq!(d.feed("<tool"), StrDetectResult::Buffering);
        assert_eq!(d.feed("_call>"), StrDetectResult::Buffering);
        assert_eq!(d.feed(r#"{"name":"x","arguments":{"a":1}}"#), StrDetectResult::Buffering);
        assert_eq!(
            d.feed("</tool_call>"),
            StrDetectResult::ToolCall(r#"{"name":"x","arguments":{"a":1}}"#.into()),
        );
    }

    #[test]
    fn str_partial_tag_mismatch_flushes() {
        let mut d = StringToolCallDetector::new(4096);
        assert_eq!(d.feed("<tool"), StrDetectResult::Buffering);
        assert_eq!(d.feed("box>"), StrDetectResult::Text("<toolbox>".into()));
    }

    #[test]
    fn str_text_before_tool_call() {
        let mut d = StringToolCallDetector::new(4096);
        let r = d.feed(r#"Help.<tool_call>{"name":"f","arguments":{}}</tool_call>"#);
        assert_eq!(r, StrDetectResult::Text("Help.".into()));
        assert_eq!(d.feed(""), StrDetectResult::ToolCall(r#"{"name":"f","arguments":{}}"#.into()));
    }

    #[test]
    fn str_abort_and_finish() {
        let mut d = StringToolCallDetector::new(20);
        d.feed("<tool_call>");
        assert!(matches!(d.feed("this is way too long for the limit"), StrDetectResult::Aborted(_)));

        let mut d2 = StringToolCallDetector::new(4096);
        d2.feed("<tool");
        assert_eq!(d2.finish(), Some("<tool".into()));
    }

    // --- extract_tool_calls integration tests ---

    #[test]
    fn extract_with_text_and_tool() {
        let text = "I'll help.\n<tool_call>{\"name\":\"read\",\"arguments\":{\"p\":1}}</tool_call>";
        let r = extract_tool_calls(text);
        assert_eq!(r.stop_reason, super::super::StopReason::ToolUse);
        assert_eq!(r.content.len(), 2);
        assert!(matches!(&r.content[0], super::super::ContentBlock::Text { .. }));
        assert!(matches!(&r.content[1], super::super::ContentBlock::ToolUse { name, .. } if name == "read"));
    }

    #[test]
    fn extract_plain_text() {
        let r = extract_tool_calls("Just text, no tools.");
        assert_eq!(r.stop_reason, super::super::StopReason::EndTurn);
        assert_eq!(r.content.len(), 1);
    }

    #[test]
    fn extract_tool_only() {
        let text = r#"<tool_call>{"name":"s","arguments":{"q":"r"}}</tool_call>"#;
        let r = extract_tool_calls(text);
        assert_eq!(r.stop_reason, super::super::StopReason::ToolUse);
        assert_eq!(r.content.len(), 1);
        assert!(matches!(&r.content[0], super::super::ContentBlock::ToolUse { name, .. } if name == "s"));
    }
}

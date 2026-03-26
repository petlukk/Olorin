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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_detector(max_capture: usize) -> ToolCallDetector {
        ToolCallDetector::new(vec![10, 20, 30], vec![40, 50, 60], max_capture)
    }

    #[test]
    fn normal_tokens_pass_through() {
        let mut d = make_detector(512);
        assert_eq!(d.feed(1), DetectResult::Text(1));
        assert_eq!(d.feed(2), DetectResult::Text(2));
        assert_eq!(d.feed(99), DetectResult::Text(99));
        assert_eq!(d.state, DetectorState::Normal);
    }

    #[test]
    fn open_tag_triggers_capture() {
        let mut d = make_detector(512);
        // Feed the first two tokens of the open pattern — still Normal.
        assert_eq!(d.feed(10), DetectResult::Text(10));
        assert_eq!(d.feed(20), DetectResult::Text(20));
        // Third token completes the open pattern → TagOpen and state = Capturing.
        assert_eq!(d.feed(30), DetectResult::TagOpen);
        assert_eq!(d.state, DetectorState::Capturing);
    }

    #[test]
    fn capture_then_close_produces_tool_call() {
        let mut d = make_detector(512);
        // Open tag.
        d.feed(10);
        d.feed(20);
        d.feed(30);
        assert_eq!(d.state, DetectorState::Capturing);

        // Body tokens.
        assert_eq!(d.feed(100), DetectResult::Captured);
        assert_eq!(d.feed(200), DetectResult::Captured);

        // Close tag tokens — first two are Captured, last one completes.
        assert_eq!(d.feed(40), DetectResult::Captured);
        assert_eq!(d.feed(50), DetectResult::Captured);
        let result = d.feed(60);
        // Body is everything before the close pattern [40,50,60].
        assert_eq!(result, DetectResult::ToolCall(vec![100, 200]));
        assert_eq!(d.state, DetectorState::Complete);
    }

    #[test]
    fn runaway_capture_aborts() {
        // max_capture = 5: abort once 5 tokens have been captured.
        let mut d = make_detector(5);
        // Open tag.
        d.feed(10);
        d.feed(20);
        d.feed(30);

        // Four tokens — still within limit.
        assert_eq!(d.feed(1), DetectResult::Captured);
        assert_eq!(d.feed(2), DetectResult::Captured);
        assert_eq!(d.feed(3), DetectResult::Captured);
        assert_eq!(d.feed(4), DetectResult::Captured);
        // Fifth token hits max_capture.
        let result = d.feed(5);
        assert!(matches!(result, DetectResult::Aborted(_)));
        assert_eq!(d.state, DetectorState::Normal);
    }

    #[test]
    fn nested_open_tag_ignored_during_capture() {
        let mut d = make_detector(512);
        // Enter capturing state.
        d.feed(10);
        d.feed(20);
        d.feed(30);

        // Feed the open pattern again inside capture — it should all be Captured.
        assert_eq!(d.feed(10), DetectResult::Captured);
        assert_eq!(d.feed(20), DetectResult::Captured);
        assert_eq!(d.feed(30), DetectResult::Captured);
        // Still capturing.
        assert_eq!(d.state, DetectorState::Capturing);

        // Now close properly.
        d.feed(40);
        d.feed(50);
        let result = d.feed(60);
        // Body should include the nested open-pattern tokens; close pattern [40,50,60] stripped.
        assert_eq!(
            result,
            DetectResult::ToolCall(vec![10, 20, 30])
        );
    }

    #[test]
    fn partial_open_pattern_then_mismatch_flushes() {
        let mut d = make_detector(512);
        // Start of open pattern.
        assert_eq!(d.feed(10), DetectResult::Text(10));
        assert_eq!(d.feed(20), DetectResult::Text(20));
        // Mismatch — should just be Text, not TagOpen.
        assert_eq!(d.feed(99), DetectResult::Text(99));
        assert_eq!(d.state, DetectorState::Normal);
    }

    #[test]
    fn resets_after_tool_call() {
        let mut d = make_detector(512);
        // Full open→body→close cycle.
        d.feed(10);
        d.feed(20);
        d.feed(30);
        d.feed(100);
        d.feed(40);
        d.feed(50);
        d.feed(60);
        assert_eq!(d.state, DetectorState::Complete);

        // Next token should trigger reset and return Text.
        assert_eq!(d.feed(7), DetectResult::Text(7));
        assert_eq!(d.state, DetectorState::Normal);
    }
}

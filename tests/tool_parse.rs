use olorin::core::tool_parse::{ToolCallDetector, DetectResult, extract_tool_calls};
use olorin::core::llm::{ContentBlock, StopReason};

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

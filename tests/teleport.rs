use olorin::kernels::ffi;
use olorin::core::router::DispatchContext;

// Contract: DispatchContext processes messages even when teleported=true.
// The dormant-message-to-caller behavior belongs at server.rs (HTTP entry
// points), not at the Pipe. The WhatsApp loop itself depends on dispatch()
// working while teleported=true.
#[test]
fn dispatch_while_teleported_still_processes() {
    ffi::init().unwrap();
    let mut ctx = DispatchContext::new(None, None);
    ctx.teleported = true;
    let resp = ctx.dispatch("/help");
    // /help always lists the full command set; dormant message does not.
    assert!(resp.text.contains("/clear") && resp.text.contains("/tools"),
        "help should be returned while teleported, got: {}", resp.text);
}

// Contract: /teleport while already teleported must not spawn a second
// bridge — handle_teleport() itself guards against double-teleport.
#[test]
fn teleport_command_while_already_teleported_refuses() {
    ffi::init().unwrap();
    std::env::set_var("OLORIN_BRIDGE", "/nonexistent/wa-bridge");
    let mut ctx = DispatchContext::new(None, None);
    ctx.teleported = true;
    let resp = ctx.dispatch("/teleport");
    assert!(resp.text.to_lowercase().contains("already"),
        "expected 'already on WhatsApp' refusal, got: {}", resp.text);
    assert!(ctx.teleported, "flag should remain true");
}

use olorin::interface::whatsapp::{strip_trigger, TRIGGER_TELEPORT};

#[test]
fn trigger_at_olorin() {
    assert_eq!(strip_trigger("@olorin what time is it?"), Some("what time is it?"));
}

#[test]
fn trigger_bang_olorin() {
    assert_eq!(strip_trigger("!olorin hello"), Some("hello"));
}

#[test]
fn trigger_olorin_space() {
    assert_eq!(strip_trigger("olorin tell me a joke"), Some("tell me a joke"));
}

#[test]
fn trigger_case_insensitive() {
    assert_eq!(strip_trigger("@Olorin hi"), Some("hi"));
}

#[test]
fn trigger_no_match() {
    assert_eq!(strip_trigger("hello world"), None);
}

#[test]
fn trigger_teleport_command() {
    assert_eq!(strip_trigger("/teleport"), Some(TRIGGER_TELEPORT));
}

#[test]
fn trigger_just_olorin_no_space() {
    assert_eq!(strip_trigger("olorin"), None);
}

#[test]
fn teleport_flag_cleared_after_bridge_not_found() {
    ffi::init().unwrap();
    std::env::set_var("OLORIN_BRIDGE", "/nonexistent/wa-bridge");
    let mut ctx = DispatchContext::new(None, None);
    assert!(!ctx.teleported);
    let resp = ctx.dispatch("/teleport");
    // Bridge not found — should return error and NOT leave teleported=true
    assert!(!ctx.teleported, "teleported should be false after bridge failure");
    assert!(resp.text.contains("not found") || resp.text.contains("Build"),
        "unexpected: {}", resp.text);
}

#[test]
fn normal_dispatch_unaffected() {
    ffi::init().unwrap();
    let mut ctx = DispatchContext::new(None, None);
    // /help should work normally when not teleported
    let resp = ctx.dispatch("/help");
    assert!(resp.text.contains("/teleport"), "help should mention /teleport");
    assert!(!ctx.teleported);
}

// Contract: dispatch_streaming also processes commands while teleported=true.
// Dormant gating is the server's job at the HTTP handler layer, not the Pipe's.
#[test]
fn dispatch_streaming_while_teleported_still_processes() {
    ffi::init().unwrap();
    let mut ctx = DispatchContext::new(None, None);
    ctx.teleported = true;
    let (tx, rx) = std::sync::mpsc::channel();
    ctx.dispatch_streaming("/help", tx);
    let mut collected = String::new();
    for event in rx {
        match event {
            olorin::core::router::StreamEvent::Token(t) => collected.push_str(&t),
            olorin::core::router::StreamEvent::Done { .. } => break,
            _ => {}
        }
    }
    assert!(collected.contains("/clear") && collected.contains("/tools"),
        "streaming should emit help output while teleported, got: {collected}");
}

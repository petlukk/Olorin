use olorin::kernels::ffi;
use olorin::core::router::DispatchContext;

#[test]
fn dispatch_while_teleported_returns_dormant() {
    ffi::init().unwrap();
    let mut ctx = DispatchContext::new(None, None, None, None);
    ctx.teleported = true;
    let resp = ctx.dispatch("hello world");
    assert!(resp.text.contains("WhatsApp"), "should mention WhatsApp: {}", resp.text);
    assert!(!resp.blocked);
}

#[test]
fn dispatch_teleport_command_while_already_teleported() {
    ffi::init().unwrap();
    let mut ctx = DispatchContext::new(None, None, None, None);
    ctx.teleported = true;
    let resp = ctx.dispatch("/teleport");
    assert!(resp.text.contains("WhatsApp"), "should mention WhatsApp: {}", resp.text);
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
    let mut ctx = DispatchContext::new(None, None, None, None);
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
    let mut ctx = DispatchContext::new(None, None, None, None);
    // /help should work normally when not teleported
    let resp = ctx.dispatch("/help");
    assert!(resp.text.contains("/teleport"), "help should mention /teleport");
    assert!(!ctx.teleported);
}

#[test]
fn dispatch_streaming_while_teleported_sends_dormant() {
    ffi::init().unwrap();
    let mut ctx = DispatchContext::new(None, None, None, None);
    ctx.teleported = true;
    let (tx, rx) = std::sync::mpsc::channel();
    ctx.dispatch_streaming("hello", tx);
    let mut got_dormant = false;
    for event in rx {
        match event {
            olorin::core::router::StreamEvent::Token(t) => {
                if t.contains("WhatsApp") { got_dormant = true; }
            }
            olorin::core::router::StreamEvent::Done { .. } => break,
            _ => {}
        }
    }
    assert!(got_dormant, "streaming should return dormant message");
}

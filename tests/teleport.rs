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

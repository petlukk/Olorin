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

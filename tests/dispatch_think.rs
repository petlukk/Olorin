//! Guards the `/think` slash command: the command_router kernel must match it
//! to CMD_THINK, carry its arg, and not collide with the other /t* commands.

use olorin::core::dispatch;

#[test]
fn think_command_routes() {
    olorin::kernels::ffi::init().unwrap();

    // bare toggle
    let (id, arg) = dispatch::match_command(b"/think");
    assert_eq!(id, dispatch::CMD_THINK, "/think must route to CMD_THINK");
    assert_eq!(arg, b"", "bare /think has no arg");

    // with explicit on/off arg
    let (id, arg) = dispatch::match_command(b"/think off");
    assert_eq!(id, dispatch::CMD_THINK);
    assert_eq!(arg, b"off");

    // name verification rejects a near-miss that hash-collides on 4 bytes
    let (id, _) = dispatch::match_command(b"/thinker");
    assert_eq!(id, dispatch::CMD_NONE, "/thinker is not /think");

    // no collision with the other /t* commands
    assert_eq!(dispatch::match_command(b"/time").0, dispatch::CMD_TIME);
    assert_eq!(dispatch::match_command(b"/tools").0, dispatch::CMD_TOOLS);
    assert_eq!(dispatch::match_command(b"/tokens").0, dispatch::CMD_TOKENS);
    assert_eq!(dispatch::match_command(b"/teleport").0, dispatch::CMD_TELEPORT);
}

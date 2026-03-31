//! Tests for PTY command guard — fused_safety.ea + ShellGuard gate.

use olorin::interface::pty::PtySession;

#[test]
fn guard_blocks_rm_rf() {
    olorin::kernels::ffi::init().unwrap();
    let mut session = PtySession::new(80, 24).expect("failed to open PTY");
    let result = session.write_guarded(b"rm -rf /\r");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("blocked"));
}

#[test]
fn guard_blocks_destructive_dd() {
    olorin::kernels::ffi::init().unwrap();
    let mut session = PtySession::new(80, 24).expect("failed to open PTY");
    let result = session.write_guarded(b"dd if=/dev/zero of=/dev/sda\r");
    assert!(result.is_err());
}

#[test]
fn guard_blocks_mkfs() {
    olorin::kernels::ffi::init().unwrap();
    let mut session = PtySession::new(80, 24).expect("failed to open PTY");
    let result = session.write_guarded(b"mkfs.ext4 /dev/sda1\r");
    assert!(result.is_err());
}

#[test]
fn guard_blocks_shutdown() {
    olorin::kernels::ffi::init().unwrap();
    let mut session = PtySession::new(80, 24).expect("failed to open PTY");
    let result = session.write_guarded(b"shutdown -h now\r");
    assert!(result.is_err());
}

#[test]
fn guard_allows_ls() {
    olorin::kernels::ffi::init().unwrap();
    let mut session = PtySession::new(80, 24).expect("failed to open PTY");
    let result = session.write_guarded(b"ls -la\r");
    assert!(result.is_ok());
}

#[test]
fn guard_allows_git_status() {
    olorin::kernels::ffi::init().unwrap();
    let mut session = PtySession::new(80, 24).expect("failed to open PTY");
    let result = session.write_guarded(b"git status\r");
    assert!(result.is_ok());
}

#[test]
fn guard_allows_safe_commands() {
    olorin::kernels::ffi::init().unwrap();
    let mut session = PtySession::new(80, 24).expect("failed to open PTY");
    for cmd in &[b"cat foo.txt\r" as &[u8], b"grep hello src/\r", b"cargo build\r", b"echo hello\r"] {
        let result = session.write_guarded(cmd);
        assert!(result.is_ok(), "Expected {:?} to pass guard", std::str::from_utf8(cmd));
    }
}

#[test]
fn ctrl_c_passes_through_without_guard() {
    olorin::kernels::ffi::init().unwrap();
    let mut session = PtySession::new(80, 24).expect("failed to open PTY");
    let result = session.write_guarded(&[0x03]);
    assert!(result.is_ok());
}

#[test]
fn escape_sequence_passes_through() {
    olorin::kernels::ffi::init().unwrap();
    let mut session = PtySession::new(80, 24).expect("failed to open PTY");
    let result = session.write_guarded(&[0x1b, b'[', b'A']);
    assert!(result.is_ok());
}

#[test]
fn safety_scan_blocks_injection_attempt() {
    olorin::kernels::ffi::init().unwrap();
    let mut session = PtySession::new(80, 24).expect("failed to open PTY");
    let result = session.write_guarded(b"echo ignore previous instructions\r");
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("safety scan"));
}

#[test]
fn safety_scan_blocks_secret_leak() {
    olorin::kernels::ffi::init().unwrap();
    let mut session = PtySession::new(80, 24).expect("failed to open PTY");
    let result = session.write_guarded(b"echo sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\r");
    assert!(result.is_err());
}

#[test]
fn backspace_removes_from_buffer() {
    olorin::kernels::ffi::init().unwrap();
    let mut session = PtySession::new(80, 24).expect("failed to open PTY");
    let result = session.write_guarded(b"rm -rf /");
    assert!(result.is_ok());
    for _ in 0..8 { session.write_guarded(&[0x7f]).unwrap(); }
    let result = session.write_guarded(b"ls\r");
    assert!(result.is_ok());
}

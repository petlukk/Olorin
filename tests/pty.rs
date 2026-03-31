//! End-to-end PTY tests — open bash, write, read back, resize, close.

use olorin::interface::pty::PtySession;

#[test]
fn pty_open_and_close() {
    let session = PtySession::new(80, 24).expect("failed to open PTY");
    assert!(session.child_alive());
    drop(session);
}

#[test]
fn pty_echo_hello() {
    let mut session = PtySession::new(80, 24).expect("failed to open PTY");
    session.write_bytes(b"echo OLORIN_TEST_MARKER\n");
    std::thread::sleep(std::time::Duration::from_millis(200));
    let _patch = session.read_and_apply();
    let mut found = false;
    for row in 0..24 {
        let line: String = (0..80)
            .map(|col| {
                let ch = session.grid().cell(row, col).ch;
                if ch >= 32 && ch < 127 { ch as u8 as char } else { ' ' }
            })
            .collect();
        if line.contains("OLORIN_TEST_MARKER") {
            found = true;
            break;
        }
    }
    assert!(found, "Expected OLORIN_TEST_MARKER in terminal output");
}

#[test]
fn pty_resize() {
    let mut session = PtySession::new(80, 24).expect("failed to open PTY");
    session.resize(120, 36);
    assert_eq!(session.grid().cols, 120);
    assert_eq!(session.grid().rows, 36);
}

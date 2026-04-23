//! Integration test: full terminal pipeline from PTY to diff.

use olorin::interface::pty::PtySession;

#[test]
fn full_pipeline_echo_produces_dirty_cells() {
    olorin::kernels::ffi::init().unwrap();

    let mut session = PtySession::new(80, 24).expect("failed to open PTY");
    session.write_bytes(b"echo hello\n");
    std::thread::sleep(std::time::Duration::from_millis(300));

    let dirty = session.read_and_apply();
    let dirty_count = dirty.iter().filter(|&&d| d != 0).count();

    assert!(dirty_count > 0, "Expected dirty cells after echo, got 0");
}

#[test]
fn ansi_color_in_pipeline() {
    olorin::kernels::ffi::init().unwrap();

    let mut session = PtySession::new(80, 24).expect("failed to open PTY");
    session.write_bytes(b"printf '\\033[31mRED\\033[0m'\n");

    // /bin/bash --login on WSL can take >300ms to reach the prompt.
    // Poll instead of a fixed sleep: read + scan the grid until 'RED'
    // appears colored, or 2s elapses.
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(2000);
    let mut found_colored = false;
    while std::time::Instant::now() < deadline {
        let _ = session.read_and_apply();
        let grid = session.grid();
        'scan: for row in 0..24u16 {
            for col in 0..77u16 {
                let c0 = grid.cell(row, col);
                let c1 = grid.cell(row, col + 1);
                let c2 = grid.cell(row, col + 2);
                if c0.ch == b'R' as u32 && c1.ch == b'E' as u32 && c2.ch == b'D' as u32
                    && c0.fg != 0x00cdd6f4
                {
                    found_colored = true;
                    break 'scan;
                }
            }
        }
        if found_colored { break; }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(found_colored, "Expected colored 'RED' text in grid");
}

#[test]
fn resize_updates_grid_dimensions() {
    olorin::kernels::ffi::init().unwrap();

    let mut session = PtySession::new(80, 24).expect("failed to open PTY");
    session.resize(120, 36);

    assert_eq!(session.grid().cols, 120);
    assert_eq!(session.grid().rows, 36);

    session.write_bytes(b"echo AFTER_RESIZE\n");
    std::thread::sleep(std::time::Duration::from_millis(300));

    let dirty = session.read_and_apply();
    let dirty_count = dirty.iter().filter(|&&d| d != 0).count();
    assert!(dirty_count > 0, "Expected dirty cells after resize + echo");
}

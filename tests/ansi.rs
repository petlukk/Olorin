//! Tests for ANSI state machine — SGR, cursor movement, erase.

mod common {
    use olorin::interface::ansi::TermGrid;

    pub fn make_grid(cols: u16, rows: u16) -> TermGrid {
        TermGrid::new(cols, rows)
    }

    /// Feed raw bytes through the classifier + state machine.
    pub fn feed(grid: &mut TermGrid, input: &[u8]) {
        let mut scan_buf = vec![0u8; input.len()];
        grid.feed(input, &mut scan_buf);
    }
}

#[test]
fn plain_text_writes_cells() {
    let mut g = common::make_grid(80, 24);
    common::feed(&mut g, b"Hello");
    assert_eq!(g.cell(0, 0).ch, b'H' as u32);
    assert_eq!(g.cell(0, 1).ch, b'e' as u32);
    assert_eq!(g.cell(0, 2).ch, b'l' as u32);
    assert_eq!(g.cell(0, 3).ch, b'l' as u32);
    assert_eq!(g.cell(0, 4).ch, b'o' as u32);
    assert_eq!(g.cursor(), (0, 5));
}

#[test]
fn sgr_red_foreground() {
    let mut g = common::make_grid(80, 24);
    common::feed(&mut g, b"\x1b[31mX");
    let cell = g.cell(0, 0);
    assert_eq!(cell.ch, b'X' as u32);
    assert_ne!(cell.fg, 0x00cdd6f4);
}

#[test]
fn sgr_reset() {
    let mut g = common::make_grid(80, 24);
    common::feed(&mut g, b"\x1b[31mA\x1b[0mB");
    let a = g.cell(0, 0);
    let b = g.cell(0, 1);
    assert_ne!(a.fg, b.fg);
}

#[test]
fn cursor_movement_csi_h() {
    let mut g = common::make_grid(80, 24);
    common::feed(&mut g, b"\x1b[3;5HX");
    assert_eq!(g.cell(2, 4).ch, b'X' as u32);
    assert_eq!(g.cursor(), (2, 5));
}

#[test]
fn erase_display_csi_2j() {
    let mut g = common::make_grid(80, 24);
    common::feed(&mut g, b"ABCDE");
    common::feed(&mut g, b"\x1b[2J");
    assert_eq!(g.cell(0, 0).ch, b' ' as u32);
    assert_eq!(g.cell(0, 4).ch, b' ' as u32);
}

#[test]
fn erase_line_csi_2k() {
    let mut g = common::make_grid(80, 24);
    common::feed(&mut g, b"ABCDE\x1b[2K");
    for col in 0..5 {
        assert_eq!(g.cell(0, col).ch, b' ' as u32);
    }
}

#[test]
fn newline_and_carriage_return() {
    let mut g = common::make_grid(80, 24);
    common::feed(&mut g, b"AB\r\nCD");
    assert_eq!(g.cell(0, 0).ch, b'A' as u32);
    assert_eq!(g.cell(0, 1).ch, b'B' as u32);
    assert_eq!(g.cell(1, 0).ch, b'C' as u32);
    assert_eq!(g.cell(1, 1).ch, b'D' as u32);
}

#[test]
fn cursor_up_down_forward_back() {
    let mut g = common::make_grid(80, 24);
    common::feed(&mut g, b"\x1b[5;6H");
    common::feed(&mut g, b"\x1b[2A");
    common::feed(&mut g, b"\x1b[3C");
    common::feed(&mut g, b"X");
    assert_eq!(g.cell(2, 8).ch, b'X' as u32);
}

#[test]
fn sgr_256_color() {
    let mut g = common::make_grid(80, 24);
    common::feed(&mut g, b"\x1b[38;5;196mX");
    let cell = g.cell(0, 0);
    assert_eq!(cell.ch, b'X' as u32);
    assert_ne!(cell.fg, 0x00cdd6f4);
}

#[test]
fn sgr_truecolor() {
    let mut g = common::make_grid(80, 24);
    common::feed(&mut g, b"\x1b[38;2;255;128;0mX");
    let cell = g.cell(0, 0);
    assert_eq!(cell.fg, 0x00ff8000);
}

#[test]
fn bold_flag() {
    let mut g = common::make_grid(80, 24);
    common::feed(&mut g, b"\x1b[1mX");
    assert!(g.cell(0, 0).flags & 0x01 != 0);
}

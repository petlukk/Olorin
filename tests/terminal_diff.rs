//! Tests for the terminal_diff SIMD kernel.

use olorin::interface::ansi::Cell;
use olorin::kernels::ffi;

fn default_cell() -> Cell {
    Cell { ch: b' ' as u32, fg: 0x00cdd6f4, bg: 0x001e1e2e, flags: 0, _pad: [0; 3] }
}

fn cell_with_ch(ch: u8) -> Cell {
    Cell { ch: ch as u32, fg: 0x00cdd6f4, bg: 0x001e1e2e, flags: 0, _pad: [0; 3] }
}

#[test]
fn identical_grids_produce_no_dirty() {
    olorin::kernels::ffi::init().unwrap();
    let cells: Vec<Cell> = vec![default_cell(); 80 * 24];
    let mut dirty = vec![0u8; 80 * 24];

    unsafe {
        ffi::terminal_diff(
            cells.as_ptr() as *const u8,
            cells.as_ptr() as *const u8,
            dirty.as_mut_ptr(),
            (80 * 24) as i32,
        );
    }

    assert!(dirty.iter().all(|&d| d == 0));
}

#[test]
fn changed_cell_detected() {
    olorin::kernels::ffi::init().unwrap();
    let old: Vec<Cell> = vec![default_cell(); 80 * 24];
    let mut new = old.clone();
    new[42] = cell_with_ch(b'X');

    let mut dirty = vec![0u8; 80 * 24];
    unsafe {
        ffi::terminal_diff(
            old.as_ptr() as *const u8,
            new.as_ptr() as *const u8,
            dirty.as_mut_ptr(),
            (80 * 24) as i32,
        );
    }

    assert_eq!(dirty[42], 1);
    assert_eq!(dirty[0], 0);
    assert_eq!(dirty[41], 0);
    assert_eq!(dirty[43], 0);
}

#[test]
fn multiple_changes_detected() {
    olorin::kernels::ffi::init().unwrap();
    let old: Vec<Cell> = vec![default_cell(); 10];
    let mut new = old.clone();
    new[0] = cell_with_ch(b'A');
    new[5] = cell_with_ch(b'B');
    new[9] = cell_with_ch(b'C');

    let mut dirty = vec![0u8; 10];
    unsafe {
        ffi::terminal_diff(
            old.as_ptr() as *const u8,
            new.as_ptr() as *const u8,
            dirty.as_mut_ptr(),
            10,
        );
    }

    assert_eq!(dirty, &[1, 0, 0, 0, 0, 1, 0, 0, 0, 1]);
}

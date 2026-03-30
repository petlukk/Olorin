# Terminal Kernel Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SIMD-accelerated interactive terminal (PTY) tiles to Olorin's web-UI, with `ansi_parser.ea` for bulk byte classification and `terminal_diff.ea` for cell-grid diffing.

**Architecture:** Rust owns all stateful logic (PTY lifecycle, ANSI state machine, cell grid). Two Eä kernels handle the parallelizable hot paths: `ansi_parser.ea` classifies raw bytes via SIMD, `terminal_diff.ea` compares cell grids. Transport uses existing SSE-down + POST-up pattern. Canvas renders in the browser.

**Tech Stack:** Rust (libc: openpty/fork/ioctl/poll), Eä SIMD kernels, HTML5 Canvas, SSE

**Spec:** `docs/superpowers/specs/2026-03-30-terminal-kernel-pipeline-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `kernels/ansi_parser.ea` | SIMD byte classifier: ESC, control, digit, final, high-byte detection |
| `kernels/terminal_diff.ea` | SIMD cell-grid comparator: XOR 16-byte cells, produce dirty bitmap |
| `src/interface/pty.rs` | PtySession: openpty, fork/exec bash, read/write, resize, Drop |
| `src/interface/ansi.rs` | ANSI state machine: parse classifier output, update cell grid |
| `tests/pty.rs` | End-to-end PTY tests: open, echo, read back, resize, close |
| `tests/ansi.rs` | ANSI parser tests: SGR colors, cursor movement, ED/EL |
| `tests/terminal_diff.rs` | Diff kernel tests: identical grids, changed cells, full dirty |

### Modified Files

| File | Change |
|------|--------|
| `src/interface/mod.rs` | Add `pub mod pty; pub mod ansi;` |
| `src/kernels/ffi.rs` | Add type aliases, KernelTable fields, load + public wrappers for 2 new kernels |
| `src/interface/server.rs` | Add 5 terminal endpoints + SSE loop thread |
| `web/chat.html` | Add `createTermTile()`, Canvas renderer, `Alt+S` keybinding |

---

### Task 1: ansi_parser.ea Kernel

**Files:**
- Create: `kernels/ansi_parser.ea`
- Test: `tests/ansi_parser_kernel.rs` (via FFI after Task 3)

- [ ] **Step 1: Create the kernel file**

```ea
// ANSI byte classifier — categorize every byte for terminal parsing.
//
// Classes: printable(0), ESC(1), bracket(2), digit(3), semicolon(4),
//          final(5), control(6), high-byte(7)
// Single streaming pass over u8x16 vectors.

export func ansi_classify(data: *u8, out classes: *mut u8, len: i32) {
    let v_esc: u8x16 = splat(27)
    let v_bracket: u8x16 = splat(91)
    let v_semicolon: u8x16 = splat(59)
    let v_0: u8x16 = splat(48)
    let v_9: u8x16 = splat(57)
    let v_at: u8x16 = splat(64)
    let v_tilde: u8x16 = splat(126)
    let v_1a: u8x16 = splat(26)
    let v_7f: u8x16 = splat(127)
    let v_zero: u8x16 = splat(0)
    let v_ones: u8x16 = splat(255)

    let c1: u8x16 = splat(1)
    let c2: u8x16 = splat(2)
    let c3: u8x16 = splat(3)
    let c4: u8x16 = splat(4)
    let c5: u8x16 = splat(5)
    let c6: u8x16 = splat(6)
    let c7: u8x16 = splat(7)

    let n_full: i32 = (len / 16) * 16
    for i in 0..n_full step 16 {
        let b: u8x16 = load(data, i)

        // Start with class 0 (printable) as default
        let mut cls: u8x16 = v_zero

        // Class 7: high byte (>= 0x80)
        let high: u8x16 = select(b .> v_7f, v_ones, v_zero)
        cls = select(high .== v_ones, c7, cls)

        // Class 6: control (0x00-0x1A, excluding ESC=0x1B)
        let ctrl: u8x16 = select(b .<= v_1a, v_ones, v_zero)
        cls = select(ctrl .== v_ones, c6, cls)

        // Class 1: ESC (0x1B)
        let esc: u8x16 = select(b .== v_esc, v_ones, v_zero)
        cls = select(esc .== v_ones, c1, cls)

        // Class 2: bracket '[' (0x5B) — Rust state machine decides if CSI
        let brk: u8x16 = select(b .== v_bracket, v_ones, v_zero)
        cls = select(brk .== v_ones, c2, cls)

        // Class 3: digit (0x30-0x39)
        let ge0: u8x16 = select(b .>= v_0, v_ones, v_zero)
        let le9: u8x16 = select(b .<= v_9, v_ones, v_zero)
        let dig: u8x16 = ge0 .& le9
        cls = select(dig .== v_ones, c3, cls)

        // Class 4: semicolon (0x3B)
        let semi: u8x16 = select(b .== v_semicolon, v_ones, v_zero)
        cls = select(semi .== v_ones, c4, cls)

        // Class 5: final byte (0x40-0x7E)
        let ge_at: u8x16 = select(b .>= v_at, v_ones, v_zero)
        let le_tilde: u8x16 = select(b .<= v_tilde, v_ones, v_zero)
        let final_byte: u8x16 = ge_at .& le_tilde
        // Exclude bracket (0x5B) from final — it's class 2
        let not_brk: u8x16 = select(b .== v_bracket, v_zero, v_ones)
        let final_clean: u8x16 = final_byte .& not_brk
        cls = select(final_clean .== v_ones, c5, cls)

        store(classes, i, cls)
    }

    // Scalar tail
    for i in n_full..len {
        let b: u8 = data[i]
        let mut c: u8 = 0
        if b == 27 {
            c = 1
        } else {
            if b == 91 {
                c = 2
            } else {
                if b >= 48 && b <= 57 {
                    c = 3
                } else {
                    if b == 59 {
                        c = 4
                    } else {
                        if b >= 64 && b <= 126 {
                            c = 5
                        } else {
                            if b <= 26 {
                                c = 6
                            } else {
                                if b > 127 {
                                    c = 7
                                }
                            }
                        }
                    }
                }
            }
        }
        classes[i] = c
    }
}
```

- [ ] **Step 2: Verify it compiles**

Run from project root:
```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" ea kernels/ansi_parser.ea --lib -o /tmp/libansi_parser.so
```
Expected: no errors, `/tmp/libansi_parser.so` created.

- [ ] **Step 3: Commit**

```bash
git add kernels/ansi_parser.ea
git commit -m "feat: add ansi_parser.ea SIMD byte classifier kernel"
```

---

### Task 2: terminal_diff.ea Kernel

**Files:**
- Create: `kernels/terminal_diff.ea`

- [ ] **Step 1: Create the kernel file**

Each Cell is 16 bytes (`#[repr(C)]`: u32 ch + u32 fg + u32 bg + u8 flags + 3 pad). Load as `u8x16`, XOR, check if any byte differs.

```ea
// Terminal cell-grid diff — compare old and new grids, produce dirty bitmap.
//
// Each cell is 16 bytes. XOR old vs new, movemask to detect any difference.
// Output: dirty[i] = 1 if cell i changed, 0 otherwise.

export func terminal_diff(old_grid: *u8, new_grid: *u8, out dirty: *mut u8, n_cells: i32) {
    let v_zero: u8x16 = splat(0)
    let v_one_byte: u8x16 = splat(1)

    // Each cell = 16 bytes, so byte offset = cell_index * 16
    for i in 0..n_cells {
        let offset: i32 = i * 16
        let old_cell: u8x16 = load(old_grid, offset)
        let new_cell: u8x16 = load(new_grid, offset)
        let diff: u8x16 = old_cell .^ new_cell
        let any_diff: i32 = movemask(diff)
        if any_diff != 0 {
            dirty[i] = 1
        } else {
            dirty[i] = 0
        }
    }
}
```

- [ ] **Step 2: Verify it compiles**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" ea kernels/terminal_diff.ea --lib -o /tmp/libterminal_diff.so
```
Expected: no errors, `/tmp/libterminal_diff.so` created.

- [ ] **Step 3: Commit**

```bash
git add kernels/terminal_diff.ea
git commit -m "feat: add terminal_diff.ea SIMD cell-grid comparator kernel"
```

---

### Task 3: FFI Wrappers for New Kernels

**Files:**
- Modify: `src/kernels/ffi.rs`

- [ ] **Step 1: Add type aliases**

Add after line 26 (`type ZeroizeFn`) in `src/kernels/ffi.rs`:

```rust
type AnsiClassifyFn     = unsafe extern "C" fn(*const u8, *mut u8, i32);
type TerminalDiffFn     = unsafe extern "C" fn(*const u8, *const u8, *mut u8, i32);
```

- [ ] **Step 2: Add fields to KernelTable**

Add before `}` closing the `KernelTable` struct (after line 85, the `pretokenize` field):

```rust
    pub ansi_classify:            AnsiClassifyFn,
    pub terminal_diff:            TerminalDiffFn,
```

- [ ] **Step 3: Load the new kernel libraries in `load_kernels()`**

Add after line 174 (`let pretokenize_lib = load("pretokenize")?;`):

```rust
    let ansi_parser_lib  = load("ansi_parser")?;
    let terminal_diff_lib = load("terminal_diff")?;
```

- [ ] **Step 4: Add symbol resolution in the KernelTable constructor**

Add after line 244 (`pretokenize: std::mem::transmute(sym(&pretokenize_lib, b"pretokenize\0")?),`):

```rust
            ansi_classify: std::mem::transmute(
                sym(&ansi_parser_lib, b"ansi_classify\0")?),
            terminal_diff: std::mem::transmute(
                sym(&terminal_diff_lib, b"terminal_diff\0")?),
```

- [ ] **Step 5: Add libraries to the libs vec**

Add `ansi_parser_lib, terminal_diff_lib,` to the `libs: vec![...]` (after `pretokenize_lib,` on line 249).

- [ ] **Step 6: Add public wrappers**

Add at the end of the file, before the final closing (after the `chacha20_search_v2` wrapper):

```rust
/// SIMD-accelerated ANSI byte classification.
/// Classifies each byte: 0=printable, 1=ESC, 2=bracket, 3=digit, 4=semicolon,
/// 5=final, 6=control, 7=high-byte.
/// # Safety
/// `data` must be valid for `len` bytes. `classes` must be valid for `len` bytes.
pub unsafe fn ansi_classify(data: *const u8, classes: *mut u8, len: i32) {
    (k().ansi_classify)(data, classes, len);
}

/// SIMD-accelerated terminal cell-grid diff.
/// Compares old_grid vs new_grid (each cell = 16 bytes), writes dirty bitmap.
/// # Safety
/// `old_grid` and `new_grid` must be valid for `n_cells * 16` bytes.
/// `dirty` must be valid for `n_cells` bytes.
pub unsafe fn terminal_diff(
    old_grid: *const u8, new_grid: *const u8, dirty: *mut u8, n_cells: i32,
) {
    (k().terminal_diff)(old_grid, new_grid, dirty, n_cells);
}
```

- [ ] **Step 7: Build to verify**

```bash
cd /home/peter/projects/olorin1
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build 2>&1
```
Expected: compiles without errors.

- [ ] **Step 8: Commit**

```bash
git add src/kernels/ffi.rs
git commit -m "feat: add FFI wrappers for ansi_parser and terminal_diff kernels"
```

---

### Task 4: Cell Types and ANSI State Machine (`ansi.rs`)

**Files:**
- Create: `src/interface/ansi.rs`
- Modify: `src/interface/mod.rs`

- [ ] **Step 1: Write failing test for SGR color parsing**

Create `tests/ansi.rs`:

```rust
//! Tests for ANSI state machine — SGR, cursor movement, erase.

mod common {
    use olorin::interface::ansi::{Cell, CellAttrs, TermGrid};

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
    // Cursor should be at col 5
    assert_eq!(g.cursor(), (0, 5));
}

#[test]
fn sgr_red_foreground() {
    let mut g = common::make_grid(80, 24);
    // ESC[31m = set fg to red (ANSI color 1)
    common::feed(&mut g, b"\x1b[31mX");
    let cell = g.cell(0, 0);
    assert_eq!(cell.ch, b'X' as u32);
    // ANSI red = index 1 → typically #ff0000 or similar
    assert_ne!(cell.fg, 0x00cdd6f4); // not default
}

#[test]
fn sgr_reset() {
    let mut g = common::make_grid(80, 24);
    common::feed(&mut g, b"\x1b[31mA\x1b[0mB");
    let a = g.cell(0, 0);
    let b = g.cell(0, 1);
    assert_ne!(a.fg, b.fg); // A is red, B is default
}

#[test]
fn cursor_movement_csi_h() {
    let mut g = common::make_grid(80, 24);
    // ESC[3;5H = move cursor to row 3, col 5 (1-indexed)
    common::feed(&mut g, b"\x1b[3;5HX");
    assert_eq!(g.cell(2, 4).ch, b'X' as u32);
    assert_eq!(g.cursor(), (2, 5));
}

#[test]
fn erase_display_csi_2j() {
    let mut g = common::make_grid(80, 24);
    common::feed(&mut g, b"ABCDE");
    common::feed(&mut g, b"\x1b[2J");
    // All cells should be cleared
    assert_eq!(g.cell(0, 0).ch, b' ' as u32);
    assert_eq!(g.cell(0, 4).ch, b' ' as u32);
}

#[test]
fn erase_line_csi_2k() {
    let mut g = common::make_grid(80, 24);
    common::feed(&mut g, b"ABCDE\x1b[2K");
    // Entire first row cleared
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
    // Move to (0,5), then CSI 2A = up 2, CSI 3C = forward 3
    common::feed(&mut g, b"\x1b[5;6H");   // row 5, col 6 (1-indexed)
    common::feed(&mut g, b"\x1b[2A");      // up 2 → row 3
    common::feed(&mut g, b"\x1b[3C");      // forward 3 → col 9
    common::feed(&mut g, b"X");
    assert_eq!(g.cell(2, 8).ch, b'X' as u32); // row 2 (0-idx), col 8 (0-idx)
}

#[test]
fn sgr_256_color() {
    let mut g = common::make_grid(80, 24);
    // ESC[38;5;196m = fg color 196 (bright red in 256-color palette)
    common::feed(&mut g, b"\x1b[38;5;196mX");
    let cell = g.cell(0, 0);
    assert_eq!(cell.ch, b'X' as u32);
    assert_ne!(cell.fg, 0x00cdd6f4); // not default
}

#[test]
fn sgr_truecolor() {
    let mut g = common::make_grid(80, 24);
    // ESC[38;2;255;128;0m = fg truecolor RGB(255,128,0)
    common::feed(&mut g, b"\x1b[38;2;255;128;0mX");
    let cell = g.cell(0, 0);
    assert_eq!(cell.fg, 0x00ff8000);
}

#[test]
fn bold_flag() {
    let mut g = common::make_grid(80, 24);
    common::feed(&mut g, b"\x1b[1mX");
    assert!(g.cell(0, 0).flags & 0x01 != 0); // BOLD bit
}
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
cd /home/peter/projects/olorin1
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test --test ansi 2>&1 | head -20
```
Expected: compilation error — `olorin::interface::ansi` module doesn't exist.

- [ ] **Step 3: Add module declaration**

Edit `src/interface/mod.rs` to:

```rust
pub mod terminal;
pub mod server;
pub mod exec;
pub mod pty;
pub mod ansi;
```

- [ ] **Step 4: Write `ansi.rs` — types and TermGrid**

Create `src/interface/ansi.rs`:

```rust
//! ANSI terminal state machine — driven by ansi_parser.ea classifier output.
//!
//! TermGrid owns the cell buffer. `feed()` takes raw PTY bytes, runs the SIMD
//! classifier, then interprets the result to update cells, cursor, and attributes.

use crate::kernels::ffi;

// ── Cell representation (16 bytes, SIMD-friendly) ────────────────────────────

/// A single terminal cell. 16 bytes, #[repr(C)] for SIMD alignment.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Cell {
    pub ch:    u32, // Unicode codepoint
    pub fg:    u32, // 0x00RRGGBB
    pub bg:    u32, // 0x00RRGGBB
    pub flags: u8,  // bit 0=bold, 1=italic, 2=underline, 3=inverse, 4=dim
    pub _pad:  [u8; 3],
}

const DEFAULT_FG: u32 = 0x00cdd6f4; // Catppuccin text
const DEFAULT_BG: u32 = 0x001e1e2e; // Catppuccin base

impl Default for Cell {
    fn default() -> Self {
        Self { ch: b' ' as u32, fg: DEFAULT_FG, bg: DEFAULT_BG, flags: 0, _pad: [0; 3] }
    }
}

pub const BOLD: u8      = 0x01;
pub const ITALIC: u8    = 0x02;
pub const UNDERLINE: u8 = 0x04;
pub const INVERSE: u8   = 0x08;
pub const DIM: u8       = 0x10;

// ── Current attributes ───────────────────────────────────────────────────────

#[derive(Clone, Copy)]
pub struct CellAttrs {
    pub fg:    u32,
    pub bg:    u32,
    pub flags: u8,
}

impl Default for CellAttrs {
    fn default() -> Self {
        Self { fg: DEFAULT_FG, bg: DEFAULT_BG, flags: 0 }
    }
}

// ── ANSI parse state ─────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq)]
enum ParseState {
    Ground,
    Escape,
    Csi,
    OscSkip,
}

// ── TermGrid ─────────────────────────────────────────────────────────────────

pub struct TermGrid {
    pub cols: u16,
    pub rows: u16,
    cells: Vec<Cell>,
    cursor_row: u16,
    cursor_col: u16,
    attrs: CellAttrs,
    state: ParseState,
    params: [u16; 16],
    param_idx: usize,
    question_mark: bool,
    cursor_visible: bool,
}

impl TermGrid {
    pub fn new(cols: u16, rows: u16) -> Self {
        let n = cols as usize * rows as usize;
        Self {
            cols,
            rows,
            cells: vec![Cell::default(); n],
            cursor_row: 0,
            cursor_col: 0,
            attrs: CellAttrs::default(),
            state: ParseState::Ground,
            params: [0; 16],
            param_idx: 0,
            question_mark: false,
            cursor_visible: true,
        }
    }

    pub fn cell(&self, row: u16, col: u16) -> Cell {
        self.cells[row as usize * self.cols as usize + col as usize]
    }

    pub fn cursor(&self) -> (u16, u16) {
        (self.cursor_row, self.cursor_col)
    }

    pub fn cursor_visible(&self) -> bool {
        self.cursor_visible
    }

    pub fn cells_ptr(&self) -> *const u8 {
        self.cells.as_ptr() as *const u8
    }

    pub fn cells_mut_ptr(&mut self) -> *mut u8 {
        self.cells.as_mut_ptr() as *mut u8
    }

    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Feed raw PTY bytes through SIMD classifier + state machine.
    /// `scan_buf` must be at least `data.len()` bytes (reusable scratch).
    pub fn feed(&mut self, data: &[u8], scan_buf: &mut [u8]) {
        let len = data.len();
        if len == 0 { return; }

        // SIMD classify
        unsafe {
            ffi::ansi_classify(
                data.as_ptr(),
                scan_buf.as_mut_ptr(),
                len as i32,
            );
        }

        // Drive state machine from classifier output
        for i in 0..len {
            let b = data[i];
            let cls = scan_buf[i];
            self.step(b, cls);
        }
    }

    fn step(&mut self, b: u8, cls: u8) {
        match self.state {
            ParseState::Ground => self.ground(b, cls),
            ParseState::Escape => self.escape(b, cls),
            ParseState::Csi    => self.csi(b, cls),
            ParseState::OscSkip => {
                // Skip until BEL (0x07) or ST (ESC \)
                if b == 0x07 || b == 0x9C { self.state = ParseState::Ground; }
                if b == 0x1B { self.state = ParseState::Escape; }
            }
        }
    }

    fn ground(&mut self, b: u8, cls: u8) {
        match cls {
            0 | 2 | 3 | 4 | 5 => self.put_char(b as u32), // printable
            1 => self.state = ParseState::Escape,            // ESC
            6 => self.control(b),                            // control char
            7 => self.put_char(b as u32),                    // high byte (simplified: treat as printable for now)
            _ => {}
        }
    }

    fn escape(&mut self, b: u8, _cls: u8) {
        match b {
            b'[' => {
                self.state = ParseState::Csi;
                self.params = [0; 16];
                self.param_idx = 0;
                self.question_mark = false;
            }
            b']' => { self.state = ParseState::OscSkip; }
            _ => { self.state = ParseState::Ground; }
        }
    }

    fn csi(&mut self, b: u8, cls: u8) {
        match cls {
            3 => {
                // digit — accumulate parameter
                if self.param_idx < 16 {
                    self.params[self.param_idx] = self.params[self.param_idx]
                        .saturating_mul(10)
                        .saturating_add((b - b'0') as u16);
                }
            }
            4 => {
                // semicolon — advance to next parameter
                if self.param_idx < 15 { self.param_idx += 1; }
            }
            5 | 0 => {
                // final byte or other — execute CSI command
                if b == b'?' {
                    self.question_mark = true;
                    return;
                }
                let n_params = self.param_idx + 1;
                self.execute_csi(b, n_params);
                self.state = ParseState::Ground;
            }
            _ => { self.state = ParseState::Ground; }
        }
    }

    fn execute_csi(&mut self, cmd: u8, n_params: usize) {
        let p0 = self.params[0];
        let p1 = self.params[1];

        match cmd {
            b'm' => self.sgr(n_params),
            b'H' | b'f' => {
                // CUP — cursor position (1-indexed, default 1;1)
                let row = if p0 == 0 { 0 } else { (p0 - 1).min(self.rows - 1) };
                let col = if p1 == 0 { 0 } else { (p1 - 1).min(self.cols - 1) };
                self.cursor_row = row;
                self.cursor_col = col;
            }
            b'A' => { // CUU — cursor up
                let n = p0.max(1);
                self.cursor_row = self.cursor_row.saturating_sub(n);
            }
            b'B' => { // CUD — cursor down
                let n = p0.max(1);
                self.cursor_row = (self.cursor_row + n).min(self.rows - 1);
            }
            b'C' => { // CUF — cursor forward
                let n = p0.max(1);
                self.cursor_col = (self.cursor_col + n).min(self.cols - 1);
            }
            b'D' => { // CUB — cursor back
                let n = p0.max(1);
                self.cursor_col = self.cursor_col.saturating_sub(n);
            }
            b'J' => self.erase_display(p0),
            b'K' => self.erase_line(p0),
            b'h' | b'l' => {
                if self.question_mark && p0 == 25 {
                    self.cursor_visible = cmd == b'h';
                }
            }
            _ => {} // Unknown CSI — ignore
        }
    }

    fn sgr(&mut self, n_params: usize) {
        let mut i = 0;
        while i < n_params {
            match self.params[i] {
                0 => self.attrs = CellAttrs::default(),
                1 => self.attrs.flags |= BOLD,
                3 => self.attrs.flags |= ITALIC,
                4 => self.attrs.flags |= UNDERLINE,
                7 => self.attrs.flags |= INVERSE,
                2 => self.attrs.flags |= DIM,
                22 => self.attrs.flags &= !(BOLD | DIM),
                23 => self.attrs.flags &= !ITALIC,
                24 => self.attrs.flags &= !UNDERLINE,
                27 => self.attrs.flags &= !INVERSE,
                // Standard fg colors (30-37)
                c @ 30..=37 => self.attrs.fg = ansi_color(c - 30),
                39 => self.attrs.fg = DEFAULT_FG,
                // Standard bg colors (40-47)
                c @ 40..=47 => self.attrs.bg = ansi_color(c - 40),
                49 => self.attrs.bg = DEFAULT_BG,
                // Bright fg (90-97)
                c @ 90..=97 => self.attrs.fg = ansi_bright_color(c - 90),
                // Bright bg (100-107)
                c @ 100..=107 => self.attrs.bg = ansi_bright_color(c - 100),
                // 256-color or truecolor fg
                38 => {
                    if i + 1 < n_params && self.params[i + 1] == 5 && i + 2 < n_params {
                        self.attrs.fg = color_256(self.params[i + 2]);
                        i += 2;
                    } else if i + 1 < n_params && self.params[i + 1] == 2 && i + 4 < n_params {
                        let r = self.params[i + 2] as u32;
                        let g = self.params[i + 3] as u32;
                        let b = self.params[i + 4] as u32;
                        self.attrs.fg = (r << 16) | (g << 8) | b;
                        i += 4;
                    }
                }
                // 256-color or truecolor bg
                48 => {
                    if i + 1 < n_params && self.params[i + 1] == 5 && i + 2 < n_params {
                        self.attrs.bg = color_256(self.params[i + 2]);
                        i += 2;
                    } else if i + 1 < n_params && self.params[i + 1] == 2 && i + 4 < n_params {
                        let r = self.params[i + 2] as u32;
                        let g = self.params[i + 3] as u32;
                        let b = self.params[i + 4] as u32;
                        self.attrs.bg = (r << 16) | (g << 8) | b;
                        i += 4;
                    }
                }
                _ => {}
            }
            i += 1;
        }
    }

    fn put_char(&mut self, ch: u32) {
        if self.cursor_col >= self.cols {
            self.cursor_col = 0;
            self.cursor_row += 1;
            if self.cursor_row >= self.rows {
                self.scroll_up();
                self.cursor_row = self.rows - 1;
            }
        }
        let idx = self.cursor_row as usize * self.cols as usize + self.cursor_col as usize;
        self.cells[idx] = Cell {
            ch,
            fg: self.attrs.fg,
            bg: self.attrs.bg,
            flags: self.attrs.flags,
            _pad: [0; 3],
        };
        self.cursor_col += 1;
    }

    fn control(&mut self, b: u8) {
        match b {
            b'\n' => {
                self.cursor_row += 1;
                if self.cursor_row >= self.rows {
                    self.scroll_up();
                    self.cursor_row = self.rows - 1;
                }
            }
            b'\r' => { self.cursor_col = 0; }
            b'\t' => {
                let next_tab = ((self.cursor_col / 8) + 1) * 8;
                self.cursor_col = next_tab.min(self.cols - 1);
            }
            8 => { // backspace
                self.cursor_col = self.cursor_col.saturating_sub(1);
            }
            _ => {} // bell, other — ignore
        }
    }

    fn erase_display(&mut self, mode: u16) {
        let default = Cell::default();
        match mode {
            0 => {
                // Erase from cursor to end
                let start = self.cursor_row as usize * self.cols as usize + self.cursor_col as usize;
                for cell in &mut self.cells[start..] { *cell = default; }
            }
            1 => {
                // Erase from start to cursor
                let end = self.cursor_row as usize * self.cols as usize + self.cursor_col as usize + 1;
                let end = end.min(self.cells.len());
                for cell in &mut self.cells[..end] { *cell = default; }
            }
            2 | 3 => {
                // Erase entire display
                for cell in &mut self.cells { *cell = default; }
            }
            _ => {}
        }
    }

    fn erase_line(&mut self, mode: u16) {
        let default = Cell::default();
        let row_start = self.cursor_row as usize * self.cols as usize;
        match mode {
            0 => {
                let start = row_start + self.cursor_col as usize;
                let end = row_start + self.cols as usize;
                for cell in &mut self.cells[start..end] { *cell = default; }
            }
            1 => {
                let end = row_start + self.cursor_col as usize + 1;
                for cell in &mut self.cells[row_start..end] { *cell = default; }
            }
            2 => {
                let end = row_start + self.cols as usize;
                for cell in &mut self.cells[row_start..end] { *cell = default; }
            }
            _ => {}
        }
    }

    fn scroll_up(&mut self) {
        let cols = self.cols as usize;
        self.cells.copy_within(cols.., 0);
        let start = (self.rows as usize - 1) * cols;
        for cell in &mut self.cells[start..] { *cell = Cell::default(); }
    }

    /// Swap cells into prev buffer, return previous buffer pointer.
    /// Used by PtySession to feed terminal_diff.
    pub fn swap_prev(&mut self, prev: &mut Vec<Cell>) {
        std::mem::swap(&mut self.cells, prev);
        // Copy current state back — prev now has old, cells has new (empty)
        // Actually we want: prev = old snapshot, cells = current
        // So copy cells (which are now "old" after swap) back
        prev.copy_from_slice(&self.cells);
        // Wait — that defeats the purpose. The correct pattern:
        // Before feed: copy cells → prev. After feed: diff cells vs prev.
        // This is handled in PtySession, not here.
    }
}

// ── Color tables ─────────────────────────────────────────────────────────────

fn ansi_color(idx: u16) -> u32 {
    match idx {
        0 => 0x0045475a, // black (surface1)
        1 => 0x00f38ba8, // red
        2 => 0x00a6e3a1, // green
        3 => 0x00f9e2af, // yellow
        4 => 0x0089b4fa, // blue
        5 => 0x00cba6f7, // magenta/mauve
        6 => 0x0094e2d5, // cyan/teal
        7 => 0x00bac2de, // white (subtext1)
        _ => DEFAULT_FG,
    }
}

fn ansi_bright_color(idx: u16) -> u32 {
    match idx {
        0 => 0x00585b70, // bright black (surface2)
        1 => 0x00f38ba8, // bright red
        2 => 0x00a6e3a1, // bright green
        3 => 0x00f9e2af, // bright yellow
        4 => 0x0089b4fa, // bright blue
        5 => 0x00cba6f7, // bright magenta
        6 => 0x0094e2d5, // bright cyan
        7 => 0x00cdd6f4, // bright white (text)
        _ => DEFAULT_FG,
    }
}

fn color_256(idx: u16) -> u32 {
    match idx {
        0..=7   => ansi_color(idx),
        8..=15  => ansi_bright_color(idx - 8),
        16..=231 => {
            // 6x6x6 color cube
            let n = idx - 16;
            let b = (n % 6) as u32;
            let g = ((n / 6) % 6) as u32;
            let r = (n / 36) as u32;
            let scale = |v: u32| if v == 0 { 0 } else { 55 + 40 * v };
            (scale(r) << 16) | (scale(g) << 8) | scale(b)
        }
        232..=255 => {
            // Grayscale ramp
            let v = (8 + 10 * (idx - 232)) as u32;
            (v << 16) | (v << 8) | v
        }
        _ => DEFAULT_FG,
    }
}
```

- [ ] **Step 5: Run tests**

```bash
cd /home/peter/projects/olorin1
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test --test ansi 2>&1
```
Expected: all 11 tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/interface/ansi.rs src/interface/mod.rs tests/ansi.rs
git commit -m "feat: add ANSI state machine with TermGrid, SGR, cursor, erase"
```

---

### Task 5: PTY Session (`pty.rs`)

**Files:**
- Create: `src/interface/pty.rs`
- Create: `tests/pty.rs`

- [ ] **Step 1: Write failing PTY test**

Create `tests/pty.rs`:

```rust
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
    // Send a command that produces known output
    session.write_bytes(b"echo OLORIN_TEST_MARKER\n");
    // Give bash time to process
    std::thread::sleep(std::time::Duration::from_millis(200));
    // Read and apply — should populate cell grid
    let patch = session.read_and_apply();
    // The grid should contain our marker somewhere
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
```

- [ ] **Step 2: Run tests, verify they fail**

```bash
cd /home/peter/projects/olorin1
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test --test pty 2>&1 | head -10
```
Expected: compilation error — `olorin::interface::pty` doesn't exist.

- [ ] **Step 3: Write `pty.rs`**

Create `src/interface/pty.rs`:

```rust
//! PTY session — openpty + fork/exec bash with SIMD-accelerated I/O.
//!
//! Each PtySession owns a master fd, child process, cell grid, and scratch
//! buffers. `read_and_apply()` reads from the PTY, runs the ANSI pipeline,
//! and returns a dirty bitmap for diffing.

use crate::interface::ansi::{Cell, TermGrid};
use crate::kernels::ffi;
use std::io;

/// A live PTY session with its own bash process and terminal state.
pub struct PtySession {
    master_fd: i32,
    child_pid: i32,
    grid: TermGrid,
    prev_cells: Vec<Cell>,
    scan_buf: Vec<u8>,
    read_buf: Vec<u8>,
    dirty_buf: Vec<u8>,
}

impl PtySession {
    /// Open a new PTY, fork bash, set window size.
    pub fn new(cols: u16, rows: u16) -> io::Result<Self> {
        let mut master: i32 = 0;
        let mut slave: i32 = 0;

        let ret = unsafe { libc::openpty(&mut master, &mut slave, std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut()) };
        if ret != 0 {
            return Err(io::Error::last_os_error());
        }

        // Set window size
        let ws = libc::winsize {
            ws_row: rows,
            ws_col: cols,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        unsafe { libc::ioctl(master, libc::TIOCSWINSZ, &ws) };

        let pid = unsafe { libc::fork() };
        if pid < 0 {
            unsafe { libc::close(master); libc::close(slave); }
            return Err(io::Error::last_os_error());
        }

        if pid == 0 {
            // Child
            unsafe {
                libc::close(master);
                libc::setsid();
                libc::ioctl(slave, libc::TIOCSCTTY, 0i32);
                libc::dup2(slave, 0);
                libc::dup2(slave, 1);
                libc::dup2(slave, 2);
                if slave > 2 { libc::close(slave); }

                // Exec bash
                let shell = b"/bin/bash\0".as_ptr() as *const libc::c_char;
                let argv: [*const libc::c_char; 2] = [shell, std::ptr::null()];
                libc::execvp(shell, argv.as_ptr());
                libc::_exit(127);
            }
        }

        // Parent
        unsafe { libc::close(slave); }

        // Set master to non-blocking for polling
        unsafe {
            let flags = libc::fcntl(master, libc::F_GETFL);
            libc::fcntl(master, libc::F_SETFL, flags | libc::O_NONBLOCK);
        }

        let n_cells = cols as usize * rows as usize;
        Ok(Self {
            master_fd: master,
            child_pid: pid,
            grid: TermGrid::new(cols, rows),
            prev_cells: vec![Cell::default(); n_cells],
            scan_buf: vec![0u8; 8192],
            read_buf: vec![0u8; 8192],
            dirty_buf: vec![0u8; n_cells],
        })
    }

    pub fn grid(&self) -> &TermGrid {
        &self.grid
    }

    pub fn child_alive(&self) -> bool {
        let mut status: i32 = 0;
        let r = unsafe { libc::waitpid(self.child_pid, &mut status, libc::WNOHANG) };
        r == 0
    }

    /// Write raw bytes to the PTY (keyboard input from web client).
    pub fn write_bytes(&mut self, data: &[u8]) {
        let mut written = 0;
        while written < data.len() {
            let n = unsafe {
                libc::write(
                    self.master_fd,
                    data[written..].as_ptr() as *const libc::c_void,
                    data.len() - written,
                )
            };
            if n <= 0 { break; }
            written += n as usize;
        }
    }

    /// Read from PTY, run ANSI pipeline, return dirty cell indices.
    /// Returns a slice of the dirty bitmap (1=changed, 0=unchanged).
    pub fn read_and_apply(&mut self) -> &[u8] {
        // Snapshot current grid into prev
        self.prev_cells.copy_from_slice(
            unsafe { std::slice::from_raw_parts(self.grid.cells_ptr() as *const Cell, self.grid.cell_count()) }
        );

        // Read all available data from PTY
        let mut total_read = 0;
        loop {
            let n = unsafe {
                libc::read(
                    self.master_fd,
                    self.read_buf[total_read..].as_mut_ptr() as *mut libc::c_void,
                    self.read_buf.len() - total_read,
                )
            };
            if n <= 0 { break; }
            total_read += n as usize;
            if total_read >= self.read_buf.len() { break; }
        }

        if total_read == 0 {
            // No data — clear dirty
            for d in &mut self.dirty_buf { *d = 0; }
            return &self.dirty_buf;
        }

        // Ensure scan_buf is large enough
        if self.scan_buf.len() < total_read {
            self.scan_buf.resize(total_read, 0);
        }

        // Feed through ANSI pipeline
        self.grid.feed(&self.read_buf[..total_read], &mut self.scan_buf);

        // SIMD diff
        let n_cells = self.grid.cell_count();
        unsafe {
            ffi::terminal_diff(
                self.prev_cells.as_ptr() as *const u8,
                self.grid.cells_ptr(),
                self.dirty_buf.as_mut_ptr(),
                n_cells as i32,
            );
        }

        &self.dirty_buf
    }

    /// Resize the PTY and terminal grid.
    pub fn resize(&mut self, cols: u16, rows: u16) {
        let ws = libc::winsize {
            ws_row: rows,
            ws_col: cols,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        unsafe {
            libc::ioctl(self.master_fd, libc::TIOCSWINSZ, &ws);
            libc::kill(self.child_pid, libc::SIGWINCH);
        }

        self.grid = TermGrid::new(cols, rows);
        let n_cells = cols as usize * rows as usize;
        self.prev_cells = vec![Cell::default(); n_cells];
        self.dirty_buf = vec![0u8; n_cells];
    }

    /// Get the master fd for polling.
    pub fn master_fd(&self) -> i32 {
        self.master_fd
    }
}

impl Drop for PtySession {
    fn drop(&mut self) {
        unsafe {
            libc::kill(self.child_pid, libc::SIGTERM);
            libc::close(self.master_fd);
            libc::waitpid(self.child_pid, std::ptr::null_mut(), libc::WNOHANG);
        }
    }
}
```

- [ ] **Step 4: Run tests**

```bash
cd /home/peter/projects/olorin1
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test --test pty 2>&1
```
Expected: all 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/interface/pty.rs tests/pty.rs
git commit -m "feat: add PtySession with openpty, fork/exec bash, SIMD diff"
```

---

### Task 6: Terminal Diff Kernel Test

**Files:**
- Create: `tests/terminal_diff.rs`

- [ ] **Step 1: Write diff kernel tests**

Create `tests/terminal_diff.rs`:

```rust
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
```

- [ ] **Step 2: Run tests**

```bash
cd /home/peter/projects/olorin1
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test --test terminal_diff 2>&1
```
Expected: all 3 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/terminal_diff.rs
git commit -m "test: add terminal_diff kernel tests"
```

---

### Task 7: Server Endpoints for Terminal Sessions

**Files:**
- Modify: `src/interface/server.rs`

- [ ] **Step 1: Add terminal session state**

Add after the imports at the top of `server.rs` (after line 10):

```rust
use crate::interface::pty::PtySession;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

static NEXT_TERM_ID: AtomicU32 = AtomicU32::new(0);

type TermSessions = Arc<Mutex<HashMap<u32, Arc<Mutex<PtySession>>>>>;

fn term_sessions() -> &'static TermSessions {
    static SESSIONS: OnceLock<TermSessions> = OnceLock::new();
    SESSIONS.get_or_init(|| Arc::new(Mutex::new(HashMap::new())))
}
```

Add `use std::sync::OnceLock;` to the existing imports.

- [ ] **Step 2: Add route matching**

In `handle_connection()`, add new routes before the `_ => 404` match arm (before line 101):

```rust
        ("POST", "/api/term/open") => {
            handle_term_open(stream);
        }
        ("POST", path) if path.starts_with("/api/term/") && path.ends_with("/input") => {
            let id = parse_term_id(path);
            handle_term_input(stream, req, buf, n, id);
        }
        ("POST", path) if path.starts_with("/api/term/") && path.ends_with("/resize") => {
            let id = parse_term_id(path);
            handle_term_resize(stream, req, buf, n, id);
        }
        ("POST", path) if path.starts_with("/api/term/") && path.ends_with("/close") => {
            let id = parse_term_id(path);
            handle_term_close(stream, id);
        }
        ("GET", path) if path.starts_with("/api/term/") && path.ends_with("/stream") => {
            let id = parse_term_id(path);
            handle_term_stream(stream, id);
        }
```

- [ ] **Step 3: Add helper to parse term ID from path**

Add after `escape_json()` (at the end of the file):

```rust
// ── Terminal session handlers ────────────────────────────────────────────────

fn parse_term_id(path: &str) -> u32 {
    // Path format: /api/term/{id}/action
    let parts: Vec<&str> = path.split('/').collect();
    // ["", "api", "term", "{id}", "action"]
    parts.get(3).and_then(|s| s.parse().ok()).unwrap_or(0)
}

fn handle_term_open(stream: &mut std::net::TcpStream) {
    let id = NEXT_TERM_ID.fetch_add(1, Ordering::Relaxed);
    match PtySession::new(80, 24) {
        Ok(session) => {
            let session = Arc::new(Mutex::new(session));
            term_sessions().lock().unwrap().insert(id, session);
            let body = format!("{{\"id\":{id}}}");
            serve_json(stream, &body);
        }
        Err(e) => {
            let escaped = escape_json(&format!("{e}"));
            let body = format!("{{\"error\":\"{escaped}\"}}");
            serve_json(stream, &body);
        }
    }
}

fn handle_term_input(stream: &mut std::net::TcpStream, req: &str, buf: &[u8], n: usize, id: u32) {
    let body_bytes = read_body(stream, req, buf, n);
    let sessions = term_sessions().lock().unwrap();
    if let Some(session) = sessions.get(&id) {
        let mut s = session.lock().unwrap();
        s.write_bytes(&body_bytes);
    }
    serve_json(stream, r#"{"ok":true}"#);
}

fn handle_term_resize(stream: &mut std::net::TcpStream, req: &str, buf: &[u8], n: usize, id: u32) {
    let body_bytes = read_body(stream, req, buf, n);
    let body_str = std::str::from_utf8(&body_bytes).unwrap_or("");
    let cols: u16 = extract_json_number(body_str, "cols").unwrap_or(80) as u16;
    let rows: u16 = extract_json_number(body_str, "rows").unwrap_or(24) as u16;

    let sessions = term_sessions().lock().unwrap();
    if let Some(session) = sessions.get(&id) {
        let mut s = session.lock().unwrap();
        s.resize(cols, rows);
    }
    serve_json(stream, r#"{"ok":true}"#);
}

fn handle_term_close(stream: &mut std::net::TcpStream, id: u32) {
    term_sessions().lock().unwrap().remove(&id);
    serve_json(stream, r#"{"ok":true}"#);
}

fn extract_json_number(json: &str, key: &str) -> Option<u32> {
    let pattern = format!("\"{}\"", key);
    let start = json.find(&pattern)?;
    let after_key = &json[start + pattern.len()..];
    let colon = after_key.find(':')?;
    let rest = after_key[colon + 1..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn handle_term_stream(stream: &mut std::net::TcpStream, id: u32) {
    // SSE headers
    let _ = write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\
         Cache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\n\
         Connection: keep-alive\r\n\r\n"
    );
    let _ = stream.flush();

    let session = {
        let sessions = term_sessions().lock().unwrap();
        match sessions.get(&id) {
            Some(s) => s.clone(),
            None => {
                let _ = write!(stream, "data: {{\"type\":\"error\",\"msg\":\"no such session\"}}\n\n");
                return;
            }
        }
    };

    // Disable read timeout for SSE
    let _ = stream.set_read_timeout(None);

    let mut pollfd = libc::pollfd {
        fd: session.lock().unwrap().master_fd(),
        events: libc::POLLIN,
        revents: 0,
    };

    loop {
        // Poll with 16ms timeout (~60fps)
        let ret = unsafe { libc::poll(&mut pollfd, 1, 16) };

        if ret > 0 && pollfd.revents & libc::POLLIN != 0 {
            let mut s = session.lock().unwrap();
            let dirty = s.read_and_apply();

            // Check if any cells are dirty
            if dirty.iter().any(|&d| d != 0) {
                let grid = s.grid();
                let cols = grid.cols;
                let (crow, ccol) = grid.cursor();

                // Build cell patch JSON
                let mut cells_json = String::with_capacity(1024);
                cells_json.push('[');
                let mut first = true;
                for (i, &d) in dirty.iter().enumerate() {
                    if d != 0 {
                        let row = i / cols as usize;
                        let col = i % cols as usize;
                        let cell = grid.cell(row as u16, col as u16);
                        if !first { cells_json.push(','); }
                        first = false;
                        let ch = if cell.ch >= 32 && cell.ch < 127 {
                            let c = cell.ch as u8 as char;
                            if c == '"' { "\\\"".to_string() }
                            else if c == '\\' { "\\\\".to_string() }
                            else { c.to_string() }
                        } else if cell.ch == 0 || cell.ch == 32 {
                            " ".to_string()
                        } else {
                            char::from_u32(cell.ch).map(|c| c.to_string()).unwrap_or(" ".to_string())
                        };
                        cells_json.push_str(&format!(
                            "{{\"r\":{row},\"c\":{col},\"ch\":\"{ch}\",\"fg\":\"#{:06x}\",\"bg\":\"#{:06x}\",\"fl\":{}}}",
                            cell.fg, cell.bg, cell.flags
                        ));
                    }
                }
                cells_json.push(']');

                let frame = format!(
                    "data: {{\"type\":\"frame\",\"cursor\":[{ccol},{crow}],\"cells\":{cells_json}}}\n\n"
                );
                if write!(stream, "{frame}").is_err() { break; }
                if stream.flush().is_err() { break; }
            }

            // Check if child is still alive
            if !s.child_alive() {
                let _ = write!(stream, "data: {{\"type\":\"exit\",\"code\":0}}\n\n");
                let _ = stream.flush();
                break;
            }
        }

        if ret < 0 { break; }

        // Check if SSE connection is still alive (try zero-length write)
        if pollfd.revents & (libc::POLLHUP | libc::POLLERR) != 0 {
            break;
        }
    }
}
```

- [ ] **Step 4: Build to verify**

```bash
cd /home/peter/projects/olorin1
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build 2>&1
```
Expected: compiles without errors.

- [ ] **Step 5: Commit**

```bash
git add src/interface/server.rs
git commit -m "feat: add terminal session endpoints (open/input/resize/close/stream)"
```

---

### Task 8: Web-UI Terminal Tile

**Files:**
- Modify: `web/chat.html`

- [ ] **Step 1: Add terminal CSS**

Add before `</style>` (after line 49, the `.repl-cursor` rule):

```css
.term-canvas{flex:1;min-height:0;display:block;background:var(--base);image-rendering:pixelated}
```

- [ ] **Step 2: Add `createTermTile()` function**

Add after the `createReplTile()` function (after line 150):

```javascript
function createTermTile(id){
  const t=el('div','tile');t.dataset.id=id;
  const hdr=el('div','tile-header');
  const hl=el('span','');const ht=span('c-green','term');hl.appendChild(ht);hl.appendChild(document.createTextNode(' shell'));
  hdr.appendChild(hl);
  const cb=el('button','tile-close','×');cb.onclick=()=>{
    fetch('/api/term/'+t._termId+'/close',{method:'POST'});closeTile(id);
  };hdr.appendChild(cb);
  t.appendChild(hdr);
  const canvas=document.createElement('canvas');canvas.className='term-canvas';
  t.appendChild(canvas);
  t._canvas=canvas;t._termId=null;t._cells={};t._cursor=[0,0];t._cols=80;t._rows=24;
  t._cellW=0;t._cellH=0;
  // Open PTY session
  fetch('/api/term/open',{method:'POST'}).then(r=>r.json()).then(d=>{
    t._termId=d.id;
    initTermCanvas(t);
    startTermStream(t);
  });
  return t;
}

function initTermCanvas(t){
  const canvas=t._canvas;
  const rect=canvas.parentElement.getBoundingClientRect();
  const ctx=canvas.getContext('2d');
  ctx.font='13px "JetBrains Mono",monospace';
  const m=ctx.measureText('M');
  t._cellW=Math.ceil(m.width);
  t._cellH=Math.ceil(13*1.4);
  t._cols=Math.floor(rect.width/t._cellW)||80;
  t._rows=Math.floor((rect.height-28)/t._cellH)||24;
  canvas.width=t._cols*t._cellW;
  canvas.height=t._rows*t._cellH;
  canvas.style.width=canvas.width+'px';
  canvas.style.height=canvas.height+'px';
  // Clear
  ctx.fillStyle='#1e1e2e';ctx.fillRect(0,0,canvas.width,canvas.height);
  // Resize PTY
  if(t._termId!=null){
    fetch('/api/term/'+t._termId+'/resize',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({cols:t._cols,rows:t._rows})});
  }
  // Keyboard
  canvas.tabIndex=0;
  canvas.addEventListener('keydown',e=>{
    e.preventDefault();
    let seq='';
    if(e.key.length===1&&!e.ctrlKey&&!e.altKey){seq=e.key;}
    else if(e.key==='Enter'){seq='\r';}
    else if(e.key==='Backspace'){seq='\x7f';}
    else if(e.key==='Tab'){seq='\t';}
    else if(e.key==='Escape'){seq='\x1b';}
    else if(e.key==='ArrowUp'){seq='\x1b[A';}
    else if(e.key==='ArrowDown'){seq='\x1b[B';}
    else if(e.key==='ArrowRight'){seq='\x1b[C';}
    else if(e.key==='ArrowLeft'){seq='\x1b[D';}
    else if(e.key==='Home'){seq='\x1b[H';}
    else if(e.key==='End'){seq='\x1b[F';}
    else if(e.key==='Delete'){seq='\x1b[3~';}
    else if(e.ctrlKey&&e.key.length===1){
      const code=e.key.toLowerCase().charCodeAt(0)-96;
      if(code>0&&code<27)seq=String.fromCharCode(code);
    }
    if(seq&&t._termId!=null){
      const enc=new TextEncoder();
      fetch('/api/term/'+t._termId+'/input',{method:'POST',body:enc.encode(seq)});
    }
  });
  // ResizeObserver
  new ResizeObserver(()=>{
    const r2=canvas.parentElement.getBoundingClientRect();
    const nc=Math.floor(r2.width/t._cellW)||80;
    const nr=Math.floor((r2.height-28)/t._cellH)||24;
    if(nc!==t._cols||nr!==t._rows){
      t._cols=nc;t._rows=nr;
      canvas.width=nc*t._cellW;canvas.height=nr*t._cellH;
      canvas.style.width=canvas.width+'px';canvas.style.height=canvas.height+'px';
      const ctx2=canvas.getContext('2d');
      ctx2.fillStyle='#1e1e2e';ctx2.fillRect(0,0,canvas.width,canvas.height);
      if(t._termId!=null){
        fetch('/api/term/'+t._termId+'/resize',{method:'POST',
          headers:{'Content-Type':'application/json'},
          body:JSON.stringify({cols:nc,rows:nr})});
      }
    }
  }).observe(canvas.parentElement);
}

function startTermStream(t){
  const es=new EventSource('/api/term/'+t._termId+'/stream');
  const canvas=t._canvas;
  es.onmessage=function(ev){
    const d=JSON.parse(ev.data);
    if(d.type==='frame'){
      const ctx=canvas.getContext('2d');
      ctx.font='13px "JetBrains Mono",monospace';
      ctx.textBaseline='top';
      const cw=t._cellW,ch=t._cellH;
      // Draw changed cells
      for(const c of d.cells){
        ctx.fillStyle=c.bg;
        ctx.fillRect(c.c*cw,c.r*ch,cw,ch);
        ctx.fillStyle=c.fg;
        if(c.fl&1)ctx.font='bold 13px "JetBrains Mono",monospace';
        ctx.fillText(c.ch,c.c*cw,c.r*ch+2);
        if(c.fl&1)ctx.font='13px "JetBrains Mono",monospace';
      }
      // Cursor
      if(t._prevCursor){
        const[pc,pr]=t._prevCursor;
        // Redraw prev cursor cell without cursor highlight
        ctx.fillStyle='#1e1e2e';ctx.fillRect(pc*cw,pr*ch,cw,ch);
      }
      const[cc,cr]=d.cursor;
      ctx.fillStyle='#cdd6f4';ctx.globalAlpha=0.7;
      ctx.fillRect(cc*cw,cr*ch,cw,ch);ctx.globalAlpha=1.0;
      t._prevCursor=d.cursor;
    }else if(d.type==='exit'){
      es.close();
    }
  };
  es.onerror=function(){es.close();};
  t._eventSource=es;
}
```

- [ ] **Step 3: Update `openTile()` and keybinding**

Modify `openTile()` (line 188) to handle the new type:

```javascript
function openTile(type){
  const id=nextId++;
  const t=type==='chat'?createChatTile(id):type==='repl'?createReplTile(id):createTermTile(id);
  tiles.push({id,type,el:t});
  document.getElementById('tiles').appendChild(t);
  updateGrid();focusTile(id);
}
```

Add `Alt+S` keybinding in the `keydown` handler (after the `Alt+T` handler at line 215):

```javascript
  else if(e.key==='s'){e.preventDefault();openTile('term')}
```

- [ ] **Step 4: Update tile focus to handle canvas**

Modify `focusTile()` (line 200) to focus canvas in term tiles:

```javascript
function focusTile(id){
  tiles.forEach(t=>t.el.classList.toggle('focused',t.id===id));focusedId=id;
  const tile=tiles.find(t=>t.id===id);
  if(tile){
    const canvas=tile.el.querySelector('canvas');
    if(canvas){canvas.focus()}
    else{const inp=tile.el.querySelector('input');if(inp)inp.focus()}
  }
}
```

- [ ] **Step 5: Build and verify**

```bash
cd /home/peter/projects/olorin1
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build 2>&1
```
Expected: compiles. (Manual verification: run `cargo run -- --web 8080`, open browser, press `Alt+S`.)

- [ ] **Step 6: Commit**

```bash
git add web/chat.html
git commit -m "feat: add terminal tile with Canvas rendering and Alt+S keybinding"
```

---

### Task 9: Integration Test — Full Pipeline

**Files:**
- Create: `tests/term_pipeline.rs`

- [ ] **Step 1: Write end-to-end pipeline test**

Create `tests/term_pipeline.rs`:

```rust
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

    // Bash prompt + "echo hello" + output "hello" = some dirty cells
    assert!(dirty_count > 0, "Expected dirty cells after echo, got 0");
}

#[test]
fn ansi_color_in_pipeline() {
    olorin::kernels::ffi::init().unwrap();

    let mut session = PtySession::new(80, 24).expect("failed to open PTY");
    // printf with ANSI color
    session.write_bytes(b"printf '\\033[31mRED\\033[0m'\n");
    std::thread::sleep(std::time::Duration::from_millis(300));

    let _dirty = session.read_and_apply();

    // Find "RED" in the grid and check it has non-default color
    let grid = session.grid();
    let mut found_colored = false;
    for row in 0..24u16 {
        for col in 0..77u16 {
            let c0 = grid.cell(row, col);
            let c1 = grid.cell(row, col + 1);
            let c2 = grid.cell(row, col + 2);
            if c0.ch == b'R' as u32 && c1.ch == b'E' as u32 && c2.ch == b'D' as u32 {
                // Should have red-ish fg (not default 0x00cdd6f4)
                if c0.fg != 0x00cdd6f4 {
                    found_colored = true;
                }
                break;
            }
        }
        if found_colored { break; }
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

    // Write after resize and verify grid is functional
    session.write_bytes(b"echo AFTER_RESIZE\n");
    std::thread::sleep(std::time::Duration::from_millis(300));

    let dirty = session.read_and_apply();
    let dirty_count = dirty.iter().filter(|&&d| d != 0).count();
    assert!(dirty_count > 0, "Expected dirty cells after resize + echo");
}
```

- [ ] **Step 2: Run all tests**

```bash
cd /home/peter/projects/olorin1
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test 2>&1
```
Expected: all tests pass (existing + new).

- [ ] **Step 3: Commit**

```bash
git add tests/term_pipeline.rs
git commit -m "test: add full terminal pipeline integration tests"
```

---

### Task 10: Remove swap_prev Dead Code + Final Cleanup

**Files:**
- Modify: `src/interface/ansi.rs`

- [ ] **Step 1: Remove unused `swap_prev` method**

Delete the `swap_prev` method and its comment from `ansi.rs` — the snapshotting is done in `pty.rs` via `copy_from_slice`, not via swap.

- [ ] **Step 2: Run full test suite**

```bash
cd /home/peter/projects/olorin1
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test 2>&1
```
Expected: all tests pass, no warnings about dead code.

- [ ] **Step 3: Verify no file exceeds 500 lines**

```bash
find /home/peter/projects/olorin1/src -name '*.rs' -exec wc -l {} + | sort -rn | head -10
```
Expected: no file > 500 lines. If `server.rs` exceeds 500, split terminal handlers into `interface/term_stream.rs`.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove dead code, verify 500-line limit"
```

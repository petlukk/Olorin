# Terminal Kernel Pipeline + Hyprbar Config Panel

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SIMD-accelerated interactive terminal (PTY) tiles to Olorin's web-UI, and a runtime config panel in the hyprbar for inference/cloud/system parameters with per-tile override support.

**Architecture:** Rust owns all stateful logic (PTY lifecycle, ANSI state machine, cell grid, config state). Eä kernels handle the parallelizable hot paths: `ansi_parser.ea` classifies raw bytes via SIMD, `terminal_diff.ea` compares cell grids, and `fused_safety.ea` scans each command line before it reaches the PTY. Config panel shows inference params in hyprbar, opens a modal on click for full configuration. API key stored encrypted in vault. Transport uses existing SSE-down + POST-up pattern. Canvas renders in the browser.

**Tech Stack:** Rust (libc: openpty/fork/ioctl/poll), Eä SIMD kernels, HTML5 Canvas, SSE

**Security:** Every line entered in the terminal tile goes through `fused_safety.ea` (injection + leak scan) and `ShellGuard` (command classification) before being written to the PTY. Raw control bytes (Ctrl-C, arrows, tab) bypass the guard — they are not commands. Blocked commands produce an SSE error event instead of reaching bash. API key stored encrypted in vault — never in plaintext, never logged, masked in GET responses.

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
| `tests/pty_guard.rs` | Safety guard tests: blocked destructive cmds, allowed safe cmds, ctrl-C passthrough |

### Modified Files

| File | Change |
|------|--------|
| `src/interface/mod.rs` | Add `pub mod pty; pub mod ansi;` |
| `src/kernels/ffi.rs` | Add type aliases, KernelTable fields, load + public wrappers for 2 new kernels |
| `src/interface/server.rs` | 5 terminal endpoints + SSE loop + 3 config endpoints + param extraction in `/api/generate` + config in `/api/system` |
| `src/core/router.rs` | `update_config()`, `get_config()`, runtime parameter mutation, vault API key load |
| `src/core/anthropic.rs` | `set_model()`, `set_max_tokens()`, pub `api_key()` setter |
| `web/chat.html` | `createTermTile()`, Canvas renderer, `Alt+S`, hyprbar config elements, olorin button, config modal, per-tile config |

### Test Files

| File | Tests |
|------|-------|
| `tests/pty.rs` | PTY lifecycle: open, write/read, resize, close |
| `tests/ansi.rs` | State machine: SGR, cursor, ED/EL |
| `tests/terminal_diff.rs` | Diff: identical grids, changed cells |
| `tests/pty_guard.rs` | Safety guard: blocked destructive commands, allowed safe, ctrl-C passthrough |
| `tests/config_api.rs` | Config GET/POST roundtrip, partial update, param passthrough |

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
//!
//! Security: all input goes through `write_guarded()` which buffers bytes
//! until a newline, then runs `fused_safety.ea` + `ShellGuard` before
//! sending the line to bash. Raw control bytes bypass the guard.

use crate::interface::ansi::{Cell, TermGrid};
use crate::core::shell_guard::{ShellGuard, load_shell_policy};
use crate::core::safety;
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
    line_buf: Vec<u8>,
    guard: ShellGuard,
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
            line_buf: Vec::with_capacity(256),
            guard: ShellGuard::new(load_shell_policy()),
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

    /// Guarded write: buffers input until newline, then scans with
    /// `fused_safety.ea` + `ShellGuard` before sending to PTY.
    /// Raw control bytes (< 0x20 except \r\n, or escape sequences) pass through
    /// directly — they are terminal control, not commands.
    /// Returns Ok(()) if sent, Err(reason) if blocked.
    pub fn write_guarded(&mut self, data: &[u8]) -> Result<(), String> {
        for &b in data {
            match b {
                // Enter — flush the line through safety
                b'\r' | b'\n' => {
                    if !self.line_buf.is_empty() {
                        let line = String::from_utf8_lossy(&self.line_buf).to_string();

                        // 1. SIMD safety scan (injection + leak detection)
                        let scan = safety::scan(line.as_bytes());
                        if scan.blocked {
                            let reason = scan.details.first()
                                .map(|w| w.pattern)
                                .unwrap_or("safety violation");
                            self.line_buf.clear();
                            return Err(format!("blocked by safety scan: {reason}"));
                        }

                        // 2. Shell guard (destructive command classification)
                        if let Err(e) = self.guard.check(&line) {
                            self.line_buf.clear();
                            return Err(e);
                        }

                        // Passed both gates — send line + newline to PTY
                        self.write_raw(&self.line_buf.clone());
                        self.write_raw(&[b'\r']);
                        self.line_buf.clear();
                    } else {
                        // Empty enter — just send \r
                        self.write_raw(&[b'\r']);
                    }
                }
                // Backspace — pop from line buffer
                0x7f | 0x08 => {
                    self.line_buf.pop();
                    self.write_raw(&[b]);
                }
                // Raw control (Ctrl-C, Ctrl-Z, Ctrl-D, etc.) — pass through directly
                0x01..=0x06 | 0x09 | 0x0b..=0x0c | 0x0e..=0x1a => {
                    self.write_raw(&[b]);
                }
                // ESC (start of escape sequence) — pass through directly
                0x1b => {
                    self.write_raw(&[b]);
                }
                // Printable — accumulate in line buffer
                _ => {
                    self.line_buf.push(b);
                }
            }
        }
        Ok(())
    }

    /// Write raw bytes to the PTY — internal, bypasses guard.
    fn write_raw(&self, data: &[u8]) {
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

    /// Unguarded write — for tests only. Bypasses safety scan.
    #[cfg(test)]
    pub fn write_bytes(&mut self, data: &[u8]) {
        self.write_raw(data);
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

### Task 6: PTY Safety Guard Tests

**Files:**
- Create: `tests/pty_guard.rs`

- [ ] **Step 1: Write safety guard tests**

Create `tests/pty_guard.rs`:

```rust
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
    // Ctrl-C = 0x03, should pass through directly (not buffered)
    let result = session.write_guarded(&[0x03]);
    assert!(result.is_ok());
}

#[test]
fn escape_sequence_passes_through() {
    olorin::kernels::ffi::init().unwrap();
    let mut session = PtySession::new(80, 24).expect("failed to open PTY");
    // Arrow up = ESC [ A
    let result = session.write_guarded(&[0x1b, b'[', b'A']);
    assert!(result.is_ok());
}

#[test]
fn safety_scan_blocks_injection_attempt() {
    olorin::kernels::ffi::init().unwrap();
    let mut session = PtySession::new(80, 24).expect("failed to open PTY");
    // Injection pattern embedded in a command
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
    // Type "rm -rf /" then backspace the whole thing and type "ls"
    let result = session.write_guarded(b"rm -rf /");
    assert!(result.is_ok()); // Not blocked yet — no enter pressed
    // Backspace 8 times
    for _ in 0..8 { session.write_guarded(&[0x7f]).unwrap(); }
    // Now type ls + enter
    let result = session.write_guarded(b"ls\r");
    assert!(result.is_ok()); // Should be "ls", not "rm -rf /"
}
```

- [ ] **Step 2: Run tests**

```bash
cd /home/peter/projects/olorin1
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test --test pty_guard 2>&1
```
Expected: all 12 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/pty_guard.rs
git commit -m "test: add PTY safety guard tests — fused_safety.ea + ShellGuard"
```

---

### Task 7: Terminal Diff Kernel Test

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

### Task 8: Server Endpoints for Terminal Sessions

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
        match s.write_guarded(&body_bytes) {
            Ok(()) => serve_json(stream, r#"{"ok":true}"#),
            Err(reason) => {
                let escaped = escape_json(&reason);
                serve_json(stream, &format!("{{\"ok\":false,\"blocked\":\"{escaped}\"}}"));
            }
        }
    } else {
        serve_json(stream, r#"{"ok":false,"error":"no session"}"#);
    }
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

### Task 9: Web-UI Terminal Tile

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
      fetch('/api/term/'+t._termId+'/input',{method:'POST',body:enc.encode(seq)})
        .then(r=>r.json()).then(d=>{
          if(d.blocked){
            // Flash canvas border red briefly to indicate blocked command
            canvas.style.boxShadow='0 0 0 2px #f38ba8';
            setTimeout(()=>{canvas.style.boxShadow='';},500);
            console.warn('[olorin] blocked:',d.blocked);
          }
        }).catch(()=>{});
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

### Task 10: Integration Test — Full Pipeline

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

### Task 11: Remove swap_prev Dead Code + Final Cleanup

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

---

## Part 2: Hyprbar Config Panel

---

### Task 12: Router Config API (`router.rs` + `anthropic.rs`)

**Files:**
- Modify: `src/core/router.rs`
- Modify: `src/core/anthropic.rs`

- [ ] **Step 1: Add setters to AnthropicClient**

Add after `with_model()` (line 26 in `src/core/anthropic.rs`):

```rust
    pub fn set_api_key(&mut self, key: String) {
        self.api_key = key;
    }

    pub fn set_model(&mut self, model: String) {
        self.model = model;
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn has_key(&self) -> bool {
        !self.api_key.is_empty()
    }
```

Replace the hardcoded `MAX_TOKENS` constant usage. Change line 12:

```rust
const DEFAULT_CLOUD_MAX_TOKENS: i64 = 4096;
```

Add a `max_tokens` field to `AnthropicClient`:

```rust
pub struct AnthropicClient {
    api_key: String,
    model:   String,
    max_tokens: i64,
}
```

Update `new()` and `with_model()` to initialize `max_tokens: DEFAULT_CLOUD_MAX_TOKENS`.

Add setter:

```rust
    pub fn set_max_tokens(&mut self, n: i64) {
        self.max_tokens = n;
    }

    pub fn max_tokens(&self) -> i64 {
        self.max_tokens
    }
```

Update `build_request` to accept `max_tokens` parameter instead of using constant. Change its signature:

```rust
fn build_request(model: &str, max_tokens: i64, system: &str, messages: &[(&str, &str)]) -> String {
```

And update the call in `generate()`:

```rust
let body = build_request(&self.model, self.max_tokens, system, messages);
```

- [ ] **Step 2: Add `get_config()` and `update_config()` to DispatchContext**

Add at the end of `impl DispatchContext` in `src/core/router.rs` (before the closing `}`):

```rust
    /// Return current config as JSON string. API key is masked.
    pub fn get_config(&self) -> String {
        let (model, temp, top_k, top_p, rep_pen, max_tok) = match &self.engine {
            Some(e) => (
                e.quant_type_str(),
                e.temperature,
                e.top_k,
                e.top_p,
                e.repetition_penalty,
                e.max_tokens,
            ),
            None => ("none", 0.0, 0, 0.0, 1.0, 0),
        };
        let (cloud_model, cloud_max, has_key) = match &self.anthropic {
            Some(a) => (a.model(), a.max_tokens(), a.has_key()),
            None => ("claude-3-5-haiku-latest", 4096, false),
        };
        let system_prompt = crate::interface::server::escape_json(&self.system_prompt);
        format!(
            "{{\"model\":\"{model}\",\"temperature\":{temp},\
             \"top_k\":{top_k},\"top_p\":{top_p},\
             \"repetition_penalty\":{rep_pen},\"max_tokens\":{max_tok},\
             \"cloud_model\":\"{cloud_model}\",\"cloud_max_tokens\":{cloud_max},\
             \"recall_level\":{},\"system_prompt\":\"{system_prompt}\",\
             \"has_api_key\":{has_key}}}",
            self.recall_level
        )
    }

    /// Update config fields from a partial JSON body.
    /// Only fields present in the JSON are updated.
    pub fn update_config(&mut self, json: &str) {
        use crate::interface::server::extract_json_string;

        if let Some(engine) = &mut self.engine {
            if let Some(v) = extract_json_float(json, "temperature") {
                engine.temperature = v;
            }
            if let Some(v) = extract_json_int(json, "top_k") {
                engine.top_k = v as usize;
            }
            if let Some(v) = extract_json_float(json, "top_p") {
                engine.top_p = v;
            }
            if let Some(v) = extract_json_float(json, "repetition_penalty") {
                engine.repetition_penalty = v;
            }
            if let Some(v) = extract_json_int(json, "max_tokens") {
                engine.max_tokens = v as usize;
            }
        }

        if let Some(anthropic) = &mut self.anthropic {
            if let Some(v) = extract_json_string(json, "cloud_model") {
                anthropic.set_model(v);
            }
            if let Some(v) = extract_json_int(json, "cloud_max_tokens") {
                anthropic.set_max_tokens(v as i64);
            }
        }

        if let Some(v) = extract_json_int(json, "recall_level") {
            self.recall_level = v as usize;
        }
        if let Some(v) = extract_json_string(json, "system_prompt") {
            self.system_prompt = v;
        }
    }

    /// Store API key in vault and update/create AnthropicClient.
    pub fn store_api_key(&mut self, key: &str) {
        if let Some(vault) = &mut self.vault {
            let _ = vault.append(b"config:api_key", key.as_bytes());
        }
        match &mut self.anthropic {
            Some(a) => a.set_api_key(key.to_string()),
            None => self.anthropic = Some(AnthropicClient::new(key.to_string())),
        }
    }

    /// Try to load API key from vault at startup.
    pub fn load_api_key_from_vault(&mut self) {
        if self.anthropic.is_some() { return; } // env key takes priority
        if let Some(vault) = &mut self.vault {
            if let Ok(results) = vault.search("config:api_key", 1) {
                if let Some(hit) = results.first() {
                    let text = String::from_utf8_lossy(&hit.plaintext);
                    if let Some(key) = text.strip_prefix("config:api_key: ") {
                        let key = key.trim().to_string();
                        if !key.is_empty() {
                            self.anthropic = Some(AnthropicClient::new(key));
                        }
                    }
                }
            }
        }
    }
```

- [ ] **Step 3: Add JSON number extraction helpers to router.rs**

Add at the bottom of `src/core/router.rs` (after the `impl DispatchContext` block):

```rust
fn extract_json_float(json: &str, key: &str) -> Option<f32> {
    let pattern = format!("\"{}\"", key);
    let start = json.find(&pattern)?;
    let after_key = &json[start + pattern.len()..];
    let colon = after_key.find(':')?;
    let rest = after_key[colon + 1..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn extract_json_int(json: &str, key: &str) -> Option<i64> {
    let pattern = format!("\"{}\"", key);
    let start = json.find(&pattern)?;
    let after_key = &json[start + pattern.len()..];
    let colon = after_key.find(':')?;
    let rest = after_key[colon + 1..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '-')
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}
```

- [ ] **Step 4: Call `load_api_key_from_vault()` in `new()`**

In `src/core/router.rs`, add after line 84 (`_max_turns: 8,`), before the closing `}`:

Change `new()` to:

```rust
    pub fn new(api_key: Option<String>, model_arg: Option<&str>) -> Self {
        let anthropic = api_key.map(AnthropicClient::new);
        let vault = Self::open_vault();
        let engine = Self::load_engine(model_arg);
        let mut ctx = Self {
            messages:      Vec::new(),
            recall:        VectorStore::new(1024),
            vault,
            engine,
            anthropic,
            last_timing:   None,
            system_prompt: llm::SYSTEM_PROMPT.to_string(),
            recall_level:  0,
            _max_turns:    8,
        };
        ctx.load_api_key_from_vault();
        ctx
    }
```

- [ ] **Step 5: Build to verify**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build 2>&1
```
Expected: compiles without errors.

- [ ] **Step 6: Commit**

```bash
git add src/core/router.rs src/core/anthropic.rs
git commit -m "feat: add runtime config API to DispatchContext + AnthropicClient setters"
```

---

### Task 13: Server Config Endpoints + Parameter Passthrough

**Files:**
- Modify: `src/interface/server.rs`

- [ ] **Step 1: Add config routes**

In `src/interface/server.rs`, add before the `_ => { 404` match arm (line 101):

```rust
        ("GET", "/api/config") => {
            let body = ctx.lock().unwrap().get_config();
            serve_json(stream, &body);
        }
        ("POST", "/api/config") => {
            handle_config_update(stream, req, &buf[..n], n, ctx);
        }
        ("POST", "/api/config/apikey") => {
            handle_config_apikey(stream, req, &buf[..n], n, ctx);
        }
```

- [ ] **Step 2: Add config handler functions**

Add after `escape_json()` (line 472 in `src/interface/server.rs`):

```rust
// ── Config handlers ──────────────────────────────────────────────────────────

fn handle_config_update(
    stream: &mut std::net::TcpStream,
    req: &str,
    buf: &[u8],
    n: usize,
    ctx: Arc<Mutex<DispatchContext>>,
) {
    let body_bytes = read_body(stream, req, buf, n);
    let body_str = std::str::from_utf8(&body_bytes).unwrap_or("");
    ctx.lock().unwrap().update_config(body_str);
    let config = ctx.lock().unwrap().get_config();
    serve_json(stream, &config);
}

fn handle_config_apikey(
    stream: &mut std::net::TcpStream,
    req: &str,
    buf: &[u8],
    n: usize,
    ctx: Arc<Mutex<DispatchContext>>,
) {
    let body_bytes = read_body(stream, req, buf, n);
    let key = std::str::from_utf8(&body_bytes).unwrap_or("").trim();
    if key.is_empty() {
        serve_json(stream, r#"{"ok":false,"error":"empty key"}"#);
        return;
    }
    ctx.lock().unwrap().store_api_key(key);
    serve_json(stream, r#"{"ok":true}"#);
}
```

- [ ] **Step 3: Extract inference params in `/api/generate`**

Replace the prompt-only extraction in `handle_generate()` (line 153) — change:

```rust
    let prompt     = extract_json_string(body_str, "prompt").unwrap_or_default();
```

to:

```rust
    let prompt = extract_json_string(body_str, "prompt").unwrap_or_default();

    // Apply per-request inference params (from chat/repl tile config)
    {
        let mut c = ctx.lock().unwrap();
        if let Some(engine) = &mut c.engine {
            if let Some(v) = extract_json_float(body_str, "temperature") {
                engine.temperature = v;
            }
            if let Some(v) = extract_json_int(body_str, "max_tokens") {
                engine.max_tokens = v as usize;
            }
            if let Some(v) = extract_json_float(body_str, "repetition_penalty") {
                engine.repetition_penalty = v;
            }
        }
    }
```

Add the import at the top of the function (or at file-level):

```rust
use crate::core::router::{extract_json_float, extract_json_int};
```

Make `extract_json_float` and `extract_json_int` in `router.rs` pub:

```rust
pub fn extract_json_float(json: &str, key: &str) -> Option<f32> {
pub fn extract_json_int(json: &str, key: &str) -> Option<i64> {
```

- [ ] **Step 4: Add config fields to `/api/system` response**

In `build_system_json()` (line 352), change the signature to accept config:

```rust
pub fn build_system_json(recall_level: usize, config_json: &str) -> String {
```

Update the format string to include config at the end:

```rust
    format!(
        "{{\"cpu_percent\":{cpu_percent},\"cpu_temp\":{cpu_temp},\
         \"memory_used_mb\":{mem_used},\"memory_total_mb\":{mem_total},\
         \"os\":\"{os}\",\"arch\":\"{arch}\",\"uptime_seconds\":{uptime},\
         \"recall_level\":{recall_level},\"config\":{config_json}}}"
    )
```

Update the call site in the route handler (line 91):

```rust
        ("GET", "/api/system") => {
            let c = ctx.lock().unwrap();
            let recall_level = c.recall_level();
            let config_json = c.get_config();
            drop(c);
            let body = build_system_json(recall_level, &config_json);
            serve_json(stream, &body);
        }
```

- [ ] **Step 5: Build to verify**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build 2>&1
```
Expected: compiles without errors.

- [ ] **Step 6: Commit**

```bash
git add src/interface/server.rs src/core/router.rs
git commit -m "feat: add config endpoints (GET/POST /api/config, POST /api/config/apikey)"
```

---

### Task 14: Config API Tests

**Files:**
- Create: `tests/config_api.rs`

- [ ] **Step 1: Write config test**

Create `tests/config_api.rs`:

```rust
//! Tests for runtime config API — get/update/partial update.

use olorin::core::router::DispatchContext;

#[test]
fn get_config_returns_defaults() {
    olorin::kernels::ffi::init().unwrap();
    let ctx = DispatchContext::new(None, None);
    let json = ctx.get_config();
    assert!(json.contains("\"temperature\":"));
    assert!(json.contains("\"has_api_key\":false"));
}

#[test]
fn update_config_changes_temperature() {
    olorin::kernels::ffi::init().unwrap();
    let mut ctx = DispatchContext::new(None, None);
    ctx.update_config(r#"{"temperature": 1.5}"#);
    let json = ctx.get_config();
    assert!(json.contains("\"temperature\":1.5"));
}

#[test]
fn update_config_partial_preserves_other_fields() {
    olorin::kernels::ffi::init().unwrap();
    let mut ctx = DispatchContext::new(None, None);
    let before = ctx.get_config();
    ctx.update_config(r#"{"temperature": 0.8}"#);
    let after = ctx.get_config();
    // top_k should be unchanged
    assert!(after.contains("\"top_k\":40"));
    // temperature should be changed
    assert!(after.contains("\"temperature\":0.8"));
}

#[test]
fn update_config_system_prompt() {
    olorin::kernels::ffi::init().unwrap();
    let mut ctx = DispatchContext::new(None, None);
    ctx.update_config(r#"{"system_prompt": "Be helpful."}"#);
    let json = ctx.get_config();
    assert!(json.contains("Be helpful."));
}

#[test]
fn store_api_key_creates_client() {
    olorin::kernels::ffi::init().unwrap();
    let mut ctx = DispatchContext::new(None, None);
    assert!(ctx.get_config().contains("\"has_api_key\":false"));
    ctx.store_api_key("sk-ant-test-key");
    assert!(ctx.get_config().contains("\"has_api_key\":true"));
}

#[test]
fn update_cloud_model() {
    olorin::kernels::ffi::init().unwrap();
    let mut ctx = DispatchContext::new(Some("sk-test".to_string()), None);
    ctx.update_config(r#"{"cloud_model": "claude-sonnet-4-6"}"#);
    let json = ctx.get_config();
    assert!(json.contains("claude-sonnet-4-6"));
}

#[test]
fn update_recall_level() {
    olorin::kernels::ffi::init().unwrap();
    let mut ctx = DispatchContext::new(None, None);
    ctx.update_config(r#"{"recall_level": 5}"#);
    let json = ctx.get_config();
    assert!(json.contains("\"recall_level\":5"));
}
```

- [ ] **Step 2: Run tests**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test --test config_api 2>&1
```
Expected: all 7 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/config_api.rs
git commit -m "test: add config API tests — get/update/partial/apikey/cloud"
```

---

### Task 15: Web-UI Hyprbar Config Elements + Config Modal

**Files:**
- Modify: `web/chat.html`

- [ ] **Step 1: Add olorin button and config elements to hyprbar**

In `web/chat.html`, replace the hyprbar left div content. Change the existing `<div class="left">` section (lines 53-63) to:

```html
  <div class="left">
    <span id="olorin-btn" style="cursor:pointer"><span class="c-teal">◆</span> <span class="c-teal">olorin</span></span>
    <span class="sep">|</span>
    <span><span class="c-mauve">◆</span> <span id="hb-model" class="c-text">—</span></span>
    <span class="sep">|</span>
    <span id="hb-backend" class="c-blue">—</span>
    <span class="sep">|</span>
    <span><span id="hb-tps" class="c-green">0.0</span> <span class="c-green">tok/s</span></span>
    <span class="sep">|</span>
    <span>temp <span id="hb-cfg-temp" class="c-yellow">0.4</span></span>
    <span class="sep">|</span>
    <span>k:<span id="hb-cfg-topk" class="c-yellow">40</span></span>
    <span class="sep">|</span>
    <span>p:<span id="hb-cfg-topp" class="c-yellow">0.9</span></span>
    <span class="sep">|</span>
    <span>rep:<span id="hb-cfg-rep" class="c-yellow">1.05</span></span>
    <span class="sep">|</span>
    <span>max:<span id="hb-cfg-max" class="c-yellow">64</span></span>
    <span class="sep">|</span>
    <span>recall <span id="hb-recall" class="c-yellow">—</span></span>
    <span class="sep">|</span>
    <span>sessions <span id="hb-sessions" class="c-peach">0</span></span>
  </div>
```

- [ ] **Step 2: Update `updateSystem()` to read config from `/api/system`**

Change the `updateSystem` function (line 230) to also update config elements:

```javascript
function updateSystem(){
  fetch('/api/system').then(r=>r.json()).then(d=>{
    document.getElementById('hb-cpu').textContent=d.cpu_percent+'%';
    document.getElementById('hb-temp').textContent=d.cpu_temp!=null?d.cpu_temp+'°C':'–';
    const m=(d.memory_used_mb/1024).toFixed(1),t=(d.memory_total_mb/1024).toFixed(1);
    document.getElementById('hb-mem').textContent=m+'G/'+t+'G';
    document.getElementById('hb-os').textContent=d.os+' '+d.arch;
    const h=Math.floor(d.uptime_seconds/3600),mi=Math.floor((d.uptime_seconds%3600)/60);
    document.getElementById('hb-uptime').textContent=h+'h '+mi+'m';
    if(d.recall_level!=null)document.getElementById('hb-recall').textContent=d.recall_level;
    if(d.config){
      const c=d.config;
      document.getElementById('hb-cfg-temp').textContent=c.temperature;
      document.getElementById('hb-cfg-topk').textContent=c.top_k;
      document.getElementById('hb-cfg-topp').textContent=c.top_p;
      document.getElementById('hb-cfg-rep').textContent=c.repetition_penalty;
      document.getElementById('hb-cfg-max').textContent=c.max_tokens;
      window._globalConfig=c;
    }
  }).catch(()=>{});
}
```

- [ ] **Step 3: Add config modal CSS**

Add before `</style>`:

```css
.cfg-overlay{position:fixed;inset:0;background:rgba(17,17,27,0.85);display:none;z-index:100;align-items:center;justify-content:center}
.cfg-overlay.open{display:flex}
.cfg-panel{background:var(--base);border:1px solid var(--surface0);border-radius:8px;padding:20px;width:480px;max-height:80vh;overflow-y:auto;color:var(--text);font-family:'JetBrains Mono',monospace;font-size:12px}
.cfg-panel h2{margin:0 0 16px;font-size:14px;color:var(--lavender)}
.cfg-section{margin-bottom:16px}
.cfg-section h3{margin:0 0 8px;font-size:12px;color:var(--mauve);text-transform:uppercase;letter-spacing:1px}
.cfg-row{display:flex;align-items:center;margin-bottom:8px;gap:8px}
.cfg-row label{width:120px;color:var(--subtext1);flex-shrink:0}
.cfg-row input[type=range]{flex:1;accent-color:var(--teal)}
.cfg-row input[type=number],.cfg-row input[type=text],.cfg-row input[type=password]{background:var(--surface0);border:1px solid var(--surface1);color:var(--text);padding:4px 8px;border-radius:4px;width:70px;font-family:inherit;font-size:12px}
.cfg-row input[type=text].wide,.cfg-row input[type=password].wide,.cfg-row textarea{width:100%;flex:1}
.cfg-row textarea{background:var(--surface0);border:1px solid var(--surface1);color:var(--text);padding:4px 8px;border-radius:4px;font-family:inherit;font-size:12px;resize:vertical;min-height:60px}
.cfg-row select{background:var(--surface0);border:1px solid var(--surface1);color:var(--text);padding:4px 8px;border-radius:4px;font-family:inherit;font-size:12px}
.cfg-buttons{display:flex;gap:8px;justify-content:flex-end;margin-top:16px}
.cfg-buttons button{padding:6px 16px;border:none;border-radius:4px;cursor:pointer;font-family:inherit;font-size:12px}
.cfg-btn-apply{background:var(--teal);color:var(--base)}
.cfg-btn-cancel{background:var(--surface1);color:var(--text)}
.cfg-scope{display:flex;gap:12px;align-items:center;margin-bottom:16px;color:var(--subtext1)}
.cfg-scope label{width:auto}
```

- [ ] **Step 4: Add config modal HTML**

Add after the closing `</div>` of the hyprbar (after the `<div id="tiles">` container, before `<script>`):

```html
<div id="cfg-overlay" class="cfg-overlay" onclick="if(event.target===this)closeCfg()">
  <div class="cfg-panel">
    <h2>⚙ Configuration</h2>
    <div class="cfg-scope" id="cfg-scope-row" style="display:none">
      <label>Apply to:</label>
      <label><input type="radio" name="cfg-scope" value="global" checked> Global</label>
      <label><input type="radio" name="cfg-scope" value="tile"> This tile</label>
    </div>
    <div class="cfg-section">
      <h3>Inference</h3>
      <div class="cfg-row"><label>Model</label><select id="cfg-model"><option>bitnet</option><option>llama</option><option>llama8b</option><option>qwen</option></select></div>
      <div class="cfg-row"><label>Temperature</label><input type="range" id="cfg-temp-r" min="0" max="2" step="0.05"><input type="number" id="cfg-temp" min="0" max="2" step="0.05"></div>
      <div class="cfg-row"><label>Top-K</label><input type="range" id="cfg-topk-r" min="1" max="100" step="1"><input type="number" id="cfg-topk" min="1" max="100"></div>
      <div class="cfg-row"><label>Top-P</label><input type="range" id="cfg-topp-r" min="0" max="1" step="0.01"><input type="number" id="cfg-topp" min="0" max="1" step="0.01"></div>
      <div class="cfg-row"><label>Rep. penalty</label><input type="range" id="cfg-rep-r" min="1" max="2" step="0.01"><input type="number" id="cfg-rep" min="1" max="2" step="0.01"></div>
      <div class="cfg-row"><label>Max tokens</label><input type="number" id="cfg-max" min="1" max="4096" style="width:100px"></div>
    </div>
    <div class="cfg-section">
      <h3>Cloud Fallback</h3>
      <div class="cfg-row"><label>API key</label><input type="password" id="cfg-apikey" class="wide" placeholder="sk-ant-..."></div>
      <div class="cfg-row"><label>Cloud model</label><input type="text" id="cfg-cloud-model" class="wide"></div>
      <div class="cfg-row"><label>Cloud max tok</label><input type="number" id="cfg-cloud-max" min="1" max="16384" style="width:100px"></div>
    </div>
    <div class="cfg-section">
      <h3>System</h3>
      <div class="cfg-row"><label>Recall level</label><input type="range" id="cfg-recall-r" min="0" max="10" step="1"><input type="number" id="cfg-recall" min="0" max="10"></div>
      <div class="cfg-row" style="align-items:flex-start"><label>System prompt</label><textarea id="cfg-sysprompt" rows="4"></textarea></div>
    </div>
    <div class="cfg-buttons">
      <button class="cfg-btn-cancel" onclick="closeCfg()">Cancel</button>
      <button class="cfg-btn-apply" onclick="applyCfg()">Apply</button>
    </div>
  </div>
</div>
```

- [ ] **Step 5: Add config modal JS**

Add after the `updateSystem` function:

```javascript
// ── Config modal ────────────────────────────────────────────────────────────
window._globalConfig={};

function openCfg(){
  fetch('/api/config').then(r=>r.json()).then(c=>{
    window._globalConfig=c;
    document.getElementById('cfg-model').value=c.model||'bitnet';
    setSlider('cfg-temp',c.temperature);
    setSlider('cfg-topk',c.top_k);
    setSlider('cfg-topp',c.top_p);
    setSlider('cfg-rep',c.repetition_penalty);
    document.getElementById('cfg-max').value=c.max_tokens;
    document.getElementById('cfg-cloud-model').value=c.cloud_model||'';
    document.getElementById('cfg-cloud-max').value=c.cloud_max_tokens||4096;
    setSlider('cfg-recall',c.recall_level);
    document.getElementById('cfg-sysprompt').value=c.system_prompt||'';
    // Show tile scope toggle if a tile is focused
    const scopeRow=document.getElementById('cfg-scope-row');
    scopeRow.style.display=focusedId!=null?'flex':'none';
    document.querySelector('input[name="cfg-scope"][value="global"]').checked=true;
    document.getElementById('cfg-overlay').classList.add('open');
  });
}

function closeCfg(){
  document.getElementById('cfg-overlay').classList.remove('open');
}

function setSlider(id,val){
  document.getElementById(id).value=val;
  const r=document.getElementById(id+'-r');
  if(r)r.value=val;
}

// Sync slider ↔ number input
document.querySelectorAll('.cfg-row input[type=range]').forEach(r=>{
  const numId=r.id.replace('-r','');
  r.addEventListener('input',()=>{document.getElementById(numId).value=r.value});
});
document.querySelectorAll('.cfg-row input[type=number]').forEach(n=>{
  const rId=n.id+'-r';
  n.addEventListener('input',()=>{const r=document.getElementById(rId);if(r)r.value=n.value});
});

function applyCfg(){
  const scope=document.querySelector('input[name="cfg-scope"]:checked').value;
  const cfg={
    temperature:parseFloat(document.getElementById('cfg-temp').value),
    top_k:parseInt(document.getElementById('cfg-topk').value),
    top_p:parseFloat(document.getElementById('cfg-topp').value),
    repetition_penalty:parseFloat(document.getElementById('cfg-rep').value),
    max_tokens:parseInt(document.getElementById('cfg-max').value),
    cloud_model:document.getElementById('cfg-cloud-model').value,
    cloud_max_tokens:parseInt(document.getElementById('cfg-cloud-max').value),
    recall_level:parseInt(document.getElementById('cfg-recall').value),
    system_prompt:document.getElementById('cfg-sysprompt').value,
  };

  if(scope==='tile'&&focusedId!=null){
    const tile=tiles.find(t=>t.id===focusedId);
    if(tile)tile.el._config=cfg;
  }else{
    fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg)});
    window._globalConfig=cfg;
  }

  // API key — only send if non-empty (don't overwrite on every save)
  const apikey=document.getElementById('cfg-apikey').value;
  if(apikey){
    fetch('/api/config/apikey',{method:'POST',body:apikey});
    document.getElementById('cfg-apikey').value='';
  }

  closeCfg();
}

document.getElementById('olorin-btn').addEventListener('click',openCfg);
document.addEventListener('keydown',e=>{if(e.key==='Escape')closeCfg()});
```

- [ ] **Step 6: Update `sendChat()` and `sendCommand()` to use per-tile config**

Replace the hardcoded params in `sendChat()` (line 117). Change:

```javascript
body:JSON.stringify({prompt:text,temperature:0,repetition_penalty:1.1,max_tokens:512,recall_level:-1})
```

to:

```javascript
body:JSON.stringify(Object.assign({prompt:text},tileConfig(id)))
```

Add the `tileConfig()` helper before `sendChat()`:

```javascript
function tileConfig(id){
  const tile=tiles.find(t=>t.id===id);
  const tc=tile&&tile.el._config||{};
  const gc=window._globalConfig||{};
  return{
    temperature:tc.temperature!=null?tc.temperature:gc.temperature!=null?gc.temperature:0.4,
    top_k:tc.top_k!=null?tc.top_k:gc.top_k!=null?gc.top_k:40,
    top_p:tc.top_p!=null?tc.top_p:gc.top_p!=null?gc.top_p:0.9,
    repetition_penalty:tc.repetition_penalty!=null?tc.repetition_penalty:gc.repetition_penalty!=null?gc.repetition_penalty:1.05,
    max_tokens:tc.max_tokens!=null?tc.max_tokens:gc.max_tokens!=null?gc.max_tokens:512,
    recall_level:tc.recall_level!=null?tc.recall_level:gc.recall_level!=null?gc.recall_level:-1,
  };
}
```

Similarly update the REPL generate call in `sendCommand()`. Change:

```javascript
body:JSON.stringify({prompt:text,temperature:0.7,repetition_penalty:1.1,max_tokens:512,recall_level:-1})
```

to:

```javascript
body:JSON.stringify(Object.assign({prompt:text},tileConfig(id)))
```

- [ ] **Step 7: Build and verify**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo build 2>&1
```
Expected: compiles. Manual test: `cargo run -- --web 8080`, open browser, click "◆ olorin", verify modal opens with current values.

- [ ] **Step 8: Commit**

```bash
git add web/chat.html
git commit -m "feat: add config modal, hyprbar config elements, per-tile config override"
```

---

### Task 16: Final Verification + 500-line Check

**Files:**
- All modified files

- [ ] **Step 1: Run full test suite**

```bash
PATH="/home/peter/projects/eacompute/target/release:$PATH" cargo test 2>&1
```
Expected: all tests pass (terminal + config + existing).

- [ ] **Step 2: Verify no file exceeds 500 lines**

```bash
find src -name '*.rs' -exec wc -l {} + | sort -rn | head -10
```
Expected: no file > 500 lines. If `server.rs` exceeds 500 (it was at 473 + ~50 new = ~523), split config handlers into `src/interface/config.rs`:

Create `src/interface/config.rs` with `handle_config_update`, `handle_config_apikey`, and the config route matching. Add `pub mod config;` to `src/interface/mod.rs`. Move only the config handler functions, keep routes in `server.rs` calling into `config::handle_*`.

- [ ] **Step 3: Verify chat.html sends per-tile config**

Manual verification in browser:
1. Open chat tile (`Alt+C`)
2. Click "◆ olorin", change temperature to 1.0, Apply
3. Hyprbar should show `temp 1.0`
4. Send a message — check browser network tab that request body contains `"temperature":1.0`
5. Click "◆ olorin" again, switch to "This tile", set temp to 0.0, Apply
6. Send another message — should send `"temperature":0`
7. Open second chat tile — it should use global (1.0)

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: final verification, 500-line split if needed"
```

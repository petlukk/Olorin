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
    utf8_buf: [u8; 4],
    utf8_len: u8,
    utf8_need: u8,
}

impl TermGrid {
    pub fn new(cols: u16, rows: u16) -> Self {
        let n = cols as usize * rows as usize;
        Self {
            cols, rows,
            cells: vec![Cell::default(); n],
            cursor_row: 0, cursor_col: 0,
            attrs: CellAttrs::default(),
            state: ParseState::Ground,
            params: [0; 16], param_idx: 0,
            question_mark: false, cursor_visible: true,
            utf8_buf: [0; 4], utf8_len: 0, utf8_need: 0,
        }
    }

    pub fn cell(&self, row: u16, col: u16) -> Cell {
        self.cells[row as usize * self.cols as usize + col as usize]
    }

    pub fn cursor(&self) -> (u16, u16) { (self.cursor_row, self.cursor_col) }
    pub fn cursor_visible(&self) -> bool { self.cursor_visible }
    pub fn cells_ptr(&self) -> *const u8 { self.cells.as_ptr() as *const u8 }
    pub fn cells_mut_ptr(&mut self) -> *mut u8 { self.cells.as_mut_ptr() as *mut u8 }
    pub fn cell_count(&self) -> usize { self.cells.len() }

    /// Feed raw PTY bytes through SIMD classifier + state machine.
    pub fn feed(&mut self, data: &[u8], scan_buf: &mut [u8]) {
        let len = data.len();
        if len == 0 { return; }
        unsafe { ffi::ansi_classify(data.as_ptr(), scan_buf.as_mut_ptr(), len as i32); }
        for i in 0..len { self.step(data[i], scan_buf[i]); }
    }

    fn step(&mut self, b: u8, cls: u8) {
        match self.state {
            ParseState::Ground => self.ground(b, cls),
            ParseState::Escape => self.escape(b, cls),
            ParseState::Csi    => self.csi(b, cls),
            ParseState::OscSkip => {
                if b == 0x07 || b == 0x9C { self.state = ParseState::Ground; }
                if b == 0x1B { self.state = ParseState::Escape; }
            }
        }
    }

    fn ground(&mut self, b: u8, cls: u8) {
        // If we're accumulating a UTF-8 sequence, handle continuation bytes
        if self.utf8_need > 0 {
            if b & 0xC0 == 0x80 {
                self.utf8_buf[self.utf8_len as usize] = b;
                self.utf8_len += 1;
                if self.utf8_len == self.utf8_need {
                    let s = std::str::from_utf8(&self.utf8_buf[..self.utf8_len as usize]);
                    let cp = s.ok().and_then(|s| s.chars().next()).unwrap_or('\u{FFFD}') as u32;
                    self.utf8_len = 0;
                    self.utf8_need = 0;
                    self.put_char(cp);
                }
            } else {
                // Invalid continuation — emit replacement, reprocess this byte
                self.utf8_len = 0;
                self.utf8_need = 0;
                self.put_char(0xFFFD);
                self.ground(b, cls);
            }
            return;
        }

        // Check for UTF-8 lead bytes (high bit set, not a control/escape)
        if b >= 0xC0 && b < 0xFE && cls != 1 && cls != 6 {
            let need = if b < 0xE0 { 2 } else if b < 0xF0 { 3 } else { 4u8 };
            self.utf8_buf[0] = b;
            self.utf8_len = 1;
            self.utf8_need = need;
            return;
        }

        match cls {
            0 | 2 | 3 | 4 | 5 => self.put_char(b as u32),
            1 => self.state = ParseState::Escape,
            6 => self.control(b),
            7 => self.put_char(b as u32),
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
                if self.param_idx < 16 {
                    self.params[self.param_idx] = self.params[self.param_idx]
                        .saturating_mul(10)
                        .saturating_add((b - b'0') as u16);
                }
            }
            4 => { if self.param_idx < 15 { self.param_idx += 1; } }
            5 | 0 => {
                if b == b'?' { self.question_mark = true; return; }
                self.execute_csi(b, self.param_idx + 1);
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
                self.cursor_row = if p0 == 0 { 0 } else { (p0 - 1).min(self.rows - 1) };
                self.cursor_col = if p1 == 0 { 0 } else { (p1 - 1).min(self.cols - 1) };
            }
            b'A' => { self.cursor_row = self.cursor_row.saturating_sub(p0.max(1)); }
            b'B' => { self.cursor_row = (self.cursor_row + p0.max(1)).min(self.rows - 1); }
            b'C' => { self.cursor_col = (self.cursor_col + p0.max(1)).min(self.cols - 1); }
            b'D' => { self.cursor_col = self.cursor_col.saturating_sub(p0.max(1)); }
            b'J' => self.erase_display(p0),
            b'K' => self.erase_line(p0),
            b'h' | b'l' => {
                if self.question_mark && p0 == 25 {
                    self.cursor_visible = cmd == b'h';
                }
            }
            _ => {}
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
                c @ 30..=37 => self.attrs.fg = ansi_color(c - 30),
                39 => self.attrs.fg = DEFAULT_FG,
                c @ 40..=47 => self.attrs.bg = ansi_color(c - 40),
                49 => self.attrs.bg = DEFAULT_BG,
                c @ 90..=97 => self.attrs.fg = ansi_bright_color(c - 90),
                c @ 100..=107 => self.attrs.bg = ansi_bright_color(c - 100),
                38 => {
                    if i + 1 < n_params && self.params[i + 1] == 5 && i + 2 < n_params {
                        self.attrs.fg = color_256(self.params[i + 2]);
                        i += 2;
                    } else if i + 1 < n_params && self.params[i + 1] == 2 && i + 4 < n_params {
                        self.attrs.fg = (self.params[i + 2] as u32) << 16
                            | (self.params[i + 3] as u32) << 8
                            | self.params[i + 4] as u32;
                        i += 4;
                    }
                }
                48 => {
                    if i + 1 < n_params && self.params[i + 1] == 5 && i + 2 < n_params {
                        self.attrs.bg = color_256(self.params[i + 2]);
                        i += 2;
                    } else if i + 1 < n_params && self.params[i + 1] == 2 && i + 4 < n_params {
                        self.attrs.bg = (self.params[i + 2] as u32) << 16
                            | (self.params[i + 3] as u32) << 8
                            | self.params[i + 4] as u32;
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
        let wide = is_wide_char(ch);
        // Wide chars need 2 columns — wrap if only 1 left
        if wide && self.cursor_col + 1 >= self.cols {
            let idx = self.cursor_row as usize * self.cols as usize + self.cursor_col as usize;
            self.cells[idx] = Cell::default();
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
        // Wide char: place a zero-width spacer in the next cell
        if wide && self.cursor_col < self.cols {
            let idx2 = self.cursor_row as usize * self.cols as usize + self.cursor_col as usize;
            self.cells[idx2] = Cell {
                ch: 0,
                fg: self.attrs.fg,
                bg: self.attrs.bg,
                flags: self.attrs.flags,
                _pad: [0; 3],
            };
            self.cursor_col += 1;
        }
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
                self.cursor_col = (((self.cursor_col / 8) + 1) * 8).min(self.cols - 1);
            }
            8 => { self.cursor_col = self.cursor_col.saturating_sub(1); }
            _ => {}
        }
    }

    fn erase_display(&mut self, mode: u16) {
        let d = Cell::default();
        match mode {
            0 => {
                let s = self.cursor_row as usize * self.cols as usize
                    + self.cursor_col as usize;
                for c in &mut self.cells[s..] { *c = d; }
            }
            1 => {
                let e = (self.cursor_row as usize * self.cols as usize
                    + self.cursor_col as usize + 1)
                    .min(self.cells.len());
                for c in &mut self.cells[..e] { *c = d; }
            }
            2 | 3 => { for c in &mut self.cells { *c = d; } }
            _ => {}
        }
    }

    fn erase_line(&mut self, mode: u16) {
        let d = Cell::default();
        let rs = self.cursor_row as usize * self.cols as usize;
        match mode {
            0 => {
                for c in &mut self.cells[rs + self.cursor_col as usize..rs + self.cols as usize] {
                    *c = d;
                }
            }
            1 => {
                for c in &mut self.cells[rs..rs + self.cursor_col as usize + 1] {
                    *c = d;
                }
            }
            2 => {
                for c in &mut self.cells[rs..rs + self.cols as usize] {
                    *c = d;
                }
            }
            _ => {}
        }
    }

    fn scroll_up(&mut self) {
        let cols = self.cols as usize;
        self.cells.copy_within(cols.., 0);
        let start = (self.rows as usize - 1) * cols;
        for c in &mut self.cells[start..] { *c = Cell::default(); }
    }

    /// Swap cells into prev buffer — used by PtySession for terminal_diff.
    pub fn swap_prev(&mut self, prev: &mut Vec<Cell>) {
        std::mem::swap(&mut self.cells, prev);
        prev.copy_from_slice(&self.cells);
    }
}

// ── Wide character detection ─────────────────────────────────────────────────

/// Returns true for characters that occupy 2 terminal columns:
/// CJK, emoji, fullwidth forms, etc.
fn is_wide_char(cp: u32) -> bool {
    matches!(cp,
        // CJK Unified Ideographs + Extension A
        0x3400..=0x4DBF | 0x4E00..=0x9FFF |
        // CJK Compatibility Ideographs
        0xF900..=0xFAFF |
        // CJK Unified Extension B-F
        0x20000..=0x2FA1F |
        // Fullwidth Forms
        0xFF01..=0xFF60 | 0xFFE0..=0xFFE6 |
        // Hangul Syllables
        0xAC00..=0xD7AF |
        // Emoji (common ranges)
        0x1F300..=0x1F9FF | 0x1FA00..=0x1FA6F | 0x1FA70..=0x1FAFF |
        // Misc symbols, dingbats, emoticons
        0x2600..=0x27BF
    )
}

// ── Color tables ─────────────────────────────────────────────────────────────

fn ansi_color(idx: u16) -> u32 {
    match idx {
        0 => 0x0045475a, 1 => 0x00f38ba8, 2 => 0x00a6e3a1, 3 => 0x00f9e2af,
        4 => 0x0089b4fa, 5 => 0x00cba6f7, 6 => 0x0094e2d5, 7 => 0x00bac2de,
        _ => DEFAULT_FG,
    }
}

fn ansi_bright_color(idx: u16) -> u32 {
    match idx {
        0 => 0x00585b70, 1 => 0x00f38ba8, 2 => 0x00a6e3a1, 3 => 0x00f9e2af,
        4 => 0x0089b4fa, 5 => 0x00cba6f7, 6 => 0x0094e2d5, 7 => 0x00cdd6f4,
        _ => DEFAULT_FG,
    }
}

fn color_256(idx: u16) -> u32 {
    match idx {
        0..=7   => ansi_color(idx),
        8..=15  => ansi_bright_color(idx - 8),
        16..=231 => {
            let n = idx - 16;
            let b = (n % 6) as u32;
            let g = ((n / 6) % 6) as u32;
            let r = (n / 36) as u32;
            let scale = |v: u32| if v == 0 { 0 } else { 55 + 40 * v };
            (scale(r) << 16) | (scale(g) << 8) | scale(b)
        }
        232..=255 => {
            let v = (8 + 10 * (idx - 232)) as u32;
            (v << 16) | (v << 8) | v
        }
        _ => DEFAULT_FG,
    }
}

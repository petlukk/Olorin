//! PTY session — cross-platform. The platform-specific bits (openpty
//! vs ConPTY, raw fd vs HANDLE, poll vs WaitForSingleObject) live behind
//! the `PtyBackend` trait; this file owns the SIMD ANSI pipeline, the
//! safety-line buffer, the ShellGuard, and the cell grid.
//!
//! Security: input from the web-terminal goes through `write_guarded`,
//! which buffers bytes until a newline, then runs `fused_safety.ea` +
//! `ShellGuard` before sending the line to the backend. Raw control
//! bytes bypass the guard so interactive programs work normally.

use crate::interface::ansi::{Cell, TermGrid};
use crate::core::shell_guard::{ShellGuard, load_shell_policy};
use crate::core::safety;
use crate::kernels::ffi;
use std::io;

pub trait PtyBackend: Send + Sync {
    fn read(&self, buf: &mut [u8]) -> io::Result<usize>;
    fn write(&self, data: &[u8]);
    fn resize(&self, cols: u16, rows: u16);
    fn child_alive(&self) -> bool;
    fn wait_readable(&self, timeout_ms: i32) -> bool;
}

#[cfg(unix)]
pub fn default_backend(cols: u16, rows: u16) -> io::Result<Box<dyn PtyBackend>> {
    super::pty_unix::open(cols, rows).map(|b| Box::new(b) as Box<dyn PtyBackend>)
}

#[cfg(windows)]
pub fn default_backend(cols: u16, rows: u16) -> io::Result<Box<dyn PtyBackend>> {
    super::pty_windows::open(cols, rows).map(|b| Box::new(b) as Box<dyn PtyBackend>)
}

pub struct PtySession {
    backend: Box<dyn PtyBackend>,
    grid: TermGrid,
    prev_cells: Vec<Cell>,
    scan_buf: Vec<u8>,
    read_buf: Vec<u8>,
    dirty_buf: Vec<u8>,
    line_buf: Vec<u8>,
    guard: ShellGuard,
}

impl PtySession {
    pub fn new(cols: u16, rows: u16) -> io::Result<Self> {
        let backend = default_backend(cols, rows)?;
        let n_cells = cols as usize * rows as usize;
        Ok(Self {
            backend,
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
        self.backend.child_alive()
    }

    pub fn wait_readable(&self, timeout_ms: i32) -> bool {
        self.backend.wait_readable(timeout_ms)
    }

    /// Guarded write: buffers input until newline, then scans with
    /// `fused_safety.ea` + `ShellGuard` before sending to the backend.
    /// Raw control bytes (Ctrl-C, arrows, tab, ESC) pass through directly.
    /// Returns Ok(()) if sent, Err(reason) if blocked.
    pub fn write_guarded(&mut self, data: &[u8]) -> Result<(), String> {
        for &b in data {
            match b {
                b'\r' | b'\n' => {
                    if !self.line_buf.is_empty() {
                        let line = String::from_utf8_lossy(&self.line_buf).to_string();

                        let scan = safety::scan(line.as_bytes());
                        if scan.blocked {
                            let reason = scan.details.first()
                                .map(|w| w.pattern)
                                .unwrap_or("safety violation");
                            self.line_buf.clear();
                            // Ctrl-U then Ctrl-C — kill the line bash already echoed
                            self.backend.write(&[0x15, 0x03]);
                            return Err(format!("blocked by safety scan: {reason}"));
                        }

                        if let Err(e) = self.guard.check(&line) {
                            self.line_buf.clear();
                            self.backend.write(&[0x15, 0x03]);
                            return Err(e);
                        }

                        self.backend.write(&[b'\r']);
                        self.line_buf.clear();
                    } else {
                        self.backend.write(&[b'\r']);
                    }
                }
                0x7f | 0x08 => {
                    self.line_buf.pop();
                    self.backend.write(&[b]);
                }
                0x01..=0x06 | 0x09 | 0x0b..=0x0c | 0x0e..=0x1a => {
                    self.backend.write(&[b]);
                }
                0x1b => {
                    self.backend.write(&[b]);
                }
                _ => {
                    self.line_buf.push(b);
                    self.backend.write(&[b]);
                }
            }
        }
        Ok(())
    }

    /// Write raw bytes directly — bypasses the safety guard. Public so
    /// integration tests can drive the PTY.
    pub fn write_bytes(&self, data: &[u8]) {
        self.backend.write(data);
    }

    /// Read all available bytes from the backend, run them through the
    /// ANSI pipeline, and return a dirty-cell bitmap for diffing.
    pub fn read_and_apply(&mut self) -> &[u8] {
        self.prev_cells.copy_from_slice(unsafe {
            std::slice::from_raw_parts(
                self.grid.cells_ptr() as *const Cell,
                self.grid.cell_count(),
            )
        });

        let mut total_read = 0;
        loop {
            let n = self.backend
                .read(&mut self.read_buf[total_read..])
                .unwrap_or(0);
            if n == 0 { break; }
            total_read += n;
            if total_read >= self.read_buf.len() { break; }
        }

        if total_read == 0 {
            for d in &mut self.dirty_buf { *d = 0; }
            return &self.dirty_buf;
        }

        if self.scan_buf.len() < total_read {
            self.scan_buf.resize(total_read, 0);
        }

        self.grid.feed(&self.read_buf[..total_read], &mut self.scan_buf);

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

    pub fn resize(&mut self, cols: u16, rows: u16) {
        self.backend.resize(cols, rows);

        self.grid = TermGrid::new(cols, rows);
        let n_cells = cols as usize * rows as usize;
        self.prev_cells = vec![Cell::default(); n_cells];
        self.dirty_buf = vec![0u8; n_cells];
    }
}

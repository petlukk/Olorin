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

        let ret = unsafe {
            libc::openpty(
                &mut master, &mut slave,
                std::ptr::null_mut(), std::ptr::null_mut(), std::ptr::null_mut(),
            )
        };
        if ret != 0 {
            return Err(io::Error::last_os_error());
        }

        // Set window size on the slave
        let ws = libc::winsize {
            ws_row: rows,
            ws_col: cols,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        unsafe { libc::ioctl(slave, libc::TIOCSWINSZ, &ws) };

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

                let shell = b"/bin/bash\0".as_ptr() as *const libc::c_char;
                let argv: [*const libc::c_char; 2] = [shell, std::ptr::null()];
                libc::execvp(shell, argv.as_ptr());
                libc::_exit(127);
            }
        }

        // Parent — close slave, set master non-blocking
        unsafe { libc::close(slave); }
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
    /// Raw control bytes (Ctrl-C, arrows, tab, ESC) pass through directly.
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
                        self.write_raw(&[b'\r']);
                    }
                }
                // Backspace — pop from line buffer
                0x7f | 0x08 => {
                    self.line_buf.pop();
                    self.write_raw(&[b]);
                }
                // Raw control (Ctrl-A..F, tab, VT, FF, Ctrl-N..Ctrl-Z) — pass through
                0x01..=0x06 | 0x09 | 0x0b..=0x0c | 0x0e..=0x1a => {
                    self.write_raw(&[b]);
                }
                // ESC — pass through directly
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

    /// Write raw bytes to the PTY — bypasses safety guard.
    /// Public so integration tests can drive the PTY directly.
    pub fn write_bytes(&mut self, data: &[u8]) {
        self.write_raw(data);
    }

    /// Write raw bytes to the PTY master fd.
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

    /// Read from PTY, run ANSI pipeline, return dirty cell bitmap.
    /// Each byte in the returned slice is 1 if that cell changed, 0 otherwise.
    pub fn read_and_apply(&mut self) -> &[u8] {
        // Snapshot current grid into prev
        self.prev_cells.copy_from_slice(unsafe {
            std::slice::from_raw_parts(
                self.grid.cells_ptr() as *const Cell,
                self.grid.cell_count(),
            )
        });

        // Read all available data from PTY (non-blocking)
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
            for d in &mut self.dirty_buf { *d = 0; }
            return &self.dirty_buf;
        }

        // Grow scan_buf if needed
        if self.scan_buf.len() < total_read {
            self.scan_buf.resize(total_read, 0);
        }

        // Feed through ANSI state machine (SIMD classifier + Rust interpreter)
        self.grid.feed(&self.read_buf[..total_read], &mut self.scan_buf);

        // SIMD diff: compare prev vs current grid
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

    /// Resize the PTY and re-allocate terminal grid + buffers.
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

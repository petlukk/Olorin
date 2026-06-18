//! Streaming file tail for the logwatch palantír: poll a file's size, read the
//! bytes appended since the last poll, and hand back the complete lines among
//! them. Survives truncation and rotation (size shrinks → reopen from the top).
//!
//! Pure std + polling, no inotify dependency — matches Olorin's zero-deps ethos
//! and works the same on every platform. A live log is written as events
//! happen, so a line's arrival time is a faithful stand-in for its event time;
//! the detector keys its lag window off arrival time and never has to parse a
//! timestamp in the hot loop.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub struct Tailer {
    path:    PathBuf,
    offset:  u64,
    partial: Vec<u8>, // bytes after the last newline — not yet a complete line
}

impl Tailer {
    /// Begin tailing from the current END of the file, so only lines appended
    /// after this point are reported. Existing content is read separately for
    /// the learn pass.
    pub fn at_end(path: impl Into<PathBuf>) -> Self {
        let path = path.into();
        let offset = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        Self { path, offset, partial: Vec::new() }
    }

    pub fn path(&self) -> &Path { &self.path }

    /// Read whatever has been appended since the last call and return the
    /// complete lines (newline-terminated, trailing `\r` stripped). A partial
    /// trailing line is buffered until its newline arrives. Returns an empty Vec
    /// when nothing new arrived or the file is momentarily unavailable.
    pub fn poll(&mut self) -> Vec<String> {
        let len = match std::fs::metadata(&self.path) {
            Ok(m) => m.len(),
            Err(_) => return Vec::new(), // gone mid-rotation; retry next tick
        };
        if len < self.offset {
            // Truncated or rotated in place — start over from the top.
            self.offset = 0;
            self.partial.clear();
        }
        if len == self.offset {
            return Vec::new();
        }
        let mut f = match File::open(&self.path) {
            Ok(f) => f,
            Err(_) => return Vec::new(),
        };
        if f.seek(SeekFrom::Start(self.offset)).is_err() {
            return Vec::new();
        }
        let mut buf = Vec::new();
        if f.read_to_end(&mut buf).is_err() {
            return Vec::new();
        }
        self.offset += buf.len() as u64;
        self.partial.extend_from_slice(&buf);
        self.drain_complete_lines()
    }

    fn drain_complete_lines(&mut self) -> Vec<String> {
        let mut lines = Vec::new();
        let mut start = 0;
        for i in 0..self.partial.len() {
            if self.partial[i] == b'\n' {
                let raw = &self.partial[start..i];
                let line = String::from_utf8_lossy(raw).trim_end_matches('\r').to_string();
                lines.push(line);
                start = i + 1;
            }
        }
        if start > 0 {
            self.partial.drain(0..start);
        }
        lines
    }
}

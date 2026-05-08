//! Windows `Spawner` backend. Wraps `std::process::Command` (which calls
//! `CreateProcessW` underneath) so callers see the cross-platform
//! `Spawner` / `ChildProcess` traits defined in `spawner.rs`.
//!
//! No FFI — Rust stdlib already provides the syscall surface. The Unix
//! backend avoids `std::process::Command` only to dodge `pidfd_spawnp@GLIBC_2.39`;
//! that constraint doesn't apply on Windows.

use std::io::{self, BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

use super::spawner::{ChildProcess, Output, Spawner};

pub struct WindowsSpawner;

impl Spawner for WindowsSpawner {
    fn run(&self, argv: &[&str]) -> io::Result<Output> {
        if argv.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "empty argv"));
        }
        let out = Command::new(argv[0]).args(&argv[1..]).output()?;
        Ok(Output {
            stdout:    out.stdout,
            stderr:    out.stderr,
            exit_code: out.status.code().unwrap_or(-1),
        })
    }

    fn shell(&self, cmd: &str) -> io::Result<Output> {
        let out = Command::new("cmd").args(["/C", cmd]).output()?;
        Ok(Output {
            stdout:    out.stdout,
            stderr:    out.stderr,
            exit_code: out.status.code().unwrap_or(-1),
        })
    }

    fn spawn(&self, argv: &[&str]) -> io::Result<Box<dyn ChildProcess>> {
        if argv.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "empty argv"));
        }
        let mut child = Command::new(argv[0])
            .args(&argv[1..])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;

        let stdin = child.stdin.take()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "no stdin pipe"))?;
        let stdout = child.stdout.take()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "no stdout pipe"))?;

        Ok(Box::new(WindowsChild {
            id:     child.id() as u64,
            child:  Mutex::new(Some(child)),
            stdin:  Mutex::new(stdin),
            stdout: Mutex::new(BufReader::new(stdout)),
        }))
    }
}

pub struct WindowsChild {
    id:     u64,
    child:  Mutex<Option<Child>>,
    stdin:  Mutex<ChildStdin>,
    stdout: Mutex<BufReader<ChildStdout>>,
}

impl ChildProcess for WindowsChild {
    /// Read one line from child stdout. Strips trailing CRLF / LF so the
    /// returned string matches the Unix backend (no terminator). The
    /// WhatsApp JSONL parser depends on this — leaving `\r` in would
    /// break `extract_json_string` in `server.rs`.
    fn read_line(&self, buf: &mut String) -> io::Result<usize> {
        buf.clear();
        let mut reader = self.stdout.lock().unwrap();
        let n = reader.read_line(buf)?;
        if n == 0 {
            return Ok(0);
        }
        if buf.ends_with('\n') { buf.pop(); }
        if buf.ends_with('\r') { buf.pop(); }
        Ok(buf.len())
    }

    fn write_line(&self, line: &str) -> io::Result<()> {
        let mut stdin = self.stdin.lock().unwrap();
        stdin.write_all(line.as_bytes())?;
        stdin.write_all(b"\n")?;
        Ok(())
    }

    fn wait(&self) -> i32 {
        let mut guard = self.child.lock().unwrap();
        match guard.take() {
            Some(mut child) => child.wait()
                .ok()
                .and_then(|s| s.code())
                .unwrap_or(-1),
            None => -1,
        }
    }

    fn id(&self) -> u64 {
        self.id
    }
}

impl Drop for WindowsChild {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.child.lock() {
            if let Some(mut child) = guard.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

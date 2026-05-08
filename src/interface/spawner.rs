//! Process-spawn abstraction. One trait per role: oneshot `run`/`shell`,
//! and long-lived `spawn` returning a line-oriented `ChildProcess`.
//!
//! Unix backend lives in `spawner_unix.rs` (raw fork+exec via `exec.rs`).
//! Windows backend lives in `spawner_windows.rs` (`std::process::Command`
//! over `CreateProcessW`).

use std::io;

pub struct Output {
    pub stdout:    Vec<u8>,
    pub stderr:    Vec<u8>,
    pub exit_code: i32,
}

pub trait Spawner: Send + Sync {
    fn run(&self, argv: &[&str]) -> io::Result<Output>;
    fn shell(&self, cmd: &str) -> io::Result<Output>;
    fn spawn(&self, argv: &[&str]) -> io::Result<Box<dyn ChildProcess>>;
}

pub trait ChildProcess: Send + Sync {
    fn read_line(&self, buf: &mut String) -> io::Result<usize>;
    fn write_line(&self, line: &str) -> io::Result<()>;
    fn wait(&self) -> i32;
    fn id(&self) -> u64;
}

#[cfg(unix)]
pub fn default_spawner() -> Box<dyn Spawner> {
    Box::new(super::spawner_unix::UnixSpawner)
}

#[cfg(windows)]
pub fn default_spawner() -> Box<dyn Spawner> {
    Box::new(super::spawner_windows::WindowsSpawner)
}

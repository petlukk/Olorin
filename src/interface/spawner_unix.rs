//! Unix `Spawner` backend. Wraps the raw fork+exec implementation in
//! `exec.rs` so callers can use the cross-platform `Spawner` /
//! `ChildProcess` traits without losing the no-`pidfd_spawnp` property.

use std::io;

use super::exec;
use super::spawner::{ChildProcess, Output, Spawner};

pub struct UnixSpawner;

impl Spawner for UnixSpawner {
    fn run(&self, argv: &[&str]) -> io::Result<Output> {
        exec::run(argv)
    }

    fn shell(&self, cmd: &str) -> io::Result<Output> {
        exec::shell(cmd)
    }

    fn spawn(&self, argv: &[&str]) -> io::Result<Box<dyn ChildProcess>> {
        exec::spawn(argv).map(|c| Box::new(c) as Box<dyn ChildProcess>)
    }
}

impl ChildProcess for exec::Child {
    fn read_line(&self, buf: &mut String) -> io::Result<usize> {
        exec::Child::read_line(self, buf)
    }

    fn write_line(&self, line: &str) -> io::Result<()> {
        exec::Child::write_line(self, line)
    }

    fn wait(&self) -> i32 {
        exec::Child::wait(self)
    }

    fn id(&self) -> u64 {
        self.pid as u64
    }
}

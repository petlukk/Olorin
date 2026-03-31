//! Raw fork+exec process spawning — no std::process::Command, no pidfd.
//!
//! Uses libc fork/execvp/waitpid directly. Works on glibc 2.17+ (any Linux
//! since 2013). Avoids Rust std's posix_spawn path which pulls in
//! pidfd_spawnp@GLIBC_2.39.

use std::ffi::CString;
use std::io;

// ── One-shot run ──────────────────────────────────────────────────────────────

/// Output from a completed child process.
pub struct Output {
    pub stdout:    Vec<u8>,
    pub stderr:    Vec<u8>,
    pub exit_code: i32,
}

/// Run a command, capture stdout+stderr, wait for completion.
/// `argv[0]` is the program, rest are arguments.
pub fn run(argv: &[&str]) -> io::Result<Output> {
    if argv.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "empty argv"));
    }

    let c_args: Vec<CString> = argv.iter().map(|s| {
        CString::new(*s).map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "NUL byte in argument"))
    }).collect::<io::Result<Vec<CString>>>()?;
    let c_ptrs: Vec<*const libc::c_char> = c_args
        .iter()
        .map(|s| s.as_ptr())
        .chain(std::iter::once(std::ptr::null()))
        .collect();

    let mut stdout_pipe = [0i32; 2];
    let mut stderr_pipe = [0i32; 2];
    unsafe {
        if libc::pipe(stdout_pipe.as_mut_ptr()) != 0 {
            return Err(io::Error::last_os_error());
        }
        if libc::pipe(stderr_pipe.as_mut_ptr()) != 0 {
            libc::close(stdout_pipe[0]);
            libc::close(stdout_pipe[1]);
            return Err(io::Error::last_os_error());
        }
    }

    let pid = unsafe { libc::fork() };
    if pid < 0 {
        return Err(io::Error::last_os_error());
    }

    if pid == 0 {
        // Child
        unsafe {
            libc::close(stdout_pipe[0]);
            libc::close(stderr_pipe[0]);
            libc::dup2(stdout_pipe[1], 1);
            libc::dup2(stderr_pipe[1], 2);
            libc::close(stdout_pipe[1]);
            libc::close(stderr_pipe[1]);
            libc::execvp(c_ptrs[0], c_ptrs.as_ptr());
            libc::_exit(127);
        }
    }

    // Parent
    unsafe {
        libc::close(stdout_pipe[1]);
        libc::close(stderr_pipe[1]);
    }

    let stdout = read_fd(stdout_pipe[0]);
    let stderr = read_fd(stderr_pipe[0]);

    unsafe {
        libc::close(stdout_pipe[0]);
        libc::close(stderr_pipe[0]);
    }

    let mut status: i32 = 0;
    unsafe { libc::waitpid(pid, &mut status, 0) };

    let exit_code = if libc::WIFEXITED(status) {
        libc::WEXITSTATUS(status)
    } else {
        -1
    };

    Ok(Output { stdout, stderr, exit_code })
}

/// Convenience: run "sh -c <cmd>"
pub fn shell(cmd: &str) -> io::Result<Output> {
    run(&["sh", "-c", cmd])
}

// ── Long-lived subprocess ─────────────────────────────────────────────────────

/// A spawned child with piped stdin/stdout for long-lived subprocesses.
pub struct Child {
    pub pid:       i32,
    pub stdin_fd:  i32,
    pub stdout_fd: i32,
}

impl Child {
    /// Read one line (up to newline or EOF) from child stdout.
    pub fn read_line(&self, buf: &mut String) -> io::Result<usize> {
        buf.clear();
        let mut byte = [0u8; 1];
        let mut count = 0;
        loop {
            let n = unsafe {
                libc::read(self.stdout_fd, byte.as_mut_ptr() as *mut libc::c_void, 1)
            };
            if n <= 0 {
                break;
            }
            count += 1;
            if byte[0] == b'\n' {
                break;
            }
            buf.push(byte[0] as char);
        }
        Ok(count)
    }

    /// Write a line (+ newline) to child stdin.
    pub fn write_line(&self, line: &str) -> io::Result<()> {
        let bytes = line.as_bytes();
        let mut written = 0;
        while written < bytes.len() {
            let n = unsafe {
                libc::write(
                    self.stdin_fd,
                    bytes[written..].as_ptr() as *const libc::c_void,
                    bytes.len() - written,
                )
            };
            if n <= 0 {
                return Err(io::Error::last_os_error());
            }
            written += n as usize;
        }
        let nl = b"\n";
        unsafe { libc::write(self.stdin_fd, nl.as_ptr() as *const libc::c_void, 1) };
        Ok(())
    }

    /// Wait for the child to exit and return its exit code.
    pub fn wait(&self) -> i32 {
        let mut status: i32 = 0;
        unsafe { libc::waitpid(self.pid, &mut status, 0) };
        if libc::WIFEXITED(status) {
            libc::WEXITSTATUS(status)
        } else {
            -1
        }
    }
}

impl Drop for Child {
    fn drop(&mut self) {
        unsafe {
            libc::close(self.stdin_fd);
            libc::close(self.stdout_fd);
            libc::kill(self.pid, libc::SIGTERM);
            libc::waitpid(self.pid, std::ptr::null_mut(), libc::WNOHANG);
        }
    }
}

/// Spawn a long-lived subprocess with piped stdin/stdout.
/// stderr is inherited (goes to parent's stderr).
pub fn spawn(argv: &[&str]) -> io::Result<Child> {
    if argv.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "empty argv"));
    }

    let c_args: Vec<CString> = argv.iter().map(|s| {
        CString::new(*s).map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "NUL byte in argument"))
    }).collect::<io::Result<Vec<CString>>>()?;
    let c_ptrs: Vec<*const libc::c_char> = c_args
        .iter()
        .map(|s| s.as_ptr())
        .chain(std::iter::once(std::ptr::null()))
        .collect();

    let mut stdin_pipe  = [0i32; 2]; // parent writes → child reads
    let mut stdout_pipe = [0i32; 2]; // child writes → parent reads
    unsafe {
        if libc::pipe(stdin_pipe.as_mut_ptr()) != 0 {
            return Err(io::Error::last_os_error());
        }
        if libc::pipe(stdout_pipe.as_mut_ptr()) != 0 {
            libc::close(stdin_pipe[0]);
            libc::close(stdin_pipe[1]);
            return Err(io::Error::last_os_error());
        }
    }

    let pid = unsafe { libc::fork() };
    if pid < 0 {
        return Err(io::Error::last_os_error());
    }

    if pid == 0 {
        unsafe {
            libc::close(stdin_pipe[1]);
            libc::close(stdout_pipe[0]);
            libc::dup2(stdin_pipe[0], 0);
            libc::dup2(stdout_pipe[1], 1);
            libc::close(stdin_pipe[0]);
            libc::close(stdout_pipe[1]);
            libc::execvp(c_ptrs[0], c_ptrs.as_ptr());
            libc::_exit(127);
        }
    }

    // Parent
    unsafe {
        libc::close(stdin_pipe[0]);
        libc::close(stdout_pipe[1]);
    }

    Ok(Child { pid, stdin_fd: stdin_pipe[1], stdout_fd: stdout_pipe[0] })
}

// ── Internal ──────────────────────────────────────────────────────────────────

fn read_fd(fd: i32) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4096);
    let mut tmp = [0u8; 4096];
    loop {
        let n = unsafe {
            libc::read(fd, tmp.as_mut_ptr() as *mut libc::c_void, tmp.len())
        };
        if n <= 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n as usize]);
    }
    buf
}

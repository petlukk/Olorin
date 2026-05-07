//! Unix `PtyBackend` — openpty + fork+exec bash, raw read/write/poll on
//! the master fd. The cross-platform `PtySession` in `pty.rs` owns the
//! terminal grid, ShellGuard, and safety pipeline and talks to this
//! backend purely through the trait surface.

use std::io;

use super::pty::PtyBackend;

pub struct UnixPtyBackend {
    master_fd: i32,
    child_pid: i32,
}

pub fn open(cols: u16, rows: u16) -> io::Result<UnixPtyBackend> {
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
        unsafe {
            libc::close(master);
            libc::setsid();
            libc::ioctl(slave, libc::TIOCSCTTY, 0i32);
            libc::dup2(slave, 0);
            libc::dup2(slave, 1);
            libc::dup2(slave, 2);
            if slave > 2 { libc::close(slave); }

            let term = b"TERM=xterm-256color\0".as_ptr() as *const libc::c_char;
            libc::putenv(term as *mut libc::c_char);

            let shell = b"/bin/bash\0".as_ptr() as *const libc::c_char;
            let login = b"--login\0".as_ptr() as *const libc::c_char;
            let argv: [*const libc::c_char; 3] = [shell, login, std::ptr::null()];
            libc::execvp(shell, argv.as_ptr());
            libc::_exit(127);
        }
    }

    unsafe { libc::close(slave); }
    unsafe {
        let flags = libc::fcntl(master, libc::F_GETFL);
        libc::fcntl(master, libc::F_SETFL, flags | libc::O_NONBLOCK);
    }

    Ok(UnixPtyBackend { master_fd: master, child_pid: pid })
}

impl PtyBackend for UnixPtyBackend {
    fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        let n = unsafe {
            libc::read(
                self.master_fd,
                buf.as_mut_ptr() as *mut libc::c_void,
                buf.len(),
            )
        };
        if n <= 0 { Ok(0) } else { Ok(n as usize) }
    }

    fn write(&self, data: &[u8]) {
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

    fn resize(&self, cols: u16, rows: u16) {
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
    }

    fn child_alive(&self) -> bool {
        let mut status: i32 = 0;
        let r = unsafe { libc::waitpid(self.child_pid, &mut status, libc::WNOHANG) };
        r == 0
    }

    fn wait_readable(&self, timeout_ms: i32) -> bool {
        let mut pollfd = libc::pollfd {
            fd: self.master_fd,
            events: libc::POLLIN,
            revents: 0,
        };
        let ret = unsafe { libc::poll(&mut pollfd, 1, timeout_ms) };
        ret > 0 && (pollfd.revents & libc::POLLIN) != 0
    }
}

impl Drop for UnixPtyBackend {
    fn drop(&mut self) {
        unsafe {
            libc::kill(self.child_pid, libc::SIGTERM);
            libc::close(self.master_fd);
            libc::waitpid(self.child_pid, std::ptr::null_mut(), libc::WNOHANG);
        }
    }
}

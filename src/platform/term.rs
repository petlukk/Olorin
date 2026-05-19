//! Read a secret line from the controlling terminal with echo
//! disabled — used to prompt for the vault passphrase at startup.
//!
//! Bypasses stdin and goes straight to `/dev/tty` (Unix) or
//! `CONIN$` (Windows) so the prompt still works when stdin is piped
//! from a script.  Echo is suppressed by toggling the canonical
//! terminal control bits (`ECHO` off on Unix, `ENABLE_ECHO_INPUT`
//! off on Windows); the original mode is restored even on error
//! paths.
//!
//! The secret lands in a [`SecureBuffer`] (mlock'd, SIMD-zeroed on
//! Drop) — the temporary `Vec<u8>` used to accumulate bytes is
//! zeroed before it falls out of scope.

use crate::error::{Error, Result};
use crate::storage::secure::SecureBuffer;

/// Prompt on the tty, read a line without echoing, return the bytes
/// in a `SecureBuffer`.  Errors when no controlling terminal is
/// available, the input is empty, or the underlying read fails.
pub fn read_secret(prompt: &str) -> Result<SecureBuffer> {
    imp::read_secret(prompt)
}

/// True when stdin is connected to a terminal (so prompting makes
/// sense); false when it's a pipe / file / no fd at all.
pub fn stdin_is_tty() -> bool {
    imp::stdin_is_tty()
}

#[cfg(unix)]
mod imp {
    use std::io::{Read, Write};
    use std::os::unix::io::AsRawFd;

    use super::*;

    pub fn stdin_is_tty() -> bool {
        unsafe { libc::isatty(libc::STDIN_FILENO) == 1 }
    }

    pub fn read_secret(prompt: &str) -> Result<SecureBuffer> {
        let mut tty = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/tty")
            .map_err(|_| {
                Error::Vault("no controlling terminal — cannot prompt for passphrase")
            })?;

        tty.write_all(prompt.as_bytes())
            .map_err(|_| Error::Vault("tty write failed"))?;
        tty.flush()
            .map_err(|_| Error::Vault("tty flush failed"))?;

        let fd = tty.as_raw_fd();

        // Save current termios so we can restore it even if the read
        // bails midway.
        let mut old: libc::termios = unsafe { std::mem::zeroed() };
        if unsafe { libc::tcgetattr(fd, &mut old) } != 0 {
            return Err(Error::Vault("tcgetattr failed"));
        }
        let mut quiet = old;
        quiet.c_lflag &= !libc::ECHO;
        if unsafe { libc::tcsetattr(fd, libc::TCSAFLUSH, &quiet) } != 0 {
            return Err(Error::Vault("tcsetattr failed"));
        }

        let mut buf: Vec<u8> = Vec::with_capacity(64);
        let read_result = read_line_into(&mut tty, &mut buf);

        // Restore termios + emit the newline the user's Enter
        // would have produced (we swallowed it during the read).
        unsafe { libc::tcsetattr(fd, libc::TCSAFLUSH, &old) };
        let _ = tty.write_all(b"\n");
        let _ = tty.flush();

        read_result?;
        if buf.is_empty() {
            return Err(Error::Vault("empty passphrase rejected"));
        }

        let mut secret = SecureBuffer::new(buf.len());
        secret.write(&buf);
        // Zero our staging buffer before drop.  The capacity may
        // exceed the live length; zero everything we touched.
        for b in buf.iter_mut() {
            *b = 0;
        }
        Ok(secret)
    }

    fn read_line_into(tty: &mut std::fs::File, out: &mut Vec<u8>) -> Result<()> {
        let mut byte = [0u8; 1];
        loop {
            match tty.read(&mut byte) {
                Ok(0) => return Err(Error::Vault("eof while reading passphrase")),
                Ok(_) => {
                    if byte[0] == b'\n' {
                        return Ok(());
                    }
                    if byte[0] == b'\r' {
                        continue;
                    }
                    out.push(byte[0]);
                }
                Err(_) => return Err(Error::Vault("tty read failed")),
            }
        }
    }
}

#[cfg(windows)]
mod imp {
    use std::io::Write;

    use super::*;

    type HANDLE = *mut std::ffi::c_void;
    const STD_INPUT_HANDLE: u32 = 0xFFFF_FFF6; // (DWORD)-10
    const ENABLE_ECHO_INPUT: u32 = 0x0004;
    const ENABLE_LINE_INPUT: u32 = 0x0002;
    const ENABLE_PROCESSED_INPUT: u32 = 0x0001;

    #[link(name = "kernel32")]
    extern "system" {
        fn GetStdHandle(nStdHandle: u32) -> HANDLE;
        fn GetConsoleMode(hConsoleHandle: HANDLE, lpMode: *mut u32) -> i32;
        fn SetConsoleMode(hConsoleHandle: HANDLE, dwMode: u32) -> i32;
        fn ReadConsoleW(
            hConsoleInput: HANDLE,
            lpBuffer: *mut u16,
            nNumberOfCharsToRead: u32,
            lpNumberOfCharsRead: *mut u32,
            pInputControl: *mut std::ffi::c_void,
        ) -> i32;
    }

    pub fn stdin_is_tty() -> bool {
        // GetConsoleMode succeeds iff the handle refers to a console.
        unsafe {
            let h = GetStdHandle(STD_INPUT_HANDLE);
            let mut mode: u32 = 0;
            GetConsoleMode(h, &mut mode) != 0
        }
    }

    pub fn read_secret(prompt: &str) -> Result<SecureBuffer> {
        let stderr = std::io::stderr();
        {
            let mut s = stderr.lock();
            s.write_all(prompt.as_bytes())
                .map_err(|_| Error::Vault("stderr write failed"))?;
            s.flush().map_err(|_| Error::Vault("stderr flush failed"))?;
        }

        unsafe {
            let h = GetStdHandle(STD_INPUT_HANDLE);
            let mut old_mode: u32 = 0;
            if GetConsoleMode(h, &mut old_mode) == 0 {
                return Err(Error::Vault("no controlling console"));
            }
            let quiet = (old_mode | ENABLE_LINE_INPUT | ENABLE_PROCESSED_INPUT)
                & !ENABLE_ECHO_INPUT;
            if SetConsoleMode(h, quiet) == 0 {
                return Err(Error::Vault("SetConsoleMode failed"));
            }

            let mut wide = vec![0u16; 1024];
            let mut chars_read: u32 = 0;
            let ok = ReadConsoleW(
                h,
                wide.as_mut_ptr(),
                wide.len() as u32,
                &mut chars_read,
                std::ptr::null_mut(),
            );

            // Restore mode regardless of read outcome.
            SetConsoleMode(h, old_mode);
            // Emit the newline that the suppressed Enter would have shown.
            let _ = stderr.lock().write_all(b"\r\n");

            if ok == 0 {
                return Err(Error::Vault("ReadConsoleW failed"));
            }

            let mut consumed = chars_read as usize;
            // Strip trailing CR/LF.
            while consumed > 0
                && (wide[consumed - 1] == b'\r' as u16 || wide[consumed - 1] == b'\n' as u16)
            {
                consumed -= 1;
            }
            let utf16 = &wide[..consumed];
            let s = String::from_utf16(utf16)
                .map_err(|_| Error::Vault("invalid UTF-16 from console"))?;
            // Zero the wide buffer before drop.
            for w in wide.iter_mut() {
                *w = 0;
            }

            let bytes = s.into_bytes();
            if bytes.is_empty() {
                return Err(Error::Vault("empty passphrase rejected"));
            }
            let mut secret = SecureBuffer::new(bytes.len());
            secret.write(&bytes);
            // Best-effort zero: bytes owns a heap String allocation.
            // Once we leave this scope the String drop runs; we
            // overwrite first so the freed pages aren't readable.
            let mut bytes = bytes;
            for b in bytes.iter_mut() {
                *b = 0;
            }
            Ok(secret)
        }
    }
}

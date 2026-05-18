//! Cryptographically secure random bytes from the OS entropy pool.
//!
//! Used to generate the per-vault Argon2id salt at first vault create.
//! One-shot at startup; no userspace PRNG state to keep zeroed, which
//! is why we go straight to the kernel here instead of layering a
//! ChaCha20-DRBG on top.

use crate::error::{Error, Result};

/// Fill `buf` with cryptographically secure random bytes.
/// Errors if the OS entropy source is unavailable.
pub fn fill_bytes(buf: &mut [u8]) -> Result<()> {
    imp::fill_bytes(buf)
}

#[cfg(unix)]
mod imp {
    use super::*;
    use std::io::Read;

    pub fn fill_bytes(buf: &mut [u8]) -> Result<()> {
        // Prefer the getrandom(2) syscall (Linux 3.17+; available on
        // all modern Linux + macOS via getentropy compat).  Fall back
        // to /dev/urandom if the syscall is missing — should never
        // happen on supported targets but means the function still
        // does something useful inside minimal containers without a
        // syscall filter for getrandom.
        if syscall_fill(buf).is_ok() {
            return Ok(());
        }
        let mut f = std::fs::File::open("/dev/urandom")
            .map_err(|_| Error::Vault("no OS entropy source"))?;
        f.read_exact(buf)
            .map_err(|_| Error::Vault("urandom short read"))?;
        Ok(())
    }

    #[cfg(target_os = "linux")]
    fn syscall_fill(buf: &mut [u8]) -> std::result::Result<(), ()> {
        // SYS_getrandom = 318 on x86_64, 278 on aarch64.
        const SYS_GETRANDOM: i64 = if cfg!(target_arch = "x86_64") { 318 } else { 278 };
        let mut filled = 0;
        while filled < buf.len() {
            let n = unsafe {
                libc::syscall(
                    SYS_GETRANDOM,
                    buf[filled..].as_mut_ptr(),
                    buf.len() - filled,
                    0u32,
                )
            };
            if n < 0 {
                return Err(());
            }
            filled += n as usize;
        }
        Ok(())
    }

    #[cfg(not(target_os = "linux"))]
    fn syscall_fill(_buf: &mut [u8]) -> std::result::Result<(), ()> {
        Err(()) // fall through to /dev/urandom on non-Linux Unix.
    }
}

#[cfg(windows)]
mod imp {
    use super::*;

    #[link(name = "bcrypt")]
    extern "system" {
        fn BCryptGenRandom(
            hAlgorithm: *mut std::ffi::c_void,
            pbBuffer: *mut u8,
            cbBuffer: u32,
            dwFlags: u32,
        ) -> i32;
    }
    const BCRYPT_USE_SYSTEM_PREFERRED_RNG: u32 = 0x00000002;

    pub fn fill_bytes(buf: &mut [u8]) -> Result<()> {
        let status = unsafe {
            BCryptGenRandom(
                std::ptr::null_mut(),
                buf.as_mut_ptr(),
                buf.len() as u32,
                BCRYPT_USE_SYSTEM_PREFERRED_RNG,
            )
        };
        if status != 0 {
            return Err(Error::Vault("BCryptGenRandom failed"));
        }
        Ok(())
    }
}

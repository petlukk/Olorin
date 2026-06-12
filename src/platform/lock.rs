//! Page locking — keeps sensitive memory off the swap file.
//!
//! Unix: `mlock` / `munlock`. May fail without `CAP_IPC_LOCK` on Linux;
//! callers treat this as best-effort.
//! Windows: `VirtualLock` / `VirtualUnlock`. Working-set-sized, also
//! best-effort.

/// Best-effort: lock `len` bytes at `ptr` into RAM. Errors are ignored —
/// SecureBuffer still SIMD-zeroes on Drop, so a failed lock means
/// "weaker against swap" but never "leaks plaintext."
#[inline]
pub fn lock_pages(ptr: *const u8, len: usize) {
    #[cfg(unix)]
    unsafe {
        libc::mlock(ptr as *const libc::c_void, len);
    }
    #[cfg(windows)]
    unsafe {
        extern "system" {
            fn VirtualLock(addr: *const core::ffi::c_void, size: usize) -> i32;
        }
        VirtualLock(ptr as *const core::ffi::c_void, len);
    }
}

/// Best-effort: release a previous `lock_pages` over the same region.
#[inline]
pub fn unlock_pages(ptr: *const u8, len: usize) {
    #[cfg(unix)]
    unsafe {
        libc::munlock(ptr as *const libc::c_void, len);
    }
    #[cfg(windows)]
    unsafe {
        extern "system" {
            fn VirtualUnlock(addr: *const core::ffi::c_void, size: usize) -> i32;
        }
        VirtualUnlock(ptr as *const core::ffi::c_void, len);
    }
}

/// Try to take a whole-file **exclusive advisory lock**, non-blocking.
/// Returns `true` if acquired, `false` if another process already holds it.
///
/// Released automatically when `file`'s descriptor/handle is closed or the
/// process dies (`flock` / `LockFileEx` semantics) — so a crashed holder
/// never leaves a stale lock, unlike a PID lockfile. Advisory: it only
/// coordinates between processes that also call this, and never blocks the
/// vault's own reads/writes.
pub fn try_lock_file_exclusive(file: &std::fs::File) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::io::AsRawFd;
        // flock is per-open-file-description, so even a second open() within
        // the same process is correctly rejected (LOCK_NB → EWOULDBLOCK).
        unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) == 0 }
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::AsRawHandle;
        const LOCKFILE_FAIL_IMMEDIATELY: u32 = 0x0000_0001;
        const LOCKFILE_EXCLUSIVE_LOCK: u32 = 0x0000_0002;
        #[repr(C)]
        struct Overlapped {
            internal: usize,
            internal_high: usize,
            offset: u32,
            offset_high: u32,
            h_event: *mut core::ffi::c_void,
        }
        extern "system" {
            fn LockFileEx(
                handle: *mut core::ffi::c_void,
                flags: u32,
                reserved: u32,
                bytes_low: u32,
                bytes_high: u32,
                overlapped: *mut Overlapped,
            ) -> i32;
        }
        let mut ov = Overlapped {
            internal: 0,
            internal_high: 0,
            offset: 0,
            offset_high: 0,
            h_event: core::ptr::null_mut(),
        };
        unsafe {
            LockFileEx(
                file.as_raw_handle() as *mut core::ffi::c_void,
                LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY,
                0,
                1,
                0,
                &mut ov,
            ) != 0
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        // Unknown platform: don't gate vault usage on a lock we can't take.
        let _ = file;
        true
    }
}

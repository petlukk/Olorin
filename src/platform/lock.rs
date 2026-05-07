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

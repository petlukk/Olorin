//! Futex-equivalent block / wake primitives over a `u32` location.
//!
//! Linux: direct `SYS_futex` syscall via libc, with FUTEX_PRIVATE_FLAG
//! for process-private waiters (cheaper kernel path).
//! Windows: `WaitOnAddress` / `WakeByAddressAll` from kernel32, available
//! since Windows 8. Same compare-and-block semantics over a 4-byte slot.
//!
//! `wait_call_count()` is exposed so the threadpool's preemption-
//! robustness test can assert the slow path was actually taken.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

static WAIT_CALLS: AtomicUsize = AtomicUsize::new(0);

/// Total number of `wait` invocations since process start.
pub fn wait_call_count() -> usize {
    WAIT_CALLS.load(Ordering::Relaxed)
}

/// Block until `*addr != expected` or a wake arrives. Returns promptly
/// on EAGAIN (value already changed) / EINTR (signal); callers re-check
/// the condition in a loop.
#[inline(never)]
pub fn wait(addr: &AtomicU32, expected: u32) {
    WAIT_CALLS.fetch_add(1, Ordering::Relaxed);
    let ptr = addr as *const AtomicU32 as *const u32;

    #[cfg(target_os = "linux")]
    unsafe {
        const FUTEX_WAIT: libc::c_int = 0;
        const FUTEX_PRIVATE_FLAG: libc::c_int = 128;
        const FUTEX_WAIT_PRIVATE: libc::c_int = FUTEX_WAIT | FUTEX_PRIVATE_FLAG;
        libc::syscall(
            libc::SYS_futex,
            ptr,
            FUTEX_WAIT_PRIVATE,
            expected as libc::c_int,
            std::ptr::null::<libc::timespec>(),
        );
    }
    #[cfg(windows)]
    unsafe {
        const INFINITE: u32 = 0xFFFFFFFF;
        extern "system" {
            fn WaitOnAddress(
                addr: *const core::ffi::c_void,
                compare: *const core::ffi::c_void,
                size: usize,
                ms: u32,
            ) -> i32;
        }
        let expected = expected;
        WaitOnAddress(
            ptr as *const core::ffi::c_void,
            &expected as *const u32 as *const core::ffi::c_void,
            4,
            INFINITE,
        );
    }
}

/// Wake all waiters blocked on `addr`.
#[inline(never)]
pub fn wake_all(addr: &AtomicU32) {
    let ptr = addr as *const AtomicU32 as *const u32;

    #[cfg(target_os = "linux")]
    unsafe {
        const FUTEX_WAKE: libc::c_int = 1;
        const FUTEX_PRIVATE_FLAG: libc::c_int = 128;
        const FUTEX_WAKE_PRIVATE: libc::c_int = FUTEX_WAKE | FUTEX_PRIVATE_FLAG;
        libc::syscall(
            libc::SYS_futex,
            ptr,
            FUTEX_WAKE_PRIVATE,
            i32::MAX,
        );
    }
    #[cfg(windows)]
    unsafe {
        extern "system" {
            fn WakeByAddressAll(addr: *const core::ffi::c_void);
        }
        WakeByAddressAll(ptr as *const core::ffi::c_void);
    }
}

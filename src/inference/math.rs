//! Shared math utilities for inference forward passes.

pub(crate) fn wipe_f32(buf: &mut [f32]) {
    unsafe { std::ptr::write_bytes(buf.as_mut_ptr(), 0, buf.len()); }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

pub(crate) fn wipe_i8(buf: &mut [i8]) {
    unsafe { std::ptr::write_bytes(buf.as_mut_ptr(), 0, buf.len()); }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

pub(crate) fn wipe_i32(buf: &mut [i32]) {
    unsafe { std::ptr::write_bytes(buf.as_mut_ptr(), 0, buf.len()); }
    std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);
}

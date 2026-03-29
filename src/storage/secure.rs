//! SecureBuffer — mlock'd memory with SIMD-zeroed Drop.
//!
//! Fixed-size buffer for sensitive data. Memory is locked (mlock) to prevent
//! swap and SIMD-zeroed on drop via the Eä `zeroize` kernel.

use crate::kernels::ffi;

/// Fixed-size secure buffer. Memory is mlock'd and SIMD-zeroed on Drop.
pub struct SecureBuffer {
    ptr: *mut u8,
    len: usize,
}

impl SecureBuffer {
    /// Allocate `len` bytes, mlock the region. Memory is zeroed on allocation.
    pub fn new(len: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(len, 16)
            .expect("invalid layout");
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        // Lock pages — best effort (may fail without CAP_IPC_LOCK).
        unsafe { libc::mlock(ptr as *const libc::c_void, len); }
        Self { ptr, len }
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
}

// SAFETY: We own the pointer exclusively; no shared aliasing.
unsafe impl Send for SecureBuffer {}

impl Drop for SecureBuffer {
    fn drop(&mut self) {
        if self.len > 0 {
            unsafe {
                ffi::zeroize(self.ptr, self.len as i32);
                libc::munlock(self.ptr as *const libc::c_void, self.len);
                let layout = std::alloc::Layout::from_size_align_unchecked(self.len, 16);
                std::alloc::dealloc(self.ptr, layout);
            }
        }
    }
}

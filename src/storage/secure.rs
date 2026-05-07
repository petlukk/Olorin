//! SecureBuffer — mlock'd memory with SIMD-zeroed Drop.
//!
//! Fixed-size buffer for sensitive data. Memory is locked (mlock) to prevent
//! swap and SIMD-zeroed on drop via the Eä `zeroize` kernel.

use crate::kernels::ffi;
use crate::platform::lock::{lock_pages, unlock_pages};

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
        lock_pages(ptr, len);
        Self { ptr, len }
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Copy `src` into the buffer. Panics if `src.len() != self.len`.
    pub fn write(&mut self, src: &[u8]) {
        assert_eq!(src.len(), self.len, "SecureBuffer::write length mismatch");
        self.as_mut_slice().copy_from_slice(src);
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
                unlock_pages(self.ptr, self.len);
                let layout = std::alloc::Layout::from_size_align_unchecked(self.len, 16);
                std::alloc::dealloc(self.ptr, layout);
            }
        }
    }
}

//! Send/Sync pointer wrappers for thread::scope dispatch.
//!
//! Raw pointers are not Send/Sync, but scoped threads guarantee the
//! pointed-to data outlives the scope. These wrappers encode that.

#[derive(Clone, Copy)]
pub(crate) struct SendPtr<T>(pub *const T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    pub fn ptr(self) -> *const T { self.0 }
}

#[derive(Clone, Copy)]
pub(crate) struct SendMutPtr<T>(pub *mut T);
unsafe impl<T> Send for SendMutPtr<T> {}
unsafe impl<T> Sync for SendMutPtr<T> {}

impl<T> SendMutPtr<T> {
    pub fn ptr(self) -> *mut T { self.0 }
}

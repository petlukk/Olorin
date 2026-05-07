//! Read-only file mapping. Used by `inference/gguf.rs` to keep model
//! weights in the OS page cache instead of heap.
//!
//! Unix: `mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0)` + `munmap`.
//! Windows: `CreateFileW` + `CreateFileMappingW` + `MapViewOfFile` —
//! lands in a follow-up commit alongside the other Windows backends.

#[cfg(unix)]
use std::io;
#[cfg(unix)]
use std::path::Path;

pub struct MapView {
    ptr: *mut u8,
    len: usize,
}

unsafe impl Send for MapView {}
unsafe impl Sync for MapView {}

impl MapView {
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }
}

/// Map the file at `path` read-only into virtual memory.
#[cfg(unix)]
pub fn map_file_readonly(path: &Path) -> io::Result<MapView> {
    use std::os::unix::io::AsRawFd;

    let file = std::fs::File::open(path)?;
    let len = file.metadata()?.len() as usize;
    if len == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "empty file"));
    }
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_PRIVATE,
            file.as_raw_fd(),
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        return Err(io::Error::last_os_error());
    }
    Ok(MapView { ptr: ptr as *mut u8, len })
}

#[cfg(unix)]
impl Drop for MapView {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.len > 0 {
            unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.len); }
        }
    }
}

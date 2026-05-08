//! Read-only file mapping. Used by `inference/gguf.rs` to keep model
//! weights in the OS page cache instead of heap.
//!
//! Unix:    `mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, 0)` + `munmap`.
//! Windows: `CreateFileW` + `CreateFileMappingW` + `MapViewOfFile`.

use std::io;
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

// ── Windows ───────────────────────────────────────────────────────────────────

#[cfg(windows)]
mod win32 {
    use std::ffi::c_void;

    pub type HANDLE = *mut c_void;
    pub const INVALID_HANDLE_VALUE: HANDLE = -1isize as HANDLE;

    pub const GENERIC_READ:         u32 = 0x80000000;
    pub const FILE_SHARE_READ:      u32 = 0x00000001;
    pub const OPEN_EXISTING:        u32 = 3;
    pub const FILE_ATTRIBUTE_NORMAL: u32 = 0x80;
    pub const PAGE_READONLY:        u32 = 0x02;
    pub const FILE_MAP_READ:        u32 = 0x0004;

    extern "system" {
        pub fn CreateFileW(
            lpFileName:            *const u16,
            dwDesiredAccess:       u32,
            dwShareMode:           u32,
            lpSecurityAttributes:  *mut c_void,
            dwCreationDisposition: u32,
            dwFlagsAndAttributes:  u32,
            hTemplateFile:         HANDLE,
        ) -> HANDLE;

        pub fn GetFileSizeEx(hFile: HANDLE, lpFileSize: *mut i64) -> i32;

        pub fn CreateFileMappingW(
            hFile:                   HANDLE,
            lpFileMappingAttributes: *mut c_void,
            flProtect:               u32,
            dwMaximumSizeHigh:       u32,
            dwMaximumSizeLow:        u32,
            lpName:                  *const u16,
        ) -> HANDLE;

        pub fn MapViewOfFile(
            hFileMappingObject:    HANDLE,
            dwDesiredAccess:       u32,
            dwFileOffsetHigh:      u32,
            dwFileOffsetLow:       u32,
            dwNumberOfBytesToMap:  usize,
        ) -> *mut c_void;

        pub fn UnmapViewOfFile(lpBaseAddress: *const c_void) -> i32;
        pub fn CloseHandle(hObject: HANDLE) -> i32;
    }
}

#[cfg(windows)]
pub fn map_file_readonly(path: &Path) -> io::Result<MapView> {
    use std::os::windows::ffi::OsStrExt;

    let wide: Vec<u16> = path.as_os_str()
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();

    unsafe {
        let file = win32::CreateFileW(
            wide.as_ptr(),
            win32::GENERIC_READ,
            win32::FILE_SHARE_READ,
            std::ptr::null_mut(),
            win32::OPEN_EXISTING,
            win32::FILE_ATTRIBUTE_NORMAL,
            std::ptr::null_mut(),
        );
        if file == win32::INVALID_HANDLE_VALUE {
            return Err(io::Error::last_os_error());
        }

        let mut size_i64: i64 = 0;
        if win32::GetFileSizeEx(file, &mut size_i64) == 0 {
            let err = io::Error::last_os_error();
            win32::CloseHandle(file);
            return Err(err);
        }
        if size_i64 <= 0 {
            win32::CloseHandle(file);
            return Err(io::Error::new(io::ErrorKind::InvalidData, "empty file"));
        }
        let len = size_i64 as usize;

        let mapping = win32::CreateFileMappingW(
            file,
            std::ptr::null_mut(),
            win32::PAGE_READONLY,
            0, 0,                     // 0,0 → mapping size = file size
            std::ptr::null(),
        );
        if mapping.is_null() {
            let err = io::Error::last_os_error();
            win32::CloseHandle(file);
            return Err(err);
        }

        let view = win32::MapViewOfFile(mapping, win32::FILE_MAP_READ, 0, 0, 0);
        // Close the section + file handles now: the view stays valid until
        // UnmapViewOfFile, which Drop will call.
        win32::CloseHandle(mapping);
        win32::CloseHandle(file);
        if view.is_null() {
            return Err(io::Error::last_os_error());
        }

        Ok(MapView { ptr: view as *mut u8, len })
    }
}

#[cfg(windows)]
impl Drop for MapView {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.len > 0 {
            unsafe { win32::UnmapViewOfFile(self.ptr as *const std::ffi::c_void); }
        }
    }
}

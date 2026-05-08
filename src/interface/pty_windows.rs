//! Windows `PtyBackend` — Win32 ConPTY (Windows 10 1809+).
//!
//! Pipeline: `CreatePipe` (×2) → `CreatePseudoConsole` → `STARTUPINFOEX`
//! with `PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE` → `CreateProcessW`. The
//! attribute list is freed right after `CreateProcessW`; only the
//! HPCON, parent-side pipe handles, and child process handle live in
//! the backend struct.
//!
//! Non-blocking reads: ConPTY pipes are blocking by default, so we
//! `PeekNamedPipe` first and only `ReadFile` when bytes are available —
//! mirrors the Unix backend's `O_NONBLOCK` master fd.

use std::ffi::c_void;
use std::io;

use super::pty::PtyBackend;

#[allow(non_snake_case)]
mod win32 {
    use std::ffi::c_void;

    pub type HANDLE  = *mut c_void;
    pub type HPCON   = *mut c_void;
    pub type HRESULT = i32;

    pub const STILL_ACTIVE:                       u32   = 259;
    pub const EXTENDED_STARTUPINFO_PRESENT:       u32   = 0x00080000;
    pub const PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE: usize = 0x00020016;

    #[repr(C)]
    pub struct COORD { pub X: i16, pub Y: i16 }

    #[repr(C)]
    pub struct STARTUPINFOW {
        pub cb:              u32,
        pub lpReserved:      *mut u16,
        pub lpDesktop:       *mut u16,
        pub lpTitle:         *mut u16,
        pub dwX:             u32,
        pub dwY:             u32,
        pub dwXSize:         u32,
        pub dwYSize:         u32,
        pub dwXCountChars:   u32,
        pub dwYCountChars:   u32,
        pub dwFillAttribute: u32,
        pub dwFlags:         u32,
        pub wShowWindow:     u16,
        pub cbReserved2:     u16,
        pub lpReserved2:     *mut u8,
        pub hStdInput:       HANDLE,
        pub hStdOutput:      HANDLE,
        pub hStdError:       HANDLE,
    }

    #[repr(C)]
    pub struct STARTUPINFOEXW {
        pub StartupInfo:     STARTUPINFOW,
        pub lpAttributeList: *mut c_void,
    }

    #[repr(C)]
    pub struct PROCESS_INFORMATION {
        pub hProcess:    HANDLE,
        pub hThread:     HANDLE,
        pub dwProcessId: u32,
        pub dwThreadId:  u32,
    }

    extern "system" {
        pub fn CreatePipe(
            hReadPipe:        *mut HANDLE,
            hWritePipe:       *mut HANDLE,
            lpPipeAttributes: *mut c_void,
            nSize:            u32,
        ) -> i32;

        pub fn CloseHandle(hObject: HANDLE) -> i32;

        pub fn CreatePseudoConsole(
            size:    COORD,
            hInput:  HANDLE,
            hOutput: HANDLE,
            dwFlags: u32,
            phPC:    *mut HPCON,
        ) -> HRESULT;

        pub fn ResizePseudoConsole(hPC: HPCON, size: COORD) -> HRESULT;
        pub fn ClosePseudoConsole(hPC: HPCON);

        pub fn InitializeProcThreadAttributeList(
            lpAttributeList:  *mut c_void,
            dwAttributeCount: u32,
            dwFlags:          u32,
            lpSize:           *mut usize,
        ) -> i32;

        pub fn UpdateProcThreadAttribute(
            lpAttributeList: *mut c_void,
            dwFlags:         u32,
            Attribute:       usize,
            lpValue:         *mut c_void,
            cbSize:          usize,
            lpPreviousValue: *mut c_void,
            lpReturnSize:    *mut usize,
        ) -> i32;

        pub fn DeleteProcThreadAttributeList(lpAttributeList: *mut c_void);

        pub fn CreateProcessW(
            lpApplicationName:    *const u16,
            lpCommandLine:        *mut u16,
            lpProcessAttributes:  *mut c_void,
            lpThreadAttributes:   *mut c_void,
            bInheritHandles:      i32,
            dwCreationFlags:      u32,
            lpEnvironment:        *mut c_void,
            lpCurrentDirectory:   *const u16,
            lpStartupInfo:        *mut STARTUPINFOEXW,
            lpProcessInformation: *mut PROCESS_INFORMATION,
        ) -> i32;

        pub fn ReadFile(
            hFile:                HANDLE,
            lpBuffer:             *mut c_void,
            nNumberOfBytesToRead: u32,
            lpNumberOfBytesRead:  *mut u32,
            lpOverlapped:         *mut c_void,
        ) -> i32;

        pub fn WriteFile(
            hFile:                  HANDLE,
            lpBuffer:               *const c_void,
            nNumberOfBytesToWrite:  u32,
            lpNumberOfBytesWritten: *mut u32,
            lpOverlapped:           *mut c_void,
        ) -> i32;

        pub fn PeekNamedPipe(
            hNamedPipe:             HANDLE,
            lpBuffer:               *mut c_void,
            nBufferSize:            u32,
            lpBytesRead:            *mut u32,
            lpTotalBytesAvail:      *mut u32,
            lpBytesLeftThisMessage: *mut u32,
        ) -> i32;

        pub fn TerminateProcess(hProcess: HANDLE, uExitCode: u32) -> i32;
        pub fn GetExitCodeProcess(hProcess: HANDLE, lpExitCode: *mut u32) -> i32;
        pub fn Sleep(dwMilliseconds: u32);
    }
}

pub struct WindowsPtyBackend {
    hpcon:       win32::HPCON,
    stdin_write: win32::HANDLE,
    stdout_read: win32::HANDLE,
    process:     win32::HANDLE,
}

unsafe impl Send for WindowsPtyBackend {}
unsafe impl Sync for WindowsPtyBackend {}

pub fn open(cols: u16, rows: u16) -> io::Result<WindowsPtyBackend> {
    unsafe {
        let mut input_read:   win32::HANDLE = std::ptr::null_mut();
        let mut input_write:  win32::HANDLE = std::ptr::null_mut();
        let mut output_read:  win32::HANDLE = std::ptr::null_mut();
        let mut output_write: win32::HANDLE = std::ptr::null_mut();

        if win32::CreatePipe(&mut input_read, &mut input_write, std::ptr::null_mut(), 0) == 0 {
            return Err(io::Error::last_os_error());
        }
        if win32::CreatePipe(&mut output_read, &mut output_write, std::ptr::null_mut(), 0) == 0 {
            let err = io::Error::last_os_error();
            win32::CloseHandle(input_read);
            win32::CloseHandle(input_write);
            return Err(err);
        }

        let size = win32::COORD { X: cols as i16, Y: rows as i16 };
        let mut hpcon: win32::HPCON = std::ptr::null_mut();
        let hr = win32::CreatePseudoConsole(size, input_read, output_write, 0, &mut hpcon);
        // ConPTY duplicates the child-side handles internally; close ours.
        win32::CloseHandle(input_read);
        win32::CloseHandle(output_write);
        if hr < 0 {
            win32::CloseHandle(input_write);
            win32::CloseHandle(output_read);
            return Err(io::Error::from_raw_os_error(hr));
        }

        // ── STARTUPINFOEX attribute list ──────────────────────────────
        let mut attr_size: usize = 0;
        win32::InitializeProcThreadAttributeList(std::ptr::null_mut(), 1, 0, &mut attr_size);
        let mut attr_buf: Vec<u8> = vec![0u8; attr_size];
        let attr_list = attr_buf.as_mut_ptr() as *mut c_void;

        if win32::InitializeProcThreadAttributeList(attr_list, 1, 0, &mut attr_size) == 0 {
            let err = io::Error::last_os_error();
            win32::ClosePseudoConsole(hpcon);
            win32::CloseHandle(input_write);
            win32::CloseHandle(output_read);
            return Err(err);
        }

        if win32::UpdateProcThreadAttribute(
            attr_list,
            0,
            win32::PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE,
            &mut hpcon as *mut _ as *mut c_void,
            std::mem::size_of::<win32::HPCON>(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ) == 0 {
            let err = io::Error::last_os_error();
            win32::DeleteProcThreadAttributeList(attr_list);
            win32::ClosePseudoConsole(hpcon);
            win32::CloseHandle(input_write);
            win32::CloseHandle(output_read);
            return Err(err);
        }

        // ── Spawn cmd.exe under the pseudoconsole ─────────────────────
        let mut si: win32::STARTUPINFOEXW = std::mem::zeroed();
        si.StartupInfo.cb = std::mem::size_of::<win32::STARTUPINFOEXW>() as u32;
        si.lpAttributeList = attr_list;

        let mut cmdline: Vec<u16> = "cmd.exe"
            .encode_utf16()
            .chain(std::iter::once(0))
            .collect();

        let mut pi: win32::PROCESS_INFORMATION = std::mem::zeroed();
        let ok = win32::CreateProcessW(
            std::ptr::null(),
            cmdline.as_mut_ptr(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            win32::EXTENDED_STARTUPINFO_PRESENT,
            std::ptr::null_mut(),
            std::ptr::null(),
            &mut si,
            &mut pi,
        );

        // Attribute list is dormant after CreateProcessW; free it now so
        // we don't have to track its lifetime in the backend struct.
        win32::DeleteProcThreadAttributeList(attr_list);
        drop(attr_buf);

        if ok == 0 {
            let err = io::Error::last_os_error();
            win32::ClosePseudoConsole(hpcon);
            win32::CloseHandle(input_write);
            win32::CloseHandle(output_read);
            return Err(err);
        }

        win32::CloseHandle(pi.hThread);

        Ok(WindowsPtyBackend {
            hpcon,
            stdin_write: input_write,
            stdout_read: output_read,
            process:     pi.hProcess,
        })
    }
}

impl PtyBackend for WindowsPtyBackend {
    fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        unsafe {
            let mut avail: u32 = 0;
            let peeked = win32::PeekNamedPipe(
                self.stdout_read,
                std::ptr::null_mut(), 0, std::ptr::null_mut(),
                &mut avail, std::ptr::null_mut(),
            );
            if peeked == 0 || avail == 0 {
                return Ok(0);
            }
            let want = (buf.len() as u32).min(avail);
            let mut read: u32 = 0;
            if win32::ReadFile(
                self.stdout_read,
                buf.as_mut_ptr() as *mut c_void,
                want,
                &mut read,
                std::ptr::null_mut(),
            ) == 0 {
                return Ok(0);
            }
            Ok(read as usize)
        }
    }

    fn write(&self, data: &[u8]) {
        let mut written = 0usize;
        while written < data.len() {
            let mut n: u32 = 0;
            let ok = unsafe {
                win32::WriteFile(
                    self.stdin_write,
                    data[written..].as_ptr() as *const c_void,
                    (data.len() - written) as u32,
                    &mut n,
                    std::ptr::null_mut(),
                )
            };
            if ok == 0 || n == 0 { break; }
            written += n as usize;
        }
    }

    fn resize(&self, cols: u16, rows: u16) {
        let size = win32::COORD { X: cols as i16, Y: rows as i16 };
        unsafe { win32::ResizePseudoConsole(self.hpcon, size); }
    }

    fn child_alive(&self) -> bool {
        unsafe {
            let mut code: u32 = 0;
            if win32::GetExitCodeProcess(self.process, &mut code) == 0 {
                return false;
            }
            code == win32::STILL_ACTIVE
        }
    }

    fn wait_readable(&self, timeout_ms: i32) -> bool {
        let infinite = timeout_ms < 0;
        let mut elapsed: i32 = 0;
        loop {
            let mut avail: u32 = 0;
            let ok = unsafe {
                win32::PeekNamedPipe(
                    self.stdout_read,
                    std::ptr::null_mut(), 0, std::ptr::null_mut(),
                    &mut avail, std::ptr::null_mut(),
                )
            };
            if ok != 0 && avail > 0 { return true; }
            if !infinite && elapsed >= timeout_ms { return false; }
            unsafe { win32::Sleep(10); }
            elapsed = elapsed.saturating_add(10);
        }
    }
}

impl Drop for WindowsPtyBackend {
    fn drop(&mut self) {
        unsafe {
            win32::TerminateProcess(self.process, 1);
            win32::CloseHandle(self.process);
            win32::ClosePseudoConsole(self.hpcon);
            win32::CloseHandle(self.stdin_write);
            win32::CloseHandle(self.stdout_read);
        }
    }
}

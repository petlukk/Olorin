//! Windows `PtyBackend` — Win32 ConPTY (Windows 10 1809+).
//!
//! Pipeline: `CreatePipe` (×2) → `CreatePseudoConsole` → `STARTUPINFOEX`
//! with `PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE` → `CreateProcessW`.
//!
//! Reads run on a dedicated thread that does blocking `ReadFile` and
//! pushes bytes into a `VecDeque` guarded by a `Condvar`. Polling the
//! pipe with `PeekNamedPipe` from the consumer thread doesn't survive
//! ConPTY's buffering behavior — the canonical pattern (Windows
//! Terminal, conpty-rs, MS ConPTY samples) is reader-thread + queue.

use std::collections::VecDeque;
use std::ffi::c_void;
use std::io;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

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

        pub fn TerminateProcess(hProcess: HANDLE, uExitCode: u32) -> i32;
        pub fn GetExitCodeProcess(hProcess: HANDLE, lpExitCode: *mut u32) -> i32;
        pub fn SetHandleInformation(hObject: HANDLE, dwMask: u32, dwFlags: u32) -> i32;
        pub fn GetStdHandle(nStdHandle: u32) -> HANDLE;
        pub fn SetStdHandle(nStdHandle: u32, hHandle: HANDLE) -> i32;
    }

    pub const HANDLE_FLAG_INHERIT: u32 = 0x00000001;
    pub const STD_OUTPUT_HANDLE:   u32 = 0xFFFFFFF5;
    pub const STD_ERROR_HANDLE:    u32 = 0xFFFFFFF4;
}

// ── Inbox: bytes from the reader thread, queued for the consumer ──────────────

struct Inbox {
    state: Mutex<InboxState>,
    cvar:  Condvar,
}

struct InboxState {
    buf:    VecDeque<u8>,
    closed: bool,
}

impl Inbox {
    fn new() -> Self {
        Self {
            state: Mutex::new(InboxState { buf: VecDeque::new(), closed: false }),
            cvar:  Condvar::new(),
        }
    }
    fn push(&self, data: &[u8]) {
        let mut s = self.state.lock().unwrap();
        s.buf.extend(data.iter().copied());
        self.cvar.notify_all();
    }
    fn close(&self) {
        let mut s = self.state.lock().unwrap();
        s.closed = true;
        self.cvar.notify_all();
    }
    fn pop(&self, dst: &mut [u8]) -> usize {
        let mut s = self.state.lock().unwrap();
        let n = dst.len().min(s.buf.len());
        for slot in dst.iter_mut().take(n) {
            *slot = s.buf.pop_front().unwrap();
        }
        n
    }
    /// Returns true if data is available (or the pipe closed). Negative
    /// timeout = block forever.
    fn wait(&self, timeout_ms: i32) -> bool {
        let s = self.state.lock().unwrap();
        if !s.buf.is_empty() || s.closed { return !s.buf.is_empty() || s.closed; }
        if timeout_ms == 0 { return false; }
        if timeout_ms < 0 {
            let s = self.cvar.wait_while(s, |st| st.buf.is_empty() && !st.closed).unwrap();
            !s.buf.is_empty() || s.closed
        } else {
            let deadline = Instant::now() + Duration::from_millis(timeout_ms as u64);
            let mut s = s;
            loop {
                let now = Instant::now();
                if now >= deadline { return !s.buf.is_empty() || s.closed; }
                let (ns, _) = self.cvar.wait_timeout(s, deadline - now).unwrap();
                s = ns;
                if !s.buf.is_empty() || s.closed {
                    return !s.buf.is_empty() || s.closed;
                }
            }
        }
    }
}

// ── Backend ───────────────────────────────────────────────────────────────────

pub struct WindowsPtyBackend {
    hpcon:       win32::HPCON,
    stdin_write: win32::HANDLE,
    process:     win32::HANDLE,
    inbox:       Arc<Inbox>,
    reader:      Mutex<Option<JoinHandle<()>>>,
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
        if hr < 0 {
            win32::CloseHandle(input_read);
            win32::CloseHandle(output_write);
            win32::CloseHandle(input_write);
            win32::CloseHandle(output_read);
            return Err(io::Error::from_raw_os_error(hr));
        }
        // output_write must stay open until after CreateProcessW so we can
        // pass it as the child's stdout/stderr via STARTF_USESTDHANDLES.
        // It needs HANDLE_FLAG_INHERIT for bInheritHandles=TRUE to copy it.
        // input_read stays NOT inheritable — we don't want the child to
        // get its own copy and race with ConPTY for the input bytes.
        win32::SetHandleInformation(output_write, win32::HANDLE_FLAG_INHERIT, win32::HANDLE_FLAG_INHERIT);

        // STARTUPINFOEX attribute list
        let mut attr_size: usize = 0;
        win32::InitializeProcThreadAttributeList(std::ptr::null_mut(), 1, 0, &mut attr_size);
        let mut attr_buf: Vec<u8> = vec![0u8; attr_size];
        let attr_list = attr_buf.as_mut_ptr() as *mut c_void;

        if win32::InitializeProcThreadAttributeList(attr_list, 1, 0, &mut attr_size) == 0 {
            let err = io::Error::last_os_error();
            win32::ClosePseudoConsole(hpcon);
            win32::CloseHandle(input_read);
            win32::CloseHandle(output_write);
            win32::CloseHandle(input_write);
            win32::CloseHandle(output_read);
            return Err(err);
        }

        // PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE is value-as-pointer: pass
        // the HPCON itself, not its address. Tutorials and conpty-rs
        // confirm this even though UpdateProcThreadAttribute's docs read
        // generically as "pointer to data". Address-of caused cmd.exe to
        // fall back to its own console and leave our pipes empty.
        if win32::UpdateProcThreadAttribute(
            attr_list,
            0,
            win32::PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE,
            hpcon as *mut c_void,
            std::mem::size_of::<win32::HPCON>(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        ) == 0 {
            let err = io::Error::last_os_error();
            win32::DeleteProcThreadAttributeList(attr_list);
            win32::ClosePseudoConsole(hpcon);
            win32::CloseHandle(input_read);
            win32::CloseHandle(output_write);
            win32::CloseHandle(input_write);
            win32::CloseHandle(output_read);
            return Err(err);
        }

        // Spawn cmd.exe under the pseudoconsole. The trick for getting both
        // output redirection AND working input:
        //   - Don't set STARTF_USESTDHANDLES — it overrides the pseudoconsole's
        //     input wiring even when we pass INVALID_HANDLE_VALUE for stdin.
        //   - Instead, temporarily SetStdHandle our own STDOUT/STDERR to the
        //     ConPTY output pipe. The kernel's default stdio propagation copies
        //     these to the child via standard inheritance.
        //   - PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE provides the input wiring
        //     (cmd.exe reads via ReadConsoleW from the pseudoconsole).
        let prev_stdout = win32::GetStdHandle(win32::STD_OUTPUT_HANDLE);
        let prev_stderr = win32::GetStdHandle(win32::STD_ERROR_HANDLE);
        win32::SetStdHandle(win32::STD_OUTPUT_HANDLE, output_write);
        win32::SetStdHandle(win32::STD_ERROR_HANDLE,  output_write);

        let mut si: win32::STARTUPINFOEXW = std::mem::zeroed();
        si.StartupInfo.cb  = std::mem::size_of::<win32::STARTUPINFOEXW>() as u32;
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
            1,                                   // bInheritHandles=TRUE
            win32::EXTENDED_STARTUPINFO_PRESENT,
            std::ptr::null_mut(),
            std::ptr::null(),
            &mut si,
            &mut pi,
        );

        // Restore our stdio immediately — only the spawn needed it redirected.
        win32::SetStdHandle(win32::STD_OUTPUT_HANDLE, prev_stdout);
        win32::SetStdHandle(win32::STD_ERROR_HANDLE,  prev_stderr);

        win32::DeleteProcThreadAttributeList(attr_list);
        drop(attr_buf);

        // ConPTY duplicated input_read/output_write internally and the
        // child also inherited duplicates. Close our copies now.
        win32::CloseHandle(input_read);
        win32::CloseHandle(output_write);

        if ok == 0 {
            let err = io::Error::last_os_error();
            win32::ClosePseudoConsole(hpcon);
            win32::CloseHandle(input_write);
            win32::CloseHandle(output_read);
            return Err(err);
        }

        win32::CloseHandle(pi.hThread);

        // ── Reader thread: blocking ReadFile on output_read into Inbox ────
        let inbox = Arc::new(Inbox::new());
        let inbox_for_reader = inbox.clone();
        // HANDLE is pointer-sized; pass through as usize so the closure
        // captures a Send type instead of *mut c_void.
        let stdout_addr = output_read as usize;

        let reader_handle = thread::Builder::new()
            .name("conpty-reader".to_string())
            .spawn(move || {
                let h = stdout_addr as win32::HANDLE;
                let mut buf = [0u8; 4096];
                loop {
                    let mut n: u32 = 0;
                    let ok = win32::ReadFile(
                        h, buf.as_mut_ptr() as *mut c_void,
                        buf.len() as u32, &mut n, std::ptr::null_mut(),
                    );
                    if ok == 0 || n == 0 { break; }
                    inbox_for_reader.push(&buf[..n as usize]);
                }
                inbox_for_reader.close();
                win32::CloseHandle(h);
            })
            .map_err(|e| {
                let kind_err = io::Error::other(format!("conpty reader thread spawn: {e}"));
                win32::ClosePseudoConsole(hpcon);
                win32::CloseHandle(input_write);
                win32::CloseHandle(output_read);
                win32::TerminateProcess(pi.hProcess, 1);
                win32::CloseHandle(pi.hProcess);
                kind_err
            })?;

        Ok(WindowsPtyBackend {
            hpcon,
            stdin_write: input_write,
            process:     pi.hProcess,
            inbox,
            reader:      Mutex::new(Some(reader_handle)),
        })
    }
}

impl PtyBackend for WindowsPtyBackend {
    fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        Ok(self.inbox.pop(buf))
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
        self.inbox.wait(timeout_ms)
    }
}

impl Drop for WindowsPtyBackend {
    fn drop(&mut self) {
        unsafe {
            // Order matters: stop the child + console first so the
            // reader thread's ReadFile returns EOF and exits cleanly.
            win32::CloseHandle(self.stdin_write);
            win32::ClosePseudoConsole(self.hpcon);
            win32::TerminateProcess(self.process, 1);
            win32::CloseHandle(self.process);
        }
        // Reader thread closes stdout_read on exit. Don't join — if the
        // pipe is still draining we'd hang. The inbox.close() it emits
        // will wake any outstanding wait_readable callers.
        if let Some(h) = self.reader.lock().unwrap().take() {
            std::mem::drop(h);
        }
    }
}

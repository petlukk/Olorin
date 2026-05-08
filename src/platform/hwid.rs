//! Per-machine identifier for vault-key derivation.
//!
//! Linux:   /sys/class/dmi/id/product_uuid → /etc/machine-id → /etc/hostname.
//! Windows: HKLM\SOFTWARE\Microsoft\Cryptography\MachineGuid — a per-install
//!          UUID generated at OS install time, the standard cross-platform
//!          analog to /etc/machine-id.

/// Returns a stable per-machine identifier, or None if no source is
/// available. Caller falls back to a universal constant which trades
/// vault-portability for security degradation.
pub fn machine_id() -> Option<String> { imp::machine_id() }

#[cfg(unix)]
mod imp {
    pub fn machine_id() -> Option<String> {
        std::fs::read_to_string("/sys/class/dmi/id/product_uuid")
            .or_else(|_| std::fs::read_to_string("/etc/machine-id"))
            .or_else(|_| std::fs::read_to_string("/etc/hostname"))
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    }
}

#[cfg(windows)]
mod imp {
    use std::ffi::c_void;

    type HKEY = *mut c_void;
    const HKEY_LOCAL_MACHINE: HKEY = 0x80000002 as usize as HKEY;
    const KEY_READ:           u32  = 0x20019;
    const REG_SZ:             u32  = 1;
    const ERROR_SUCCESS:      i32  = 0;

    // Registry APIs live in advapi32, not kernel32 — mingw won't pull
    // them in unless we ask explicitly.
    #[link(name = "advapi32")]
    extern "system" {
        fn RegOpenKeyExW(
            hKey:       HKEY,
            lpSubKey:   *const u16,
            ulOptions:  u32,
            samDesired: u32,
            phkResult:  *mut HKEY,
        ) -> i32;

        fn RegQueryValueExW(
            hKey:        HKEY,
            lpValueName: *const u16,
            lpReserved:  *mut u32,
            lpType:      *mut u32,
            lpData:      *mut u8,
            lpcbData:    *mut u32,
        ) -> i32;

        fn RegCloseKey(hKey: HKEY) -> i32;
    }

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    pub fn machine_id() -> Option<String> {
        let subkey = wide("SOFTWARE\\Microsoft\\Cryptography");
        let value  = wide("MachineGuid");

        unsafe {
            let mut hkey: HKEY = std::ptr::null_mut();
            if RegOpenKeyExW(HKEY_LOCAL_MACHINE, subkey.as_ptr(), 0, KEY_READ, &mut hkey)
                != ERROR_SUCCESS
            {
                return None;
            }

            // First call: query size. lpData=null → returns required bytes.
            let mut size: u32 = 0;
            let mut ty:   u32 = 0;
            if RegQueryValueExW(
                hkey, value.as_ptr(), std::ptr::null_mut(),
                &mut ty, std::ptr::null_mut(), &mut size,
            ) != ERROR_SUCCESS || ty != REG_SZ || size == 0 {
                RegCloseKey(hkey);
                return None;
            }

            // size is in bytes; UTF-16 → halve for u16 count.
            let mut buf: Vec<u16> = vec![0u16; (size as usize) / 2];
            let mut buf_size = size;
            let rc = RegQueryValueExW(
                hkey, value.as_ptr(), std::ptr::null_mut(),
                std::ptr::null_mut(),
                buf.as_mut_ptr() as *mut u8,
                &mut buf_size,
            );
            RegCloseKey(hkey);
            if rc != ERROR_SUCCESS { return None; }

            // Strip trailing NUL(s) that REG_SZ values include.
            while buf.last() == Some(&0) { buf.pop(); }
            String::from_utf16(&buf).ok().filter(|s| !s.is_empty())
        }
    }
}

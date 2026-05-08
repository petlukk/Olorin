//! System telemetry (memory, uptime, CPU%, optional CPU temp).
//!
//! Linux:   /proc/meminfo, /proc/uptime, /proc/stat,
//!          /sys/class/thermal/thermal_zone0/temp.
//! Windows: GlobalMemoryStatusEx, GetTickCount64, GetSystemTimes.
//!          CPU temp has no standard Win32 API → returns None.

/// Returns (used_mb, total_mb) of physical memory.
pub fn memory_usage_mb() -> Option<(u64, u64)> { imp::memory_usage_mb() }

/// Returns CPU temperature in degrees C. None if unavailable.
pub fn cpu_temp_c() -> Option<u32> { imp::cpu_temp_c() }

/// Returns the percentage of CPU time spent non-idle since the
/// previous call. The first call returns 0 (no baseline yet).
pub fn cpu_percent() -> Option<u32> { imp::cpu_percent() }

/// Returns seconds since system boot.
pub fn uptime_seconds() -> Option<u64> { imp::uptime_seconds() }

/// CPU brand string (e.g. "AMD Ryzen 9 7950X 16-Core Processor").
/// Read via cpuid leaves 0x80000002/3/4 on x86_64.
pub fn cpu_model() -> Option<String> {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use std::arch::x86_64::__cpuid;
        if __cpuid(0x80000000).eax < 0x80000004 { return None; }
        let mut bytes = [0u8; 48];
        for (i, leaf) in [0x80000002u32, 0x80000003, 0x80000004].iter().enumerate() {
            let r = __cpuid(*leaf);
            let off = i * 16;
            bytes[off..off + 4].copy_from_slice(&r.eax.to_le_bytes());
            bytes[off + 4..off + 8].copy_from_slice(&r.ebx.to_le_bytes());
            bytes[off + 8..off + 12].copy_from_slice(&r.ecx.to_le_bytes());
            bytes[off + 12..off + 16].copy_from_slice(&r.edx.to_le_bytes());
        }
        let s = std::str::from_utf8(&bytes).ok()?
            .trim_end_matches('\0').trim().to_string();
        if s.is_empty() { None } else { Some(s) }
    }
    #[cfg(all(not(target_arch = "x86_64"), target_os = "linux"))]
    {
        let s = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        for line in s.lines() {
            if line.starts_with("model name") {
                return line.split(':').nth(1).map(|v| v.trim().to_string());
            }
        }
        None
    }
    #[cfg(all(not(target_arch = "x86_64"), not(target_os = "linux")))]
    { None }
}

/// Logical CPU count. Falls back to 0 if unavailable (very rare).
pub fn cpu_cores() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0)
}

/// Load averages over 1m / 5m / 15m. Linux only — Windows has no
/// native equivalent (NtQuerySystemInformation gives instantaneous
/// data, not a moving average).
pub fn load_average() -> Option<(f64, f64, f64)> {
    #[cfg(target_os = "linux")]
    {
        let s = std::fs::read_to_string("/proc/loadavg").ok()?;
        let parts: Vec<f64> = s.split_whitespace().take(3)
            .filter_map(|v| v.parse().ok()).collect();
        if parts.len() == 3 {
            return Some((parts[0], parts[1], parts[2]));
        }
    }
    None
}

#[cfg(unix)]
mod imp {
    pub fn memory_usage_mb() -> Option<(u64, u64)> {
        let s = std::fs::read_to_string("/proc/meminfo").ok()?;
        let mut total_kb = 0u64;
        let mut avail_kb = 0u64;
        for line in s.lines() {
            if line.starts_with("MemTotal:") {
                total_kb = line.split_whitespace().nth(1)?.parse().ok()?;
            } else if line.starts_with("MemAvailable:") {
                avail_kb = line.split_whitespace().nth(1)?.parse().ok()?;
            }
        }
        Some((total_kb / 1024 - avail_kb / 1024, total_kb / 1024))
    }

    pub fn cpu_temp_c() -> Option<u32> {
        let s = std::fs::read_to_string("/sys/class/thermal/thermal_zone0/temp").ok()?;
        Some(s.trim().parse::<u32>().ok()? / 1000)
    }

    fn parse_proc_stat() -> Option<(u64, u64)> {
        let s = std::fs::read_to_string("/proc/stat").ok()?;
        let line = s.lines().find(|l| l.starts_with("cpu "))?;
        let vals: Vec<u64> = line.split_whitespace().skip(1)
            .filter_map(|v| v.parse().ok()).collect();
        if vals.len() < 4 { return None; }
        let total: u64 = vals.iter().sum();
        let idle = vals[3];
        Some((total, idle))
    }

    pub fn cpu_percent() -> Option<u32> {
        use std::sync::Mutex;
        static PREV: Mutex<(u64, u64, u32)> = Mutex::new((0, 0, 0));
        let (t2, i2) = parse_proc_stat()?;
        let mut prev = PREV.lock().ok()?;
        let (t1, i1, last_pct) = *prev;
        let dt = t2.saturating_sub(t1);
        let di = i2.saturating_sub(i1);
        *prev = if dt > 0 {
            (t2, i2, (100 * (dt - di) / dt) as u32)
        } else {
            (t2, i2, last_pct)
        };
        Some(prev.2)
    }

    pub fn uptime_seconds() -> Option<u64> {
        let s = std::fs::read_to_string("/proc/uptime").ok()?;
        Some(s.split_whitespace().next()?.parse::<f64>().ok()? as u64)
    }
}

#[cfg(windows)]
mod imp {
    #[repr(C)]
    #[allow(non_snake_case)]
    struct MEMORYSTATUSEX {
        dwLength:                u32,
        dwMemoryLoad:            u32,
        ullTotalPhys:            u64,
        ullAvailPhys:            u64,
        ullTotalPageFile:        u64,
        ullAvailPageFile:        u64,
        ullTotalVirtual:         u64,
        ullAvailVirtual:         u64,
        ullAvailExtendedVirtual: u64,
    }

    #[repr(C)]
    struct FILETIME { low: u32, high: u32 }

    impl FILETIME {
        fn as_u64(&self) -> u64 { ((self.high as u64) << 32) | (self.low as u64) }
    }

    extern "system" {
        fn GlobalMemoryStatusEx(lpBuffer: *mut MEMORYSTATUSEX) -> i32;
        fn GetTickCount64() -> u64;
        fn GetSystemTimes(
            lpIdleTime:   *mut FILETIME,
            lpKernelTime: *mut FILETIME,
            lpUserTime:   *mut FILETIME,
        ) -> i32;
    }

    pub fn memory_usage_mb() -> Option<(u64, u64)> {
        unsafe {
            let mut s: MEMORYSTATUSEX = std::mem::zeroed();
            s.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
            if GlobalMemoryStatusEx(&mut s) == 0 { return None; }
            let total = s.ullTotalPhys / (1024 * 1024);
            let used  = (s.ullTotalPhys - s.ullAvailPhys) / (1024 * 1024);
            Some((used, total))
        }
    }

    pub fn cpu_temp_c() -> Option<u32> {
        // No standard Win32 API for CPU temperature. Available via WMI
        // or vendor drivers (Open Hardware Monitor); not worth the
        // dependency surface yet.
        None
    }

    pub fn cpu_percent() -> Option<u32> {
        // Linux's parse_proc_stat returns (total_ticks, idle_ticks).
        // GetSystemTimes gives kernel + user + idle in 100ns units; we
        // build the same shape so the diff math matches the Unix path.
        unsafe {
            let mut idle:   FILETIME = std::mem::zeroed();
            let mut kernel: FILETIME = std::mem::zeroed();
            let mut user:   FILETIME = std::mem::zeroed();
            if GetSystemTimes(&mut idle, &mut kernel, &mut user) == 0 {
                return None;
            }
            // kernel includes idle, so total = kernel + user.
            let total = kernel.as_u64() + user.as_u64();
            let idle  = idle.as_u64();

            use std::sync::Mutex;
            static PREV: Mutex<(u64, u64, u32)> = Mutex::new((0, 0, 0));
            let mut prev = PREV.lock().ok()?;
            let (t1, i1, last_pct) = *prev;
            let dt = total.saturating_sub(t1);
            let di = idle.saturating_sub(i1);
            *prev = if dt > 0 {
                (total, idle, (100 * (dt - di) / dt) as u32)
            } else {
                (total, idle, last_pct)
            };
            Some(prev.2)
        }
    }

    pub fn uptime_seconds() -> Option<u64> {
        Some(unsafe { GetTickCount64() } / 1000)
    }
}

use super::ToolResult;

pub fn run(_args: &str) -> ToolResult {
    #[cfg(unix)]
    let result = {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let secs = now.as_secs() as i64;
        let mut tm: libc::tm = unsafe { std::mem::zeroed() };
        unsafe { libc::localtime_r(&secs, &mut tm) };
        format!(
            "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
            tm.tm_year + 1900,
            tm.tm_mon + 1,
            tm.tm_mday,
            tm.tm_hour,
            tm.tm_min,
            tm.tm_sec,
        )
    };

    #[cfg(windows)]
    let result = {
        #[repr(C)]
        struct SystemTime {
            year:         u16,
            month:        u16,
            day_of_week:  u16,
            day:          u16,
            hour:         u16,
            minute:       u16,
            second:       u16,
            milliseconds: u16,
        }
        extern "system" {
            fn GetLocalTime(lp_system_time: *mut SystemTime);
        }
        let mut st = SystemTime {
            year: 0, month: 0, day_of_week: 0, day: 0,
            hour: 0, minute: 0, second: 0, milliseconds: 0,
        };
        unsafe { GetLocalTime(&mut st); }
        format!(
            "{:04}-{:02}-{:02} {:02}:{:02}:{:02}",
            st.year, st.month, st.day, st.hour, st.minute, st.second,
        )
    };

    ToolResult { output: result, success: true }
}

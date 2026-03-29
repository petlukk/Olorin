use super::ToolResult;
use std::time::Duration;

pub fn run(args: &str) -> ToolResult {
    // Format: "<duration> <message>"
    let args = args.trim();
    let (time_str, message) = match args.find(' ') {
        Some(pos) => (&args[..pos], args[pos + 1..].trim()),
        None => return ToolResult {
            output: "usage: remind <duration> <message>  (e.g. remind 5m Check the oven)".to_string(),
            success: false,
        },
    };

    if message.is_empty() {
        return ToolResult { output: "usage: remind <duration> <message>".to_string(), success: false };
    }

    let duration = match parse_duration(time_str) {
        Ok(d) => d,
        Err(e) => return ToolResult { output: e, success: false },
    };

    if duration.as_secs() > 24 * 60 * 60 {
        return ToolResult { output: "maximum reminder duration is 24 hours".to_string(), success: false };
    }

    // Synchronous sleep — blocks the thread.
    // For long reminders this is intentionally blocking; callers should run on a thread.
    std::thread::sleep(duration);

    ToolResult { output: format!("Reminder: {message}"), success: true }
}

fn parse_duration(s: &str) -> Result<Duration, String> {
    let s = s.trim().to_lowercase();
    if s.is_empty() {
        return Err("empty duration".to_string());
    }

    let mut total_secs: u64 = 0;
    let mut num_buf = String::new();

    for c in s.chars() {
        if c.is_ascii_digit() {
            num_buf.push(c);
        } else {
            if num_buf.is_empty() {
                return Err(format!("invalid duration: '{s}'. Use e.g. '30s', '5m', '2h', '1h30m'"));
            }
            let n: u64 = num_buf.parse()
                .map_err(|_| format!("invalid number in duration: '{s}'"))?;
            num_buf.clear();
            match c {
                's' => total_secs += n,
                'm' => total_secs += n * 60,
                'h' => total_secs += n * 3600,
                _ => return Err(format!("unknown unit '{c}' in duration. Use s/m/h.")),
            }
        }
    }

    // Bare number defaults to minutes
    if !num_buf.is_empty() {
        let n: u64 = num_buf.parse()
            .map_err(|_| format!("invalid duration: '{s}'"))?;
        total_secs += n * 60;
    }

    if total_secs == 0 {
        return Err("duration must be > 0".to_string());
    }

    Ok(Duration::from_secs(total_secs))
}

//! Tool parameter building and invocation.
//!
//! Builds tool name + params from SIMD-classified command IDs and intent IDs.
//! Two-stage matching: SIMD hash → full name verification.
//! Synchronous — no async.

use crate::error::{Error, Result};
use crate::kernels::ffi;

// ── Command ID constants ─────────────────────────────────────────────────────
// These match the values produced by the match_command SIMD kernel exactly.

pub const CMD_NONE:      i32 = -1;
// Meta commands
pub const CMD_HELP:      i32 = 0;
pub const CMD_QUIT:      i32 = 1;
pub const CMD_TOOLS:     i32 = 2;
pub const CMD_CLEAR:     i32 = 3;
pub const CMD_MODEL:     i32 = 4;
pub const CMD_PROFILE:   i32 = 5;
pub const CMD_TASKS:     i32 = 18;
pub const CMD_RECALL:    i32 = 19;
// Tool commands
pub const CMD_TIME:      i32 = 6;
pub const CMD_CALC:      i32 = 7;
pub const CMD_HTTP:      i32 = 8;
pub const CMD_SHELL:     i32 = 9;
pub const CMD_MEMORY:    i32 = 10;
pub const CMD_READ:      i32 = 11;
pub const CMD_WRITE:     i32 = 12;
pub const CMD_LS:        i32 = 13;
pub const CMD_JSON:      i32 = 14;
pub const CMD_CPU:       i32 = 15;
pub const CMD_TOKENS:    i32 = 16;
pub const CMD_BENCH:     i32 = 17;
pub const CMD_WEATHER:   i32 = 20;
pub const CMD_TRANSLATE: i32 = 21;
pub const CMD_DEFINE:    i32 = 22;
pub const CMD_SUMMARIZE: i32 = 23;
pub const CMD_GREP:      i32 = 24;
pub const CMD_GIT:       i32 = 25;
pub const CMD_REMIND:    i32 = 26;
pub const CMD_TELEPORT:  i32 = 27;
// Range
pub const CMD_TOOL_FIRST: i32 = CMD_TIME;
pub const CMD_TOOL_LAST:  i32 = CMD_REMIND;

// Intent constants from classify_intent kernel
pub const INTENT_NONE:    i32 = 0;
pub const INTENT_CALC:    i32 = 1;
pub const INTENT_TIME:    i32 = 2;
pub const INTENT_CPU:     i32 = 3;
pub const INTENT_WEATHER: i32 = 4;

/// Full command names for two-stage verification.
const ALL_CMD_NAMES: &[(i32, &str)] = &[
    (CMD_HELP, "help"),     (CMD_QUIT, "quit"),       (CMD_TOOLS, "tools"),
    (CMD_CLEAR, "clear"),   (CMD_MODEL, "model"),     (CMD_PROFILE, "profile"),
    (CMD_TIME, "time"),     (CMD_CALC, "calc"),       (CMD_HTTP, "http"),
    (CMD_SHELL, "shell"),   (CMD_MEMORY, "memory"),   (CMD_READ, "read"),
    (CMD_WRITE, "write"),   (CMD_LS, "ls"),           (CMD_JSON, "json"),
    (CMD_CPU, "cpu"),       (CMD_TOKENS, "tokens"),   (CMD_BENCH, "bench"),
    (CMD_TASKS, "tasks"),   (CMD_RECALL, "recall"),   (CMD_WEATHER, "weather"),
    (CMD_TRANSLATE, "translate"), (CMD_DEFINE, "define"),
    (CMD_SUMMARIZE, "summarize"), (CMD_GREP, "grep"),
    (CMD_GIT, "git"),       (CMD_REMIND, "remind"),   (CMD_TELEPORT, "teleport"),
];

// ── Slash command matching ───────────────────────────────────────────────────

/// Two-stage match: SIMD hash + full name verification.
/// Returns (command_id, argument_bytes).
pub fn match_command(input: &[u8]) -> (i32, &[u8]) {
    if input.is_empty() || input[0] != b'/' {
        return (CMD_NONE, &[]);
    }

    // Stage 1: SIMD hash lookup
    let mut cmd_id: i32 = CMD_NONE;
    unsafe {
        ffi::match_command(input.as_ptr(), input.len() as i32, &mut cmd_id);
    }
    if cmd_id == CMD_NONE {
        return (CMD_NONE, &[]);
    }

    // Stage 2: full name verification
    for &(id, name) in ALL_CMD_NAMES {
        if id != cmd_id { continue; }
        let expected_len = 1 + name.len(); // "/" + name
        if input.len() < expected_len { return (CMD_NONE, &[]); }
        if &input[1..expected_len] != name.as_bytes() { return (CMD_NONE, &[]); }
        // Must be exact or followed by space
        if input.len() == expected_len {
            return (cmd_id, &[]);
        }
        if input[expected_len] == b' ' {
            let arg_start = expected_len + 1;
            let arg = if arg_start < input.len() { &input[arg_start..] } else { &[] };
            return (cmd_id, arg);
        }
        // Not a match (e.g., "/timer" hash-matched "/time" but isn't "/time")
        return (CMD_NONE, &[]);
    }
    (CMD_NONE, &[])
}

/// Get command name from ID.
pub fn command_name(cmd_id: i32) -> &'static str {
    for &(id, name) in ALL_CMD_NAMES {
        if id == cmd_id { return name; }
    }
    "unknown"
}

// ── Intent classification ────────────────────────────────────────────────────

/// Classify user intent using the SIMD kernel.
/// Returns (intent_id, argument_start, argument_len).
pub fn classify_intent(input: &[u8]) -> (i32, usize, usize) {
    if input.is_empty() {
        return (INTENT_NONE, 0, 0);
    }
    let mut intent: i32 = 0;
    let mut arg_start: i32 = 0;
    let mut arg_len: i32 = 0;
    unsafe {
        ffi::classify_intent(
            input.as_ptr(), input.len() as i32,
            &mut intent, &mut arg_start, &mut arg_len,
        );
    }
    (intent, arg_start as usize, arg_len as usize)
}

/// Map an intent to the tool name that handles it.
pub fn intent_to_tool_name(intent: i32) -> Option<&'static str> {
    match intent {
        INTENT_CALC    => Some("calc"),
        INTENT_TIME    => Some("time"),
        INTENT_CPU     => Some("cpu"),
        INTENT_WEATHER => Some("weather"),
        _ => None,
    }
}

/// Build tool parameters from an intent classification.
pub fn intent_to_params(intent: i32, arg_bytes: &[u8]) -> Vec<(&'static str, String)> {
    let arg_str = std::str::from_utf8(arg_bytes).unwrap_or("").trim();
    match intent {
        INTENT_CALC => {
            let expr = extract_math_expr(arg_str);
            vec![("expr", expr)]
        }
        _ => vec![],
    }
}

// ── Tool parameter building ─────────────────────────────────────────────────

/// Build tool parameters from a slash command ID and argument string.
pub fn build_tool_params(cmd_id: i32, arg: &str) -> Result<(&'static str, Vec<(&'static str, String)>)> {
    match cmd_id {
        CMD_TIME => Ok(("time", vec![])),
        CMD_CPU  => Ok(("cpu", vec![])),
        CMD_CALC => {
            if arg.is_empty() { return Err(Error::Tool("usage: /calc <expression>".into())); }
            Ok(("calc", vec![("expr", arg.to_string())]))
        }
        CMD_HTTP => {
            if arg.is_empty() { return Err(Error::Tool("usage: /http <url>".into())); }
            Ok(("http", vec![("url", arg.to_string())]))
        }
        CMD_SHELL => {
            if arg.is_empty() { return Err(Error::Tool("usage: /shell <command>".into())); }
            Ok(("shell", vec![("command", arg.to_string())]))
        }
        CMD_MEMORY => {
            if arg.is_empty() { return Err(Error::Tool("usage: /memory <action> [key] [value]".into())); }
            Ok(("memory", vec![("args", arg.to_string())]))
        }
        CMD_READ => {
            if arg.is_empty() { return Err(Error::Tool("usage: /read <path>".into())); }
            Ok(("read", vec![("path", arg.to_string())]))
        }
        CMD_WRITE => {
            if arg.is_empty() { return Err(Error::Tool("usage: /write <path> <content>".into())); }
            Ok(("write", vec![("args", arg.to_string())]))
        }
        CMD_LS => {
            let path = if arg.is_empty() { "." } else { arg };
            Ok(("ls", vec![("path", path.to_string())]))
        }
        CMD_JSON => {
            if arg.is_empty() { return Err(Error::Tool("usage: /json <keys|get|pretty> <input>".into())); }
            Ok(("json", vec![("args", arg.to_string())]))
        }
        CMD_TOKENS => {
            if arg.is_empty() { return Err(Error::Tool("usage: /tokens <text>".into())); }
            Ok(("tokens", vec![("text", arg.to_string())]))
        }
        CMD_BENCH => {
            if arg.is_empty() { return Err(Error::Tool("usage: /bench <target>".into())); }
            Ok(("bench", vec![("target", arg.to_string())]))
        }
        CMD_WEATHER => {
            if arg.is_empty() { return Err(Error::Tool("usage: /weather <city>".into())); }
            Ok(("weather", vec![("city", arg.to_string())]))
        }
        CMD_TRANSLATE => {
            if arg.is_empty() { return Err(Error::Tool("usage: /translate <lang> <text>".into())); }
            Ok(("translate", vec![("args", arg.to_string())]))
        }
        CMD_DEFINE => {
            if arg.is_empty() { return Err(Error::Tool("usage: /define <word>".into())); }
            Ok(("define", vec![("word", arg.to_string())]))
        }
        CMD_SUMMARIZE => {
            if arg.is_empty() { return Err(Error::Tool("usage: /summarize <url>".into())); }
            Ok(("summarize", vec![("url", arg.to_string())]))
        }
        CMD_GREP => {
            if arg.is_empty() { return Err(Error::Tool("usage: /grep <pattern> [path]".into())); }
            Ok(("grep", vec![("args", arg.to_string())]))
        }
        CMD_GIT => {
            if arg.is_empty() { return Err(Error::Tool("usage: /git <subcommand> [args]".into())); }
            Ok(("git", vec![("args", arg.to_string())]))
        }
        CMD_REMIND => {
            if arg.is_empty() { return Err(Error::Tool("usage: /remind <time> <message>".into())); }
            Ok(("remind", vec![("args", arg.to_string())]))
        }
        _ => Err(Error::Tool(format!("unknown tool command: {cmd_id}"))),
    }
}

// ── Math expression extraction ───────────────────────────────────────────────

/// Extract a math expression from a natural language string.
pub fn extract_math_expr(input: &str) -> String {
    let bytes = input.as_bytes();
    let start = match bytes.iter().position(|&b| b.is_ascii_digit()) {
        Some(pos) => pos,
        None => return input.to_string(),
    };
    let end = bytes.iter()
        .rposition(|&b| b.is_ascii_digit() || b == b')')
        .unwrap_or(start);
    let slice = &input[start..=end];
    slice.chars()
        .filter(|&c| c.is_ascii_digit() || "+-*/%^(). ".contains(c))
        .collect::<String>()
        .trim()
        .to_string()
}

// ── Eval expression (SIMD kernel) ────────────────────────────────────────────

const SCALE: i64 = 1_000_000;

/// Evaluate a math expression using the SIMD kernel.
/// Returns formatted string (kernel uses fixed-point with scale 1_000_000).
pub fn eval_expr(expr: &str) -> Result<String> {
    let bytes = expr.as_bytes();
    let mut result: i64 = 0;
    let mut error: i32 = 0;
    let mut val_stack = vec![0i64; 64];
    let mut op_stack = vec![0i32; 64];
    unsafe {
        ffi::eval_expr(
            bytes.as_ptr(), bytes.len() as i32,
            &mut result, &mut error,
            val_stack.as_mut_ptr(), op_stack.as_mut_ptr(),
        );
    }
    if error != 0 {
        return Err(Error::Tool(format!("expression error: code {error}")));
    }
    Ok(format_fixed_point(result))
}

fn format_fixed_point(value: i64) -> String {
    let negative = value < 0;
    let abs = value.unsigned_abs();
    let whole = abs / SCALE as u64;
    let frac = abs % SCALE as u64;
    if frac == 0 {
        if negative { format!("-{whole}") } else { format!("{whole}") }
    } else {
        let frac_str = format!("{:06}", frac);
        let trimmed = frac_str.trim_end_matches('0');
        if negative { format!("-{whole}.{trimmed}") } else { format!("{whole}.{trimmed}") }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_math_simple() {
        assert_eq!(extract_math_expr("6*7"), "6*7");
    }

    #[test]
    fn extract_math_natural_language() {
        assert_eq!(extract_math_expr("what is 6 * 7"), "6 * 7");
    }

    #[test]
    fn command_name_known() {
        assert_eq!(command_name(CMD_HELP), "help");
        assert_eq!(command_name(CMD_CALC), "calc");
    }

    #[test]
    fn command_name_unknown() {
        assert_eq!(command_name(999), "unknown");
    }

    #[test]
    fn build_tool_params_calc() {
        let (name, params) = build_tool_params(CMD_CALC, "2+3").unwrap();
        assert_eq!(name, "calc");
        assert_eq!(params[0], ("expr", "2+3".to_string()));
    }

    #[test]
    fn build_tool_params_empty_calc() {
        assert!(build_tool_params(CMD_CALC, "").is_err());
    }

    #[test]
    fn intent_to_tool_name_calc() {
        assert_eq!(intent_to_tool_name(INTENT_CALC), Some("calc"));
    }

    #[test]
    fn intent_to_tool_name_none() {
        assert_eq!(intent_to_tool_name(INTENT_NONE), None);
    }
}

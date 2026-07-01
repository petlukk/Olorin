//! Tool parameter building and invocation.
//!
//! Builds tool name + params from SIMD-classified command IDs and intent IDs.
//! Two-stage matching: SIMD hash → full name verification.
//! Synchronous.

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
pub const CMD_RUNE:      i32 = 28;
pub const CMD_THINK:     i32 = 29;
// Range
pub const CMD_TOOL_FIRST: i32 = CMD_TIME;
pub const CMD_TOOL_LAST:  i32 = CMD_RUNE;

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
    (CMD_RUNE, "rune"),     (CMD_THINK, "think"),
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

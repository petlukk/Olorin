use super::ffi;

/// Command IDs returned by the SIMD router.
/// Meta commands:
pub const CMD_HELP: i32 = 0;
pub const CMD_QUIT: i32 = 1;
pub const CMD_TOOLS: i32 = 2;
pub const CMD_CLEAR: i32 = 3;
pub const CMD_MODEL: i32 = 4;
pub const CMD_PROFILE: i32 = 5;
pub const CMD_TASKS: i32 = 18;
pub const CMD_RECALL: i32 = 19;
/// Tool commands:
pub const CMD_TIME: i32 = 6;
pub const CMD_CALC: i32 = 7;
pub const CMD_HTTP: i32 = 8;
pub const CMD_SHELL: i32 = 9;
pub const CMD_MEMORY: i32 = 10;
pub const CMD_READ: i32 = 11;
pub const CMD_WRITE: i32 = 12;
pub const CMD_LS: i32 = 13;
pub const CMD_JSON: i32 = 14;
pub const CMD_CPU: i32 = 15;
pub const CMD_TOKENS: i32 = 16;
pub const CMD_BENCH: i32 = 17;
pub const CMD_WEATHER: i32 = 20;
pub const CMD_TRANSLATE: i32 = 21;
pub const CMD_DEFINE: i32 = 22;
pub const CMD_SUMMARIZE: i32 = 23;
pub const CMD_GREP: i32 = 24;
pub const CMD_GIT: i32 = 25;
pub const CMD_REMIND: i32 = 26;
/// No match:
pub const CMD_NONE: i32 = -1;

/// First and last tool command IDs.
pub const CMD_TOOL_FIRST: i32 = CMD_TIME;
pub const CMD_TOOL_LAST: i32 = CMD_REMIND;

/// Full command names for two-stage verification.
const ALL_CMD_NAMES: &[(i32, &str)] = &[
    (CMD_HELP, "help"),
    (CMD_QUIT, "quit"),
    (CMD_TOOLS, "tools"),
    (CMD_CLEAR, "clear"),
    (CMD_MODEL, "model"),
    (CMD_PROFILE, "profile"),
    (CMD_TIME, "time"),
    (CMD_CALC, "calc"),
    (CMD_HTTP, "http"),
    (CMD_SHELL, "shell"),
    (CMD_MEMORY, "memory"),
    (CMD_READ, "read"),
    (CMD_WRITE, "write"),
    (CMD_LS, "ls"),
    (CMD_JSON, "json"),
    (CMD_CPU, "cpu"),
    (CMD_TOKENS, "tokens"),
    (CMD_BENCH, "bench"),
    (CMD_TASKS, "tasks"),
    (CMD_RECALL, "recall"),
    (CMD_WEATHER, "weather"),
    (CMD_TRANSLATE, "translate"),
    (CMD_DEFINE, "define"),
    (CMD_SUMMARIZE, "summarize"),
    (CMD_GREP, "grep"),
    (CMD_GIT, "git"),
    (CMD_REMIND, "remind"),
];

/// Match a slash command using the SIMD kernel.
/// Returns a command ID or CMD_NONE (-1).
pub fn match_command(text: &[u8]) -> i32 {
    let len = text.len();
    let mut result: i32 = CMD_NONE;
    unsafe {
        ffi::match_command(text.as_ptr(), len as i32, &mut result);
    }
    result
}

/// Two-stage match: SIMD hash + full name verification.
/// Verifies the full command name matches to handle potential hash collisions.
/// Returns (command_id, argument) where argument is the text after the command.
pub fn match_command_verified(text: &[u8]) -> (i32, &[u8]) {
    let cmd_id = match_command(text);
    if cmd_id == CMD_NONE {
        return (CMD_NONE, &[]);
    }

    // Verify the full command name for ALL commands
    for &(id, name) in ALL_CMD_NAMES {
        if id != cmd_id {
            continue;
        }
        // Expected: "/" + name, optionally followed by " " + args
        let expected_len = 1 + name.len(); // "/" + name
        if text.len() < expected_len {
            return (CMD_NONE, &[]);
        }
        if &text[1..expected_len] != name.as_bytes() {
            return (CMD_NONE, &[]);
        }
        // Must be exact or followed by space
        if text.len() == expected_len {
            return (cmd_id, &[]);
        }
        if text[expected_len] == b' ' {
            let arg_start = expected_len + 1;
            let arg = if arg_start < text.len() {
                &text[arg_start..]
            } else {
                &[]
            };
            return (cmd_id, arg);
        }
        // Not a match (e.g., "/timer" matched "/time" hash but isn't "/time")
        return (CMD_NONE, &[]);
    }

    (CMD_NONE, &[])
}

/// Convert command ID to name.
pub fn command_name(id: i32) -> Option<&'static str> {
    match id {
        CMD_HELP => Some("help"),
        CMD_QUIT => Some("quit"),
        CMD_TOOLS => Some("tools"),
        CMD_CLEAR => Some("clear"),
        CMD_MODEL => Some("model"),
        CMD_PROFILE => Some("profile"),
        CMD_TIME => Some("time"),
        CMD_CALC => Some("calc"),
        CMD_HTTP => Some("http"),
        CMD_SHELL => Some("shell"),
        CMD_MEMORY => Some("memory"),
        CMD_READ => Some("read"),
        CMD_WRITE => Some("write"),
        CMD_LS => Some("ls"),
        CMD_JSON => Some("json"),
        CMD_CPU => Some("cpu"),
        CMD_TOKENS => Some("tokens"),
        CMD_BENCH => Some("bench"),
        CMD_TASKS => Some("tasks"),
        CMD_RECALL => Some("recall"),
        CMD_WEATHER => Some("weather"),
        CMD_TRANSLATE => Some("translate"),
        CMD_DEFINE => Some("define"),
        CMD_SUMMARIZE => Some("summarize"),
        CMD_GREP => Some("grep"),
        CMD_GIT => Some("git"),
        CMD_REMIND => Some("remind"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- SIMD hash tests (meta commands) ---

    #[test]
    fn test_help() {
        assert_eq!(match_command(b"/help"), CMD_HELP);
    }

    #[test]
    fn test_quit() {
        assert_eq!(match_command(b"/quit"), CMD_QUIT);
    }

    #[test]
    fn test_tools() {
        assert_eq!(match_command(b"/tools"), CMD_TOOLS);
    }

    #[test]
    fn test_clear() {
        assert_eq!(match_command(b"/clear"), CMD_CLEAR);
    }

    #[test]
    fn test_model() {
        assert_eq!(match_command(b"/model"), CMD_MODEL);
    }

    #[test]
    fn test_profile() {
        assert_eq!(match_command(b"/profile"), CMD_PROFILE);
    }

    #[test]
    fn test_no_match() {
        assert_eq!(match_command(b"/unknown"), CMD_NONE);
    }

    #[test]
    fn test_too_short() {
        assert_eq!(match_command(b"/h"), CMD_NONE);
    }

    #[test]
    fn test_no_slash() {
        assert_eq!(match_command(b"help"), CMD_NONE);
    }

    #[test]
    fn test_empty() {
        assert_eq!(match_command(b""), CMD_NONE);
    }

    // --- SIMD hash tests for tool commands ---

    #[test]
    fn test_time() {
        assert_eq!(match_command(b"/time"), CMD_TIME);
    }

    #[test]
    fn test_calc() {
        assert_eq!(match_command(b"/calc"), CMD_CALC);
    }

    #[test]
    fn test_http() {
        assert_eq!(match_command(b"/http"), CMD_HTTP);
    }

    #[test]
    fn test_shell() {
        assert_eq!(match_command(b"/shell"), CMD_SHELL);
    }

    #[test]
    fn test_memory() {
        assert_eq!(match_command(b"/memory"), CMD_MEMORY);
    }

    #[test]
    fn test_read() {
        assert_eq!(match_command(b"/read"), CMD_READ);
    }

    #[test]
    fn test_write() {
        assert_eq!(match_command(b"/write"), CMD_WRITE);
    }

    #[test]
    fn test_ls() {
        assert_eq!(match_command(b"/ls"), CMD_LS);
    }

    #[test]
    fn test_json() {
        assert_eq!(match_command(b"/json"), CMD_JSON);
    }

    #[test]
    fn test_cpu() {
        assert_eq!(match_command(b"/cpu"), CMD_CPU);
    }

    #[test]
    fn test_tokens() {
        assert_eq!(match_command(b"/tokens"), CMD_TOKENS);
    }

    #[test]
    fn test_bench() {
        assert_eq!(match_command(b"/bench"), CMD_BENCH);
    }

    // --- Two-stage verified match tests ---

    #[test]
    fn test_tasks() {
        assert_eq!(match_command(b"/tasks"), CMD_TASKS);
    }

    #[test]
    fn test_verified_time_no_arg() {
        let (id, arg) = match_command_verified(b"/time");
        assert_eq!(id, CMD_TIME);
        assert!(arg.is_empty());
    }

    #[test]
    fn test_verified_calc_with_arg() {
        let (id, arg) = match_command_verified(b"/calc 2 + 3");
        assert_eq!(id, CMD_CALC);
        assert_eq!(arg, b"2 + 3");
    }

    #[test]
    fn test_verified_shell_with_arg() {
        let (id, arg) = match_command_verified(b"/shell ls -la");
        assert_eq!(id, CMD_SHELL);
        assert_eq!(arg, b"ls -la");
    }

    #[test]
    fn test_verified_http_with_url() {
        let (id, arg) = match_command_verified(b"/http https://example.com");
        assert_eq!(id, CMD_HTTP);
        assert_eq!(arg, b"https://example.com");
    }

    #[test]
    fn test_verified_memory_with_arg() {
        let (id, arg) = match_command_verified(b"/memory list");
        assert_eq!(id, CMD_MEMORY);
        assert_eq!(arg, b"list");
    }

    #[test]
    fn test_verified_read_with_path() {
        let (id, arg) = match_command_verified(b"/read /etc/hosts");
        assert_eq!(id, CMD_READ);
        assert_eq!(arg, b"/etc/hosts");
    }

    #[test]
    fn test_verified_write_with_args() {
        let (id, arg) = match_command_verified(b"/write file.txt hello");
        assert_eq!(id, CMD_WRITE);
        assert_eq!(arg, b"file.txt hello");
    }

    #[test]
    fn test_verified_ls_no_arg() {
        let (id, arg) = match_command_verified(b"/ls");
        assert_eq!(id, CMD_LS);
        assert!(arg.is_empty());
    }

    #[test]
    fn test_verified_ls_with_arg() {
        let (id, arg) = match_command_verified(b"/ls /tmp");
        assert_eq!(id, CMD_LS);
        assert_eq!(arg, b"/tmp");
    }

    #[test]
    fn test_verified_json_with_arg() {
        let (id, arg) = match_command_verified(b"/json keys {}");
        assert_eq!(id, CMD_JSON);
        assert_eq!(arg, b"keys {}");
    }

    #[test]
    fn test_verified_cpu_no_arg() {
        let (id, arg) = match_command_verified(b"/cpu");
        assert_eq!(id, CMD_CPU);
        assert!(arg.is_empty());
    }

    #[test]
    fn test_verified_tokens_with_arg() {
        let (id, arg) = match_command_verified(b"/tokens hello world");
        assert_eq!(id, CMD_TOKENS);
        assert_eq!(arg, b"hello world");
    }

    #[test]
    fn test_verified_bench_with_arg() {
        let (id, arg) = match_command_verified(b"/bench safety");
        assert_eq!(id, CMD_BENCH);
        assert_eq!(arg, b"safety");
    }

    #[test]
    fn test_verified_rejects_longer_name() {
        let (id, _) = match_command_verified(b"/timer");
        assert_eq!(id, CMD_NONE);
    }

    #[test]
    fn test_verified_meta_commands_pass_through() {
        let (id, _) = match_command_verified(b"/help");
        assert_eq!(id, CMD_HELP);
    }

    #[test]
    fn test_verified_calc_space_no_arg() {
        let (id, arg) = match_command_verified(b"/calc ");
        assert_eq!(id, CMD_CALC);
        assert!(arg.is_empty());
    }

    #[test]
    fn test_weather() {
        assert_eq!(match_command(b"/weather"), CMD_WEATHER);
    }

    #[test]
    fn test_verified_weather_with_arg() {
        let (id, arg) = match_command_verified(b"/weather London");
        assert_eq!(id, CMD_WEATHER);
        assert_eq!(arg, b"London");
    }

    #[test]
    fn test_translate() {
        assert_eq!(match_command(b"/translate"), CMD_TRANSLATE);
    }

    #[test]
    fn test_verified_translate_with_arg() {
        let (id, arg) = match_command_verified(b"/translate Spanish hello");
        assert_eq!(id, CMD_TRANSLATE);
        assert_eq!(arg, b"Spanish hello");
    }

    #[test]
    fn test_define() {
        assert_eq!(match_command(b"/define"), CMD_DEFINE);
    }

    #[test]
    fn test_verified_define_with_arg() {
        let (id, arg) = match_command_verified(b"/define hello");
        assert_eq!(id, CMD_DEFINE);
        assert_eq!(arg, b"hello");
    }

    #[test]
    fn test_summarize() {
        assert_eq!(match_command(b"/summarize"), CMD_SUMMARIZE);
    }

    #[test]
    fn test_verified_summarize_with_arg() {
        let (id, arg) = match_command_verified(b"/summarize https://example.com");
        assert_eq!(id, CMD_SUMMARIZE);
        assert_eq!(arg, b"https://example.com");
    }

    #[test]
    fn test_grep() {
        assert_eq!(match_command(b"/grep"), CMD_GREP);
    }

    #[test]
    fn test_verified_grep_with_arg() {
        let (id, arg) = match_command_verified(b"/grep TODO src/");
        assert_eq!(id, CMD_GREP);
        assert_eq!(arg, b"TODO src/");
    }

    #[test]
    fn test_git() {
        assert_eq!(match_command(b"/git"), CMD_GIT);
    }

    #[test]
    fn test_verified_git_with_arg() {
        let (id, arg) = match_command_verified(b"/git log --oneline -5");
        assert_eq!(id, CMD_GIT);
        assert_eq!(arg, b"log --oneline -5");
    }

    #[test]
    fn test_remind() {
        assert_eq!(match_command(b"/remind"), CMD_REMIND);
    }

    #[test]
    fn test_verified_remind_with_arg() {
        let (id, arg) = match_command_verified(b"/remind 5m check deploy");
        assert_eq!(id, CMD_REMIND);
        assert_eq!(arg, b"5m check deploy");
    }
}

use crate::kernels::command_router;

/// Command enum for cleaner matching.
#[derive(Debug, PartialEq)]
pub enum Command {
    // Meta commands
    Help,
    Quit,
    Tools,
    Clear,
    Model,
    Profile,
    Tasks,
    // Tool commands (with optional argument)
    Time,
    Calc(String),
    Http(String),
    Shell(String),
    Memory(String),
    Read(String),
    Write(String),
    Ls(String),
    Json(String),
    Cpu,
    Tokens(String),
    Bench(String),
    // Fallback
    Unknown(String),
    NotACommand,
}

/// Parse a command from input text using the SIMD kernel + verification.
pub fn parse_command(text: &str, prefix: &str) -> Command {
    if !text.starts_with(prefix) {
        return Command::NotACommand;
    }

    let (cmd_id, arg) = command_router::match_command_verified(text.as_bytes());
    let arg_str = || String::from_utf8_lossy(arg).into_owned();

    match cmd_id {
        command_router::CMD_HELP => Command::Help,
        command_router::CMD_QUIT => Command::Quit,
        command_router::CMD_TOOLS => Command::Tools,
        command_router::CMD_CLEAR => Command::Clear,
        command_router::CMD_MODEL => Command::Model,
        command_router::CMD_PROFILE => Command::Profile,
        command_router::CMD_TASKS => Command::Tasks,
        command_router::CMD_TIME => Command::Time,
        command_router::CMD_CALC => Command::Calc(arg_str()),
        command_router::CMD_HTTP => Command::Http(arg_str()),
        command_router::CMD_SHELL => Command::Shell(arg_str()),
        command_router::CMD_MEMORY => Command::Memory(arg_str()),
        command_router::CMD_READ => Command::Read(arg_str()),
        command_router::CMD_WRITE => Command::Write(arg_str()),
        command_router::CMD_LS => Command::Ls(arg_str()),
        command_router::CMD_JSON => Command::Json(arg_str()),
        command_router::CMD_CPU => Command::Cpu,
        command_router::CMD_TOKENS => Command::Tokens(arg_str()),
        command_router::CMD_BENCH => Command::Bench(arg_str()),
        _ => Command::Unknown(text.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_help() {
        assert_eq!(parse_command("/help", "/"), Command::Help);
    }

    #[test]
    fn test_parse_quit() {
        assert_eq!(parse_command("/quit", "/"), Command::Quit);
    }

    #[test]
    fn test_not_a_command() {
        assert_eq!(parse_command("hello", "/"), Command::NotACommand);
    }

    #[test]
    fn test_unknown_command() {
        assert_eq!(
            parse_command("/foobar", "/"),
            Command::Unknown("/foobar".to_string())
        );
    }

    #[test]
    fn test_parse_time() {
        assert_eq!(parse_command("/time", "/"), Command::Time);
    }

    #[test]
    fn test_parse_calc_with_arg() {
        assert_eq!(
            parse_command("/calc 2+3", "/"),
            Command::Calc("2+3".to_string())
        );
    }

    #[test]
    fn test_parse_shell_with_arg() {
        assert_eq!(
            parse_command("/shell ls -la", "/"),
            Command::Shell("ls -la".to_string())
        );
    }

    #[test]
    fn test_parse_memory_list() {
        assert_eq!(
            parse_command("/memory list", "/"),
            Command::Memory("list".to_string())
        );
    }

    #[test]
    fn test_parse_ls() {
        assert_eq!(parse_command("/ls", "/"), Command::Ls(String::new()));
    }

    #[test]
    fn test_parse_ls_with_path() {
        assert_eq!(
            parse_command("/ls /tmp", "/"),
            Command::Ls("/tmp".to_string())
        );
    }

    #[test]
    fn test_parse_cpu() {
        assert_eq!(parse_command("/cpu", "/"), Command::Cpu);
    }

    #[test]
    fn test_parse_read_with_path() {
        assert_eq!(
            parse_command("/read file.txt", "/"),
            Command::Read("file.txt".to_string())
        );
    }

    #[test]
    fn test_parse_bench() {
        assert_eq!(
            parse_command("/bench safety", "/"),
            Command::Bench("safety".to_string())
        );
    }

    #[test]
    fn test_parse_tokens() {
        assert_eq!(
            parse_command("/tokens hello", "/"),
            Command::Tokens("hello".to_string())
        );
    }

    #[test]
    fn test_pipeline_detection() {
        // A pipeline should still parse the first command
        let text = "/shell ls | /tokens";
        assert!(text.contains(" | /"));
        // First segment routes correctly
        let (id, _) = command_router::match_command_verified(b"/shell ls");
        assert_eq!(id, command_router::CMD_SHELL);
    }

    #[test]
    fn test_pipeline_second_stage() {
        let (id, arg) = command_router::match_command_verified(b"/tokens");
        assert_eq!(id, command_router::CMD_TOKENS);
        assert!(arg.is_empty());
    }

    #[test]
    fn test_parse_tasks() {
        assert_eq!(parse_command("/tasks", "/"), Command::Tasks);
    }

    #[test]
    fn test_parse_json() {
        assert_eq!(
            parse_command("/json keys {}", "/"),
            Command::Json("keys {}".to_string())
        );
    }
}

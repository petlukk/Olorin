use olorin::interface::exec::{run, shell, spawn};

#[test]
fn test_run_echo() {
    let out = run(&["echo", "hello"]).unwrap();
    assert_eq!(out.exit_code, 0);
    assert_eq!(out.stdout.trim_ascii(), b"hello");
}

#[test]
fn test_run_exit_code() {
    let out = run(&["sh", "-c", "exit 42"]).unwrap();
    assert_eq!(out.exit_code, 42);
}

#[test]
fn test_shell_stdout() {
    let out = shell("echo world").unwrap();
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.starts_with(b"world"));
}

#[test]
fn test_shell_stderr() {
    let out = shell("echo err >&2").unwrap();
    assert!(out.stderr.starts_with(b"err"));
}

#[test]
fn test_run_empty_argv_error() {
    let result = run(&[]);
    assert!(result.is_err());
}

#[test]
fn test_spawn_empty_argv_error() {
    let result = spawn(&[]);
    assert!(result.is_err());
}

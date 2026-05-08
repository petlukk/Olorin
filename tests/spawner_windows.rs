#![cfg(windows)]

use olorin::interface::spawner::default_spawner;

#[test]
fn test_run_echo() {
    let s = default_spawner();
    let out = s.run(&["cmd", "/C", "echo hello"]).unwrap();
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.starts_with(b"hello"));
}

#[test]
fn test_run_exit_code() {
    let s = default_spawner();
    let out = s.shell("exit 42").unwrap();
    assert_eq!(out.exit_code, 42);
}

#[test]
fn test_shell_stdout() {
    let s = default_spawner();
    let out = s.shell("echo world").unwrap();
    assert_eq!(out.exit_code, 0);
    assert!(out.stdout.starts_with(b"world"));
}

#[test]
fn test_shell_stderr() {
    let s = default_spawner();
    let out = s.shell("echo err 1>&2").unwrap();
    assert!(out.stderr.starts_with(b"err"));
}

#[test]
fn test_run_empty_argv_error() {
    let s = default_spawner();
    assert!(s.run(&[]).is_err());
}

#[test]
fn test_spawn_empty_argv_error() {
    let s = default_spawner();
    assert!(s.spawn(&[]).is_err());
}

#[test]
fn test_spawn_line_io_strips_crlf() {
    // findstr /N "^" copies stdin to stdout with line numbers — predictable
    // way to verify the read_line CRLF strip without depending on a Go bridge.
    let s = default_spawner();
    let child = s.spawn(&["findstr", "/N", "^"]).unwrap();
    child.write_line("alpha").unwrap();
    child.write_line("beta").unwrap();

    let mut buf = String::new();
    let n1 = child.read_line(&mut buf).unwrap();
    assert!(buf.ends_with("alpha"));
    assert_eq!(n1, buf.len());
    assert!(!buf.ends_with('\r'));
    assert!(!buf.ends_with('\n'));

    let n2 = child.read_line(&mut buf).unwrap();
    assert!(buf.ends_with("beta"));
    assert_eq!(n2, buf.len());

    let _ = child.wait();
}
